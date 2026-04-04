from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fusion_depth.metrics import compute_depth_metrics, tensor_to_metric_dict
from fusion_depth.model import FusionDepthModel
from fusion_depth.runtime import (
    cleanup_distributed,
    get_autocast_dtype,
    init_distributed,
    instantiate_from_config,
    is_main_process,
    load_model_checkpoint,
    move_batch_to_device,
    seed_everything,
)
from fusion_depth.visualization import save_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline evaluation for fusion depth checkpoints.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(output_dir: Path, main_process: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if main_process:
        handlers.append(logging.FileHandler(output_dir / "eval.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def build_dataloader(dataset, batch_size: int, num_workers: int, distributed: bool):
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )
    return loader


def reduce_sum(tensor: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    distributed, rank, world_size, local_rank, device = init_distributed(
        timeout_minutes=int(cfg["eval"].get("ddp_timeout_minutes", 180))
    )
    output_dir = Path(cfg["output_dir"])
    setup_logging(output_dir, is_main_process(rank))
    seed_everything(int(cfg.get("seed", 42)) + rank)

    eval_dataset = instantiate_from_config(cfg["data"]["eval"])
    eval_loader = build_dataloader(
        eval_dataset,
        batch_size=int(cfg["eval"].get("batch_size", 1)),
        num_workers=int(cfg["eval"].get("num_workers", 2)),
        distributed=distributed,
    )

    model = FusionDepthModel(cfg["model"]).to(device)
    checkpoint_path = args.checkpoint or cfg["eval"]["checkpoint_path"]
    load_model_checkpoint(model, checkpoint_path, strict=False)
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    model.eval()

    amp_enabled = bool(cfg["eval"].get("amp", True)) and device.type == "cuda"
    amp_dtype = get_autocast_dtype(cfg["eval"].get("amp_dtype", "bfloat16"))
    max_batches = cfg["eval"].get("max_batches")
    save_visualizations = bool(cfg["eval"].get("save_visualizations", False))
    max_visualizations = int(cfg["eval"].get("max_visualizations", 20))
    vis_saved = 0

    metric_sums = torch.zeros(8, device=device)
    processed_batches = 0
    for batch_idx, batch in enumerate(eval_loader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["images"])

        metric_sums += compute_depth_metrics(
            outputs["pred_depth"],
            batch["depths"],
            batch["valid_masks"],
            min_depth=float(cfg["loss"].get("min_depth", 1e-3)),
            max_depth=float(cfg["loss"].get("max_depth", 200.0)),
            min_valid_pixels=int(cfg["eval"].get("min_valid_pixels", 64)),
        )
        processed_batches += 1

        if save_visualizations and is_main_process(rank) and vis_saved < max_visualizations:
            vis_batch = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in batch.items()
            }
            vis_outputs = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in outputs.items()
            }
            save_visualization(
                vis_batch,
                vis_outputs,
                output_dir / "visualizations" / f"batch_{batch_idx:06d}.png",
                max_frames=int(cfg["eval"].get("max_visualization_frames", 6)),
            )
            vis_saved += 1

    metric_sums = reduce_sum(metric_sums, distributed)
    metrics = tensor_to_metric_dict(metric_sums)

    if is_main_process(rank):
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        logging.info("evaluation finished on %d batches | %s", processed_batches, metrics)
        logging.info("metrics saved to %s", metrics_path)

    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()

