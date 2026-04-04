from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fusion_depth.data import SequenceDepthDataset
from fusion_depth.losses import FusionDepthLoss
from fusion_depth.metrics import compute_depth_metrics, tensor_to_metric_dict
from fusion_depth.model import FusionDepthModel
from fusion_depth.visualization import save_visualization


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a VGGT + DA3 fusion depth model.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(output_dir: Path, is_main_process: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if is_main_process:
        handlers.append(logging.FileHandler(output_dir / "train.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, int, torch.device]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        return True, rank, world_size, local_rank, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    return False, 0, 1, 0, device


def barrier_if_needed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.barrier()


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def reduce_mean(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    value /= dist.get_world_size()
    return value


def reduce_sum(value: torch.Tensor, distributed: bool) -> torch.Tensor:
    if not distributed:
        return value
    value = value.clone()
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def build_dataloader(
    dataset: SequenceDepthDataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
) -> Tuple[DataLoader, DistributedSampler | None]:
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        drop_last=drop_last,
    )
    return loader, sampler


def get_autocast_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name.lower() == "float16":
        return torch.float16
    return torch.bfloat16


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def build_optimizer(model: FusionDepthModel, cfg: Dict) -> torch.optim.Optimizer:
    base_lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.0))
    pretrained_lr_mult = float(cfg["optimizer"].get("pretrained_lr_mult", 0.1))
    param_groups = model.parameter_groups(
        base_lr=base_lr,
        weight_decay=weight_decay,
        pretrained_lr_mult=pretrained_lr_mult,
    )
    betas = tuple(float(x) for x in cfg["optimizer"].get("betas", [0.9, 0.999]))
    return torch.optim.AdamW(param_groups, betas=betas)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, total_steps: int):
    warmup_steps = int(cfg["scheduler"].get("warmup_steps", 0))
    min_lr_ratio = float(cfg["scheduler"].get("min_lr_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def maybe_create_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return NullSummaryWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logging.warning("tensorboard is not installed; scalar logs will only go to stdout.")
        return NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    epoch: int,
    best_abs_rel: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "epoch": epoch,
            "best_abs_rel": best_abs_rel,
        },
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Tuple[int, int, float]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    unwrap_model(model).load_state_dict(checkpoint["model"], strict=False)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    step = int(checkpoint.get("step", 0))
    epoch = int(checkpoint.get("epoch", 0))
    best_abs_rel = float(checkpoint.get("best_abs_rel", float("inf")))
    return step, epoch, best_abs_rel


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: FusionDepthLoss,
    device: torch.device,
    cfg: Dict,
    distributed: bool,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.eval()
    metric_sums = torch.zeros(8, device=device)
    loss_sum = torch.zeros(1, device=device)
    batch_count = 0
    max_batches = cfg["eval"].get("max_batches")

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batch = move_batch_to_device(batch, device)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            outputs = model(batch["images"])
            loss_dict = criterion(outputs, batch)
        metric_sums += compute_depth_metrics(
            outputs["pred_depth"],
            batch["depths"],
            batch["valid_masks"],
            min_depth=float(cfg["loss"].get("min_depth", 1e-3)),
            max_depth=float(cfg["loss"].get("max_depth", 200.0)),
            min_valid_pixels=int(cfg["eval"].get("min_valid_pixels", 64)),
        )
        loss_sum += loss_dict["loss"].detach()
        batch_count += 1

    metric_sums = reduce_sum(metric_sums, distributed)
    loss_sum = reduce_sum(loss_sum, distributed)
    batch_counter = reduce_sum(torch.tensor([batch_count], dtype=torch.float32, device=device), distributed)

    metrics = tensor_to_metric_dict(metric_sums)
    metrics["loss"] = (loss_sum / batch_counter.clamp_min(1)).item()
    return metrics


def format_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(
        [
            f"loss={metrics['loss']:.4f}",
            f"abs_rel={metrics['abs_rel']:.4f}",
            f"sq_rel={metrics['sq_rel']:.4f}",
            f"rmse={metrics['rmse']:.4f}",
            f"rmse_log={metrics['rmse_log']:.4f}",
            f"delta1={metrics['delta1']:.4f}",
            f"delta2={metrics['delta2']:.4f}",
            f"delta3={metrics['delta3']:.4f}",
        ]
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    distributed, rank, world_size, local_rank, device = init_distributed()
    output_dir = Path(cfg["output_dir"])
    setup_logging(output_dir, is_main_process(rank))

    seed_everything(int(cfg.get("seed", 42)) + rank)
    logging.info("Using device=%s rank=%d world_size=%d local_rank=%d", device, rank, world_size, local_rank)

    train_dataset = SequenceDepthDataset(
        manifest_path=cfg["data"]["train_manifest"],
        image_size=cfg["data"]["image_size"],
        sequence_length=int(cfg["data"].get("sequence_length", 6)),
        sequence_stride=int(cfg["data"].get("sequence_stride", 1)),
        depth_scale=float(cfg["data"].get("depth_scale", 1.0)),
        npz_key=cfg["data"].get("npz_key", "depth"),
        random_horizontal_flip=bool(cfg["data"].get("random_horizontal_flip", False)),
        max_samples=cfg["data"].get("max_train_samples"),
    )
    val_dataset = SequenceDepthDataset(
        manifest_path=cfg["data"]["val_manifest"],
        image_size=cfg["data"]["image_size"],
        sequence_length=int(cfg["data"].get("sequence_length", 6)),
        sequence_stride=int(cfg["data"].get("eval_sequence_stride", cfg["data"].get("sequence_stride", 1))),
        depth_scale=float(cfg["data"].get("depth_scale", 1.0)),
        npz_key=cfg["data"].get("npz_key", "depth"),
        random_horizontal_flip=False,
        max_samples=cfg["data"].get("max_val_samples"),
    )

    train_loader, train_sampler = build_dataloader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        distributed=distributed,
        shuffle=True,
        drop_last=True,
    )
    val_loader, val_sampler = build_dataloader(
        val_dataset,
        batch_size=int(cfg["eval"].get("batch_size", 1)),
        num_workers=int(cfg["eval"].get("num_workers", 2)),
        distributed=distributed,
        shuffle=False,
        drop_last=False,
    )

    model = FusionDepthModel(cfg["model"]).to(device)
    optimizer = build_optimizer(model, cfg)
    total_steps = int(cfg["train"]["max_steps"])
    scheduler = build_scheduler(optimizer, cfg, total_steps)

    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = get_autocast_dtype(cfg["train"].get("amp_dtype", "bfloat16"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = FusionDepthLoss(cfg["loss"])
    writer = maybe_create_writer(output_dir / "tensorboard", enabled=is_main_process(rank))

    start_step = 0
    start_epoch = 0
    best_abs_rel = float("inf")

    resume_path = args.resume or cfg["train"].get("resume_from")
    if resume_path:
        start_step, start_epoch, best_abs_rel = load_checkpoint(
            resume_path,
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        logging.info("Resumed from %s at step=%d epoch=%d", resume_path, start_step, start_epoch)

    if distributed:
        ddp_model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
    else:
        ddp_model = model

    if args.eval_only:
        metrics = evaluate(
            ddp_model,
            val_loader,
            criterion,
            device,
            cfg,
            distributed,
            amp_enabled,
            amp_dtype,
        )
        if is_main_process(rank):
            logging.info("Eval-only metrics: %s", format_metrics(metrics))
        cleanup_distributed(distributed)
        return

    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    clip_grad_norm = float(cfg["train"].get("clip_grad_norm", 0.0))
    log_every_steps = int(cfg["train"].get("log_every_steps", 20))
    vis_every_steps = int(cfg["train"].get("vis_every_steps", 500))
    eval_every_steps = int(cfg["train"].get("eval_every_steps", 2000))
    save_every_steps = int(cfg["train"].get("save_every_steps", 2000))

    step = start_step
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    start_time = time.time()

    while step < total_steps:
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)
        ddp_model.train()

        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = ddp_model(batch["images"])
                loss_dict = criterion(outputs, batch)
                loss = loss_dict["loss"] / grad_accum_steps

            scaler.scale(loss).backward()

            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == len(train_loader) - 1)
            if not should_step:
                continue

            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            reduced_loss = reduce_mean(loss_dict["loss"].detach(), distributed)

            if step % log_every_steps == 0:
                reduced_loss_dict = {
                    key: reduce_mean(loss_dict[key].detach(), distributed)
                    for key in sorted(loss_dict.keys())
                }

            if step % log_every_steps == 0 and is_main_process(rank):
                elapsed = time.time() - start_time
                lr = scheduler.get_last_lr()[0]
                logging.info(
                    "step=%d/%d epoch=%d loss=%.4f lr=%.6e time=%.1fs",
                    step,
                    total_steps,
                    epoch,
                    reduced_loss.item(),
                    lr,
                    elapsed,
                )
                writer.add_scalar("train/loss", reduced_loss.item(), step)
                writer.add_scalar("train/lr", lr, step)
                for key, value in reduced_loss_dict.items():
                    writer.add_scalar(f"train/{key}", value.item(), step)

            if step % vis_every_steps == 0:
                barrier_if_needed(distributed)
                if is_main_process(rank):
                    vis_outputs = {key: value.detach().cpu() if torch.is_tensor(value) else value for key, value in outputs.items()}
                    vis_batch = {
                        key: value.detach().cpu() if torch.is_tensor(value) else value
                        for key, value in batch.items()
                    }
                    save_visualization(
                        vis_batch,
                        vis_outputs,
                        output_dir / "visualizations" / f"step_{step:07d}.png",
                        max_frames=int(cfg["train"].get("max_visualization_frames", 6)),
                    )
                barrier_if_needed(distributed)

            if step % eval_every_steps == 0 or step == total_steps:
                barrier_if_needed(distributed)
                if distributed and val_sampler is not None:
                    val_sampler.set_epoch(step)
                metrics = evaluate(
                    ddp_model,
                    val_loader,
                    criterion,
                    device,
                    cfg,
                    distributed,
                    amp_enabled,
                    amp_dtype,
                )
                if is_main_process(rank):
                    logging.info("validation step=%d | %s", step, format_metrics(metrics))
                    for key, value in metrics.items():
                        writer.add_scalar(f"val/{key}", value, step)
                    metrics_path = output_dir / "metrics" / f"step_{step:07d}.json"
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(metrics_path, "w", encoding="utf-8") as file:
                        json.dump(metrics, file, indent=2)

                    if metrics["abs_rel"] < best_abs_rel:
                        best_abs_rel = metrics["abs_rel"]
                        save_checkpoint(
                            output_dir / "checkpoints" / "best.pt",
                            ddp_model,
                            optimizer,
                            scheduler,
                            scaler,
                            step,
                            epoch,
                            best_abs_rel,
                        )
                barrier_if_needed(distributed)

            if step % save_every_steps == 0 or step == total_steps:
                barrier_if_needed(distributed)
                if is_main_process(rank):
                    save_checkpoint(
                        output_dir / "checkpoints" / "latest.pt",
                        ddp_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        epoch,
                        best_abs_rel,
                    )
                    save_checkpoint(
                        output_dir / "checkpoints" / f"step_{step:07d}.pt",
                        ddp_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        epoch,
                        best_abs_rel,
                    )
                barrier_if_needed(distributed)

            if step >= total_steps:
                break

        epoch += 1

    writer.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
