from __future__ import annotations

import argparse
import logging
import math
import threading
import time
from pathlib import Path
from typing import Dict

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from fusion_depth.losses import FusionDepthLoss
from fusion_depth.model import FusionDepthModel
from fusion_depth.runtime import (
    build_checkpoint_payload,
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


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


class AsyncSaver:
    def __init__(self, name: str) -> None:
        self.name = name
        self.thread: threading.Thread | None = None

    def join(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()

    def launch(self, target, *, description: str, kwargs: Dict) -> bool:
        if self.thread is not None and self.thread.is_alive():
            logging.warning("%s save still running; skip %s", self.name, description)
            return False

        def _worker() -> None:
            target(**kwargs)

        self.thread = threading.Thread(
            target=_worker,
            name=f"{self.name}-{description}",
            daemon=True,
        )
        self.thread.start()
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified fusion depth trainer without inline eval.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint path to resume from.")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def setup_logging(output_dir: Path, main_process: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler()]
    if main_process:
        handlers.append(logging.FileHandler(output_dir / "train.log"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def maybe_create_writer(log_dir: Path, enabled: bool):
    if not enabled:
        return NullSummaryWriter()
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logging.warning("tensorboard is not installed; scalar logs will only go to stdout.")
        return NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


def build_dataloader(
    dataset,
    batch_size: int,
    num_workers: int,
    distributed: bool,
    shuffle: bool,
    drop_last: bool,
):
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


def build_optimizer(model: FusionDepthModel, cfg: Dict) -> torch.optim.Optimizer:
    base_lr = float(cfg["optimizer"]["lr"])
    weight_decay = float(cfg["optimizer"].get("weight_decay", 0.0))
    pretrained_lr_mult = float(cfg["optimizer"].get("pretrained_lr_mult", 0.1))
    betas = tuple(float(x) for x in cfg["optimizer"].get("betas", [0.9, 0.999]))
    param_groups = model.parameter_groups(
        base_lr=base_lr,
        weight_decay=weight_decay,
        pretrained_lr_mult=pretrained_lr_mult,
    )
    return torch.optim.AdamW(param_groups, betas=betas)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict, total_steps: int):
    warmup_steps = int(cfg["scheduler"].get("warmup_steps", 0))
    min_lr_ratio = float(cfg["scheduler"].get("min_lr_ratio", 0.1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_visualization_task(batch: Dict, outputs: Dict, save_path: Path, max_frames: int) -> None:
    save_visualization(batch, outputs, save_path, max_frames=max_frames)


def save_checkpoint_task(payload: Dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    distributed, rank, world_size, local_rank, device = init_distributed(
        timeout_minutes=int(cfg["train"].get("ddp_timeout_minutes", 180))
    )
    output_dir = Path(cfg["output_dir"])
    setup_logging(output_dir, is_main_process(rank))

    seed_everything(int(cfg.get("seed", 42)) + rank)
    logging.info("device=%s rank=%d world_size=%d local_rank=%d", device, rank, world_size, local_rank)

    train_dataset = instantiate_from_config(cfg["data"]["train"])
    train_loader, train_sampler = build_dataloader(
        train_dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["train"].get("num_workers", 4)),
        distributed=distributed,
        shuffle=True,
        drop_last=True,
    )

    model = FusionDepthModel(cfg["model"]).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, int(cfg["train"]["max_steps"]))
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = get_autocast_dtype(cfg["train"].get("amp_dtype", "bfloat16"))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    criterion = FusionDepthLoss(cfg["loss"])
    writer = maybe_create_writer(output_dir / "tensorboard", enabled=is_main_process(rank))

    start_step = 0
    start_epoch = 0
    resume_path = args.resume or cfg["train"].get("resume_from")
    if resume_path:
        start_step, start_epoch = load_model_checkpoint(
            model,
            resume_path,
            optimizer=optimizer,
            scaler=scaler,
            strict=False,
        )
        logging.info("resumed from %s at step=%d epoch=%d", resume_path, start_step, start_epoch)

    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    grad_accum_steps = int(cfg["train"].get("grad_accum_steps", 1))
    clip_grad_norm = float(cfg["train"].get("clip_grad_norm", 0.0))
    log_every_steps = int(cfg["train"].get("log_every_steps", 20))
    vis_every_steps = int(cfg["train"].get("vis_every_steps", 500))
    save_every_steps = int(cfg["train"].get("save_every_steps", 2000))
    save_optimizer_state = bool(cfg["train"].get("save_optimizer_state", False))
    max_steps = int(cfg["train"]["max_steps"])

    vis_saver = AsyncSaver("visualization")
    ckpt_saver = AsyncSaver("checkpoint")

    step = start_step
    epoch = start_epoch
    optimizer.zero_grad(set_to_none=True)
    train_start_time = time.time()

    while step < max_steps:
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
                outputs = model(batch["images"])
                loss_dict = criterion(outputs, batch)
                loss = loss_dict["loss"] / grad_accum_steps

            scaler.scale(loss).backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx == len(train_loader) - 1)
            if not should_step:
                continue

            if clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if is_main_process(rank) and step % log_every_steps == 0:
                elapsed = time.time() - train_start_time
                lr = scheduler.get_last_lr()[0]
                logging.info(
                    "step=%d/%d epoch=%d loss=%.4f lr=%.6e elapsed=%.1fs",
                    step,
                    max_steps,
                    epoch,
                    float(loss_dict["loss"].detach().cpu()),
                    lr,
                    elapsed,
                )
                writer.add_scalar("train/loss", float(loss_dict["loss"].detach().cpu()), step)
                writer.add_scalar("train/lr", lr, step)
                for key, value in loss_dict.items():
                    writer.add_scalar(f"train/{key}", float(value.detach().cpu()), step)

            if is_main_process(rank) and vis_every_steps > 0 and step % vis_every_steps == 0:
                vis_batch = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                vis_outputs = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in outputs.items()
                }
                vis_saver.launch(
                    save_visualization_task,
                    description=f"step_{step:07d}",
                    kwargs={
                        "batch": vis_batch,
                        "outputs": vis_outputs,
                        "save_path": output_dir / "visualizations" / f"step_{step:07d}.png",
                        "max_frames": int(cfg["train"].get("max_visualization_frames", 6)),
                    },
                )

            if is_main_process(rank) and save_every_steps > 0 and step % save_every_steps == 0:
                payload = build_checkpoint_payload(
                    model=model,
                    step=step,
                    epoch=epoch,
                    save_optimizer_state=save_optimizer_state,
                    optimizer=optimizer,
                    scaler=scaler,
                )
                ckpt_saver.launch(
                    save_checkpoint_task,
                    description=f"latest_{step:07d}",
                    kwargs={
                        "payload": payload,
                        "save_path": output_dir / "checkpoints" / "latest.pt",
                    },
                )

            if step >= max_steps:
                break
        epoch += 1

    if is_main_process(rank):
        vis_saver.join()
        payload = build_checkpoint_payload(
            model=model,
            step=step,
            epoch=epoch,
            save_optimizer_state=save_optimizer_state,
            optimizer=optimizer,
            scaler=scaler,
        )
        ckpt_saver.join()
        save_checkpoint_task(payload, output_dir / "checkpoints" / "latest.pt")
        save_checkpoint_task(payload, output_dir / "checkpoints" / f"final_step_{step:07d}.pt")

    writer.close()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
