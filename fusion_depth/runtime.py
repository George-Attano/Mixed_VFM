from __future__ import annotations

import copy
import importlib
import logging
import os
import random
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from fusion_depth.data import SequenceDepthDataset


LOGGER = logging.getLogger(__name__)


def import_string(path: str) -> Any:
    module_path, attr_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), attr_name)


def instantiate_from_config(config: Dict[str, Any]) -> Any:
    if "_target_" not in config:
        return SequenceDepthDataset(**config)
    target = config["_target_"]
    kwargs = {key: value for key, value in config.items() if key != "_target_"}
    cls_or_fn = import_string(target)
    return cls_or_fn(**kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(timeout_minutes: int = 180) -> Tuple[bool, int, int, int, torch.device]:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout_minutes))
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


def cleanup_distributed(distributed: bool) -> None:
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def get_autocast_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name.lower() == "float16":
        return torch.float16
    return torch.bfloat16


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def _to_cpu_clone(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _to_cpu_clone(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu_clone(item) for item in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu_clone(item) for item in obj)
    return copy.deepcopy(obj)


def build_checkpoint_payload(
    model: torch.nn.Module,
    step: int,
    epoch: int,
    save_optimizer_state: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": _to_cpu_clone(unwrap_model(model).state_dict()),
        "step": int(step),
        "epoch": int(epoch),
    }
    if save_optimizer_state and optimizer is not None:
        payload["optimizer"] = _to_cpu_clone(optimizer.state_dict())
    if save_optimizer_state and scaler is not None:
        payload["scaler"] = _to_cpu_clone(scaler.state_dict())
    return payload


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = False,
) -> Tuple[int, int]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    unwrap_model(model).load_state_dict(state_dict, strict=strict)

    if isinstance(checkpoint, dict):
        if optimizer is not None and "optimizer" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as exc:
                LOGGER.warning("Failed to load optimizer state from %s: %s", checkpoint_path, exc)
        if scaler is not None and "scaler" in checkpoint:
            try:
                scaler.load_state_dict(checkpoint["scaler"])
            except Exception as exc:
                LOGGER.warning("Failed to load scaler state from %s: %s", checkpoint_path, exc)
        return int(checkpoint.get("step", 0)), int(checkpoint.get("epoch", 0))
    return 0, 0

