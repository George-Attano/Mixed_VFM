from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch

from fusion_depth.paths import ensure_repo_paths

ensure_repo_paths()

from depth_anything_3.utils.model_loading import convert_metric_state_dict  # noqa: E402


def _load_raw_checkpoint(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires `safetensors` to be installed."
            ) from exc
        checkpoint = load_file(str(path))
    else:
        checkpoint = torch.load(str(path), map_location="cpu")

    while isinstance(checkpoint, dict):
        for nested_key in ("state_dict", "model", "module"):
            nested = checkpoint.get(nested_key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
        else:
            break

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}.")
    return checkpoint


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    prefix_len = len(prefix)
    return {
        key[prefix_len:] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def _best_prefix_transform(
    state_dict: Dict[str, torch.Tensor],
    target_keys: Iterable[str],
) -> Dict[str, torch.Tensor]:
    target_keys = set(target_keys)
    candidates = [
        state_dict,
        _strip_prefix(state_dict, "module."),
        _strip_prefix(state_dict, "model."),
        _strip_prefix(_strip_prefix(state_dict, "module."), "model."),
        _strip_prefix(_strip_prefix(state_dict, "model."), "module."),
    ]

    best_state = state_dict
    best_overlap = -1
    for candidate in candidates:
        overlap = sum(1 for key in candidate if key in target_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = candidate
    return best_state


def _count_overlap(state_dict: Dict[str, torch.Tensor], target_keys: Iterable[str]) -> int:
    target_keys = set(target_keys)
    return sum(1 for key in state_dict if key in target_keys)


def load_vggt_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> Tuple[list[str], list[str]]:
    state_dict = _load_raw_checkpoint(checkpoint_path)
    state_dict = _best_prefix_transform(state_dict, model.state_dict().keys())
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)


def load_da3_metric_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = False,
) -> Tuple[list[str], list[str]]:
    raw_state_dict = _load_raw_checkpoint(checkpoint_path)
    target_keys = model.state_dict().keys()

    raw_candidate = _best_prefix_transform(raw_state_dict, target_keys)
    converted_candidate = _best_prefix_transform(convert_metric_state_dict(raw_state_dict), target_keys)

    if _count_overlap(converted_candidate, target_keys) >= _count_overlap(raw_candidate, target_keys):
        state_dict = converted_candidate
    else:
        state_dict = raw_candidate

    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    return list(missing), list(unexpected)
