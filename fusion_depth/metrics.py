from __future__ import annotations

from typing import Dict

import torch


METRIC_NAMES = (
    "abs_rel",
    "sq_rel",
    "rmse",
    "rmse_log",
    "delta1",
    "delta2",
    "delta3",
)


def _canonicalize_depth_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 5:
        if tensor.shape[2] == 1:
            tensor = tensor[:, :, 0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        else:
            raise ValueError(
                f"{name} must have shape [B, S, H, W] or a singleton channel variant; got {tuple(tensor.shape)}."
            )
    elif tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 4:
        raise ValueError(
            f"{name} must have shape [B, S, H, W] or a singleton channel variant; got {tuple(tensor.shape)}."
        )
    return tensor


def compute_depth_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    min_depth: float = 1e-3,
    max_depth: float = 200.0,
    min_valid_pixels: int = 64,
) -> torch.Tensor:
    pred = _canonicalize_depth_tensor(pred, "pred")
    target = _canonicalize_depth_tensor(target, "target")
    valid_mask = _canonicalize_depth_tensor(valid_mask, "valid_mask").bool()
    if pred.shape != target.shape or pred.shape != valid_mask.shape:
        raise ValueError(
            "pred, target, and valid_mask must have the same canonicalized shape. "
            f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}, valid_mask={tuple(valid_mask.shape)}."
        )

    pred = pred.clamp(min_depth, max_depth)
    target = target.clamp(min_depth, max_depth)
    valid_mask = valid_mask & torch.isfinite(pred) & torch.isfinite(target)
    valid_mask = valid_mask & (target >= min_depth) & (target <= max_depth)

    pred = pred.reshape(-1, pred.shape[-2], pred.shape[-1])
    target = target.reshape(-1, target.shape[-2], target.shape[-1])
    valid_mask = valid_mask.reshape(-1, valid_mask.shape[-2], valid_mask.shape[-1])

    sums = pred.new_zeros(len(METRIC_NAMES) + 1)
    for frame_idx in range(pred.shape[0]):
        mask = valid_mask[frame_idx]
        if mask.sum() < min_valid_pixels:
            continue

        pred_valid = pred[frame_idx][mask]
        target_valid = target[frame_idx][mask]
        diff = pred_valid - target_valid
        ratio = torch.maximum(target_valid / pred_valid, pred_valid / target_valid)

        sums[0] += torch.mean(torch.abs(diff) / target_valid)
        sums[1] += torch.mean((diff**2) / target_valid)
        sums[2] += torch.sqrt(torch.mean(diff**2))
        sums[3] += torch.sqrt(torch.mean((torch.log(pred_valid) - torch.log(target_valid)) ** 2))
        sums[4] += torch.mean((ratio < 1.25).float())
        sums[5] += torch.mean((ratio < 1.25**2).float())
        sums[6] += torch.mean((ratio < 1.25**3).float())
        sums[7] += 1

    return sums


def tensor_to_metric_dict(metric_tensor: torch.Tensor) -> Dict[str, float]:
    count = metric_tensor[-1].item()
    if count <= 0:
        return {name: float("nan") for name in METRIC_NAMES}
    return {
        name: (metric_tensor[idx] / count).item()
        for idx, name in enumerate(METRIC_NAMES)
    }
