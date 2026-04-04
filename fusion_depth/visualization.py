from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image


def _canonicalize_depth_tensor(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim == 4:
        if tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        elif tensor.shape[-1] == 1:
            tensor = tensor[..., 0]
        else:
            raise ValueError(
                f"{name} must have shape [S, H, W] or a singleton channel variant; got {tuple(tensor.shape)}."
            )
    elif tensor.ndim != 3:
        raise ValueError(
            f"{name} must have shape [S, H, W] or a singleton channel variant; got {tuple(tensor.shape)}."
        )
    return tensor


def _colorize_depth(
    depth: torch.Tensor,
    valid_mask: torch.Tensor | None,
    vmin: float,
    vmax: float,
) -> torch.Tensor:
    denom = max(vmax - vmin, 1e-6)
    normalized = ((depth - vmin) / denom).clamp(0, 1)
    try:
        from matplotlib import cm

        colored = cm.get_cmap("magma")(normalized.cpu().numpy())[..., :3]
        colored = torch.from_numpy(colored).permute(2, 0, 1).float()
    except ImportError:
        colored = normalized.unsqueeze(0).repeat(3, 1, 1).cpu()

    if valid_mask is not None:
        colored[:, ~valid_mask.cpu()] = 0
    return colored


def save_visualization(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    save_path: str | Path,
    max_frames: int = 6,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    images = batch["images"][0].detach().cpu()
    gt_depths = _canonicalize_depth_tensor(batch["depths"][0].detach().cpu(), "gt_depths")
    valid_masks = _canonicalize_depth_tensor(batch["valid_masks"][0].detach().cpu(), "valid_masks").bool()
    pred_depths = _canonicalize_depth_tensor(outputs["pred_depth"][0].detach().cpu(), "pred_depths")
    da3_depths = _canonicalize_depth_tensor(outputs["da3_depth"][0].detach().cpu(), "da3_depths")
    vggt_depths = _canonicalize_depth_tensor(outputs["vggt_depth"][0].detach().cpu(), "vggt_depths")

    rows = []
    num_frames = min(max_frames, images.shape[0])
    for frame_idx in range(num_frames):
        valid = valid_masks[frame_idx]
        valid_depth = gt_depths[frame_idx][valid]
        if valid_depth.numel() > 0:
            vmin = float(valid_depth.quantile(0.02))
            vmax = float(valid_depth.quantile(0.98))
        else:
            vmax = float(torch.max(gt_depths[frame_idx]).item() + 1e-6)
            vmin = 0.0

        rgb = images[frame_idx].clamp(0, 1)
        gt = _colorize_depth(gt_depths[frame_idx], valid, vmin, vmax)
        pred = _colorize_depth(pred_depths[frame_idx], None, vmin, vmax)
        da3 = _colorize_depth(da3_depths[frame_idx], None, vmin, vmax)
        vggt = _colorize_depth(vggt_depths[frame_idx], None, vmin, vmax)
        row = torch.cat([rgb, gt, pred, da3, vggt], dim=2)
        rows.append(row)

    grid = torch.cat(rows, dim=1).clamp(0, 1)
    array = (grid.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(save_path)
