from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


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


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    values = values * mask
    return values.sum() / mask.sum().clamp_min(eps)


def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_mean(torch.abs(pred - target), mask)


def silog_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    lam: float = 0.85,
) -> torch.Tensor:
    pred = pred.clamp_min(1e-6)
    target = target.clamp_min(1e-6)
    log_diff = torch.log(pred) - torch.log(target)
    valid = mask > 0
    if valid.sum() == 0:
        return pred.new_tensor(0.0)
    log_diff = log_diff[valid]
    mean_sq = torch.mean(log_diff**2)
    mean = torch.mean(log_diff)
    return torch.sqrt(torch.clamp(mean_sq - lam * mean * mean, min=0.0) + 1e-6)


def multiscale_gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    scales: tuple[int, ...] = (1, 2, 4),
) -> torch.Tensor:
    pred = torch.log(pred.clamp_min(1e-6))
    target = torch.log(target.clamp_min(1e-6))

    total = pred.new_tensor(0.0)
    valid_scales = 0
    for scale in scales:
        pred_s = pred[..., ::scale, ::scale]
        target_s = target[..., ::scale, ::scale]
        mask_s = mask[..., ::scale, ::scale].bool()

        dx_mask = mask_s[..., :, 1:] & mask_s[..., :, :-1]
        dy_mask = mask_s[..., 1:, :] & mask_s[..., :-1, :]

        if dx_mask.any():
            dx_pred = pred_s[..., :, 1:] - pred_s[..., :, :-1]
            dx_target = target_s[..., :, 1:] - target_s[..., :, :-1]
            total = total + torch.abs(dx_pred[dx_mask] - dx_target[dx_mask]).mean()
            valid_scales += 1

        if dy_mask.any():
            dy_pred = pred_s[..., 1:, :] - pred_s[..., :-1, :]
            dy_target = target_s[..., 1:, :] - target_s[..., :-1, :]
            total = total + torch.abs(dy_pred[dy_mask] - dy_target[dy_mask]).mean()
            valid_scales += 1

    if valid_scales == 0:
        return pred.new_tensor(0.0)
    return total / valid_scales


class FusionDepthLoss(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.l1_weight = float(cfg.get("l1_weight", 1.0))
        self.silog_weight = float(cfg.get("silog_weight", 1.0))
        self.grad_weight = float(cfg.get("grad_weight", 0.5))
        self.aux_branch_weight = float(cfg.get("aux_branch_weight", 0.0))
        self.silog_lambda = float(cfg.get("silog_lambda", 0.85))
        self.min_depth = float(cfg.get("min_depth", 1e-3))
        self.max_depth = float(cfg.get("max_depth", 200.0))

    def _compute_single_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        pred = _canonicalize_depth_tensor(pred, "pred")
        target = _canonicalize_depth_tensor(target, "target")
        mask = _canonicalize_depth_tensor(mask, "mask").bool()
        if pred.shape != target.shape or pred.shape != mask.shape:
            raise ValueError(
                "pred, target, and mask must have the same canonicalized shape. "
                f"Got pred={tuple(pred.shape)}, target={tuple(target.shape)}, mask={tuple(mask.shape)}."
            )

        valid = mask & torch.isfinite(target)
        valid = valid & (target >= self.min_depth) & (target <= self.max_depth)
        valid_f = valid.float()

        pred = pred.clamp(self.min_depth, self.max_depth)
        target = target.clamp(self.min_depth, self.max_depth)

        l1 = masked_l1_loss(pred, target, valid_f)
        silog = silog_loss(pred, target, valid_f, lam=self.silog_lambda)
        grad = multiscale_gradient_loss(pred, target, valid_f)
        total = self.l1_weight * l1 + self.silog_weight * silog + self.grad_weight * grad
        return {
            "total": total,
            "l1": l1,
            "silog": silog,
            "grad": grad,
        }

    def forward(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        target = batch["depths"]
        mask = batch["valid_masks"]

        fused_losses = self._compute_single_loss(outputs["pred_depth"], target, mask)
        total = fused_losses["total"]
        logs: Dict[str, torch.Tensor] = {
            "loss": total,
            "loss_fused": fused_losses["total"],
            "loss_l1": fused_losses["l1"],
            "loss_silog": fused_losses["silog"],
            "loss_grad": fused_losses["grad"],
        }

        if self.aux_branch_weight > 0:
            for branch_name in ("da3_depth", "vggt_depth"):
                if branch_name not in outputs:
                    continue
                branch_losses = self._compute_single_loss(outputs[branch_name], target, mask)
                branch_total = branch_losses["total"]
                total = total + self.aux_branch_weight * branch_total
                logs[f"loss_{branch_name}"] = branch_total

        logs["loss"] = total
        return logs
