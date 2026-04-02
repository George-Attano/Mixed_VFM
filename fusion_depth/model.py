from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from fusion_depth.checkpoints import load_da3_metric_checkpoint, load_vggt_checkpoint
from fusion_depth.paths import DA3_SRC_ROOT, REPO_ROOT, ensure_repo_paths

ensure_repo_paths()

from depth_anything_3.cfg import create_object, load_config  # noqa: E402
from depth_anything_3.model.utils.head_utils import custom_interpolate as da3_interpolate  # noqa: E402
from vggt.heads.dpt_head import custom_interpolate as vggt_interpolate  # noqa: E402
from vggt.heads.head_act import activate_head  # noqa: E402
from vggt.models.vggt import VGGT  # noqa: E402


LOGGER = logging.getLogger(__name__)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        groups = 8 if out_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvNormAct(channels, channels, kernel_size=3)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(8 if channels % 8 == 0 else 1, channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class FusionDepthHead(nn.Module):
    def __init__(self, feature_dim: int = 128, hidden_dim: int = 192) -> None:
        super().__init__()
        self.vggt_proj = ConvNormAct(feature_dim, hidden_dim, kernel_size=1)
        self.da3_proj = ConvNormAct(feature_dim, hidden_dim, kernel_size=1)
        self.prior_proj = ConvNormAct(2, hidden_dim // 2, kernel_size=3)

        fused_in_dim = hidden_dim * 2 + hidden_dim // 2
        self.fuser = nn.Sequential(
            ConvNormAct(fused_in_dim, hidden_dim, kernel_size=3),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(
        self,
        vggt_feature: torch.Tensor,
        da3_feature: torch.Tensor,
        vggt_depth: torch.Tensor,
        da3_depth: torch.Tensor,
    ) -> torch.Tensor:
        log_vggt = torch.log(vggt_depth.clamp_min(1e-6)).unsqueeze(1)
        log_da3 = torch.log(da3_depth.clamp_min(1e-6)).unsqueeze(1)
        base_log_depth = 0.5 * (log_vggt + log_da3)

        fused = torch.cat(
            [
                self.vggt_proj(vggt_feature),
                self.da3_proj(da3_feature),
                self.prior_proj(torch.cat([log_vggt, log_da3], dim=1)),
            ],
            dim=1,
        )
        residual = self.fuser(fused).clamp(-4.0, 4.0)
        return torch.exp(base_log_depth + residual).squeeze(1)


def _apply_vggt_pos_embed(head: nn.Module, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if getattr(head, "pos_embed", False):
        return head._apply_pos_embed(x, width, height)
    return x


def _apply_da3_pos_embed(head: nn.Module, x: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if getattr(head, "pos_embed", False):
        return head._add_pos_embed(x, width, height)
    return x


def _extract_vggt_branch(
    depth_head: nn.Module,
    aggregated_tokens_list: List[torch.Tensor],
    images: torch.Tensor,
    patch_start_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, _, height, width = images.shape
    patch_h = height // depth_head.patch_size
    patch_w = width // depth_head.patch_size

    projected: List[torch.Tensor] = []
    for dpt_idx, layer_idx in enumerate(depth_head.intermediate_layer_idx):
        tokens = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
        tokens = tokens.reshape(batch_size * num_frames, -1, tokens.shape[-1])
        tokens = depth_head.norm(tokens)
        tokens = tokens.permute(0, 2, 1).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)
        tokens = depth_head.projects[dpt_idx](tokens)
        tokens = _apply_vggt_pos_embed(depth_head, tokens, width, height)
        tokens = depth_head.resize_layers[dpt_idx](tokens)
        projected.append(tokens)

    features = depth_head.scratch_forward(projected)
    features = vggt_interpolate(
        features,
        (int(patch_h * depth_head.patch_size), int(patch_w * depth_head.patch_size)),
        mode="bilinear",
        align_corners=True,
    )
    features = _apply_vggt_pos_embed(depth_head, features, width, height)
    logits = depth_head.scratch.output_conv2(features)
    depth, _ = activate_head(logits, activation=depth_head.activation, conf_activation=depth_head.conf_activation)

    feature_map = features.view(batch_size, num_frames, *features.shape[1:])
    depth_map = depth.view(batch_size, num_frames, *depth.shape[1:]).squeeze(-1)
    return feature_map, depth_map


def _extract_da3_branch(
    depth_head: nn.Module,
    feats: Sequence[Tuple[torch.Tensor, torch.Tensor]],
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_frames, num_tokens, channels = feats[0][0].shape
    patch_h = height // depth_head.patch_size
    patch_w = width // depth_head.patch_size

    resized_feats: List[torch.Tensor] = []
    flat_feats = [feat[0].reshape(batch_size * num_frames, num_tokens, channels) for feat in feats]
    for stage_idx, take_idx in enumerate(depth_head.intermediate_layer_idx):
        x = flat_feats[take_idx]
        x = depth_head.norm(x[:, 0:])
        x = x.permute(0, 2, 1).contiguous().reshape(batch_size * num_frames, channels, patch_h, patch_w)
        x = depth_head.projects[stage_idx](x)
        x = _apply_da3_pos_embed(depth_head, x, width, height)
        x = depth_head.resize_layers[stage_idx](x)
        resized_feats.append(x)

    feature_map = depth_head._fuse(resized_feats)
    feature_map = depth_head.scratch.output_conv1(feature_map)
    feature_map = da3_interpolate(
        feature_map,
        size=(int(patch_h * depth_head.patch_size), int(patch_w * depth_head.patch_size)),
        mode="bilinear",
        align_corners=True,
    )
    feature_map = _apply_da3_pos_embed(depth_head, feature_map, width, height)

    logits = depth_head.scratch.output_conv2(feature_map)
    depth_map = depth_head._apply_activation_single(logits, depth_head.activation).squeeze(1)
    feature_map = feature_map.view(batch_size, num_frames, *feature_map.shape[1:])
    depth_map = depth_map.view(batch_size, num_frames, *depth_map.shape[1:])
    return feature_map, depth_map


class FusionDepthModel(nn.Module):
    def __init__(self, cfg: Dict) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer(
            "da3_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "da3_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.vggt = VGGT(
            img_size=int(cfg.get("img_size", 518)),
            enable_camera=False,
            enable_depth=True,
            enable_point=False,
            enable_track=False,
        )

        da3_config_path = cfg.get(
            "da3_config",
            str(DA3_SRC_ROOT / "depth_anything_3" / "configs" / "da3metric-large.yaml"),
        )
        da3_config_path = Path(da3_config_path)
        if not da3_config_path.is_absolute():
            da3_config_path = REPO_ROOT / da3_config_path
        da3_config = load_config(str(da3_config_path))
        self.da3 = create_object(da3_config)

        self.fusion_head = FusionDepthHead(
            feature_dim=int(cfg.get("feature_dim", 128)),
            hidden_dim=int(cfg.get("fusion_hidden_dim", 192)),
        )

        self._load_pretrained_weights(cfg)
        self._maybe_freeze_modules(cfg)

    def _load_pretrained_weights(self, cfg: Dict) -> None:
        vggt_ckpt = cfg.get("vggt_checkpoint")
        if vggt_ckpt:
            missing, unexpected = load_vggt_checkpoint(
                self.vggt,
                vggt_ckpt,
                strict=bool(cfg.get("strict_load", False)),
            )
            LOGGER.info("Loaded VGGT checkpoint. Missing=%d Unexpected=%d", len(missing), len(unexpected))

        da3_ckpt = cfg.get("da3_checkpoint")
        if da3_ckpt:
            missing, unexpected = load_da3_metric_checkpoint(
                self.da3,
                da3_ckpt,
                strict=bool(cfg.get("strict_load", False)),
            )
            LOGGER.info("Loaded DA3 checkpoint. Missing=%d Unexpected=%d", len(missing), len(unexpected))

    def _maybe_freeze_modules(self, cfg: Dict) -> None:
        if cfg.get("freeze_vggt_aggregator", False):
            for parameter in self.vggt.aggregator.parameters():
                parameter.requires_grad = False

        if cfg.get("freeze_vggt_depth_head", False):
            for parameter in self.vggt.depth_head.parameters():
                parameter.requires_grad = False

        if cfg.get("freeze_da3_backbone", False):
            for parameter in self.da3.backbone.parameters():
                parameter.requires_grad = False

        if cfg.get("freeze_da3_head", False):
            for parameter in self.da3.head.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(images)
        vggt_feature, vggt_depth = _extract_vggt_branch(
            self.vggt.depth_head,
            aggregated_tokens_list,
            images,
            patch_start_idx,
        )

        da3_images = (images - self.da3_mean) / self.da3_std
        da3_feats, _ = self.da3.backbone(da3_images, export_feat_layers=[], ref_view_strategy="first")
        da3_feature, da3_depth = _extract_da3_branch(self.da3.head, da3_feats, images.shape[-2], images.shape[-1])

        batch_size, num_frames, channels, height, width = vggt_feature.shape
        pred_depth = self.fusion_head(
            vggt_feature.reshape(batch_size * num_frames, channels, height, width),
            da3_feature.reshape(batch_size * num_frames, channels, height, width),
            vggt_depth.reshape(batch_size * num_frames, height, width),
            da3_depth.reshape(batch_size * num_frames, height, width),
        ).view(batch_size, num_frames, height, width)
        return {
            "pred_depth": pred_depth,
            "vggt_depth": vggt_depth,
            "da3_depth": da3_depth,
        }

    def parameter_groups(self, base_lr: float, weight_decay: float, pretrained_lr_mult: float) -> List[Dict]:
        fusion_params = []
        pretrained_params = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("fusion_head."):
                fusion_params.append(parameter)
            else:
                pretrained_params.append(parameter)

        groups = []
        if pretrained_params:
            groups.append(
                {
                    "params": pretrained_params,
                    "lr": base_lr * pretrained_lr_mult,
                    "weight_decay": weight_decay,
                }
            )
        if fusion_params:
            groups.append(
                {
                    "params": fusion_params,
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                }
            )
        return groups
