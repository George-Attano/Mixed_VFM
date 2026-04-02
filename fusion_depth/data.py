from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def _load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _load_rgb(path: str | Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _load_depth(path: str | Path, depth_scale: float, npz_key: str) -> torch.Tensor:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        depth = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        npz = np.load(path)
        if npz_key in npz:
            depth = npz[npz_key].astype(np.float32)
        elif len(npz.files) == 1:
            depth = npz[npz.files[0]].astype(np.float32)
        else:
            raise KeyError(f"{path} does not contain key `{npz_key}`.")
    else:
        depth = np.asarray(Image.open(path), dtype=np.float32)

    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth * float(depth_scale)
    return torch.from_numpy(depth)


def _resize_image(image: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(
        image.unsqueeze(0),
        size=tuple(image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def _resize_depth(depth: torch.Tensor, image_size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(
        depth[None, None],
        size=tuple(image_size),
        mode="nearest",
    ).squeeze(0).squeeze(0)


class SequenceDepthDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        image_size: Sequence[int],
        sequence_length: int = 6,
        sequence_stride: int = 1,
        depth_scale: float = 1.0,
        npz_key: str = "depth",
        random_horizontal_flip: bool = False,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.image_size = tuple(int(x) for x in image_size)
        self.sequence_length = int(sequence_length)
        self.sequence_stride = int(sequence_stride)
        self.depth_scale = float(depth_scale)
        self.npz_key = npz_key
        self.random_horizontal_flip = random_horizontal_flip

        raw_manifest = _load_json(manifest_path)
        self.samples = self._build_samples(raw_manifest)
        if max_samples is not None:
            self.samples = self.samples[: int(max_samples)]

    def _build_samples(self, raw_manifest: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        for item_idx, item in enumerate(raw_manifest):
            if "images" in item and "depths" in item:
                if len(item["images"]) != len(item["depths"]):
                    raise ValueError(f"Manifest item {item_idx} has mismatched image/depth lengths.")
                if len(item["images"]) != self.sequence_length:
                    raise ValueError(
                        f"Direct sample item {item_idx} must contain exactly {self.sequence_length} frames."
                    )
                sequence_id = item.get("sequence_id", f"sample_{item_idx:06d}")
                samples.append(
                    {
                        "sample_id": item.get("sample_id", sequence_id),
                        "sequence_id": sequence_id,
                        "frames": [
                            {
                                "image": image_path,
                                "depth": depth_path,
                                "frame_id": str(frame_idx),
                            }
                            for frame_idx, (image_path, depth_path) in enumerate(
                                zip(item["images"], item["depths"])
                            )
                        ],
                    }
                )
                continue

            frames = item.get("frames")
            if not frames:
                raise ValueError(
                    "Each manifest item must either contain `images` + `depths`, or a `frames` list."
                )
            if len(frames) < self.sequence_length:
                continue

            sequence_id = item.get("sequence_id", f"sequence_{item_idx:06d}")
            for start_idx in range(0, len(frames) - self.sequence_length + 1, self.sequence_stride):
                window = frames[start_idx : start_idx + self.sequence_length]
                samples.append(
                    {
                        "sample_id": f"{sequence_id}_{start_idx:06d}",
                        "sequence_id": sequence_id,
                        "frames": window,
                    }
                )
        if not samples:
            raise ValueError(f"No samples were built from manifest {self.manifest_path}.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        images = []
        depths = []
        valid_masks = []
        frame_ids = []

        for frame in sample["frames"]:
            rgb = _resize_image(_load_rgb(frame["image"]), self.image_size)
            depth = _resize_depth(
                _load_depth(frame["depth"], self.depth_scale, self.npz_key),
                self.image_size,
            )
            valid = torch.isfinite(depth) & (depth > 0)
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

            images.append(rgb)
            depths.append(depth)
            valid_masks.append(valid)
            frame_ids.append(str(frame.get("frame_id", Path(frame["image"]).stem)))

        images_t = torch.stack(images, dim=0)
        depths_t = torch.stack(depths, dim=0)
        valid_masks_t = torch.stack(valid_masks, dim=0)

        if self.random_horizontal_flip and random.random() < 0.5:
            images_t = torch.flip(images_t, dims=[-1])
            depths_t = torch.flip(depths_t, dims=[-1])
            valid_masks_t = torch.flip(valid_masks_t, dims=[-1])

        return {
            "images": images_t,
            "depths": depths_t,
            "valid_masks": valid_masks_t,
            "sample_id": sample["sample_id"],
            "sequence_id": sample["sequence_id"],
            "frame_ids": frame_ids,
        }

