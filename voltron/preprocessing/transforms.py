"""
transforms.py

Default video/image transforms for Voltron preprocessing and training. Provides utilities for defining different scale
and crop transformations on a dataset-specific basis.

There are two key desiderata we ensure with the transforms:
    - Aspect Ratio --> We *never* naively reshape images in a way that distorts the aspect ratio; we crop instead!
    - Minimum Size --> We *never* upsample images; processing strictly reduces dimensionality!
"""
from functools import partial
from typing import Any, Callable, List, Tuple

import torch
from PIL import Image, ImageOps
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Normalize, Resize


# Simple Identity Function --> needs to be top-level/pickleable for mp/distributed.spawn()
def identity(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def scaled_center_crop(target_resolution: int, frames: List[Image.Image]) -> Image.Image:
    # Assert width >= height and height >= target_resolution
    orig_w, orig_h = frames[0].size
    assert orig_w >= orig_h >= target_resolution

    # Compute scale factor --> just a function of height and target_resolution
    scale_factor = target_resolution / orig_h
    for idx in range(len(frames)):
        frames[idx] = ImageOps.scale(frames[idx], factor=scale_factor)
        left = (frames[idx].size[0] - target_resolution) // 2
        frames[idx] = frames[idx].crop((left, 0, left + target_resolution, target_resolution))

    # Return "scaled and squared" images
    return frames


def get_preprocess_transform(
    dataset_name: str, preprocess_resolution: int
) -> Callable[[List[Image.Image]], List[Image.Image]]:
    """Returns a transform that extracts square crops of `preprocess_resolution` from videos (as [T x H x W x C])."""
    if dataset_name == "sth-sth-v2":
        return partial(scaled_center_crop, preprocess_resolution)
    else:
        raise ValueError(f"Preprocessing transform for dataset `{dataset_name}` is not defined!")


def get_online_transform(
    dataset_name: str, model_arch: str, online_resolution: int, normalization: Tuple[Any, Any]
) -> Compose:
    """Returns an "online" torchvision Transform to be applied during training (batching/inference)."""
    if dataset_name == "sth-sth-v2":
        # Note: R3M does *not* expect normalized 0-1 (then ImageNet normalized) images --> drop the identity.
        if model_arch in {"v-r3m", "v-rn3m"}:
            return Compose([Resize((online_resolution, online_resolution), antialias=True), Lambda(identity)])
        else:
            return Compose(
                [
                    Resize((online_resolution, online_resolution), antialias=True),
                    ConvertImageDtype(torch.float),
                    Normalize(mean=normalization[0], std=normalization[1]),
                ]
            )
    else:
        raise ValueError(f"Online Transforms for Dataset `{dataset_name}` not implemented!")
