"""
transforms.py

Default image/video transformations for various datasets.
"""
from typing import Any, Tuple

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, ConvertImageDtype, Lambda, Normalize


# Definitions of Video Transformations (Reference: `something-something-v2-baseline`)
class ComposeMix:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for transformation, scope in self.transforms:
            if scope == "img":
                for idx, img in enumerate(imgs):
                    imgs[idx] = transformation(img)
            elif scope == "vid":
                imgs = transformation(imgs)
            else:
                raise ValueError("Please specify a valid transformation...")
        return imgs


class RandomCropVideo:
    def __init__(self, size):
        self.size = size

    def __call__(self, imgs):
        th, tw = self.size
        h, w = imgs[0].shape[:2]
        x1, y1 = np.random.randint(0, w - tw), np.random.randint(0, h - th)
        for idx, img in enumerate(imgs):
            imgs[idx] = img[y1 : y1 + th, x1 : x1 + tw]
        return imgs


class Scale:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        return cv2.resize(img, tuple(self.size))


def identity(x):
    """Transform needs to be pickleable for multiprocessing.spawn()."""
    return x.float()


def get_pre_transform(dataset: str, resolution: int, scale_factor: float = 1.1) -> ComposeMix:
    """Defines a `pre` transform to be applied *when serializing the images* (first pass)."""
    if dataset == "sth-sth-v2":
        if scale_factor > 1:
            transform = ComposeMix(
                [
                    [Scale((int(resolution * scale_factor), int(resolution * scale_factor))), "img"],
                    [RandomCropVideo((resolution, resolution)), "vid"],
                ]
            )
        else:
            transform = ComposeMix(
                [
                    [Scale((int(resolution * scale_factor), int(resolution * scale_factor))), "img"],
                ]
            )

        return transform
    else:
        raise NotImplementedError(f"(Pre) transforms for dataset `{dataset}` not yet implemented!")


def get_online_transform(dataset: str, model_arch: str, normalization: Tuple[Any, Any]) -> Compose:
    """Defines an `online` transform to be applied *when batching the images* (during training/validation)."""
    if dataset == "sth-sth-v2":
        # Note: R3M does *not* expect normalized 0-1 (then ImageNet normalized) images --> drop the identity.
        if model_arch in {"v-r3m", "v-rn3m"}:
            return Compose([Lambda(identity)])
        else:
            return Compose([ConvertImageDtype(torch.float), Normalize(mean=normalization[0], std=normalization[1])])
    else:
        raise NotImplementedError(f"(Online) transforms for dataset `{dataset} not yet implemented!")
