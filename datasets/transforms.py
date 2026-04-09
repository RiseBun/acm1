from __future__ import annotations

import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def _ensure_rgb(pil_image: Image.Image) -> Image.Image:
    if pil_image.mode != "RGB":
        return pil_image.convert("RGB")
    return pil_image


def _mask_tensor_from_pil(pil_mask: Image.Image) -> torch.Tensor:
    m = np.array(pil_mask.convert("L"), dtype=np.float32) / 255.0
    m = (m > 0.5).astype(np.float32)
    return torch.from_numpy(m).unsqueeze(0)


def train_image_mask_transform(pil_image: Image.Image, pil_mask: Image.Image, image_size: int):
    """Same weak aug as guide: resize, random crop, horizontal flip. Mask aligned."""
    image = _ensure_rgb(pil_image)
    mask = pil_mask
    i, j, h, w = T.RandomResizedCrop.get_params(image, scale=(0.9, 1.0), ratio=(0.95, 1.05))
    image = TF.resized_crop(image, i, j, h, w, (image_size, image_size), antialias=True)
    mask = TF.resized_crop(mask, i, j, h, w, (image_size, image_size), interpolation=T.InterpolationMode.NEAREST)
    if random.random() < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
    image_t = TF.to_tensor(image)
    mask_t = _mask_tensor_from_pil(mask)
    image_t = TF.normalize(image_t, mean=CLIP_MEAN, std=CLIP_STD)
    return image_t, mask_t


def eval_image_mask_transform(pil_image: Image.Image, pil_mask: Image.Image, image_size: int):
    image = _ensure_rgb(pil_image)
    image = TF.resize(image, (image_size, image_size), antialias=True)
    mask = TF.resize(pil_mask, (image_size, image_size), interpolation=T.InterpolationMode.NEAREST)
    image_t = TF.to_tensor(image)
    mask_t = _mask_tensor_from_pil(mask)
    image_t = TF.normalize(image_t, mean=CLIP_MEAN, std=CLIP_STD)
    return image_t, mask_t
