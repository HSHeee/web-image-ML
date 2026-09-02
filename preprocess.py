"""학습과 추론에서 동일하게 쓰는 이미지 -> 텐서 변환."""

import numpy as np
import torch
from PIL import Image

from config import IMG_H, IMG_W

_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_RESAMPLE = Image.Resampling.BILINEAR


def to_tensor(img: Image.Image) -> torch.Tensor:
    """PIL 이미지를 (3, IMG_H, IMG_W) float32 텐서로. [-1, 1] 정규화."""
    img = img.convert("RGB").resize((IMG_W, IMG_H), _RESAMPLE)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))
