"""
Shadow-removal dataset using BiSeNet face parsing.

Uses BiSeNet (from Masking project) to generate mask during training,
focusing only on configured regions (face, neck, etc.).
"""
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple, Optional

from utils.config_loader import get_config

_BISENET_GEN = None


def _get_bisenet_generator():
    """Lazy load BiSeNet mask generator."""
    global _BISENET_GEN
    if _BISENET_GEN is None:
        from masking_bg.bisenet_mask import BiSeNetMaskGenerator
        config = get_config()
        _BISENET_GEN = BiSeNetMaskGenerator.from_config(config)
    return _BISENET_GEN


def _to_tensor(arr: np.ndarray) -> torch.Tensor:
    arr = arr.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def _augment_pair(
    inp: np.ndarray, tgt: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random augmentation to image pair."""
    if np.random.rand() > 0.5:
        inp = np.fliplr(inp).copy()
        tgt = np.fliplr(tgt).copy()

    if np.random.rand() > 0.5:
        brightness = np.random.uniform(0.9, 1.1)
        inp = np.clip(inp * brightness, 0, 255).astype(np.uint8)
        tgt = np.clip(tgt * brightness, 0, 255).astype(np.uint8)

    return inp, tgt


class ShadowDataset(Dataset):
    """
    Paired shadow-removal dataset with BiSeNet masking.
    
    Uses BiSeNet (from Masking project) to generate mask during training,
    focusing only on configured regions (face, neck, etc.).

    Args:
        input_list:  path to .txt file, one input  image path per line
        target_list: path to .txt file, one target image path per line
        augment:     True for training set, False for validation
        image_size:  model input resolution (square)
    """

    def __init__(
        self,
        input_list: str,
        target_list: str,
        augment: bool = False,
        image_size: int = 256,
    ):
        self.augment    = augment
        self.image_size = image_size

        with open(input_list)  as f:
            self.inputs  = [l.strip() for l in f if l.strip()]
        with open(target_list) as f:
            self.targets = [l.strip() for l in f if l.strip()]

        assert len(self.inputs) == len(self.targets), (
            f"Mismatch: {len(self.inputs)} inputs vs {len(self.targets)} targets"
        )
        
        self.bisenet = _get_bisenet_generator()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        inp_bgr = cv2.imread(self.inputs[idx])
        tgt_bgr = cv2.imread(self.targets[idx])

        if inp_bgr is None or tgt_bgr is None:
            z = torch.zeros(3, self.image_size, self.image_size)
            m = torch.zeros(1, self.image_size, self.image_size)
            return z, z, m

        inp_bgr = cv2.resize(inp_bgr, (self.image_size, self.image_size))
        tgt_bgr = cv2.resize(tgt_bgr, (self.image_size, self.image_size))

        if self.augment:
            inp_bgr, tgt_bgr = _augment_pair(inp_bgr, tgt_bgr)

        inp_rgb = cv2.cvtColor(inp_bgr, cv2.COLOR_BGR2RGB)
        mask = self.bisenet.generate_mask(inp_rgb)
        
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        
        return _to_tensor(inp_bgr), _to_tensor(tgt_bgr), mask_tensor