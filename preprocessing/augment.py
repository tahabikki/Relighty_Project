"""
Data augmentation utilities for shadow removal training.

Provides consistent augmentation applied to both input and target images,
preserving pixel-wise correspondence needed for training.
"""
import numpy as np
import cv2
from typing import Tuple, Optional


def augment_pair(
    inp: np.ndarray,
    tgt: np.ndarray,
    flip_prob: float = 0.5,
    brightness_prob: float = 0.3,
    brightness_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply consistent augmentation to input-target image pair.

    All transforms preserve the pixel-wise correspondence so the model
    learns the correct shadow removal mapping.

    Args:
        inp: Input BGR image (uint8)
        tgt: Target BGR image (uint8)
        flip_prob: Probability of horizontal flip
        brightness_prob: Probability of brightness adjustment
        brightness_range: Range for brightness factor

    Returns:
        Augmented (input, target) pair
    """
    if np.random.rand() < flip_prob:
        inp = cv2.flip(inp, 1)
        tgt = cv2.flip(tgt, 1)

    if np.random.rand() < brightness_prob:
        factor = np.random.uniform(*brightness_range)
        inp = np.clip(inp.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        tgt = np.clip(tgt.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    return inp, tgt


def random_color_jitter(
    inp: np.ndarray,
    tgt: np.ndarray,
    sat_range: Tuple[float, float] = (0.9, 1.1),
    hue_range: Tuple[float, float] = (-10, 10),
    prob: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply color jitter while maintaining input-target consistency.

    Args:
        inp: Input BGR image
        tgt: Target BGR image
        sat_range: Saturation adjustment range
        hue_range: Hue adjustment range (in degrees)
        prob: Probability of applying jitter

    Returns:
        Color-jittered (input, target) pair
    """
    if np.random.rand() > prob:
        return inp, tgt

    sat_factor = np.random.uniform(*sat_range)
    hue_shift = np.random.uniform(*hue_range)

    inp_hsv = cv2.cvtColor(inp, cv2.COLOR_BGR2HSV).astype(np.float32)
    tgt_hsv = cv2.cvtColor(tgt, cv2.COLOR_BGR2HSV).astype(np.float32)

    inp_hsv[:, :, 1] = np.clip(inp_hsv[:, :, 1] * sat_factor, 0, 255)
    tgt_hsv[:, :, 1] = np.clip(tgt_hsv[:, :, 1] * sat_factor, 0, 255)

    inp_hsv[:, :, 0] = (inp_hsv[:, :, 0] + hue_shift) % 180
    tgt_hsv[:, :, 0] = (tgt_hsv[:, :, 0] + hue_shift) % 180

    inp = cv2.cvtColor(inp_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    tgt = cv2.cvtColor(tgt_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return inp, tgt


def apply_augmentation_pipeline(
    inp: np.ndarray,
    tgt: np.ndarray,
    use_flip: bool = True,
    use_brightness: bool = True,
    use_color: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply full augmentation pipeline.

    Args:
        inp: Input BGR image
        tgt: Target BGR image
        use_flip: Enable horizontal flip
        use_brightness: Enable brightness adjustment
        use_color: Enable color jitter

    Returns:
        Augmented (input, target) pair
    """
    if use_flip or use_brightness:
        inp, tgt = augment_pair(inp, tgt, flip_prob=0.5 if use_flip else 0.0)

    if use_color:
        inp, tgt = random_color_jitter(inp, tgt)

    return inp, tgt