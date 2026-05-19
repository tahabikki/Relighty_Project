"""
Training Module - Model Training Pipeline

Provides training functionality including:
- ShadowDataset: Dataset class for paired shadow removal training
- CombinedLoss: Combined L1 + SSIM loss
- Training entry point (train.py)
"""

from .dataset import ShadowDataset
from .losses import CombinedLoss, SSIMLoss

__all__ = [
    "ShadowDataset",
    "CombinedLoss",
    "SSIMLoss",
]