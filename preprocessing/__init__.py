"""
Preprocessing Module - Data Preprocessing and Augmentation

Provides image augmentation and transformation utilities for the
shadow removal pipeline.

Functions:
    augment_pair: Basic augmentation (flip, brightness)
    random_color_jitter: Color adjustments
    apply_augmentation_pipeline: Full augmentation pipeline
"""

from .augment import (
    augment_pair,
    random_color_jitter,
    apply_augmentation_pipeline,
)

__all__ = [
    "augment_pair",
    "random_color_jitter",
    "apply_augmentation_pipeline",
]