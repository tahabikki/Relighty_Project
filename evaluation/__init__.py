"""
Evaluation Module - Model Evaluation and Inference

Provides:
- evaluate.py: Model evaluation on validation set
- inference.py: Single image or batch inference

Functions:
    load_model: Load trained model from checkpoint
    find_checkpoint: Auto-detect best checkpoint
    remove_shadow: Core shadow removal inference
"""

from .inference import (
    load_model,
    find_checkpoint,
    remove_shadow,
)

__all__ = [
    "load_model",
    "find_checkpoint",
    "remove_shadow",
]