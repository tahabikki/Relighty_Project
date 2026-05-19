"""
Deployment Module - End-to-end Inference Pipeline

Provides:
- ShadowRemovalPipeline: Complete pipeline for processing images
- load_config: Load configuration from YAML
- find_checkpoint: Auto-detect best checkpoint
- get_device: Device selection utility
"""

from .pipeline import (
    ShadowRemovalPipeline,
    load_config,
    find_checkpoint,
    get_device,
)

__all__ = [
    "ShadowRemovalPipeline",
    "load_config",
    "find_checkpoint",
    "get_device",
]