"""
Utility modules for Relighty.

Provides:
- split.py: Dataset splitting utility
- config_loader.py: Centralized configuration management
"""

from .config_loader import get_config, get_data_path, get_checkpoint_path, get_split_path, get_project_root

__all__ = [
    "get_config",
    "get_data_path",
    "get_checkpoint_path",
    "get_split_path",
    "get_project_root",
]