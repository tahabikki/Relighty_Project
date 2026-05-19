"""
Configuration loader for Relighty.

Provides centralized access to all configuration values.
Can be imported anywhere in the project.

Usage:
    from utils.config_loader import get_config, get_data_path, get_checkpoint_path

    config = get_config()
    dataset_root = get_data_path('dataset_root')
    checkpoint_dir = get_checkpoint_path()
"""
import os
import sys
from pathlib import Path
from typing import Optional, Union

import yaml


class Config:
    """Singleton configuration class."""

    _instance = None
    _config = None
    _project_root = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        self._project_root = Path(__file__).resolve().parent.parent
        config_path = self._project_root / "configs" / "config.yaml"

        if not config_path.exists():
            self._config = self._get_defaults()
            return

        with open(config_path, "r") as f:
            self._config = yaml.safe_load(f) or {}

    def _get_defaults(self) -> dict:
        """Return default configuration if config file doesn't exist."""
        return {
            "data": {
                "dataset_root": "dataset",
                "input_subdir": "input",
                "target_subdir": "target",
                "splits_dir": "dataset/splits",
                "image_size": 256,
                "train_ratio": 0.85,
                "seed": 42,
            },
            "training": {
                "batch_size": 4,
                "epochs": 200,
                "lr": 0.00001,
                "enc_lr": 0.000001,
                "device": "auto",
            },
            "checkpoint": {
                "dir": "checkpoints",
            },
            "logging": {
                "log_dir": "logs",
            },
            "inference": {
                "output_dir": "Results/output",
                "input_dir": "Results/input",
            },
        }

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root

    def get(self, key: str, default=None):
        """Get a config value using dot notation (e.g., 'data.image_size')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def get_data_path(self, path_key: str, relative_to: str = "project_root") -> Path:
        """
        Get a data path from config, resolving it relative to project root.

        Args:
            path_key: Key in data section (e.g., 'dataset_root', 'splits_dir')
            relative_to: 'project_root' or 'dataset_root'

        Returns:
            Resolved absolute Path
        """
        path_str = self.get(f"data.{path_key}", "")

        if not path_str:
            return self._project_root

        path = Path(path_str)

        if path.is_absolute():
            return path

        if relative_to == "dataset_root":
            dataset_root = self.get_data_path("dataset_root")
            return dataset_root / path_str

        return self._project_root / path_str

    def get_checkpoint_path(self, filename: Optional[str] = None) -> Path:
        """Get checkpoint directory or specific checkpoint file."""
        checkpoint_dir = self._project_root / self.get("checkpoint.dir", "checkpoints")
        if filename:
            return checkpoint_dir / filename
        return checkpoint_dir

    def get_log_path(self, filename: Optional[str] = None) -> Path:
        """Get log directory or specific log file."""
        log_dir = self._project_root / self.get("logging.log_dir", "logs")
        if filename:
            return log_dir / filename
        return log_dir

    def get_inference_input_path(self) -> Path:
        """Get inference input directory."""
        return self._project_root / self.get("inference.input_dir", "Results/input")

    def get_inference_output_path(self) -> Path:
        """Get inference output directory."""
        return self._project_root / self.get("inference.output_dir", "Results/output")

    def get_split_path(self, split_file: str) -> Path:
        """Get path to a split file (e.g., 'train_input.txt')."""
        return self.get_data_path("splits_dir") / split_file

    def reload(self):
        """Reload configuration from file."""
        self._config = None
        self._load_config()


_config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return _config


def get_data_path(path_key: str) -> Path:
    """Shortcut to get a data path."""
    return _config.get_data_path(path_key)


def get_checkpoint_path(filename: Optional[str] = None) -> Path:
    """Shortcut to get checkpoint path."""
    return _config.get_checkpoint_path(filename)


def get_log_path(filename: Optional[str] = None) -> Path:
    """Shortcut to get log path."""
    return _config.get_log_path(filename)


def get_split_path(split_file: str) -> Path:
    """Shortcut to get split file path."""
    return _config.get_split_path(split_file)


def get_inference_input_path() -> Path:
    """Shortcut to get inference input path."""
    return _config.get_inference_input_path()


def get_inference_output_path() -> Path:
    """Shortcut to get inference output path."""
    return _config.get_inference_output_path()


def get_project_root() -> Path:
    """Get project root directory."""
    return _config.project_root