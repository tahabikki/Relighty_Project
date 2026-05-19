"""
Relighty - Face and Neck Shadow Removal System

A modular computer vision pipeline for removing shadows from face and neck regions
using deep learning with MediaPipe facial landmark detection.

Modules:
    masking: Facial landmark detection and mask generation (face + neck)
    preprocessing: Data preprocessing and augmentation
    models: Neural network architectures
    training: Model training pipeline
    evaluation: Model evaluation and metrics
    postprocessing: Post-processing utilities (texture preservation, background removal)
    deployment: End-to-end inference pipelines
"""

from pathlib import Path

__version__ = "1.0.0"
__author__ = "Relighty Team"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = Path(__file__).resolve().parent

__all__ = [
    "PROJECT_ROOT",
    "SRC_DIR",
]