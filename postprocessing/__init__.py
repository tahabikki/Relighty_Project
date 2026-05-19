"""
Postprocessing Module - Post-processing Utilities

Provides:
- fix_light: Face and neck shadow removal with texture preservation
- fix_background: Background removal using rembg
- fix_icao: ICAO-compliant face cropping
- fix_blurred: Intelligent deblurring

Key Features:
- Texture preservation: Keeps original skin texture while removing shadows
- Neck region support: Extends face masks to include neck
- LAB color space: Uses L-channel for perceptually better blending
"""

from .fix_light import (
    fix_light,
    fix_background,
    fix_icao,
    fix_blurred,
    init_landmarks,
    compute_initial_data,
)

from .bg_remove import remove_background

__all__ = [
    "fix_light",
    "fix_background",
    "fix_icao",
    "fix_blurred",
    "init_landmarks",
    "compute_initial_data",
    "remove_background",
]