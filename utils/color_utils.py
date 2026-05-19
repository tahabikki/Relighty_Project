"""
Color space conversion utilities for ICAO-compliant shadow removal.

LAB Color Space: 
  L = Lightness (0-100)
  A = Green-Red axis (-128 to 127)
  B = Blue-Yellow axis (-128 to 127)

Why LAB for shadow removal?
  - Modify L (lightness) to remove shadows while preserving skin tone
  - Keep A, B channels untouched to preserve original color
  - Mimics how human eyes perceive brightness independent of color
"""

import torch
import torch.nn.functional as F
import numpy as np


def rgb_to_lab(rgb_img):
    """
    Convert RGB image to LAB color space.
    
    Args:
        rgb_img: Tensor [B, 3, H, W] or [3, H, W], values in [0, 1]
    
    Returns:
        lab_img: Tensor same shape, L in [0, 100], A/B in [-128, 127]
    """
    # Handle both batched and unbatched inputs
    is_batched = rgb_img.dim() == 4
    if not is_batched:
        rgb_img = rgb_img.unsqueeze(0)
    
    # Step 1: RGB to XYZ
    # First, linearize RGB (undo gamma correction)
    rgb_linear = torch.where(
        rgb_img > 0.04045,
        torch.pow((rgb_img + 0.055) / 1.055, 2.4),
        rgb_img / 12.92
    )
    
    # RGB to XYZ transformation matrix (D65 illuminant)
    rgb_to_xyz_matrix = torch.tensor([
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505]
    ], dtype=rgb_img.dtype, device=rgb_img.device)
    
    # Reshape for matrix multiplication
    B, C, H, W = rgb_linear.shape
    rgb_flat = rgb_linear.view(B, C, -1)  # [B, 3, H*W]
    
    xyz_flat = torch.matmul(rgb_to_xyz_matrix, rgb_flat)  # [B, 3, H*W]
    xyz = xyz_flat.view(B, 3, H, W)
    
    # Normalize by D65 reference white
    xyz_n = torch.stack([
        xyz[:, 0] / 0.95047,
        xyz[:, 1] / 1.00000,
        xyz[:, 2] / 1.08883
    ], dim=1)
    
    # Step 2: XYZ to LAB
    # Apply nonlinearity
    delta = 6.0 / 29.0
    xyz_nonlin = torch.where(
        xyz_n > delta**3,
        torch.pow(xyz_n, 1.0/3.0),
        xyz_n / (3 * delta**2) + 4.0/29.0
    )
    
    L = 116 * xyz_nonlin[:, 1] - 16
    A = 500 * (xyz_nonlin[:, 0] - xyz_nonlin[:, 1])
    B_chan = 200 * (xyz_nonlin[:, 1] - xyz_nonlin[:, 2])
    
    lab = torch.stack([L, A, B_chan], dim=1)
    
    if not is_batched:
        lab = lab.squeeze(0)
    
    return lab


def lab_to_rgb(lab_img):
    """
    Convert LAB image back to RGB color space.
    
    Args:
        lab_img: Tensor [B, 3, H, W] or [3, H, W], L in [0, 100], A/B in [-128, 127]
    
    Returns:
        rgb_img: Tensor same shape, values in [0, 1]
    """
    # Handle both batched and unbatched inputs
    is_batched = lab_img.dim() == 4
    if not is_batched:
        lab_img = lab_img.unsqueeze(0)
    
    B, C, H, W = lab_img.shape
    
    L = lab_img[:, 0]
    A = lab_img[:, 1]
    B_chan = lab_img[:, 2]
    
    # Step 1: LAB to XYZ
    fy = (L + 16) / 116
    fx = A / 500 + fy
    fz = fy - B_chan / 200
    
    delta = 6.0 / 29.0
    
    xr = torch.where(
        fx**3 > delta**3,
        fx**3,
        3 * delta**2 * (fx - 4.0/29.0)
    )
    yr = torch.where(
        fy**3 > delta**3,
        fy**3,
        3 * delta**2 * (fy - 4.0/29.0)
    )
    zr = torch.where(
        fz**3 > delta**3,
        fz**3,
        3 * delta**2 * (fz - 4.0/29.0)
    )
    
    xyz = torch.stack([
        xr * 0.95047,
        yr * 1.00000,
        zr * 1.08883
    ], dim=1)
    
    # Step 2: XYZ to RGB
    xyz_to_rgb_matrix = torch.tensor([
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.2040, 1.0570]
    ], dtype=lab_img.dtype, device=lab_img.device)
    
    xyz_flat = xyz.view(B, 3, -1)
    rgb_linear_flat = torch.matmul(xyz_to_rgb_matrix, xyz_flat)
    rgb_linear = rgb_linear_flat.view(B, 3, H, W)
    
    # Step 3: Apply gamma correction
    rgb = torch.where(
        rgb_linear > 0.0031308,
        1.055 * torch.pow(rgb_linear, 1.0/2.4) - 0.055,
        12.92 * rgb_linear
    )
    
    # Clamp to valid range
    rgb = torch.clamp(rgb, 0, 1)
    
    if not is_batched:
        rgb = rgb.squeeze(0)
    
    return rgb


def rgb_to_gray(rgb_img):
    """
    Convert RGB to grayscale using standard formula.
    
    Args:
        rgb_img: Tensor [B, 3, H, W] or [3, H, W], values in [0, 1]
    
    Returns:
        gray_img: Tensor [B, 1, H, W] or [1, H, W]
    """
    is_batched = rgb_img.dim() == 4
    if not is_batched:
        rgb_img = rgb_img.unsqueeze(0)
    
    # Standard grayscale formula
    gray = 0.299 * rgb_img[:, 0] + 0.587 * rgb_img[:, 1] + 0.114 * rgb_img[:, 2]
    gray = gray.unsqueeze(1)
    
    if not is_batched:
        gray = gray.squeeze(0)
    
    return gray


def normalize_lightness(lab_img, target_brightness=50):
    """
    Normalize lightness across image for uniform face lighting (ICAO compliance).
    
    Args:
        lab_img: LAB image [B, 3, H, W]
        target_brightness: Target L value (default 50 = mid-bright for faces)
    
    Returns:
        lab_normalized: LAB image with uniform lightness
    """
    L = lab_img[:, 0]
    
    # Compute mean brightness
    mean_L = L.mean(dim=[1, 2], keepdim=True)
    
    # Shift lightness to target
    L_normalized = L + (target_brightness - mean_L)
    L_normalized = torch.clamp(L_normalized, 0, 100)
    
    lab_normalized = lab_img.clone()
    lab_normalized[:, 0] = L_normalized
    
    return lab_normalized


# Test if conversions are working
if __name__ == "__main__":
    # Create test tensor
    rgb_test = torch.ones(1, 3, 256, 256) * 0.5
    
    # Convert RGB -> LAB -> RGB
    lab = rgb_to_lab(rgb_test)
    rgb_reconstructed = lab_to_rgb(lab)
    
    # Check reconstruction error
    error = torch.abs(rgb_test - rgb_reconstructed).mean()
    print(f"Reconstruction error: {error.item():.6f}")
    print(f"LAB shape: {lab.shape}")
    print(f"LAB range: L=[{lab[:, 0].min():.1f}, {lab[:, 0].max():.1f}], "
          f"A=[{lab[:, 1].min():.1f}, {lab[:, 1].max():.1f}], "
          f"B=[{lab[:, 2].min():.1f}, {lab[:, 2].max():.1f}]")
