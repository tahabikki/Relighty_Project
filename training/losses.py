"""
Loss functions for face shadow removal.

All losses accept an optional face mask so the model only
optimises pixels inside the face region.

Mask convention: float32 tensor of shape (B, 1, H, W), values in [0, 1].
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── SSIM ─────────────────────────────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """
    Structural Similarity loss.
    Returns  (1 - SSIM), so 0 is perfect.
    """

    def __init__(self, window_size: int = 11, channels: int = 3):
        super().__init__()
        self.window_size = window_size
        self.channels    = channels
        sigma   = 1.5
        coords  = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        g       = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
        g       = g / g.sum()
        kernel  = g.outer(g).unsqueeze(0).unsqueeze(0)   # 1 x 1 x W x W
        kernel  = kernel.repeat(channels, 1, 1, 1)        # C x 1 x W x W
        self.register_buffer("window", kernel)

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Clip to valid range to prevent NaN from invalid values
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)
        
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        w  = self.window.to(pred.device)
        p  = self.window_size // 2
        g  = self.channels

        mu1    = F.conv2d(pred,         w, padding=p, groups=g)
        mu2    = F.conv2d(target,       w, padding=p, groups=g)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu12   = mu1 * mu2
        s1     = F.conv2d(pred   ** 2,      w, padding=p, groups=g) - mu1_sq
        s2     = F.conv2d(target ** 2,      w, padding=p, groups=g) - mu2_sq
        s12    = F.conv2d(pred * target,    w, padding=p, groups=g) - mu12
        
        # Clamp variance to prevent division issues
        s1 = torch.clamp(s1, min=0.0)
        s2 = torch.clamp(s2, min=0.0)

        denom = (mu1_sq + mu2_sq + C1) * (s1 + s2 + C2)
        denom = torch.clamp(denom, min=1e-8)  # Prevent division by zero
        
        ssim_map = (
            (2.0 * mu12 + C1) * (2.0 * s12 + C2)
        ) / denom
        
        # Clamp SSIM to valid range
        ssim_map = torch.clamp(ssim_map, 0.0, 2.0)
        loss_map = 1.0 - ssim_map   # per-pixel SSIM loss in [0, 2]
        
        # Detect NaN and fallback to L1
        if torch.isnan(loss_map).any():
            return torch.abs(pred - target).mean()

        if mask is not None:
            m = mask.expand_as(loss_map)
            mask_sum = m.sum()
            if mask_sum < 1e-6:
                return torch.abs(pred - target).mean()
            loss = (loss_map * m).sum() / (mask_sum + 1e-8)
            if torch.isnan(loss) or torch.isinf(loss):
                return torch.abs(pred - target).mean()
            return loss
        
        loss = loss_map.mean()
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.abs(pred - target).mean()
        return loss


# ─── Combined loss ────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Face-aware shadow-removal loss:

        L = l1_w * masked_L1(pred, target)
          + ssim_w * masked_SSIM(pred, target)

    Both terms are weighted so gradient magnitudes are comparable.
    Default weights (1.0 L1 + 0.5 SSIM) work well empirically.
    """

    def __init__(self, l1_weight: float = 1.0, ssim_weight: float = 0.5):
        super().__init__()
        self.l1_weight   = l1_weight
        self.ssim_weight = ssim_weight
        self.ssim        = SSIMLoss()

    @staticmethod
    def _masked_l1(
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor],
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        if mask is not None:
            m = mask.expand_as(diff)
            mask_sum = m.sum()
            # Safety: if mask is empty/zero, fall back to unmasked loss
            if mask_sum < 1e-6:
                return diff.mean()
            loss = (diff * m).sum() / (mask_sum + 1e-8)
            # Safety: clip any NaN/Inf values
            if torch.isnan(loss) or torch.isinf(loss):
                return diff.mean()
            return loss
        return diff.mean()

    def forward(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        l1   = self._masked_l1(pred, target, mask)
        ssim = self.ssim(pred, target, mask)
        return self.l1_weight * l1 + self.ssim_weight * ssim

    @torch.no_grad()
    def metrics(
        self,
        pred:   torch.Tensor,
        target: torch.Tensor,
        mask:   Optional[torch.Tensor] = None,
    ) -> dict:
        return {
            "l1":   self._masked_l1(pred, target, mask).item(),
            "ssim": 1.0 - self.ssim(pred, target, mask).item(),   # higher = better
        }
