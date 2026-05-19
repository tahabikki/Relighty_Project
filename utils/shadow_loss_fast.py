import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleShadowRemovalLoss(nn.Module):
    """
    Minimal loss for fast training testing.
    Just L1 loss on masked region.
    """
    def __init__(self,
                 l1_weight=1.5,
                 perceptual_weight=0.0,
                 color_weight=0.0,
                 uniformity_weight=0.0,
                 edge_weight=0.0,
                 **kwargs):
        super().__init__()
        self.w_l1 = l1_weight
        
        print(f"Loss: FAST SIMPLE (L1 only for testing)")
        print(f"  L1 (weight={l1_weight})")

    def forward(self, pred, target, mask=None, shadow_mask=None):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            mask: Face region mask [B, 1, H, W]
            shadow_mask: Shadow-specific region mask [B, 1, H, W]
        """
        # Simple L1 loss
        if mask is not None:
            m = mask.expand_as(pred)
            loss = (torch.abs(pred - target) * m).sum() / (m.sum() + 1e-8)
        else:
            loss = F.l1_loss(pred, target)
        
        return self.w_l1 * loss
