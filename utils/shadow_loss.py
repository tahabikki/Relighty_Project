import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.color_utils import rgb_to_lab, lab_to_rgb, rgb_to_gray


class PerceptualLoss(nn.Module):
    """
    VGG16 perceptual loss.
    Compares high-level features (texture, structure) not just pixels.
    Prevents blurry outputs and preserves facial details.
    
    OPTIMIZED: Moves VGG to device ONCE at init, not every forward pass.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.slice1 = nn.Sequential(*list(vgg.features)[:9]).eval()    # relu2_2
        self.slice2 = nn.Sequential(*list(vgg.features)[9:16]).eval()  # relu3_3
        
        # Move to device ONCE at init
        self.slice1.to(device)
        self.slice2.to(device)
        
        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device))

    def forward(self, pred, target):
        # Normalize to ImageNet stats
        pred_n   = (pred   - self.mean) / self.std
        target_n = (target - self.mean) / self.std

        # Compare VGG features (no .to() needed - already on device)
        f1_p = self.slice1(pred_n)
        f1_t = self.slice1(target_n)
        f2_p = self.slice2(f1_p)
        f2_t = self.slice2(f1_t)

        return nn.functional.l1_loss(f1_p, f1_t) + \
               nn.functional.l1_loss(f2_p, f2_t)


class ShadowRemovalLoss(nn.Module):
    """
    Enhanced loss for ICAO-compliant shadow removal:
      
    Components:
      1. L1 Loss (masked)       → Pixel accuracy on shadow regions
      2. Perceptual Loss        → Texture/detail preservation (VGG)
      3. LAB Color Loss         → Skin tone preservation (L-channel only)
      4. Light Uniformity Loss  → ICAO standard compliance
      5. Edge Consistency Loss  → Sharp detail recovery
    
    Weights (default):
      α = 1.5  (L1: focus on shadow removal)
      β = 0.3  (Perceptual: maintain details)
      γ = 0.8  (Color: preserve skin tone)
      δ = 0.5  (Uniformity: ICAO standard)
      ε = 0.2  (Edges: sharpen recovered regions)
    """
    def __init__(self,
                 l1_weight           = 1.5,
                 perceptual_weight   = 0.3,
                 color_weight        = 0.8,
                 uniformity_weight   = 0.5,
                 edge_weight         = 0.2,
                 device              = 'cuda',
                 ssim_weight=0, identity_weight=0):
        super().__init__()
        self.w_l1       = l1_weight
        self.w_perc     = perceptual_weight
        self.w_color    = color_weight
        self.w_uniform  = uniformity_weight
        self.w_edge     = edge_weight
        self.device     = device
        
        # Initialize loss components
        self.perc       = PerceptualLoss(device=device)
        self.color_loss = LABColorConsistencyLoss()
        self.uniform_loss = LightUniformityLoss()
        self.edge_loss  = EdgeConsistencyLoss()
        
        print(f"Loss: Enhanced ICAO-Compliant Shadow Removal")
        print(f"  L1 (weight={l1_weight}) + Perceptual (weight={perceptual_weight})")
        print(f"  + Color (weight={color_weight}) + Uniformity (weight={uniformity_weight})")
        print(f"  + Edge (weight={edge_weight})")
        print(f"  Device: {device}")

    def forward(self, pred, target, mask=None, shadow_mask=None):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            mask: Face region mask [B, 1, H, W]
            shadow_mask: Shadow-specific region mask [B, 1, H, W]
        """
        # 1. L1 Loss (masked by face region)
        if mask is not None:
            m = mask.expand_as(pred)
            l1 = (torch.abs(pred - target) * m).sum() / (m.sum() + 1e-8)
        else:
            l1 = F.l1_loss(pred, target)
        
        # 2. Perceptual Loss (full image)
        perc = self.perc(pred, target)
        
        # 3. LAB Color Consistency Loss (preserve skin tone)
        color = self.color_loss(pred, target, mask)
        
        # 4. Light Uniformity Loss (ICAO compliance)
        uniform = self.uniform_loss(pred, target, mask)
        
        # 5. Edge Consistency Loss (sharp details)
        edge = self.edge_loss(pred, target, mask)
        
        # Combine all losses
        total_loss = (
            self.w_l1 * l1 +
            self.w_perc * perc +
            self.w_color * color +
            self.w_uniform * uniform +
            self.w_edge * edge
        )
        
        return total_loss

    def get_metrics(self, pred, target, mask=None):
        """Return individual loss components for monitoring."""
        if mask is not None:
            m = mask.expand_as(pred)
            l1 = (torch.abs(pred - target) * m).sum() / (m.sum() + 1e-8)
        else:
            l1 = F.l1_loss(pred, target)
        
        perc = self.perc(pred, target)
        color = self.color_loss(pred, target, mask)
        uniform = self.uniform_loss(pred, target, mask)
        edge = self.edge_loss(pred, target, mask)
        
        return {
            'l1': l1.item(),
            'perceptual': perc.item(),
            'color': color.item(),
            'uniformity': uniform.item(),
            'edge': edge.item()
        }


class LABColorConsistencyLoss(nn.Module):
    """
    Preserve skin tone while removing shadows using LAB color space.
    
    Strategy:
      - Convert RGB to LAB
      - Allow L (lightness) to change freely → removes shadows
      - Penalize changes to A, B (color) → preserves skin tone
      - Weight them appropriately for ICAO compliance
    """
    def __init__(self, ab_weight=0.5):
        super().__init__()
        self.ab_weight = ab_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            mask: Face region mask [B, 1, H, W]
        """
        # Convert to LAB
        pred_lab = rgb_to_lab(pred)
        target_lab = rgb_to_lab(target)
        
        # L channel loss (lightness) - allow to change for shadow removal
        L_loss = F.l1_loss(pred_lab[:, 0], target_lab[:, 0])
        
        # A, B channels loss (color) - preserve original color
        AB_loss = F.l1_loss(pred_lab[:, 1:], target_lab[:, 1:])
        
        # If face mask provided, apply it
        if mask is not None:
            m = mask.squeeze(1)  # [B, H, W]
            
            # Weighted L loss
            L_diff = torch.abs(pred_lab[:, 0] - target_lab[:, 0])
            L_loss = (L_diff * m).sum() / (m.sum() + 1e-8)
            
            # Weighted AB loss
            AB_diff = torch.abs(pred_lab[:, 1:] - target_lab[:, 1:])
            m_exp = m.unsqueeze(1).expand_as(AB_diff)
            AB_loss = (AB_diff * m_exp).sum() / (m_exp.sum() + 1e-8)
        
        # Combined: allow L change but penalize color shift
        return L_loss + self.ab_weight * AB_loss


class LightUniformityLoss(nn.Module):
    """
    ICAO standard: Enforce uniform face lighting to avoid shadows.
    
    Strategy:
      - Compute target image lightness (L in LAB)
      - Penalize non-uniformity (high variance)
      - Encourages smooth, even illumination
    """
    def __init__(self, uniformity_factor=0.1):
        super().__init__()
        self.uniformity_factor = uniformity_factor
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            target: Target ground truth [B, 3, H, W]
            mask: Face region mask [B, 1, H, W]
        """
        # Convert target to LAB and extract lightness
        target_lab = rgb_to_lab(target)
        L = target_lab[:, 0]  # [B, H, W]
        
        # Compute local variance (using Laplacian as proxy for non-uniformity)
        # Kernel: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]]
        laplacian_kernel = torch.tensor(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
            dtype=torch.float32, device=L.device
        ).unsqueeze(0).unsqueeze(0)
        
        L_exp = L.unsqueeze(1)
        non_uniformity = F.conv2d(L_exp, laplacian_kernel, padding=1)
        
        # Compute uniformity loss
        uniformity_loss = non_uniformity.abs().mean()
        
        # Apply mask if provided
        if mask is not None:
            m = mask.squeeze(1)  # [B, H, W]
            non_uniformity_masked = non_uniformity.squeeze(1)  # [B, H, W]
            # No padding needed - conv2d with padding=1 maintains spatial dimensions
            uniformity_loss = (non_uniformity_masked.abs() * m).sum() / (m.sum() + 1e-8)
        
        return self.uniformity_factor * uniformity_loss


class EdgeConsistencyLoss(nn.Module):
    """
    Preserve sharp edges in recovered shadow regions.
    
    Strategy:
      - Compute edge map (Sobel) for both predicted and target
      - Penalize edge mismatch
      - Ensures details aren't blurred during shadow removal
    """
    def __init__(self, edge_weight=1.0):
        super().__init__()
        self.edge_weight = edge_weight
    
    def forward(self, pred, target, mask=None):
        """
        Args:
            pred: Predicted image [B, 3, H, W]
            target: Target image [B, 3, H, W]
            mask: Face region mask [B, 1, H, W]
        """
        # Convert to grayscale for edge detection
        pred_gray = rgb_to_gray(pred)  # [B, 1, H, W]
        target_gray = rgb_to_gray(target)
        
        # Sobel edge detection
        pred_edges = self._compute_edges(pred_gray)
        target_edges = self._compute_edges(target_gray)
        
        # Edge loss
        edge_loss = F.l1_loss(pred_edges, target_edges)
        
        # Apply mask if provided
        if mask is not None:
            edge_diff = torch.abs(pred_edges - target_edges)
            m = mask
            edge_loss = (edge_diff * m).sum() / (m.sum() + 1e-8)
        
        return self.edge_weight * edge_loss
    
    def _compute_edges(self, gray_img):
        """Compute edge map using Sobel operator."""
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=gray_img.device
        ).unsqueeze(0).unsqueeze(0)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=gray_img.device
        ).unsqueeze(0).unsqueeze(0)
        
        edges_x = F.conv2d(gray_img, sobel_x, padding=1)
        edges_y = F.conv2d(gray_img, sobel_y, padding=1)
        
        edges = torch.sqrt(edges_x**2 + edges_y**2 + 1e-8)
        return edges
