"""
Shadow removal model: U-Net with pretrained ResNet-34 encoder.

Design rationale
────────────────
• ResNet-34 encoder pre-trained on ImageNet → rich face/texture features
  from day one, critical when the dataset is small (~500–1 000 pairs).
• 5-level U-Net decoder with skip connections → spatial detail preserved.
• Sigmoid output → pixel values strictly in [0, 1], matching dataset range.
• Separate learning-rate groups → encoder stays close to ImageNet init
  while the decoder learns shadow removal aggressively.

Input  : (B, 3, 256, 256) float32, values in [0, 1]
Output : (B, 3, 256, 256) float32, values in [0, 1]
"""
import torch
import torch.nn as nn
import torchvision.models as models


def _up_block(in_ch: int, skip_ch: int, out_ch: int) -> nn.Sequential:
    """Conv-BN-ReLU × 2 decoder block (receives concatenated up + skip)."""
    return nn.Sequential(
        nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class ShadowRemovalNet(nn.Module):
    """
    U-Net with ResNet-34 encoder for face shadow removal.

    Parameter count  : ~29 M
    Checkpoint size  : ~115 MB
    Inference speed  : ~50 ms / image on a mid-range GPU
    Recommended epochs: 80–120
    """

    def __init__(self):
        super().__init__()

        # ── Encoder (ResNet-34, ImageNet pre-trained) ─────────────────────────
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.enc0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        #   out: 64 ch, H/2  × W/2  (128 × 128 for 256 input)
        self.pool = resnet.maxpool          # → H/4  × W/4  (64 × 64)
        self.enc1 = resnet.layer1           # 64 ch, H/4
        self.enc2 = resnet.layer2           # 128 ch, H/8
        self.enc3 = resnet.layer3           # 256 ch, H/16
        self.enc4 = resnet.layer4           # 512 ch, H/32  (8 × 8)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up4  = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = _up_block(256, 256, 256)   # H/16

        self.up3  = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = _up_block(128, 128, 128)   # H/8

        self.up2  = nn.ConvTranspose2d(128,  64, 2, stride=2)
        self.dec2 = _up_block( 64,  64,  64)   # H/4

        self.up1  = nn.ConvTranspose2d( 64,  64, 2, stride=2)
        self.dec1 = _up_block( 64,  64,  64)   # H/2

        self.up0  = nn.ConvTranspose2d( 64,  32, 2, stride=2)
        self.dec0 = nn.Sequential(              # H (full resolution)
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16,  3, 1),
            nn.Sigmoid(),       # strict [0, 1] output
        )

    # ── Encoder parameter groups (for split LR in optimizer) ─────────────────
    def encoder_parameters(self):
        for m in (self.enc0, self.enc1, self.enc2, self.enc3, self.enc4):
            yield from m.parameters()

    def decoder_parameters(self):
        encoder_ids = {id(p) for p in self.encoder_parameters()}
        for p in self.parameters():
            if id(p) not in encoder_ids:
                yield p

    # ── Forward ───────────────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)               # 64  × 128 × 128
        e1 = self.enc1(self.pool(e0))   # 64  ×  64 ×  64
        e2 = self.enc2(e1)              # 128 ×  32 ×  32
        e3 = self.enc3(e2)              # 256 ×  16 ×  16
        e4 = self.enc4(e3)              # 512 ×   8 ×   8

        # Bottleneck
        b  = self.bottleneck(e4)        # 512 ×   8 ×   8

        # Decoder (skip connections from encoder)
        d4 = self.dec4(torch.cat([self.up4(b),  e3], 1))   # 256 × 16 × 16
        d3 = self.dec3(torch.cat([self.up3(d4), e2], 1))   # 128 × 32 × 32
        d2 = self.dec2(torch.cat([self.up2(d3), e1], 1))   #  64 × 64 × 64
        d1 = self.dec1(torch.cat([self.up1(d2), e0], 1))   #  64 ×128 ×128
        d0 = self.dec0(self.up0(d1))                        #  32 ×256 ×256

        return self.head(d0)                                #   3 ×256 ×256
