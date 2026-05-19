#!/usr/bin/env python
"""
Evaluate the shadow removal model on the validation split.

Computes per-image PSNR and SSIM on the face+neck region
(using PNG with transparent background and BiSeNet face parsing).

Usage
────
    python -m evaluation.evaluate
    python -m evaluation.evaluate --checkpoint checkpoints/shadow_removal_best.pth
"""
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_split_path, get_checkpoint_path, get_inference_output_path
from models.shadow_remover import ShadowRemovalNet
from training.dataset import ShadowDataset
from training.losses import CombinedLoss

config = get_config()


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    """PSNR (dB) computed on the masked (face+neck) region only."""
    m = mask.expand_as(pred)
    mse = ((pred - target) ** 2 * m).sum() / (m.sum() + 1e-8)
    if mse < 1e-10:
        return 100.0
    return (10.0 * torch.log10(torch.tensor(1.0) / mse)).item()


def main():
    import argparse
    p = argparse.ArgumentParser(description="Shadow removal evaluation")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--img_size", type=int, default=256)
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = args.checkpoint
    if ckpt_path is None:
        best = get_checkpoint_path("shadow_removal_best.pth")
        if best.exists():
            ckpt_path = str(best)
        else:
            ckpts = sorted(get_checkpoint_path().glob("*.pth"))
            ckpt_path = str(ckpts[-1]) if ckpts else None
    if ckpt_path is None:
        print("ERROR: No checkpoint found. Train first: python -m training.train")
        sys.exit(1)

    print(f"\nEvaluating: {ckpt_path}")
    print(f"Device    : {device}\n")

    model = ShadowRemovalNet().to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw.get("model", raw))
    model.eval()

    data_cfg = config.get("data") or {}
    img_size = args.img_size or data_cfg.get("image_size", 256)

    val_ds = ShadowDataset(
        str(get_split_path("val_input.txt")),
        str(get_split_path("val_target.txt")),
        augment=False,
        image_size=img_size,
    )
    loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)
    criterion = CombinedLoss().to(device)

    total_loss = total_psnr = total_ssim = 0.0
    n = 0
    with torch.no_grad():
        for inp, tgt, mask in loader:
            inp, tgt, mask = inp.to(device), tgt.to(device), mask.to(device)
            pred = model(inp)
            loss = criterion(pred, tgt, mask)
            metrics = criterion.metrics(pred, tgt, mask)
            bs = inp.size(0)

            total_loss += loss.item() * bs
            total_psnr += masked_psnr(pred, tgt, mask) * bs
            total_ssim += metrics["ssim"] * bs
            n += bs

    avg_loss = total_loss / max(n, 1)
    avg_psnr = total_psnr / max(n, 1)
    avg_ssim = total_ssim / max(n, 1)

    print(f"{'─'*40}")
    print(f"  Images   : {n}")
    print(f"  Loss     : {avg_loss:.4f}")
    print(f"  PSNR(dB) : {avg_psnr:.2f}")
    print(f"  SSIM     : {avg_ssim:.4f}")
    print(f"{'─'*40}\n")

    results_dir = get_inference_output_path().parent
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "evaluation_results.txt"
    with open(out_file, "w") as f:
        f.write(f"Checkpoint : {ckpt_path}\n")
        f.write(f"Images     : {n}\n")
        f.write(f"Loss       : {avg_loss:.4f}\n")
        f.write(f"PSNR (dB)  : {avg_psnr:.2f}\n")
        f.write(f"SSIM       : {avg_ssim:.4f}\n")
    print(f"Results saved → {out_file}")


if __name__ == "__main__":
    main()