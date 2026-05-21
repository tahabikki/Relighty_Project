#!/usr/bin/env python
"""
Minimal inference for train branch.
Raw model output without fix_light post-processing.
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_checkpoint_path
from models.shadow_remover import ShadowRemovalNet
from masking_bg.bisenet_mask import BiSeNetMaskGenerator

config = get_config()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(ckpt_path: str, device: torch.device):
    model = ShadowRemovalNet().to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw.get("model", raw))
    model.eval()
    return model


def find_checkpoint():
    best = get_checkpoint_path("shadow_removal_best.pth")
    if best.exists():
        return best
    ckpts = sorted(get_checkpoint_path().glob("*.pth"))
    return ckpts[-1] if ckpts else None


@torch.no_grad()
def raw_inference(model, mask_gen, img_bgr, device, img_size=256):
    orig_h, orig_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (img_size, img_size))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    face_mask = mask_gen.generate_mask(rgb)

    tensor = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    pred = model(tensor)
    pred_rgb = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    if (orig_h, orig_w) != (img_size, img_size):
        pred_bgr = cv2.resize(pred_bgr, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        face_mask = cv2.resize(face_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    mask_soft = cv2.GaussianBlur(face_mask.astype(np.float32), (15, 15), 0)
    mask_soft = np.clip(mask_soft, 0.0, 1.0)[..., None]

    blended = img_bgr.astype(np.float32) * (1.0 - mask_soft) + pred_bgr.astype(np.float32) * mask_soft
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Raw shadow removal inference (no fix_light)")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--img_size", type=int, default=256)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = args.checkpoint or find_checkpoint()
    if ckpt_path is None:
        print("ERROR: No checkpoint found. Train first: python -m training.train")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print("Raw model output (no fix_light)")

    model = load_model(str(ckpt_path), device)
    mask_gen = BiSeNetMaskGenerator.from_config(config)

    inp_path = Path(args.input)

    if inp_path.is_file():
        out_path = Path(args.output) if args.output else inp_path.parent / f"clean_{inp_path.stem}.png"
        img = cv2.imread(str(inp_path))
        if img is None:
            print(f"ERROR: Cannot read {inp_path}")
            sys.exit(1)

        result = raw_inference(model, mask_gen, img, device, args.img_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), result)
        print(f"Saved -> {out_path}")

    elif inp_path.is_dir():
        out_dir = Path(args.output) if args.output else PROJECT_ROOT / "Results" / "raw_output"
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(f for f in inp_path.iterdir() if f.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"No images in {inp_path}")
            return

        print(f"Processing {len(files)} images -> {out_dir}\n")
        for i, f in enumerate(files, 1):
            img = cv2.imread(str(f))
            if img is None:
                continue
            result = raw_inference(model, mask_gen, img, device, args.img_size)
            out_file = out_dir / f"{f.stem}{f.suffix}"
            cv2.imwrite(str(out_file), result)
            print(f"  [{i:>4}/{len(files)}]  {f.name}  ->  {out_file.name}")
        print(f"\nDone. Results in {out_dir}")
    else:
        print(f"ERROR: {inp_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
