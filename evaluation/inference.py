#!/usr/bin/env python
"""
Shadow removal inference using BiSeNet face parsing.

Output:
- Original image with only the configured face/neck mask corrected.

Usage
────
    python -m evaluation.inference --input photo.jpg --output result.png
"""
import sys
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_checkpoint_path, get_inference_output_path
from models.shadow_remover import ShadowRemovalNet
from masking_bg.bisenet_mask import BiSeNetMaskGenerator
from postprocessing.fix_light import fix_light

config = get_config()
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(ckpt_path: str, device: torch.device) -> ShadowRemovalNet:
    model = ShadowRemovalNet().to(device)
    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(raw.get("model", raw))
    model.eval()
    return model


def find_checkpoint() -> Optional[Path]:
    best = get_checkpoint_path("shadow_removal_best.pth")
    if best.exists():
        return best
    ckpts = sorted(get_checkpoint_path().glob("*.pth"))
    return ckpts[-1] if ckpts else None


def _apply_texture_preservation(original, prediction, mask, dynamic_strength=0.65):
    image_f = original.astype(np.float32)
    pred_f = prediction.astype(np.float32)
    blur = cv2.GaussianBlur(image_f, (0, 0), sigmaX=2.0)
    texture_hf = np.clip(image_f - blur, -15.0, 15.0)

    orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    pred_lab = cv2.cvtColor(pred_f, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_diff = pred_lab[:, :, 0] - orig_lab[:, :, 0]

    corrected_lab = orig_lab.copy()
    corrected_lab[:, :, 0] = np.clip(orig_lab[:, :, 0] + l_diff * dynamic_strength, 0, 255)
    corrected_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    corrected_textured = corrected_bgr + texture_hf

    hard_mask = (mask > 0.5).astype(np.float32)
    alpha = (cv2.GaussianBlur(mask, (15, 15), 0) * hard_mask * dynamic_strength)[..., None]
    blended = image_f * (1.0 - alpha) + corrected_textured * alpha

    return np.clip(blended, 0, 255).astype(np.uint8)


@torch.no_grad()
def remove_shadow(model, mask_gen, img_bgr, device, img_size=256):
    """
    Remove facial shadows using BiSeNet face parsing.
    Keeps original background, applies model only on face+neck+ear.
    
    Returns:
        BGR image with original background preserved
    """
    orig_h, orig_w = img_bgr.shape[:2]
    orig_bgr = img_bgr.copy()

    resized = cv2.resize(img_bgr, (img_size, img_size))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    face_mask = mask_gen.generate_mask(rgb)

    tensor = (torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device))
    pred = model(tensor)
    pred_rgb = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask_bool = face_mask > 0.5
    std_dev = float(np.std(gray[mask_bool])) if np.any(mask_bool) else 50.0
    mean_diff = float(np.abs(pred_bgr.astype(np.float32) - resized.astype(np.float32))[mask_bool].mean()) if np.any(mask_bool) else 0.0

    if mean_diff < (std_dev * 0.15):
        result_resized = resized
    else:
        dynamic_strength = float(np.clip(0.30 + (mean_diff / 100.0), 0.35, 0.60))
        result_resized = _apply_texture_preservation(resized, pred_bgr, face_mask, dynamic_strength)

    result_resized = cv2.GaussianBlur(result_resized, (3, 3), 0)

    if (orig_h, orig_w) != (img_size, img_size):
        result_resized = cv2.resize(result_resized, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
        face_mask = cv2.resize(face_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    face_mask_hard = (face_mask > 0.5).astype(np.float32)
    face_mask_smooth = cv2.GaussianBlur(face_mask.astype(np.float32), (21, 21), 0) * face_mask_hard

    result_f = result_resized.astype(np.float32)
    orig_f = orig_bgr.astype(np.float32)

    final = orig_f * (1 - face_mask_smooth[:, :, np.newaxis]) + result_f * face_mask_smooth[:, :, np.newaxis]
    return np.clip(final, 0, 255).astype(np.uint8)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Face shadow removal")
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--img_size", type=int, default=None)
    args = p.parse_args()

    data_cfg = config.get("data") or {}
    args.img_size = args.img_size or data_cfg.get("image_size", 256)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt_path = args.checkpoint or find_checkpoint()
    if ckpt_path is None:
        print("ERROR: No checkpoint. Train first: python -m training.train")
        sys.exit(1)

    print(f"Checkpoint: {ckpt_path}")
    print(f"Device: {device}")
    print("Background: original preserved")
    print("Engine: postprocessing.fix_light")

    model = load_model(str(ckpt_path), device)
    mask_gen = BiSeNetMaskGenerator.from_config(config)
    print(f"Masking: {mask_gen.model_name}, enabled parts: {mask_gen.get_enabled_parts()}")
    print("Ready.\n")

    inp_path = Path(args.input)

    if inp_path.is_file():
        out_path = Path(args.output) if args.output else inp_path.parent / f"clean_{inp_path.stem}.png"
        img = cv2.imread(str(inp_path))
        if img is None:
            print(f"ERROR: Cannot read {inp_path}")
            sys.exit(1)
        
        result = fix_light(img, model_bundle=(model, device), mask_gen=mask_gen)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        if out_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
            out_path = out_path.with_suffix('.png')
        
        cv2.imwrite(str(out_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 0] if out_path.suffix.lower() == '.png' else [])
        print(f"Saved -> {out_path}")

    elif inp_path.is_dir():
        out_dir = Path(args.output) if args.output else get_inference_output_path()
        out_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(f for f in inp_path.iterdir() if f.suffix.lower() in IMG_EXTS)
        if not files:
            print(f"No images in {inp_path}")
            sys.exit(0)

        print(f"Processing {len(files)} images -> {out_dir}\n")
        for i, f in enumerate(files, 1):
            img = cv2.imread(str(f))
            if img is None:
                print(f"  [{i:>4}/{len(files)}]  SKIP: {f.name}")
                continue
            
            result = fix_light(img, model_bundle=(model, device), mask_gen=mask_gen)
            ext = f.suffix if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'] else '.png'
            out_file = out_dir / f"{f.stem}{ext}"
            cv2.imwrite(str(out_file), result, [cv2.IMWRITE_PNG_COMPRESSION, 0] if ext == ".png" else [])
            print(f"  [{i:>4}/{len(files)}]  {f.name}  ->  {out_file.name}")

        print(f"\nDone. Results in {out_dir}")
    else:
        print(f"ERROR: {inp_path} not found")
        sys.exit(1)


if __name__ == "__main__":
    main()
