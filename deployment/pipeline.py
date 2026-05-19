"""
End-to-end shadow removal pipeline using BiSeNet face parsing.

Uses PNG with transparent background throughout.
"""
import sys
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.shadow_remover import ShadowRemovalNet
from masking_bg.bisenet_mask import BiSeNetMaskGenerator


def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def find_checkpoint(checkpoint_dir: Optional[Path] = None) -> Optional[Path]:
    if checkpoint_dir is None:
        checkpoint_dir = PROJECT_ROOT / "checkpoints"
    best = checkpoint_dir / "shadow_removal_best.pth"
    if best.exists():
        return best
    ckpts = sorted(checkpoint_dir.glob("*.pth"))
    return ckpts[-1] if ckpts else None


def get_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


class ShadowRemovalPipeline:
    """
    End-to-end pipeline using BiSeNet face parsing.
    Outputs PNG with transparent background.
    """

    def __init__(
        self,
        checkpoint: Optional[Union[str, Path]] = None,
        device: str = "auto",
        image_size: int = 256,
    ):
        self.config = load_config()
        self.image_size = self.config.get("data", {}).get("image_size", image_size)
        self.device = get_device(device)

        self.checkpoint = Path(checkpoint) if checkpoint else find_checkpoint()
        if self.checkpoint is None:
            raise RuntimeError("No checkpoint found. Train the model first.")

        self.model = self._load_model()
        self.mask_gen = BiSeNetMaskGenerator.from_config(self.config)

    def _load_model(self) -> ShadowRemovalNet:
        model = ShadowRemovalNet().to(self.device)
        state = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        model.eval()
        return model

    @torch.no_grad()
    def process_image(self, image: np.ndarray) -> np.ndarray:
        """Process image and return BGRA (PNG with transparency)."""
        orig_h, orig_w = image.shape[:2]
        resized = cv2.resize(image, (self.image_size, self.image_size))

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        combined_mask = self.mask_gen.generate_mask(rgb, include_face=True, include_neck=True, include_mouth=True)

        tensor = (
            torch.from_numpy(rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )
        pred = self.model(tensor)
        pred_rgb = (pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pred_bgr = cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR)

        dynamic_strength = 0.65
        result_img = self._blend_with_texture(resized, pred_bgr, combined_mask, dynamic_strength)

        if (orig_h, orig_w) != (self.image_size, self.image_size):
            result_img = cv2.resize(result_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
            combined_mask = cv2.resize(combined_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        b, g, r = cv2.split(result_img)
        alpha = (cv2.GaussianBlur(combined_mask, (21, 21), 0) * 255).astype(np.uint8)
        return cv2.merge([b, g, r, alpha])

    def _blend_with_texture(self, original, prediction, mask, strength):
        image_f = original.astype(np.float32)
        pred_f = prediction.astype(np.float32)
        blur = cv2.GaussianBlur(image_f, (0, 0), sigmaX=2.0)
        texture_hf = np.clip(image_f - blur, -15.0, 15.0)

        orig_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
        pred_lab = cv2.cvtColor(pred_f, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_diff = pred_lab[:, :, 0] - orig_lab[:, :, 0]

        corrected_lab = orig_lab.copy()
        corrected_lab[:, :, 0] = np.clip(orig_lab[:, :, 0] + l_diff * strength, 0, 255)
        corrected_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        corrected = corrected_bgr + texture_hf

        alpha = (cv2.GaussianBlur(mask, (15, 15), 0) * strength)[..., None]
        return (image_f * (1.0 - alpha) + corrected * alpha).clip(0, 255).astype(np.uint8)

    def process_folder(self, input_dir: Union[str, Path], output_dir: Union[str, Path]) -> int:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        files = sorted(f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in extensions)

        processed = 0
        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                continue
            result = self.process_image(img)
            out_path = output_dir / f"{f.stem}.png"
            cv2.imwrite(str(out_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            processed += 1

        return processed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Shadow removal pipeline (PNG output)")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config()
    image_size = args.image_size or config.get("data", {}).get("image_size", 256)

    pipeline = ShadowRemovalPipeline(
        checkpoint=args.checkpoint,
        device=args.device,
        image_size=image_size,
    )

    input_path = Path(args.input)

    if input_path.is_file():
        output_path = Path(args.output) if args.output else input_path.parent / f"clean_{input_path.stem}.png"
        img = cv2.imread(str(input_path))
        if img is None:
            print(f"Error: Cannot read {input_path}")
            return 1
        result = pipeline.process_image(img)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), result, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(f"Saved → {output_path}")

    elif input_path.is_dir():
        output_path = Path(args.output) if args.output else PROJECT_ROOT / "Results" / "output"
        count = pipeline.process_folder(input_path, output_path)
        print(f"Processed {count} images → {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())