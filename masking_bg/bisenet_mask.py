"""
BiSeNet Face Parsing Mask Generator - using Masking project.
Best face parsing for face, neck, mouth, eyes, etc.

Config-based mask selection - choose which parts to include.
"""
import os
import sys
import importlib.util
import urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
MASKING_PATH = _REPO_ROOT / "Masking"

sys.modules['models.resnet'] = None
spec_resnet = importlib.util.spec_from_file_location("resnet_model", str(MASKING_PATH / "models" / "resnet.py"))
resnet_module = importlib.util.module_from_spec(spec_resnet)
spec_resnet.loader.exec_module(resnet_module)
sys.modules['models.resnet'] = resnet_module

spec_bisenet = importlib.util.spec_from_file_location("bisenet_model", str(MASKING_PATH / "models" / "bisenet.py"))
bisenet_module = importlib.util.module_from_spec(spec_bisenet)
spec_bisenet.loader.exec_module(bisenet_module)
BiSeNet = bisenet_module.BiSeNet

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


ATTRIBUTES = {
    'skin': 0, 'l_brow': 1, 'r_brow': 2, 'l_eye': 3, 'r_eye': 4,
    'eye_g': 5, 'l_ear': 6, 'r_ear': 7, 'ear_r': 8, 'nose': 9,
    'mouth': 10, 'u_lip': 11, 'l_lip': 12, 'neck': 13, 'neck_l': 14,
    'cloth': 15, 'hair': 16, 'hat': 17,
}

DEFAULT_INCLUDE = {
    'skin': True,
    'l_brow': True,
    'r_brow': True,
    'l_eye': True,
    'r_eye': True,
    'eye_g': False,
    'l_ear': False,
    'r_ear': False,
    'ear_r': False,
    'nose': True,
    'mouth': True,
    'u_lip': True,
    'l_lip': True,
    'neck': True,
    'neck_l': True,
    'cloth': False,
    'hair': False,
    'hat': False,
}

BISENET_WEIGHT_URLS = {
    "resnet18": "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.pt",
    "resnet34": "https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.pt",
}


def ensure_bisenet_weight(model_name: str) -> Path:
    """Return the local BiSeNet weight path, downloading it if needed."""
    weight_path = MASKING_PATH / "weights" / f"{model_name}.pt"
    if weight_path.exists():
        return weight_path

    url = BISENET_WEIGHT_URLS.get(model_name)
    if not url:
        available = ", ".join(sorted(BISENET_WEIGHT_URLS))
        raise ValueError(f"Unsupported BiSeNet model '{model_name}'. Available: {available}")

    weight_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = weight_path.with_suffix(".pt.download")
    print(f"BiSeNet weight missing: {weight_path}")
    print(f"Downloading {model_name} weights from {url}")

    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.replace(weight_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    return weight_path


class BiSeNetMaskGenerator:
    def __init__(
        self,
        model_name: str = "resnet34",
        device: str = None,
        include: dict = None,
    ):
        self.model_name = model_name
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.num_classes = 19
        self.input_size = (512, 512)
        self.include = include if include is not None else DEFAULT_INCLUDE
        self._model = None
        self._init()

    @classmethod
    def from_config(cls, config: dict) -> "BiSeNetMaskGenerator":
        """Create from config dict."""
        masking_config = config.get('masking', {})
        return cls(
            model_name=masking_config.get('model_name', 'resnet34'),
            include=masking_config.get('include', None),
        )

    def _init(self):
        weight_path = ensure_bisenet_weight(self.model_name)
        self._model = BiSeNet(self.num_classes, backbone_name=self.model_name)
        self._model.to(self.device)
        self._model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self._model.eval()

        self._transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        resized = image.resize(self.input_size, resample=Image.BILINEAR)
        tensor = self._transform(resized)
        return tensor.unsqueeze(0).to(self.device)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self.generate_mask(image_rgb)

    def generate_mask(
        self,
        image_rgb: np.ndarray,
        include_face: bool = None,
        include_neck: bool = None,
        include_mouth: bool = None,
    ) -> np.ndarray:
        """
        Generate mask for specified regions.
        
        Args:
            image_rgb: RGB image as numpy array (H, W, 3)
            include_face: Override config - include face/skin regions
            include_neck: Override config - include neck regions
            include_mouth: Override config - include mouth regions
        
        Returns:
            Binary mask (H, W) with values 0 or 1
        """
        h, w = image_rgb.shape[:2]
        pil_image = Image.fromarray(image_rgb)

        image_batch = self._prepare_image(pil_image)
        
        with torch.no_grad():
            output = self._model(image_batch)[0]
            predicted_mask = output.squeeze(0).cpu().numpy().argmax(0)

        model_h, model_w = predicted_mask.shape
        
        mask = np.zeros((model_h, model_w), dtype=np.float32)
        
        for part, enabled in self.include.items():
            if enabled and part in ATTRIBUTES:
                mask[predicted_mask == ATTRIBUTES[part]] = 1.0

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_resized = np.array(mask_pil.resize((w, h), resample=Image.NEAREST)) / 255.0
        
        return mask_resized.astype(np.float32)

    def get_full_parsing(self, image_rgb: np.ndarray) -> np.ndarray:
        """Get full 19-class parsing map."""
        h, w = image_rgb.shape[:2]
        pil_image = Image.fromarray(image_rgb)
        
        image_batch = self._prepare_image(pil_image)
        
        with torch.no_grad():
            output = self._model(image_batch)[0]
            parsing = output.squeeze(0).cpu().numpy().argmax(0)
        
        parsing_pil = Image.fromarray(parsing.astype(np.uint8))
        parsing_resized = np.array(parsing_pil.resize((w, h), resample=Image.NEAREST))
        
        return parsing_resized

    def get_parsing_classes(self) -> dict:
        """Return the mapping of attribute names to class indices."""
        return ATTRIBUTES.copy()

    def set_include(self, part: str, enabled: bool):
        """Enable/disable a specific part."""
        if part in self.include:
            self.include[part] = enabled

    def get_enabled_parts(self) -> list:
        """Return list of currently enabled parts."""
        return [part for part, enabled in self.include.items() if enabled]
