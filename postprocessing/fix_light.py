import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import argparse
import sys
import json
import mediapipe as mp
import subprocess
import torch
import yaml
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from rembg import remove

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.shadow_remover import ShadowRemovalNet
from masking_bg.bisenet_mask import BiSeNetMaskGenerator


def load_config() -> dict:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


_config = load_config()


def ensure_model(path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        subprocess.run(["python", "download_models.py", path])


def parse_args():
    parser = argparse.ArgumentParser(description="ICAO Image Processing Pipeline")
    parser.add_argument("--input", required=True, help="Image input")
    parser.add_argument("--mask", default=None, help="Chemin vers un masque (image)")
    parser.add_argument("--landmarks", default=None, help="Chemin vers les landmarks (JSON)")
    parser.add_argument("--fix-bg", action="store_true", help="Fix background")
    parser.add_argument("--fix-light", action="store_true", help="Fix directional light")
    parser.add_argument("--fix-icao", action="store_true", help="Fix ICAO cadre")
    parser.add_argument("--fix-blurred", action="store_true", help="Fix blurred image")
    parser.add_argument("--output", default=None, help="Chemin de sauvegarde du résultat")
    parser.add_argument("--show", action="store_true", help="Afficher l'image résultat")
    return parser.parse_args()


FACE_MODEL = str(PROJECT_ROOT / "models" / "deploy.prototxt")
FACE_WEIGHTS = str(PROJECT_ROOT / "models" / "res10_300x300_ssd_iter_140000.caffemodel")
LANDMARK_MODEL = str(PROJECT_ROOT / "models" / "lbfmodel.yaml")
SHADOW_MODEL_PATH = str(PROJECT_ROOT / "checkpoints" / "shadow_removal_best.pth")
SHADOW_INPUT_SIZE = 512
_shadow_model = None

def init_landmarks():
    ensure_model(FACE_MODEL)
    ensure_model(FACE_WEIGHTS)
    ensure_model(LANDMARK_MODEL)
    net = cv2.dnn.readNetFromCaffe(FACE_MODEL, FACE_WEIGHTS)
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(LANDMARK_MODEL)
    return net, facemark


def get_landmarks(image, net, facemark):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        1.0, (300, 300), (104.0, 177.0, 123.0)
    )
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face_rect = np.array([[x1, y1, x2 - x1, y2 - y1]])
            ok, landmarks = facemark.fit(image, face_rect)
            if ok:
                return landmarks[0][0]
    return None


#  SEGMENTATION
BASE_OPTIONS = mp_python.BaseOptions(
    model_asset_path=str(PROJECT_ROOT / "postprocessing" / "selfie_segmenter.tflite")
)
SEGMENTER_OPTIONS = vision.ImageSegmenterOptions(
    base_options=BASE_OPTIONS,
    output_category_mask=True,
    running_mode=vision.RunningMode.IMAGE,
)
def get_segmentation_mask(image):
    ensure_model(str(PROJECT_ROOT / "models" / "selfie_segmentation.tflite"))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with vision.ImageSegmenter.create_from_options(SEGMENTER_OPTIONS) as segmenter:
        result = segmenter.segment(mp_image)
        mask = result.category_mask.numpy_view().astype(np.float32)
        return mask / 255.0

def load_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Impossible de charger le masque : {path}")
    return (mask / 255.0).astype(np.float32)


def load_landmarks(path):
    with open(path, "r") as f:
        data = json.load(f)
    return np.array(data, dtype=np.float32)


def compute_initial_data(image, net, facemark, mask_path=None, landmarks_path=None):
    """Calcule ou charger masque et landmarks."""
    if mask_path:
        mask = load_mask(mask_path)
    else:
        mask = get_segmentation_mask(image)

    if landmarks_path:
        landmarks = load_landmarks(landmarks_path)
    else:
        landmarks = get_landmarks(image, net, facemark)

    return mask, landmarks


# - MASQUES DÉRIVÉS -

def smooth_mask(mask):
    return cv2.GaussianBlur(mask, (31, 31), 0)


def get_face_bbox_from_mask(mask):
    binary = (mask > 0.5).astype(np.uint8)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    mask_tete = binary[y:y + h // 4, :]
    coords_tete = cv2.findNonZero(mask_tete)
    if coords_tete is None:
        return None, None

    _, _, w2, _ = cv2.boundingRect(coords_tete)
    return y, w2

#  MASK REMBG (pour fix_background uniquement)

def get_mask_rembg(image):
    alpha = remove(image)[:, :, 3]
    mask = (alpha > 128).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    return (mask / 255.0).astype(np.float32)


def fix_background(image):
    h, w = image.shape[:2]
    cs = max(20, h // 10)

    mask = get_mask_rembg(image)

    # Eroder le masque pour couper les bords douteux
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.erode(mask, erode_k)
    mask_clean = cv2.GaussianBlur(mask_clean, (5, 5), 0)

    # Composition fond blanc directement avec masque erodé
    mask_3 = mask_clean[:, :, np.newaxis]
    result = image.astype(np.float32) * mask_3 + np.full(image.shape, 240, np.float32) * (1 - mask_3)
    return result.clip(0, 255).astype(np.uint8)

# FIX LIGHT

def load_shadow_model():
    global _shadow_model
    if _shadow_model is not None:
        return _shadow_model

    if not os.path.exists(SHADOW_MODEL_PATH):
        # On lève une exception proprement au lieu de tuer le programme
        raise FileNotFoundError(f"Erreur: modèle shadow introuvable à {SHADOW_MODEL_PATH}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    model = ShadowRemovalNet()
    state = torch.load(SHADOW_MODEL_PATH, map_location=device)

    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "model_state_dict" in state:
            state = state["model_state_dict"]
        elif "state_dict" in state:
            state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device).eval()
    _shadow_model = (model, device)
    return _shadow_model

_face_mask_gen = None

def get_face_mask_gen():
    global _face_mask_gen
    if _face_mask_gen is None:
        _face_mask_gen = BiSeNetMaskGenerator.from_config(_config)
    return _face_mask_gen


# MediaPipe jawline landmark indices (bottom of face oval, left→chin→right).
# These trace the exact chin curve for any face shape, rotation or size.
_JAWLINE = [
    132, 58, 172, 136, 150, 149, 176, 148, 152,   # left jaw → chin
    377, 400, 378, 379, 365, 397, 288, 361,         # chin → right jaw
]


# ─── EXACT SAME neck mask logic as user's code ────────────────────────────

def _normalize_luminosity(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    normalized = cv2.merge([l, a, b])
    return cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)


def _enforce_symmetry(mask, center_x):
    h, w = mask.shape
    left_half = mask[:, :center_x]
    right_half = mask[:, center_x:]
    
    # Fix: handle case when halves have different sizes
    min_w = min(left_half.shape[1], right_half.shape[1])
    if min_w == 0:
        return mask
    
    left_half = left_half[:, :min_w]
    right_half = right_half[:, :min_w]
    
    right_half_flipped = cv2.flip(right_half, 1)
    left_half_flipped = cv2.flip(left_half, 1)
    left_pixels = cv2.countNonZero(left_half)
    right_pixels = cv2.countNonZero(right_half)
    if left_pixels > right_pixels * 1.5:
        merged = np.hstack([left_half, left_half_flipped])
    elif right_pixels > left_pixels * 1.5:
        merged = np.hstack([right_half_flipped, right_half])
    else:
        combined = cv2.bitwise_or(left_half, right_half_flipped)
        combined_right = cv2.bitwise_or(right_half, left_half_flipped)
        merged = np.hstack([combined, combined_right])
    
    # Pad back to original width
    if merged.shape[1] < w:
        pad = np.zeros((h, w - merged.shape[1]), dtype=merged.dtype)
        merged = np.hstack([merged, pad])
    return merged


def _suppress_clothing_boundary(mask, image, neck_bottom_y):
    h, w = mask.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Fix: ensure same size and type
    if edges.shape != mask.shape:
        edges = cv2.resize(edges, (w, h))
    
    lower_region = np.zeros((h, w), dtype=np.uint8)
    boundary_y = neck_bottom_y + int(h * 0.15)
    lower_polygon = np.array([
        [0, boundary_y], [w, boundary_y], [w, h], [0, h]
    ], dtype=np.int32)
    cv2.fillPoly(lower_region, [lower_polygon], 255)
    edges_lower = cv2.bitwise_and(edges, lower_region)
    kernel = np.ones((5, 5), np.uint8)
    edges_dilated = cv2.dilate(edges_lower, kernel, iterations=2)
    suppressed = cv2.bitwise_and(mask, cv2.bitwise_not(edges_dilated))
    return suppressed


def _create_neck_mask_exact(image: np.ndarray, face_pts: np.ndarray) -> np.ndarray:
    """EXACT same mask logic as user's create_neck_mask function."""
    h, w = image.shape[:2]
    
    if face_pts is None or len(face_pts) < 478:
        return np.zeros((h, w), dtype=np.float32)
    
    # Jaw indices (same as user's code)
    jaw_indices = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
    jaw_pts = np.array([[int(face_pts[idx, 0]), int(face_pts[idx, 1])] for idx in jaw_indices], dtype=np.int32)
    
    chin_indices = [152, 377, 400, 148, 176, 149, 150, 136, 172, 58]
    chin_pts = np.array([[int(face_pts[idx, 0]), int(face_pts[idx, 1])] for idx in chin_indices], dtype=np.int32)
    
    face_width = np.linalg.norm(jaw_pts[0] - jaw_pts[-1])
    neck_center_x = int(np.mean(jaw_pts[:, 0]))
    neck_bottom_y = int(np.max(chin_pts[:, 1]))
    neck_top_y = int(np.min(chin_pts[:, 1]))
    
    # Sobel edge detection for jaw line (exact same)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.uint8(np.absolute(sobelx))
    _, jaw_edges = cv2.threshold(abs_sobel, 25, 255, cv2.THRESH_BINARY)
    
    jaw_line_mask = np.zeros((h, w), dtype=np.uint8)
    jaw_top = neck_top_y - 20
    jaw_bottom = neck_top_y + 30
    jaw_polygon = np.array([
        [0, max(0, jaw_top)], [w, max(0, jaw_top)], [w, min(h, jaw_bottom)], [0, min(h, jaw_bottom)]
    ], dtype=np.int32)
    cv2.fillPoly(jaw_line_mask, [jaw_polygon], 255)
    jaw_edge_detected = cv2.bitwise_and(jaw_edges, jaw_line_mask)
    
    kernel_h = np.ones((1, 5), np.uint8)
    jaw_edge_dilated = cv2.dilate(jaw_edge_detected, kernel_h, iterations=2)
    
    # Neck polygon geometry (exact same)
    shoulder_y = min(h - 1, neck_bottom_y + int(face_width * 2.5))
    shoulder_extend = int(face_width * 1.2)
    left_shoulder = [neck_center_x - shoulder_extend, shoulder_y]
    right_shoulder = [neck_center_x + shoulder_extend, shoulder_y]
    
    neck_polygon = np.vstack([jaw_pts, right_shoulder, left_shoulder])
    geo_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(geo_mask, [neck_polygon.astype(np.int32)], 255)
    
    # YCrCb skin detection - NO normalize_luminosity, direct on original
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycrcb_lower = np.array([0, 133, 77])
    ycrcb_upper = np.array([255, 173, 127])
    skin_mask = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)
    
    skin_with_jaw = cv2.bitwise_and(skin_mask, cv2.bitwise_not(jaw_edge_dilated))
    final_mask = cv2.bitwise_and(geo_mask, skin_with_jaw)
    
    # Erode (exact same)
    eroded = cv2.erode(final_mask, np.ones((11, 11), np.uint8), iterations=1)
    if cv2.countNonZero(eroded) > 100:
        final_mask = eroded
    else:
        final_mask = cv2.erode(geo_mask, np.ones((15, 15), np.uint8), iterations=2)
    
    # Enforce symmetry (exact same)
    final_mask = _enforce_symmetry(final_mask, neck_center_x)
    
    # Suppress clothing boundary (exact same)
    final_mask = _suppress_clothing_boundary(final_mask, image, neck_bottom_y)
    
    # Morphology cleanup (exact same)
    kernel = np.ones((7, 7), np.uint8)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    
    # Connected components (exact same)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    if num_labels > 1:
        jaw_y = int(np.max(jaw_pts[:, 1]))
        largest_label = 1
        largest_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            y_pos = stats[i, cv2.CC_STAT_TOP]
            if area > largest_area and y_pos < jaw_y + 150:
                largest_area = area
                largest_label = i
        final_mask = (labels == largest_label).astype(np.uint8) * 255
    
    # 101x101 blur (exact same as user's code)
    blurred = cv2.GaussianBlur(final_mask, (101, 101), 0)
    return blurred.astype(np.float32) / 255.0


# ─────────────────────────────────────────────────────────────────────────────


def _build_neck_mask_smart(face_pts: np.ndarray, h: int, w: int) -> np.ndarray:
    """
    Build a precise neck mask from actual MediaPipe jawline landmarks.

    Top boundary  = real jaw curve (follows chin shape, tilt, asymmetry).
    Bottom edge   = jaw curve translated downward by 40 % of face height.
    Result        = closed polygon filled, then soft-blurred for blending.

    Much more accurate than a fixed rectangle: works for tilted heads,
    short/long necks, beards, wide/narrow jaws.
    """
    if face_pts is None or len(face_pts) < 478:
        return np.zeros((h, w), dtype=np.float32)

    face_h = float(face_pts[:, 1].max() - face_pts[:, 1].min())
    if face_h < 10:
        return np.zeros((h, w), dtype=np.float32)

    # Extract jawline points and sort left→right for a clean polygon edge
    jaw_pts = face_pts[_JAWLINE].copy()
    jaw_pts = jaw_pts[jaw_pts[:, 0].argsort()]

    # Build neck mask from jawline - exact same logic as face mask
    jaw_pts = face_pts[_JAWLINE].copy()
    
    # Left/right: from jawline landmarks
    jaw_left_x = jaw_pts[:, 0].min()
    jaw_right_x = jaw_pts[:, 0].max()
    chin_y = jaw_pts[:, 1].max()
    
    # Same process as face: polygon, fill, blur
    neck_left = jaw_left_x
    neck_right = jaw_right_x
    neck_depth = int(face_h * 0.28)
    
    poly = np.array([
        [neck_left, chin_y],
        [neck_right, chin_y],
        [neck_right, chin_y + neck_depth],
        [neck_left, chin_y + neck_depth]
    ], dtype=np.int32)

    neck_mask = np.zeros((h, w), dtype=np.float32)
    cv2.fillPoly(neck_mask, [poly], 1.0)
    
    # Stronger at chin (shadow zone), fades toward bottom
    ys = np.arange(h, dtype=np.float32)
    grad = np.clip(1.0 - (ys - chin_y) / (neck_depth + 1) * 0.5, 0.4, 1.0)
    neck_mask = neck_mask * grad[:, None]
    
    # Same blur as face (31x31 for smooth transition)
    neck_mask = cv2.GaussianBlur(neck_mask, (31, 31), 0)
    neck_mask = np.clip(neck_mask, 0.0, 1.0)


def fix_light(image, model_bundle=None, mask_gen=None):
    """
    Face shadow removal with neck extension.
    - Face zone  : full model prediction + texture preservation
    - Neck zone  : luminance-only transfer (LAB L-channel) to avoid color
                   artifacts from the model never being trained on neck pixels
    - Gradient mask ensures smooth chin→neck transition
    """
    model, device = model_bundle if model_bundle is not None else load_shadow_model()
    orig_h, orig_w = image.shape[:2]

    # 1. Basse résolution pour l'inférence
    resized = cv2.resize(image, (SHADOW_INPUT_SIZE, SHADOW_INPUT_SIZE), interpolation=cv2.INTER_AREA)
    rgb_lr = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    # 2. Détection visage + cou avec BiSeNet
    mask_gen = mask_gen if mask_gen is not None else get_face_mask_gen()
    face_mask_lr = mask_gen.generate_mask(rgb_lr)
    face_mask_hard_lr = (face_mask_lr > 0.5).astype(np.float32)
    face_mask_lr = cv2.GaussianBlur(face_mask_lr, (31, 31), 0) * face_mask_hard_lr
    face_mask_lr = np.clip(face_mask_lr, 0.0, 1.0)
    mask_bool_lr = face_mask_lr > 0.5

    # BiSeNet already includes neck in the mask
    neck_mask_hr = np.zeros((orig_h, orig_w), dtype=np.float32)

    # 3. Inférence ShadowRemoval
    tensor = (torch.from_numpy(rgb_lr.astype(np.float32) / 255.0)
              .permute(2, 0, 1).unsqueeze(0).to(device))
    with torch.no_grad():
        pred = model(tensor)
    pred_rgb_lr = np.clip(pred.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    pred_bgr_lr = cv2.cvtColor(pred_rgb_lr, cv2.COLOR_RGB2BGR)

    # 4. Garde adaptatif
    gray_lr = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    std_dev  = float(np.std(gray_lr[mask_bool_lr])) if np.any(mask_bool_lr) else 50.0
    diff     = cv2.absdiff(resized, pred_bgr_lr)
    mean_diff = float(diff.mean(axis=2)[mask_bool_lr].mean()) if np.any(mask_bool_lr) else 0.0

    if mean_diff < (std_dev * 0.15):
        return image

    # 5. Force adaptative - higher for hard shadows (up to 80%)
    dynamic_strength = float(np.clip(0.50 + (mean_diff / 70.0), 0.55, 0.80))

    # 6. Upscale prédiction + masques (neck_mask_hr already at HR from advanced function)
    pred_bgr_hr  = cv2.resize(pred_bgr_lr,  (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    face_mask_hr = cv2.resize(face_mask_lr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # Érosion légère du masque visage
    inner_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    face_mask_hr = cv2.erode(face_mask_hr, inner_kernel)
    face_mask_hr = np.clip(face_mask_hr, 0.0, 1.0)
    
    # Same erosion for neck mask
    neck_mask_hr = cv2.erode(neck_mask_hr, inner_kernel)
    neck_mask_hr = np.clip(neck_mask_hr, 0.0, 1.0)

    # 7. Keep original texture - only use L-channel (luminance) from model
    # This removes shadows while preserving ORIGINAL skin texture and color
    image_f   = image.astype(np.float32)
    pred_f    = pred_bgr_hr.astype(np.float32)
    blur_orig = cv2.GaussianBlur(image_f, (0, 0), sigmaX=2.0)
    texture_hf = np.clip(image_f - blur_orig, -15.0, 15.0)  # ORIGINAL texture
    
    # Get L-channel (luminance) from model prediction for shadow removal
    orig_lab = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    pred_lab = cv2.cvtColor(pred_f.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    l_diff = pred_lab[:, :, 0] - orig_lab[:, :, 0]  # L difference = shadow removal
    
    # Apply L correction ONLY - preserves original texture and color
    corrected_lab = orig_lab.copy()
    corrected_lab[:, :, 0] = np.clip(orig_lab[:, :, 0] + l_diff * dynamic_strength, 0, 255)
    corrected_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    
    # Add back ORIGINAL texture (not model's)
    corrected_textured = corrected_bgr + texture_hf

    # 8. Face blend - removes shadows, keeps original texture
    hard_face_mask_hr = (face_mask_hr > 0.5).astype(np.float32)
    alpha_face = (cv2.GaussianBlur(face_mask_hr, (15, 15), 0) * hard_face_mask_hr * dynamic_strength)[..., None]
    blended    = image_f * (1.0 - alpha_face) + corrected_textured * alpha_face

    # 9. Neck blend - same L-channel approach (preserves original texture)
    if neck_mask_hr is not None and neck_mask_hr.sum() > 10:
        if neck_mask_hr.shape[:2] != (orig_h, orig_w):
            neck_mask_hr = cv2.resize(neck_mask_hr, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        hard_neck_mask_hr = (neck_mask_hr > 0.5).astype(np.float32)
        alpha_neck = (cv2.GaussianBlur(neck_mask_hr, (21, 21), 0) * hard_neck_mask_hr * dynamic_strength)[..., None]
        blended = blended * (1.0 - alpha_neck) + corrected_textured * alpha_neck

    return np.clip(blended, 0, 255).astype(np.uint8)

def fix_icao(image, landmarks, mask):
    """
    Recadre au format ICAO  (35x45mm).
    Mesure la tête réelle (sommet→menton) et calcule le cadre
    pour que la tête fasse 34mm sur 45mm avec 4mm de marge.
    """
    ECHEC = (image, 0, 0, 1.0, 1.0)
    hauteur, largeur = image.shape[:2]

    if landmarks is None:
        return ECHEC

    points = landmarks
    centre_yeux = (points[36] + points[45]) / 2.0
    centre_visage_x = (points[0][0] + points[16][0]) / 2.0
    menton = points[8]
    face_width_landmarks = np.linalg.norm(points[16] - points[0])
    haut_visage = np.linalg.norm(menton - centre_yeux)

    if haut_visage < 50:
        return ECHEC

    # Sommet de tête : fallback landmarks
    brow_top = min(points[19][1], points[24][1])
    larg_ref = max(
        face_width_landmarks * 1.05,
        np.linalg.norm(points[15] - points[1]) * 1.1
    )
    facteur_crane = np.clip(0.75 + 0.15 * (larg_ref / haut_visage), 0.75, 1.1)
    sommet_tete = max(0, brow_top - facteur_crane * haut_visage)

    # Masque améliore si fiable
    sommet_mask, largeur_mask = get_face_bbox_from_mask(mask)
    if sommet_mask is not None:
        if largeur_mask * 1.02 < face_width_landmarks * 1.8:
            sommet_tete = max(0, sommet_mask)

    # Marge accessoires
    sommet_tete = max(0, sommet_tete - haut_visage * 0.08)

    hauteur_tete = menton[1] - sommet_tete
    if hauteur_tete < 50:
        return ECHEC

    # Dimensions du cadre depuis la norme ICAO
    RATIO_TETE = 34.0 / 45.0
    RATIO_MARGE_HAUT = 4.0 / 45.0
    RATIO_CIBLE = 35.0 / 45.0

    recad_haut = int(hauteur_tete / RATIO_TETE)
    recad_larg = int(recad_haut * RATIO_CIBLE)

    if recad_larg < 200 or recad_haut < 250:
        return ECHEC

    # Positionnement
    y1 = int(sommet_tete - RATIO_MARGE_HAUT * recad_haut)
    y2 = y1 + recad_haut
    x1 = int(centre_visage_x - recad_larg / 2)
    x2 = x1 + recad_larg

    # Validation yeux dans zone ICAO
    pos_yeux_ratio = (centre_yeux[1] - y1) / recad_haut
    if not (0.40 <= pos_yeux_ratio <= 0.70):
        y1 = int(centre_yeux[1] - 0.55 * recad_haut)
        y2 = y1 + recad_haut

    # Clamp aux bords
    if y1 < 0:       y2 -= y1;            y1 = 0
    if y2 > hauteur: y1 -= y2 - hauteur;  y2 = hauteur; y1 = max(0, y1)
    if x1 < 0:       x2 -= x1;            x1 = 0
    if x2 > largeur: x1 -= x2 - largeur;  x2 = largeur; x1 = max(0, x1)

    if x2 <= x1 or y2 <= y1:
        return ECHEC
    # Vérification menton
    if menton[1] > y2 - 10:
        return ECHEC

    # Crop + ajustement ratio
    crop = image[y1:y2, x1:x2]
    h_c, w_c = crop.shape[:2]

    if h_c < 10 or w_c < 10:
        return ECHEC

    r = w_c / h_c
    if r > RATIO_CIBLE:
        nw = int(h_c * RATIO_CIBLE)
        dx = (w_c - nw) // 2
        crop = crop[:, dx:dx + nw]
        x1 += dx
    elif r < RATIO_CIBLE:
        nh = int(w_c / RATIO_CIBLE)
        dy = (h_c - nh) // 2
        crop = crop[dy:dy + nh, :]
        y1 += dy

    # Redimensionnement final
    out_h = int(np.sqrt(largeur * hauteur / RATIO_CIBLE))
    out_w = int(out_h * RATIO_CIBLE)

    scale_x = out_w / crop.shape[1]
    scale_y = out_h / crop.shape[0]

    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
    return resized, x1, y1, scale_x, scale_y

def fix_blurred(image):
    """
    Deblur intelligent (version PRO ICAO)

    ✔ Détection fiable du flou
    ✔ Sharpen uniquement si nécessaire
    ✔ Pas d'amplification du bruit
    ✔ Préserve texture naturelle
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Détection du flou (plus fiable)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()

    # seuils réalistes
    if variance > 120:
        # image déjà nette → on ne touche pas
        return image

    if variance > 80:
        strength = 0.3
    elif variance > 40:
        strength = 0.6
    else:
        strength = 1.0

    blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)

    sharpen = cv2.addWeighted(
        image, 1 + strength,
        blur, -strength,
        0
    )

    result = cv2.bilateralFilter(sharpen, 5, 20, 20)

    return result


#  MAIN

def main():
    args = parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print("Erreur image", file=sys.stderr)
        sys.exit(1)

    net, facemark = None, None
    mask, landmarks = None, None

    if args.fix_icao:
        net, facemark = init_landmarks()
        mask, landmarks = compute_initial_data(
            img, net, facemark,
            mask_path=args.mask,
            landmarks_path=args.landmarks
        )

    result = img.copy()

    if not any([args.fix_icao, args.fix_bg, args.fix_light, args.fix_blurred]):
        print("Aucun traitement appliqué")

    # 1. FIX BG D'ABORD : Détourage sur l'image entière
    if args.fix_bg:
        result = fix_background(result)

    # 2.  Recadrage de l'image déjà détourée

    if args.fix_icao:
        result, cx, cy, sx, sy = fix_icao(result, landmarks, mask)

        if cx == 0 and cy == 0 and sx == 1.0 and sy == 1.0:
            print("  fix_icao: recadrage échoué, image originale conservée")
        else:
            h_new, w_new = result.shape[:2]
            crop_h, crop_w = int(h_new / sy), int(w_new / sx)
            mask = cv2.resize(mask[cy:cy + crop_h, cx:cx + crop_w],
                              (w_new, h_new), interpolation=cv2.INTER_LINEAR)
            if landmarks is not None:
                landmarks[:, 0] = (landmarks[:, 0] - cx) * sx
                landmarks[:, 1] = (landmarks[:, 1] - cy) * sy

    # 3. FIX LIGHT : Correction lumière sur l'image recadrée
    if args.fix_light:
        result = fix_light(result)

    # 4. FIX BLURRED : Netteté à la fin

    if args.fix_blurred:
        result = fix_blurred(result)

    if args.output:
        success = cv2.imwrite(args.output, result)
        if not success:
            print(f"Erreur: échec d'écriture vers {args.output}", file=sys.stderr)
            sys.exit(1)
        print(f"Sauvegardé : {args.output}")

    if args.show:
        cv2.imshow("original", img)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
