from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_config, get_data_path

config = get_config()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite"
)
MODEL_PATH = PROJECT_ROOT / "postprocessing" / "selfie_segmenter.tflite"


def get_project_root() -> Path:
    """Return project root directory."""
    return PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    default_input_root = get_data_path("dataset_root")
    default_output_root = PROJECT_ROOT / "data" / "dataset_no_bg"

    parser = argparse.ArgumentParser(
        description="Remove image backgrounds from dataset/input and dataset/target using a MediaPipe person-segmentation task."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=default_input_root,
        help=f"Dataset root containing input/ and target/. Default: {default_input_root}",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help=f"Output dataset root. Default: {default_output_root}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Foreground confidence threshold between 0.0 and 1.0. Default: 0.6",
    )
    parser.add_argument(
        "--edge-tightness",
        type=float,
        default=1.0,
        help="Edge cleanup strength. Higher values cut halos more aggressively. Default: 1.0",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help=f"Path to the MediaPipe image segmentation model. Default: {MODEL_PATH}",
    )
    return parser.parse_args()


def collect_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_structure(root: Path) -> tuple[Path, Path]:
    input_dir = root / "input"
    target_dir = root / "target"
    missing = [str(path) for path in (input_dir, target_dir) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required dataset folders: {', '.join(missing)}")
    return input_dir, target_dir


def ensure_model(model_path: Path) -> Path:
    if model_path.exists():
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Downloading MediaPipe model to: {model_path}")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(model_path))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not download the MediaPipe segmentation model. "
            f"Download it manually from {MODEL_URL} and save it to {model_path}."
        ) from exc

    return model_path


def create_segmenter(model_path: Path) -> vision.ImageSegmenter:
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        output_confidence_masks=True,
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def extract_person_confidence(segmentation_result: vision.ImageSegmenterResult) -> np.ndarray:
    if segmentation_result.confidence_masks:
        confidence_masks = [mask.numpy_view() for mask in segmentation_result.confidence_masks]
        if len(confidence_masks) >= 2:
            person_mask = confidence_masks[-1]
        else:
            person_mask = confidence_masks[0]
        return person_mask.astype(np.float32)

    if segmentation_result.category_mask is not None:
        category_mask = segmentation_result.category_mask.numpy_view()
        return (category_mask > 0).astype(np.float32)

    raise RuntimeError("MediaPipe Task did not return any mask output.")


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels <= 1:
        return binary.astype(np.float32)

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest_label).astype(np.float32)


def build_refined_alpha(confidence: np.ndarray, threshold: float, edge_tightness: float) -> np.ndarray:
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    confidence = cv2.GaussianBlur(confidence, (7, 7), 0)

    coarse = (confidence >= threshold).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, kernel)
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, kernel)
    coarse = keep_largest_component(coarse)

    support = cv2.dilate(coarse.astype(np.uint8), kernel, iterations=2).astype(np.float32)
    low = max(0.0, threshold - (0.18 / max(edge_tightness, 0.25)))
    high = min(1.0, threshold + (0.12 * max(edge_tightness, 0.25)))
    alpha = np.clip((confidence - low) / max(high - low, 1e-6), 0.0, 1.0)
    alpha *= support

    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    shrink = min(0.2, 0.035 * edge_tightness)
    alpha = np.clip((alpha - shrink) / max(1.0 - shrink, 1e-6), 0.0, 1.0)
    alpha[alpha < 0.02] = 0.0
    alpha[alpha > 0.98] = 1.0
    return alpha


def refine_alpha_with_grabcut(
    image_bgr: np.ndarray,
    alpha: np.ndarray,
    edge_tightness: float,
) -> np.ndarray:
    hard_fg = (alpha >= 0.93).astype(np.uint8)
    hard_bg = (alpha <= 0.05).astype(np.uint8)

    core_kernel_size = max(3, int(round(5 + edge_tightness * 2)))
    if core_kernel_size % 2 == 0:
        core_kernel_size += 1
    core_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (core_kernel_size, core_kernel_size))

    fg_core = cv2.erode(hard_fg, core_kernel, iterations=1)
    bg_core = cv2.erode(hard_bg, core_kernel, iterations=1)

    grabcut_mask = np.full(alpha.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    grabcut_mask[bg_core > 0] = cv2.GC_BGD
    grabcut_mask[fg_core > 0] = cv2.GC_FGD
    grabcut_mask[(alpha >= 0.55) & (fg_core == 0)] = cv2.GC_PR_FGD
    grabcut_mask[(alpha <= 0.2) & (bg_core == 0)] = cv2.GC_PR_BGD

    if np.count_nonzero(fg_core) == 0 or np.count_nonzero(bg_core) == 0:
        return alpha

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_bgr,
            grabcut_mask,
            None,
            bgd_model,
            fgd_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return alpha

    refined_binary = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        1.0,
        0.0,
    ).astype(np.float32)

    refined_binary = keep_largest_component(refined_binary)
    refined_binary = cv2.GaussianBlur(refined_binary, (5, 5), 0)

    alpha = np.maximum(alpha, refined_binary * 0.75)
    alpha = np.minimum(alpha, cv2.GaussianBlur(refined_binary, (9, 9), 0) + 0.15)
    alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
    alpha[alpha < 0.02] = 0.0
    alpha[alpha > 0.98] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def suppress_edge_spill(image_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    result = image_bgr.astype(np.float32).copy()
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)

    opaque_mask = (alpha >= 0.98).astype(np.float32)
    if np.count_nonzero(opaque_mask) > 0:
        opaque_3 = np.repeat(opaque_mask[:, :, None], 3, axis=2)
        blur_size = 11
        blurred_color = cv2.GaussianBlur(result * opaque_3, (blur_size, blur_size), 0)
        blurred_weight = cv2.GaussianBlur(opaque_3, (blur_size, blur_size), 0)
        foreground_fill = blurred_color / np.maximum(blurred_weight, 1e-3)

        edge_mask = (alpha > 0.0) & (alpha < 1.0)
        edge_mix = np.clip((1.0 - alpha) ** 0.7, 0.0, 1.0) * 0.9
        edge_mix_3 = np.repeat(edge_mix[:, :, None], 3, axis=2)
        blended = result * (1.0 - edge_mix_3) + foreground_fill * edge_mix_3
        result[edge_mask] = blended[edge_mask]

    result[alpha_u8 <= 2] = 0
    return np.clip(result, 0, 255).astype(np.uint8)


def remove_background(
    image_bgr: np.ndarray,
    segmenter: vision.ImageSegmenter,
    threshold: float,
    edge_tightness: float,
) -> np.ndarray:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    segmentation_result = segmenter.segment(mp_image)
    confidence = extract_person_confidence(segmentation_result)
    alpha = build_refined_alpha(confidence, threshold, edge_tightness)
    alpha = refine_alpha_with_grabcut(image_bgr, alpha, edge_tightness)
    clean_bgr = suppress_edge_spill(image_bgr, alpha)
    alpha_u8 = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    b, g, r = cv2.split(clean_bgr)
    return cv2.merge((b, g, r, alpha_u8))


def process_folder(
    source_dir: Path,
    target_dir: Path,
    segmenter: vision.ImageSegmenter,
    threshold: float,
    edge_tightness: float,
) -> tuple[int, int]:
    target_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_images(source_dir)

    processed = 0
    failed = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Could not read image: {image_path}")
            failed += 1
            continue

        try:
            output = remove_background(image, segmenter, threshold, edge_tightness)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed to process {image_path.name}: {exc}")
            failed += 1
            continue

        output_path = target_dir / f"{image_path.stem}.png"
        if not cv2.imwrite(str(output_path), output):
            print(f"[WARN] Could not save output image: {output_path}")
            failed += 1
            continue

        processed += 1
        if processed % 100 == 0:
            print(f"[INFO] Processed {processed} images in {source_dir.name}")

    return processed, failed


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        print("[ERROR] --threshold must be between 0.0 and 1.0")
        return 1
    if args.edge_tightness <= 0.0:
        print("[ERROR] --edge-tightness must be greater than 0.0")
        return 1

    try:
        input_dir, target_dir = validate_structure(args.input_root)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        model_path = ensure_model(args.model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}")
        return 1

    output_input_dir = args.output_root / "input"
    output_target_dir = args.output_root / "target"

    print(f"[INFO] Input dataset : {args.input_root}")
    print(f"[INFO] Output dataset: {args.output_root}")
    print(f"[INFO] Threshold     : {args.threshold}")
    print(f"[INFO] Edge tightness: {args.edge_tightness}")
    print(f"[INFO] Model path    : {model_path}")
    print("[INFO] Output format : PNG with transparent background")
    print("[INFO] Backend       : MediaPipe Tasks API (person segmentation)")

    try:
        segmenter = create_segmenter(model_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Could not initialize MediaPipe Task: {exc}")
        return 1

    try:
        input_processed, input_failed = process_folder(
            input_dir,
            output_input_dir,
            segmenter,
            args.threshold,
            args.edge_tightness,
        )
        target_processed, target_failed = process_folder(
            target_dir,
            output_target_dir,
            segmenter,
            args.threshold,
            args.edge_tightness,
        )
    finally:
        segmenter.close()

    total_processed = input_processed + target_processed
    total_failed = input_failed + target_failed

    print(f"[INFO] Done. Processed: {total_processed}, Failed: {total_failed}")
    return 0 if total_processed > 0 and total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
