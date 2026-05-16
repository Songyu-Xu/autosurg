"""
Fully automated surgical needle detection and segmentation pipeline.
YOLOv8 detector -> Box Enlarge -> SAM3 segmentation

Usage:
    python needle_pipeline.py --image path/to/image.jpg
    python needle_pipeline.py --image path/to/image.jpg --yolo_weights path/to/best.pt

Dependencies:
    pip install ultralytics
    # SAM3 must be installed from GitHub:
    # git clone https://github.com/facebookresearch/sam3
    # cd sam3 && pip install -e .
    # HuggingFace login is required to download model weights (free):
    # huggingface-cli login
"""

import argparse
import numpy as np
import cv2
import torch
from pathlib import Path
from PIL import Image


# ─────────────────────────── Config ───────────────────────────
YOLO_WEIGHTS = "./models/yolo_weights/best.pt"  # path to trained YOLOv8 weights
YOLO_CONF    = 0.25        # YOLO detection confidence threshold
BOX_EXPAND   = 1.1         # box expansion ratio (1.4 = expand by 40%)
SAM3_TEXT    = "thin curved silver metal needle"  # text prompt passed to SAM3 (improves stability)
FIT_ARC      = True        # fit a circular arc to the mask to recover the needle midpoint
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────── YOLO Detection + Box Enlarge ─────────────────────────
def load_yolo(yolo_weights: str = YOLO_WEIGHTS):
    """Load and return a YOLOv8 model. Call once and reuse across images."""
    from ultralytics import YOLO
    return YOLO(yolo_weights)


def detect_and_enlarge(image_path: str,
                        yolo_weights: str = YOLO_WEIGHTS,
                        conf: float = YOLO_CONF,
                        expand: float = BOX_EXPAND,
                        yolo_model=None):
    """
    Run YOLOv8 detection on image and return the single enlarged bounding box
    with the highest confidence score.

    Prior knowledge: there is exactly one needle per image, so we discard all
    detections except the top-scoring one to avoid false positives.

    Args:
        yolo_model: pre-loaded YOLO instance (avoids reloading weights every call).
                    If None, loads from yolo_weights.

    Returns:
        enlarged_xyxy: np.ndarray, shape [1, 4], absolute pixel coords [x1, y1, x2, y2]
        img_w, img_h: image dimensions
    """
    if yolo_model is None:
        yolo_model = load_yolo(yolo_weights)
    results = yolo_model(image_path, conf=conf, verbose=False)[0]

    img_h, img_w = results.orig_shape
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()   # [N, 4] absolute coordinates
    confs      = results.boxes.conf.cpu().numpy()   # [N]   confidence scores

    if len(boxes_xyxy) == 0:
        print("[WARN] YOLOv8 detected no needles")
        return np.array([]), img_w, img_h

    # Prior: exactly one needle per image — keep only the highest-confidence box
    best_idx = confs.argmax()
    if len(boxes_xyxy) > 1:
        print(f"  [INFO] {len(boxes_xyxy)} detections found, keeping only the "
              f"highest-confidence one (idx={best_idx}, conf={confs[best_idx]:.3f})")

    x1, y1, x2, y2 = boxes_xyxy[best_idx]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w,  h  = (x2 - x1) * expand, (y2 - y1) * expand

    # Clamp to image boundaries
    nx1 = max(0,     cx - w / 2)
    ny1 = max(0,     cy - h / 2)
    nx2 = min(img_w, cx + w / 2)
    ny2 = min(img_h, cy + h / 2)

    print(f"  original box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]  conf={confs[best_idx]:.3f}  ->  "
          f"enlarged: [{nx1:.1f}, {ny1:.1f}, {nx2:.1f}, {ny2:.1f}]")

    return np.array([[nx1, ny1, nx2, ny2]]), img_w, img_h


def xyxy_to_xywh_norm(box_xyxy: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    Convert absolute XYXY coordinates to normalized [cx, cy, w, h] format
    required by SAM3's add_geometric_prompt.
    """
    x1, y1, x2, y2 = box_xyxy
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h
    return np.array([cx, cy, w, h], dtype=np.float32)


# ─────────────────────────── SAM3 Segmentation ───────────────────────────
def load_sam3():
    """Load and return (sam3_model, Sam3Processor). Call once and reuse across images."""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("[SAM3] Loading model (weights will be downloaded on first run)...")
    model = build_sam3_image_model().to(DEVICE)
    processor = Sam3Processor(model)
    return model, processor


def _square_crop_region(box_xyxy: np.ndarray, img_w: int, img_h: int):
    """Compute a square crop region enclosing the bounding box, clamped to image bounds."""
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) / 2
    crop_x1 = int(max(0,     cx - half))
    crop_y1 = int(max(0,     cy - half))
    crop_x2 = int(min(img_w, cx + half))
    crop_y2 = int(min(img_h, cy + half))
    return crop_x1, crop_y1, crop_x2, crop_y2


def sam3_segment(image_path: str,
                 boxes_xyxy: np.ndarray,
                 img_w: int,
                 img_h: int,
                 text_prompt: str = SAM3_TEXT,
                 sam3_model=None,
                 sam3_processor=None):
    """
    Run SAM3 fine-grained segmentation for each detected box.
    Only the text prompt is used; the box is used solely to crop the image
    region fed to SAM3, not as a geometric prompt.

    Args:
        sam3_model, sam3_processor: pre-loaded instances (avoids reloading every call).
                                    If either is None, both are loaded fresh.

    Returns:
        all_masks: list of np.ndarray, each with shape [H, W], dtype bool
        all_scores: list of float
    """
    if sam3_model is None or sam3_processor is None:
        sam3_model, sam3_processor = load_sam3()
    model, processor = sam3_model, sam3_processor

    image = Image.open(image_path).convert("RGB")

    all_masks  = []
    all_scores = []

    # Wrap entire inference loop in autocast to keep all ops in bfloat16
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i, box_xyxy in enumerate(boxes_xyxy):
            print(f"[SAM3] Processing target {i+1}/{len(boxes_xyxy)}...")

            # Compute square crop around detection box
            crop_x1, crop_y1, crop_x2, crop_y2 = _square_crop_region(box_xyxy, img_w, img_h)

            # Crop the PIL image to the square region
            cropped_image = image.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Run SAM3 on the cropped image with text prompt only (no box prompt)
            state  = processor.set_image(cropped_image)
            output = processor.set_text_prompt(state=state, prompt=text_prompt)

            # Cast to float32 before calling .numpy() (tensors are BFloat16 inside autocast)
            masks  = output["masks"].cpu().float().numpy()   # [N, 1, crop_h, crop_w]
            scores = output["scores"].cpu().float().numpy()  # [N]

            best_idx   = scores.argmax()
            crop_mask  = masks[best_idx, 0].astype(bool)    # [crop_h, crop_w]
            best_score = float(scores[best_idx])

            # Embed crop mask back into a full-size mask
            full_mask = np.zeros((img_h, img_w), dtype=bool)
            full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = crop_mask

            all_masks.append(full_mask)
            all_scores.append(best_score)
            print(f"  -> mask score: {best_score:.3f}")

    return all_masks, all_scores


# ─────────────────────────── Arc Fitting ───────────────────────────
def _clean_mask(mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    Falls back to the original mask if the largest component is tiny
    (less than `min_area_ratio` of the total), to avoid over-pruning.
    """
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    areas   = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    cleaned = (labels == largest)
    if cleaned.sum() < min_area_ratio * m.sum():
        return mask.astype(bool)
    return cleaned


def _fit_circle_lsq(xs: np.ndarray, ys: np.ndarray):
    """
    Algebraic least-squares circle fit.
    Solves x^2 + y^2 + D*x + E*y + F = 0  ->  center=(-D/2, -E/2), r=sqrt(cx^2+cy^2-F).
    Returns (cx, cy, r) or None if the fit is degenerate.
    """
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    A  = np.column_stack([xs, ys, np.ones_like(xs)])
    b  = -(xs ** 2 + ys ** 2)
    D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
    cx, cy  = -D / 2, -E / 2
    r2      = cx ** 2 + cy ** 2 - F
    if r2 <= 0:
        return None
    return float(cx), float(cy), float(np.sqrt(r2))


def fit_needle_arc(mask: np.ndarray):
    """
    Fit a circular arc to a (possibly broken) needle mask and return the arc
    midpoint plus supporting geometry.

    A suturing needle is a circular arc, so even when SAM3's mask is split
    into two disjoint segments (e.g. due to specular reflection), both
    segments still lie on the same underlying circle. The arc midpoint is
    defined along the needle body — NOT the mask centroid, which would bias
    toward the denser segment.

    Algorithm:
        1. Clean the mask (largest connected component only).
        2. Least-squares circle fit on every foreground pixel.
        3. Compute each pixel's polar angle about the fitted center.
        4. The LARGEST gap in sorted angles is the needle's opening; the two
           angles bordering that gap are the arc endpoints.
        5. The arc midpoint is halfway between the endpoints along the arc
           (not across the gap), projected back onto the circle.

    Returns a dict with:
        center       : (cx, cy)          fitted circle center in pixels
        radius       : float              fitted circle radius in pixels
        endpoints    : ((x1,y1),(x2,y2))  arc endpoints on the circle
        midpoint     : (mx, my)           needle midpoint on the arc
        arc_angles   : (a_start, a_end)   endpoint angles (radians)
        mid_angle    : float              midpoint angle (radians)
        arc_sweep    : float              angular span of the arc (radians)
        fit_residual : float              RMS radial error (pixels)
        n_points     : int                number of foreground pixels used
    Returns None if fitting fails.
    """
    cleaned = _clean_mask(mask)
    ys, xs  = np.where(cleaned)
    if len(xs) < 20:
        print("  [ArcFit] too few foreground pixels")
        return None

    fit = _fit_circle_lsq(xs, ys)
    if fit is None:
        print("  [ArcFit] circle fit failed (non-positive radius)")
        return None
    cx, cy, r = fit

    # Polar angle of every foreground pixel about the fitted center
    angles     = np.arctan2(ys - cy, xs - cx)
    sorted_ang = np.sort(angles)

    # Circular gaps between consecutive sorted angles.
    # The largest gap is the arc's opening; the arc occupies the complement.
    wrap    = sorted_ang[0] + 2 * np.pi
    gaps    = np.diff(np.concatenate([sorted_ang, [wrap]]))
    gap_idx = int(np.argmax(gaps))

    # Arc runs from a_start CCW to a_end — not across the gap
    a_end   = float(sorted_ang[gap_idx])                            # gap starts just after this
    a_start = float(sorted_ang[(gap_idx + 1) % len(sorted_ang)])    # gap ends just before this

    # Unwrap so a_end > a_start, then midpoint is the simple average
    a_end_u   = a_end if a_end > a_start else a_end + 2 * np.pi
    mid_angle = (a_start + a_end_u) / 2
    arc_sweep = a_end_u - a_start

    # Project back onto the circle
    mx, my   = cx + r * np.cos(mid_angle), cy + r * np.sin(mid_angle)
    ex1, ey1 = cx + r * np.cos(a_start),   cy + r * np.sin(a_start)
    ex2, ey2 = cx + r * np.cos(a_end),     cy + r * np.sin(a_end)

    # Fit quality: RMS radial residual
    radial_dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    fit_residual = float(np.sqrt(np.mean((radial_dists - r) ** 2)))

    result = {
        "center":       (cx, cy),
        "radius":       r,
        "endpoints":    ((ex1, ey1), (ex2, ey2)),
        "midpoint":     (mx, my),
        "arc_angles":   (a_start, a_end),
        "mid_angle":    mid_angle,
        "arc_sweep":    arc_sweep,
        "fit_residual": fit_residual,
        "n_points":     int(len(xs)),
    }

    # Sanity warnings (non-fatal)
    res_ratio = fit_residual / r
    arc_deg   = np.degrees(arc_sweep)
    print(f"  [ArcFit] center=({cx:.1f}, {cy:.1f})  r={r:.1f}  "
          f"sweep={arc_deg:.1f}°  midpoint=({mx:.1f}, {my:.1f})  "
          f"residual/r={res_ratio:.3f}")
    if res_ratio > 0.25:
        print("  [ArcFit][WARN] high residual/radius ratio — mask may be noisy")
    if arc_deg < 60 or arc_deg > 300:
        print("  [ArcFit][WARN] unusual arc sweep — check mask quality")

    return result


# ─────────────────────────── Visualization & Save ───────────────────────────
def visualize_and_save(image_path: str,
                       boxes_xyxy: np.ndarray,
                       masks: list,
                       scores: list,
                       arc_infos: list = None,
                       output_path: str = None):
    """
    Overlay bounding boxes, segmentation masks, and (optionally) fitted arc
    geometry onto the image and save the result.

    If arc_infos is provided, it should be a list aligned with masks; each
    entry is either a dict returned by fit_needle_arc() or None.
    """
    img = cv2.imread(image_path)
    overlay = img.copy()

    colors = [
        (0, 255, 100),   # green
        (0, 150, 255),   # orange
        (255, 50, 50),   # blue
        (200, 0, 200),   # purple
    ]

    # Arc-fit drawing colors (kept constant so they are easy to recognize)
    color_circle = (255, 200, 0)   # full fitted circle (thin)
    color_arc    = (0, 180, 255)   # visible arc portion (thick)
    color_mid    = (0, 0, 255)     # arc midpoint (filled red dot)
    color_end    = (255, 0, 255)   # arc endpoints (magenta dots)
    color_ctr    = (255, 255, 255) # circle center (white cross)

    for i, (box, mask, score) in enumerate(zip(boxes_xyxy, masks, scores)):
        color = colors[i % len(colors)]

        # Draw semi-transparent mask fill
        overlay[mask] = (
            overlay[mask] * 0.4 + np.array(color) * 0.6
        ).astype(np.uint8)

        # Draw enlarged box and label
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"needle {score:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # Blend overlay with original image
    result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)

    # Overlay arc-fit geometry on top of the blended result
    if arc_infos is not None:
        for arc in arc_infos:
            if arc is None:
                continue
            cx, cy = arc["center"]
            r      = arc["radius"]
            mx, my = arc["midpoint"]
            (ex1, ey1), (ex2, ey2) = arc["endpoints"]
            a_start, a_end = arc["arc_angles"]

            # Full fitted circle (thin)
            cv2.circle(result, (int(round(cx)), int(round(cy))),
                       int(round(r)), color_circle, 1, lineType=cv2.LINE_AA)

            # Visible arc segment (thick). cv2.ellipse uses degrees; its angle
            # convention matches atan2 in image coordinates (y-axis down).
            a_end_u = a_end if a_end > a_start else a_end + 2 * np.pi
            cv2.ellipse(result,
                        (int(round(cx)), int(round(cy))),
                        (int(round(r)), int(round(r))),
                        0,
                        np.degrees(a_start),
                        np.degrees(a_end_u),
                        color_arc, 3, lineType=cv2.LINE_AA)

            # Circle center (white cross)
            cv2.drawMarker(result,
                           (int(round(cx)), int(round(cy))),
                           color_ctr,
                           markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2)

            # Arc endpoints (magenta)
            for (ex, ey) in [(ex1, ey1), (ex2, ey2)]:
                cv2.circle(result,
                           (int(round(ex)), int(round(ey))), 5,
                           color_end, -1, lineType=cv2.LINE_AA)

            # Arc midpoint (red dot with white ring + label)
            mxi, myi = int(round(mx)), int(round(my))
            cv2.circle(result, (mxi, myi), 7, color_mid, -1, lineType=cv2.LINE_AA)
            cv2.circle(result, (mxi, myi), 9, (255, 255, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(result, "mid", (mxi + 10, myi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_mid, 2)

    if output_path is None:
        stem = Path(image_path).stem
        output_path = f"{stem}_result.jpg"

    cv2.imwrite(output_path, result)
    print(f"[✓] Result saved: {output_path}")
    return result


# ─────────────────────────── Main Pipeline ───────────────────────────
def run_pipeline(image_path: str,
                 yolo_weights: str = YOLO_WEIGHTS,
                 conf: float = YOLO_CONF,
                 expand: float = BOX_EXPAND,
                 fit_arc: bool = FIT_ARC,
                 output_path: str = None,
                 yolo_model=None,
                 sam3_model=None,
                 sam3_processor=None):
    """
    Run the full detection -> segmentation -> arc-fitting pipeline on a single image.

    Args:
        image_path:      path to the input image
        yolo_weights:    path to trained YOLOv8 weights (ignored if yolo_model given)
        conf:            YOLO detection confidence threshold
        expand:          box expansion ratio passed to detect_and_enlarge
        fit_arc:         if True, fit a circular arc to each mask and recover
                         the needle midpoint, radius, and endpoints
        output_path:     where to save the result image.
                         - If a directory is given, the result is saved inside it
                           as <original_stem>_result.jpg.
                         - If a full file path is given, it is used as-is.
                         - If None, the result is saved next to the input image
                           as <original_stem>_result.jpg.
        yolo_model:      pre-loaded YOLO instance; if None, loads from yolo_weights.
        sam3_model,
        sam3_processor:  pre-loaded SAM3 instances; if either is None, loads fresh.
    """
    print(f"\n{'='*50}")
    print(f"Input image   : {image_path}")
    print(f"YOLOv8 weights: {yolo_weights}")
    print(f"conf={conf}  expand={expand}  fit_arc={fit_arc}")

    # Resolve output path
    stem = Path(image_path).stem
    if output_path is None:
        resolved_output = str(Path(image_path).parent / f"{stem}_result.jpg")
    else:
        out = Path(output_path)
        if not out.suffix:
            # No file extension -> treat as directory
            out.mkdir(parents=True, exist_ok=True)
            resolved_output = str(out / f"{stem}_result.jpg")
        else:
            # Full file path with extension
            out.parent.mkdir(parents=True, exist_ok=True)
            resolved_output = str(out)

    print(f"Output path   : {resolved_output}")
    print(f"{'='*50}\n")

    # Step 1: YOLO detection + box enlarge
    print("[Step 1] YOLOv8 detection...")
    boxes_xyxy, img_w, img_h = detect_and_enlarge(
        image_path, yolo_weights, conf, expand, yolo_model=yolo_model)

    if len(boxes_xyxy) == 0:
        print("[STOP] No needles detected, exiting.")
        return

    print(f"  {len(boxes_xyxy)} target(s) detected\n")

    # Step 2: SAM3 fine-grained segmentation
    print("[Step 2] SAM3 segmentation...")
    masks, scores = sam3_segment(
        image_path, boxes_xyxy, img_w, img_h,
        sam3_model=sam3_model, sam3_processor=sam3_processor)

    # Step 3: arc fitting (optional)
    arc_infos = None
    if fit_arc:
        print("\n[Step 3] Arc fitting...")
        arc_infos = []
        for i, mask in enumerate(masks):
            print(f"  target {i+1}/{len(masks)}:")
            arc_infos.append(fit_needle_arc(mask))

    # Step 4: visualize and save result
    print("\n[Step 4] Saving result...")
    vis = visualize_and_save(image_path, boxes_xyxy, masks, scores,
                             arc_infos=arc_infos,
                             output_path=resolved_output)

    # Return everything for downstream use (e.g. robot grasping planner)
    return {
        "boxes_xyxy": boxes_xyxy,
        "masks":      masks,       # list of [H, W] bool arrays
        "scores":     scores,
        "arc_infos":  arc_infos,   # list of dicts (or None entries) or None
        "result_img": vis,         # rendered BGR image with boxes/masks drawn
    }


# ─────────────────────────── Batch Processing ───────────────────────────
def run_batch(image_dir: str,
              yolo_weights: str = YOLO_WEIGHTS,
              conf: float = YOLO_CONF,
              expand: float = BOX_EXPAND,
              fit_arc: bool = FIT_ARC,
              output_dir: str = None):
    """
    Run the pipeline on all images in a directory.
    YOLO and SAM3 are loaded exactly once and reused across all images.

    Args:
        image_dir:    directory containing input images
        yolo_weights: path to trained YOLOv8 weights
        conf:         YOLO detection confidence threshold
        expand:       box expansion ratio
        fit_arc:      whether to run arc fitting
        output_dir:   directory to save results; if None, results are saved
                      next to each source image.
    """
    image_dir = Path(image_dir)
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = sorted(f for f in image_dir.iterdir() if f.suffix.lower() in exts)

    print(f"Found {len(images)} images")

    # Load both models once before the loop
    print("\n[Batch] Loading models once for all images...")
    yolo_model = load_yolo(yolo_weights)
    sam3_model, sam3_processor = load_sam3()
    print("[Batch] Models ready.\n")

    for img_path in images:
        run_pipeline(str(img_path), yolo_weights, conf, expand,
                     fit_arc=fit_arc, output_path=output_dir,
                     yolo_model=yolo_model,
                     sam3_model=sam3_model,
                     sam3_processor=sam3_processor)


# ─────────────────────────── CLI ───────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated surgical needle detection and segmentation")
    parser.add_argument("--image",        type=str, help="Path to a single input image")
    parser.add_argument("--image_dir",    type=str, help="Directory of images for batch processing")
    parser.add_argument("--output",       type=str, default=None,
                        help="Output path: a file path (single image) or a directory (batch). "
                             "Defaults to saving next to the input image.")
    parser.add_argument("--yolo_weights", type=str, default=YOLO_WEIGHTS,
                        help="Path to YOLOv8 weights file")
    parser.add_argument("--conf",         type=float, default=YOLO_CONF,
                        help="YOLO confidence threshold")
    parser.add_argument("--expand",       type=float, default=BOX_EXPAND,
                        help="Box expansion ratio")
    parser.add_argument("--no_arc",       action="store_true",
                        help="Disable arc fitting (skip midpoint/radius recovery)")
    args = parser.parse_args()

    fit_arc = not args.no_arc

    # Pass conf and expand explicitly — do not rely on mutating module-level globals,
    # because function default values are bound at definition time and will not
    # pick up runtime reassignments.
    if args.image:
        run_pipeline(args.image, args.yolo_weights,
                     conf=args.conf, expand=args.expand,
                     fit_arc=fit_arc,
                     output_path=args.output)
    elif args.image_dir:
        run_batch(args.image_dir, args.yolo_weights,
                  conf=args.conf, expand=args.expand,
                  fit_arc=fit_arc,
                  output_dir=args.output)
    else:
        print("Please specify --image or --image_dir")
        parser.print_help()