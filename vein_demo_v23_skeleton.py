"""
vein_demo_v23_skeleton.py
=========================

Key changes from v22:
─────────────────────────────────────────────────────────────────────────────
PROBLEM 1 — Fat mask blob (Image 2):
  The mask was correct but visually bloated. CLOSE_K=9 was bridging across
  skin between vein branches, making the whole vein cluster one fat region.

PROBLEM 2 — Missed left-side veins (Image 1):
  This is a model accuracy issue (Dice ~0.63, not enough training yet).
  Post-processing cannot invent veins the model didn't detect.
  No code fix for this — it resolves as training continues past epoch 100+.

SOLUTION for Problem 1 — Skeleton rendered FROM the stable mask:
  Pipeline:
    prob_ema  →  stable binary mask (same as v22, but smaller CLOSE_K)
              →  skeletonize(mask)        ← 1-pixel centerline
              →  dilate skeleton slightly ← controlled thickness (not fat)
              →  draw as solid teal lines

  Why skeleton-from-mask instead of skeleton-from-prob:
  - Direct skeleton from raw prob flickers badly (1-px lines are hypersensitive)
  - Skeleton from stable mask inherits all EMA + morphology stability
  - The mask does the stability work; skeleton just controls visual width

  CLOSE_K reduced from 9 → 5:
  - Previously bridging across skin gaps between separate vein branches
  - Smaller kernel only bridges genuine intra-vein gaps (model resolution gaps)
  - Separate vein branches stay separate → skeleton branches stay separate

RENDERING modes (switch via RENDER_MODE):
  "skeleton"  — skeletonized centerline, dilated to SKEL_THICKNESS px (recommended)
  "outline"   — contour outline of mask (v22 style, kept for comparison)
  "both"      — skeleton + outline together (useful for debugging)
─────────────────────────────────────────────────────────────────────────────
"""

import os
import cv2
import time
import numpy as np
import torch
from collections import deque
from skimage.morphology import skeletonize as ski_skeletonize
from typing import Optional

from config import (
    MIN_SEGMENT_LEN, SPUR_PRUNE_ITERS,
    REANALYZE_EVERY, SMOOTH_WINDOW,
    MIN_VOTES_NEEDED, STABILITY_DIST_PX,
    MIN_PROB_TO_SHOW,
)
from utils.preprocessing import smooth_mask
from utils.skeleton      import prune_spurs, extract_graph_segments
from idss.features       import extract_idss_features
from idss.rules          import apply_knowledge_rules
from idss.ahp            import compute_ahp_weights
from idss.normalize      import normalize_features
from idss.topsis         import topsis_score
from idss.insertion      import find_insertion_point


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "unetplusplus.pt"
VIDEO_IN   = "vein_recording_v2_30s.mp4"
VIDEO_OUT  = "outputs/unetplusplus/result_friend_code.unetplusplus.mp4"

MODEL_H, MODEL_W = 704, 512

# -- CLAHE (must match training) -------------------------------
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE       = (8, 8)

# -- Temporal EMA (same logic as v22, proven stable) -----------
ALPHA_RISE_STILL  = 0.38
ALPHA_RISE_MOVING = 0.52
ALPHA_FALL_STILL  = 0.18   # slow fall = no flicker
ALPHA_FALL_MOVING = 0.28

# -- Gaussian pre-threshold smoothing -------------------------
GAUSS_SIGMA = 1.8

# -- Threshold ------------------------------------------------
THRESHOLD = 0.60

# -- Morphology -----------------------------------------------
# CLOSE_K reduced from 9 → 5: stops bridging skin between vein branches
# Rule of thumb: CLOSE_K should be < half the vein spacing in pixels
# At 704×512 model res, typical inter-vein gap is ~8-12px → 5 is safe
CLOSE_K  = 5
OPEN_K   = 3
DILATE_K = 0
MIN_AREA = 180   # px² at model resolution

# -- Skeleton rendering ---------------------------------------
RENDER_MODE = "skeleton"   # "skeleton" | "outline" | "both"

# Skeleton dilation: controls visual line thickness
# 3 = thin (like reference image), 5 = medium, 7 = thick
SKEL_THICKNESS = 3

# Skeleton color: teal/cyan matching reference image
SKEL_COLOR  = (210, 200, 30)   # BGR ≈ teal
SKEL_ALPHA  = 1.0              # fully opaque main line

# Optional soft glow behind skeleton line
DRAW_GLOW        = True
GLOW_THICKNESS   = SKEL_THICKNESS + 4   # wider than main line
GLOW_COLOR       = (180, 170, 20)       # slightly darker
GLOW_ALPHA       = 0.30                 # semi-transparent

# Outline style (used in "outline" and "both" modes)
CONTOUR_COLOR          = (210, 200, 30)
CONTOUR_THICK          = 2
CONTOUR_SMOOTH_EPSILON = 0.002

# -- IDSS display gate ----------------------------------------
# Only show insertion point + segment when safety score reaches this level
IDSS_MIN_DISPLAY_SCORE = 0.55

# -- Motion detection -----------------------------------------
MOTION_BLUR_K    = 21
MOTION_EMA_ALPHA = 0.20
MOTION_ON_THR    = 0.014
MOTION_OFF_THR   = 0.008

# -- Scene change reset ---------------------------------------
SCENE_LOW_CONF_THR = 0.05
SCENE_RESET_N      = 12

# -- HUD ------------------------------------------------------
SHOW_HUD = True

clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device : {DEVICE}")


# ============================================================
# LOAD MODEL
# ============================================================

print(f"Loading: {MODEL_PATH}")
model = torch.jit.load(MODEL_PATH, map_location=DEVICE).eval()

with torch.no_grad():
    _t = torch.zeros(1, 1, MODEL_H, MODEL_W).to(DEVICE)
    _o = model(_t)
    _min, _max = float(_o.min()), float(_o.max())

MODEL_HAS_SIGMOID = (0.0 <= _min and _max <= 1.0)
print(f"- Loaded | sigmoid_in_model={MODEL_HAS_SIGMOID}\n")


# ============================================================
# HELPERS
# ============================================================

def get_crop(fh: int, fw: int) -> tuple:
    ar  = MODEL_W / MODEL_H
    far = fw / fh
    if far > ar:
        ch, cw = fh, int(fh * ar)
    else:
        cw, ch = fw, int(fw / ar)
    x1 = (fw - cw) // 2
    y1 = (fh - ch) // 2
    return y1, y1 + ch, x1, x1 + cw


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def create_video_writer(video_out: str, fps: float, fw: int, fh: int):
    abs_out = os.path.abspath(video_out)
    candidates = [
        (abs_out,                                  cv2.VideoWriter_fourcc(*"avc1"), "mp4/avc1"),
        (abs_out,                                  cv2.VideoWriter_fourcc(*"mp4v"), "mp4/mp4v"),
        (os.path.splitext(abs_out)[0] + ".avi",   cv2.VideoWriter_fourcc(*"XVID"), "avi/XVID"),
        (os.path.splitext(abs_out)[0] + ".avi",   cv2.VideoWriter_fourcc(*"MJPG"), "avi/MJPG"),
    ]
    for path, fourcc, label in candidates:
        w = cv2.VideoWriter(path, fourcc, fps, (fw, fh))
        if w.isOpened():
            print(f"VideoWriter OK: {label} → {path}")
            return w, path
        w.release()
        print(f"VideoWriter failed: {label}")
    raise RuntimeError("Could not open any video writer.")


# ============================================================
# INFERENCE
# ============================================================

def run_model(frame_bgr: np.ndarray, crop: tuple):
    """Returns (prob, gray_model) both at MODEL_H×MODEL_W resolution."""
    y1, y2, x1, x2 = crop
    gray    = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    gray    = clahe.apply(gray)
    resized = cv2.resize(gray, (MODEL_W, MODEL_H), interpolation=cv2.INTER_AREA)
    x       = resized.astype(np.float32)[None, None] / 255.0

    with torch.no_grad():
        out = model(torch.from_numpy(x).to(DEVICE))
        if MODEL_HAS_SIGMOID:
            prob = out[0, 0].detach().cpu().numpy()
        else:
            prob = torch.sigmoid(out[:, :1])[0, 0].detach().cpu().numpy()

    return prob.astype(np.float32), resized


# ============================================================
# MOTION
# ============================================================

def update_motion(gray_curr, gray_prev, motion_ema_prev, is_moving_prev):
    if gray_prev is None:
        return 0.0, 0.0, False, 0.0

    b1  = cv2.GaussianBlur(gray_curr.astype(np.float32), (MOTION_BLUR_K, MOTION_BLUR_K), 0)
    b2  = cv2.GaussianBlur(gray_prev.astype(np.float32), (MOTION_BLUR_K, MOTION_BLUR_K), 0)
    raw = float(np.mean(np.abs(b1 - b2))) / 255.0
    ema = MOTION_EMA_ALPHA * raw + (1.0 - MOTION_EMA_ALPHA) * motion_ema_prev

    moving = ema > (MOTION_OFF_THR if is_moving_prev else MOTION_ON_THR)
    denom  = max(MOTION_ON_THR - MOTION_OFF_THR, 1e-6)
    norm   = float(np.clip((ema - MOTION_OFF_THR) / denom, 0.0, 1.0))

    return raw, ema, moving, norm


# ============================================================
# TEMPORAL EMA
# ============================================================

def temporal_ema(prob_new: np.ndarray,
                 prob_prev: Optional[np.ndarray],
                 motion_norm: float) -> np.ndarray:
    if prob_prev is None:
        return prob_new.copy()

    alpha_rise = ALPHA_RISE_STILL + motion_norm * (ALPHA_RISE_MOVING - ALPHA_RISE_STILL)
    alpha_fall = ALPHA_FALL_STILL + motion_norm * (ALPHA_FALL_MOVING - ALPHA_FALL_STILL)

    blended = np.where(
        prob_new >= prob_prev,
        alpha_rise * prob_new + (1.0 - alpha_rise) * prob_prev,
        alpha_fall * prob_new + (1.0 - alpha_fall) * prob_prev
    )
    return blended.astype(np.float32)


# ============================================================
# STABLE MASK
# ============================================================

def make_stable_mask(prob_ema: np.ndarray) -> np.ndarray:
    ksize    = max(3, int(GAUSS_SIGMA * 4) | 1)
    smoothed = cv2.GaussianBlur(prob_ema, (ksize, ksize), GAUSS_SIGMA)
    binary   = (smoothed >= THRESHOLD).astype(np.uint8)

    # CLOSE first (bridge intra-vein gaps), then OPEN (remove noise)
    if CLOSE_K > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, _kernel(CLOSE_K))
    if OPEN_K > 0:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  _kernel(OPEN_K))
    if DILATE_K > 0:
        binary = cv2.dilate(binary, _kernel(DILATE_K), iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = np.zeros_like(binary)
    for cid in range(1, n):
        if stats[cid, cv2.CC_STAT_AREA] >= MIN_AREA:
            cleaned[labels == cid] = 1

    return cleaned.astype(np.uint8)


# ============================================================
# IDSS ANALYSIS
# ============================================================

def run_idss_analysis(prob_ema: np.ndarray, gray_model: np.ndarray):
    """
    Run the full IDSS decision pipeline at model resolution.

    Uses prob_ema (temporally smoothed) for stable segment scoring.
    Returns (best_feat, insertion_point, best_score, mask_sm) or
    (None, None, None, None) when no viable segment is found.
    """
    mask_raw = (prob_ema > THRESHOLD).astype(np.uint8)
    mask_sm  = smooth_mask(mask_raw)
    dist_map = cv2.distanceTransform(mask_sm, cv2.DIST_L2, 5)
    skel     = ski_skeletonize(mask_sm.astype(bool)).astype(np.uint8)
    skel     = prune_spurs(skel, iters=SPUR_PRUNE_ITERS)
    segments = extract_graph_segments(skel, min_len_px=MIN_SEGMENT_LEN)

    if len(segments) == 0:
        return None, None, None, None

    features          = extract_idss_features(
        segments, skel, dist_map, prob_ema, (MODEL_H, MODEL_W)
    )
    accepted_features = []
    rule_results      = []

    for feat in features:
        acc, penalty, bonus, _ = apply_knowledge_rules(feat)
        if acc:
            accepted_features.append(feat)
            rule_results.append({"penalty": penalty, "bonus": bonus})

    if len(accepted_features) == 0:
        return None, None, None, None

    weights, _   = compute_ahp_weights(verbose=False)
    normalized   = normalize_features(accepted_features)
    t_scores     = topsis_score(normalized, weights)
    final_scores = [
        min(1.0, t_scores[k] * rule_results[k]["penalty"] * rule_results[k]["bonus"])
        for k in range(len(t_scores))
    ]

    best_idx        = int(np.argmax(final_scores))
    best_feat       = accepted_features[best_idx]
    best_score      = final_scores[best_idx]
    insertion_point = find_insertion_point(best_feat["path"], dist_map)

    return best_feat, insertion_point, best_score, mask_sm


# ============================================================
# SKELETONIZE
# ============================================================

def mask_to_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Compute 1-pixel centerline skeleton from a stable binary mask.

    Uses skimage's medial-axis-based skeletonize, which gives clean
    single-pixel paths through the center of each vein region.

    The mask does all the stability work. The skeleton just finds the
    centerline of whatever stable region the mask defined.
    """
    if mask.max() == 0:
        return np.zeros_like(mask)

    # skimage skeletonize expects bool input
    skel = ski_skeletonize(mask.astype(bool))
    return skel.astype(np.uint8)


# ============================================================
# RENDERING
# ============================================================

def render_skeleton(frame: np.ndarray,
                    skel_model: np.ndarray,
                    crop: tuple) -> np.ndarray:
    """
    Upscale skeleton to display resolution, dilate to controlled thickness,
    draw as clean teal lines.

    The skeleton is 1px at model resolution (704×512). After upscaling to
    display resolution it becomes ~2-3px wide naturally, then we dilate to
    SKEL_THICKNESS for visibility.
    """
    y1, y2, x1, x2 = crop
    crop_h, crop_w  = y2 - y1, x2 - x1

    # Upscale with INTER_NEAREST to keep 1-pixel paths crisp (no blurring)
    skel_display = cv2.resize(
        skel_model.astype(np.uint8) * 255,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )

    # Dilate to controlled visual thickness
    if SKEL_THICKNESS > 1:
        skel_display = cv2.dilate(skel_display, _kernel(SKEL_THICKNESS), iterations=1)

    # Place in full-frame canvas
    skel_full = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    skel_full[y1:y2, x1:x2] = skel_display

    out = frame.copy()

    # Optional glow layer (drawn first, underneath)
    if DRAW_GLOW and GLOW_ALPHA > 0:
        glow_mask = cv2.dilate(skel_full, _kernel(GLOW_THICKNESS), iterations=1)
        glow_layer = out.copy()
        glow_layer[glow_mask > 0] = GLOW_COLOR
        out = cv2.addWeighted(out, 1.0 - GLOW_ALPHA, glow_layer, GLOW_ALPHA, 0)

    # Main skeleton line: solid color where skeleton pixels are active
    # Use the dilated skeleton as a mask and paint directly
    # This avoids LINE_AA artifacts on the dilated blobs
    color_img = np.full_like(out, SKEL_COLOR, dtype=np.uint8)
    mask_3ch  = (skel_full[..., None] > 0).repeat(3, axis=2)
    if SKEL_ALPHA >= 1.0:
        np.copyto(out, color_img, where=mask_3ch)
    else:
        out = np.where(mask_3ch,
                       (SKEL_ALPHA * color_img + (1.0 - SKEL_ALPHA) * out).astype(np.uint8),
                       out)

    return out


def render_outline(frame: np.ndarray,
                   mask_model: np.ndarray,
                   crop: tuple) -> np.ndarray:
    """Thin contour outline (v22 style). Used in 'outline' and 'both' modes."""
    y1, y2, x1, x2 = crop
    crop_h, crop_w  = y2 - y1, x2 - x1

    mask_display = cv2.resize(
        mask_model.astype(np.uint8) * 255,
        (crop_w, crop_h),
        interpolation=cv2.INTER_NEAREST
    )
    mask_full = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    mask_full[y1:y2, x1:x2] = mask_display

    contours, _ = cv2.findContours(mask_full, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return frame

    smoothed = []
    for cnt in contours:
        if len(cnt) < 5 or CONTOUR_SMOOTH_EPSILON <= 0:
            smoothed.append(cnt)
        else:
            eps    = CONTOUR_SMOOTH_EPSILON * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, eps, True)
            smoothed.append(approx if len(approx) >= 3 else cnt)

    out = frame.copy()
    cv2.drawContours(out, smoothed, -1, CONTOUR_COLOR, CONTOUR_THICK, cv2.LINE_AA)
    return out


def render_overlay(frame: np.ndarray,
                   mask_model: np.ndarray,
                   skel_model: np.ndarray,
                   crop: tuple) -> np.ndarray:
    if RENDER_MODE == "skeleton":
        return render_skeleton(frame, skel_model, crop)
    elif RENDER_MODE == "outline":
        return render_outline(frame, mask_model, crop)
    elif RENDER_MODE == "both":
        out = render_outline(frame, mask_model, crop)
        return render_skeleton(out, skel_model, crop)
    else:
        raise ValueError(f"Unknown RENDER_MODE: {RENDER_MODE!r}")


def draw_idss_overlay(frame: np.ndarray,
                      best_feat,
                      insertion_point,
                      best_score,
                      crop: tuple) -> np.ndarray:
    """
    Draw IDSS elements on top of the skeleton overlay:
      - Best vein segment path  (green dots)
      - Insertion point         (cyan circle + red dot + arrow + label)
      - Safety score            (bottom-left, colour-coded)

    All model-resolution coordinates are mapped to crop display coordinates.
    """
    # Only show when score is confidently green
    if best_score is None or best_score < IDSS_MIN_DISPLAY_SCORE:
        return frame

    y1, y2, x1, x2 = crop
    scale_y = (y2 - y1) / MODEL_H
    scale_x = (x2 - x1) / MODEL_W

    # Best segment path
    if best_feat is not None:
        for (ym, xm) in best_feat["path"]:
            fy = y1 + int(ym * scale_y)
            fx = x1 + int(xm * scale_x)
            cv2.circle(frame, (fx, fy), 3, (0, 255, 0), -1)

    # Insertion point marker
    if insertion_point is not None:
        iy = y1 + int(insertion_point[0] * scale_y)
        ix = x1 + int(insertion_point[1] * scale_x)
        cv2.circle(frame, (ix, iy), 18, (0, 255, 255), 2)
        cv2.circle(frame, (ix, iy),  5, (0, 0, 255),  -1)
        cv2.arrowedLine(frame,
                        (ix + 45, iy - 45),
                        (ix + 19, iy - 19),
                        (0, 255, 255), 2, tipLength=0.35)
        cv2.putText(frame, "INSERT HERE",
                    (ix + 48, iy - 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2, cv2.LINE_AA)

    # Safety score — bottom-left, clear of the top HUD bar
    if best_score is not None:
        score_pct   = best_score * 100
        score_color = (
            (0, 255, 0)   if score_pct >= 75 else
            (0, 165, 255) if score_pct >= 50 else
            (0, 0, 255)
        )
        cv2.putText(frame, f"Safety: {score_pct:.1f}%",
                    (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, score_color, 2, cv2.LINE_AA)

    return frame


# ============================================================
# HUD
# ============================================================

def draw_hud(frame, frame_idx, total, prob_max, alpha_fall,
             is_moving, motion_raw, motion_ema_val, mask_coverage):
    if not SHOW_HUD:
        return
    fw = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (fw, 78), (0, 0, 0), -1)

    detected     = mask_coverage > 0.1
    motion_str   = "MOVING" if is_moving else "STILL"
    status       = f"Veins [{motion_str}]" if detected else f"No veins [{motion_str}]"
    status_color = (0, 230, 180) if detected else (80, 80, 220)

    cv2.putText(frame, status, (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, status_color, 2, cv2.LINE_AA)

    cv2.putText(frame,
        f"mode={RENDER_MODE}  prob_max={prob_max:.2f}  cov={mask_coverage:.1f}%  "
        f"thr={THRESHOLD:.2f}  fall_a={alpha_fall:.2f}  close={CLOSE_K}",
        (12, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

    cv2.putText(frame,
        f"motion={motion_raw:.4f}  m_ema={motion_ema_val:.4f}  "
        f"skel_thick={SKEL_THICKNESS}  open={OPEN_K}  min_area={MIN_AREA}",
        (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (185, 185, 185), 1, cv2.LINE_AA)

    pct = frame_idx / total * 100 if total > 0 else 0.0
    cv2.putText(frame, f"Frame {frame_idx}/{total} ({pct:.0f}%)",
        (fw - 270, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (185, 185, 185), 1, cv2.LINE_AA)


# ============================================================
# MAIN
# ============================================================

def main(video_in: str = VIDEO_IN, video_out: str = VIDEO_OUT):
    print("CWD    :", os.getcwd())
    print("Input  :", os.path.abspath(video_in))
    print("Output :", os.path.abspath(video_out))

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_in}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop         = get_crop(fh, fw)
    y1, y2, x1, x2 = crop

    print(f"Video  : {fw}×{fh} @ {fps:.1f}fps  ({total_frames} frames)")
    print(f"Crop   : x={x1}:{x2}  y={y1}:{y2}")
    print(f"Mode   : {RENDER_MODE}  skel_thickness={SKEL_THICKNESS}")
    print(f"EMA    : rise_still={ALPHA_RISE_STILL}  fall_still={ALPHA_FALL_STILL}")
    print(f"Morpho : close={CLOSE_K}  open={OPEN_K}  min_area={MIN_AREA}\n")

    out_vid, actual_out = create_video_writer(video_out, fps, fw, fh)

    prob_ema        : Optional[np.ndarray] = None
    prev_gray       : Optional[np.ndarray] = None
    motion_ema_val  : float = 0.0
    is_moving       : bool  = False
    low_conf_streak : int   = 0
    frame_count     : int   = 0

    # IDSS temporal state
    idss_frame_count   : int   = 0
    candidate_points   = deque(maxlen=SMOOTH_WINDOW)
    candidate_scores   = deque(maxlen=SMOOTH_WINDOW)
    candidate_segments = deque(maxlen=SMOOTH_WINDOW)  # segment-level voting
    stable_point       = None
    stable_score       = None
    stable_feat        = None
    idss_best_feat     = None
    idss_insertion_pt  = None
    idss_best_score    = None

    t_start = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1

        crop_bgr  = frame[y1:y2, x1:x2]
        curr_gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

        motion_raw, motion_ema_val, is_moving, motion_norm = update_motion(
            curr_gray, prev_gray, motion_ema_val, is_moving)
        prev_gray = curr_gray.copy()

        prob_raw, gray_model = run_model(frame, crop)
        prob_max = float(prob_raw.max())

        # Scene reset
        low_conf_streak = low_conf_streak + 1 if prob_max < SCENE_LOW_CONF_THR else 0
        if low_conf_streak >= SCENE_RESET_N:
            prob_ema        = None
            motion_ema_val  = 0.0
            is_moving       = False
            low_conf_streak = 0
            print(f"[{frame_count}] Scene reset")

        # EMA → stable mask → skeleton
        prob_ema = temporal_ema(prob_raw, prob_ema, motion_norm)
        mask     = make_stable_mask(prob_ema)
        skel     = mask_to_skeleton(mask)

        # Render skeleton/outline
        display = render_overlay(frame, mask, skel, crop)

        # IDSS analysis (every REANALYZE_EVERY frames, gated on confidence)
        idss_frame_count += 1
        if idss_frame_count % REANALYZE_EVERY == 0:
            if float(prob_ema.max()) >= MIN_PROB_TO_SHOW:
                new_feat, new_point, new_score, _ = run_idss_analysis(prob_ema, gray_model)
                if new_point is not None:
                    if is_moving:
                        # Hand moving: follow vein directly, reset votes
                        candidate_points.clear()
                        candidate_scores.clear()
                        candidate_segments.clear()
                        stable_point = new_point
                        stable_score = new_score
                        stable_feat  = new_feat
                    else:
                        # Hand still: vote for a stable insertion point
                        candidate_points.append(new_point)
                        candidate_scores.append(new_score)
                        if len(candidate_points) >= MIN_VOTES_NEEDED:
                            latest_y, latest_x = candidate_points[-1]
                            votes = sum(
                                1 for (py, px) in candidate_points
                                if abs(py - latest_y) < STABILITY_DIST_PX and
                                   abs(px - latest_x) < STABILITY_DIST_PX
                            )
                            if votes >= MIN_VOTES_NEEDED:
                                agreeing = [
                                    (py, px) for (py, px) in candidate_points
                                    if abs(py - latest_y) < STABILITY_DIST_PX and
                                       abs(px - latest_x) < STABILITY_DIST_PX
                                ]
                                stable_point = (
                                    int(np.mean([p[0] for p in agreeing])),
                                    int(np.mean([p[1] for p in agreeing]))
                                )
                                stable_score = float(np.mean([
                                    s for s, (py, px) in zip(candidate_scores, candidate_points)
                                    if abs(py - latest_y) < STABILITY_DIST_PX and
                                       abs(px - latest_x) < STABILITY_DIST_PX
                                ]))

                        # Vote for a stable segment (green path)
                        seg_cy = int(np.mean([p[0] for p in new_feat["path"]]))
                        seg_cx = int(np.mean([p[1] for p in new_feat["path"]]))
                        candidate_segments.append(((seg_cy, seg_cx), new_feat))
                        if len(candidate_segments) >= MIN_VOTES_NEEDED:
                            latest_cy, latest_cx = candidate_segments[-1][0]
                            agreeing_segs = [
                                feat for (cy, cx), feat in candidate_segments
                                if abs(cy - latest_cy) < STABILITY_DIST_PX and
                                   abs(cx - latest_cx) < STABILITY_DIST_PX
                            ]
                            if len(agreeing_segs) >= MIN_VOTES_NEEDED:
                                stable_feat = agreeing_segs[-1]
            else:
                candidate_points.clear()
                candidate_scores.clear()
                candidate_segments.clear()
                stable_point   = None
                stable_score   = None
                stable_feat    = None

            idss_best_feat    = stable_feat
            idss_insertion_pt = stable_point
            idss_best_score   = stable_score

        # Draw IDSS overlay on top of skeleton
        display = draw_idss_overlay(display, idss_best_feat, idss_insertion_pt,
                                    idss_best_score, crop)

        alpha_fall    = ALPHA_FALL_STILL + motion_norm * (ALPHA_FALL_MOVING - ALPHA_FALL_STILL)
        mask_coverage = float(mask.sum()) / float(mask.size) * 100.0

        draw_hud(display, frame_count, total_frames, prob_max,
                 alpha_fall, is_moving, motion_raw, motion_ema_val, mask_coverage)

        out_vid.write(display)

        if frame_count % 30 == 0:
            elapsed    = time.time() - t_start
            fps_actual = frame_count / max(elapsed, 1e-6)
            pct        = frame_count / total_frames * 100 if total_frames > 0 else 0
            print(
                f"[{frame_count:5d}/{total_frames}] {pct:5.1f}%  "
                f"{'M' if is_moving else 'S'}  "
                f"motion={motion_raw:.4f}  prob_max={prob_max:.2f}  "
                f"cov={mask_coverage:.1f}%  fps={fps_actual:.1f}"
            )

    cap.release()
    out_vid.release()

    elapsed = time.time() - t_start
    size    = os.path.getsize(actual_out) if os.path.exists(actual_out) else 0
    print(f"\nDone in {elapsed:.1f}s  |  {actual_out}  ({size:,} bytes)")
    print(f"Avg: {frame_count / max(elapsed, 1e-6):.2f} fps")


if __name__ == "__main__":
    main()
