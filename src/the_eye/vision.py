"""Deterministic OpenCV Rocket League Boost Gauge (0-100) polar scanner & HUD analyzer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from the_eye.schemas import FrameMetrics, format_timestamp

# Calibrated 16:9 Rocket League Boost Gauge geometry (verified on real SSL & Gold POVs):
# Center at (0.912*W, 0.840*H), outer tick ring radius 0.086*H..0.114*H,
# clockwise angular sweep from 102 deg (0 boost) -> 306 deg (100 boost) = 204 deg span.
GAUGE_CX_NORM = 0.912
GAUGE_CY_NORM = 0.840
GAUGE_R_IN_NORM = 0.086
GAUGE_R_OUT_NORM = 0.114
GAUGE_START_DEG = 102.0
GAUGE_SPAN_DEG = 204.0

CENTER_FIELD_ROI = (0.20, 0.25, 0.80, 0.78)


@dataclass
class SampledFrame:
    """A decoded BGR frame paired with its extracted Rocket League visual telemetry."""

    metrics: FrameMetrics
    bgr: np.ndarray


def _crop_norm(frame: np.ndarray, roi: tuple[float, float, float, float]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (
        int(roi[0] * w),
        int(roi[1] * h),
        int(roi[2] * w),
        int(roi[3] * h),
    )
    return frame[max(0, y1) : max(1, y2), max(0, x1) : max(1, x2)]


def read_boost_gauge(bgr_frame: np.ndarray) -> tuple[bool, float]:
    """Read Rocket League's bottom-right circular Boost Gauge (0-100) via polar arc scanning.

    Returns `(hud_visible, boost_pct)`. When a goal replay or menu hides the dark circular
    gauge backing, returns `(False, 0.0)` so replays don't pollute live boost statistics.
    """
    h, w = bgr_frame.shape[:2]
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    cx = int(round(GAUGE_CX_NORM * w))
    cy = int(round(GAUGE_CY_NORM * h))

    # Check dark circular gauge backing (radius ~0.065*H around center) to confirm live HUD
    r_bg = max(4, int(round(0.060 * h)))
    y1, y2 = max(0, cy - r_bg), min(h, cy + r_bg)
    x1, x2 = max(0, cx - r_bg), min(w, cx + r_bg)
    inner_v = hsv[y1:y2, x1:x2, 2]
    # Live boost gauge has a dark translucent circular backing (median V < 110)
    if inner_v.size == 0 or float(np.median(inner_v)) > 125.0:
        return False, 0.0

    r_in = GAUGE_R_IN_NORM * h
    r_out = GAUGE_R_OUT_NORM * h
    num_bins = 100
    lit_bins: list[bool] = []

    for i in range(num_bins):
        deg = GAUGE_START_DEG + (i / (num_bins - 1)) * GAUGE_SPAN_DEG
        rad = math.radians(deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        is_lit = False
        for r in np.linspace(r_in, r_out, 5):
            px = int(round(cx + r * cos_a))
            py = int(round(cy + r * sin_a))
            if 0 <= px < w and 0 <= py < h:
                H = int(hsv[py, px, 0])
                S = int(hsv[py, px, 1])
                V = int(hsv[py, px, 2])
                # Rocket League boost arc is dark-orange (low) to bright-yellow (high)
                if 2 <= H <= 38 and S >= 110 and V >= 130:
                    is_lit = True
                    break
        lit_bins.append(is_lit)

    # Walk clockwise from 0 boost allowing up to 4 dark gap bins (tick-mark spacing)
    highest_lit = -1
    gap = 0
    for idx, hit in enumerate(lit_bins):
        if hit:
            highest_lit = idx
            gap = 0
        else:
            gap += 1
            if gap > 5 and highest_lit >= 0:
                break
            if gap > 8 and highest_lit < 0:
                # No lit ticks near the bottom start -> boost is 0
                break

    if highest_lit < 0:
        return True, 0.0
    boost_pct = round(min(100.0, (highest_lit + 1) * (100.0 / num_bins)), 1)
    return True, boost_pct


def analyze_single_frame(
    bgr_frame: np.ndarray,
    *,
    frame_index: int,
    timestamp_sec: float,
    original_video_sec: float,
    prev_field_gray: np.ndarray | None = None,
    prev_boost: float | None = None,
) -> tuple[FrameMetrics, np.ndarray]:
    """Extract Rocket League boost gauge reading, brightness, and field motion for one frame."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    brightness = round(float(np.mean(hsv[:, :, 2])), 2)
    saturation = round(float(np.mean(hsv[:, :, 1])), 2)
    hud_visible, boost_pct = read_boost_gauge(bgr_frame)

    boost_delta = 0.0
    if hud_visible and prev_boost is not None:
        boost_delta = round(boost_pct - prev_boost, 1)

    field_gray = _crop_norm(gray, CENTER_FIELD_ROI)
    field_motion = 0.0
    if prev_field_gray is not None and prev_field_gray.shape == field_gray.shape:
        field_motion = round(float(np.mean(cv2.absdiff(field_gray, prev_field_gray))), 2)

    metrics = FrameMetrics(
        frame_index=frame_index,
        timestamp_sec=round(timestamp_sec, 2),
        original_video_sec=round(original_video_sec, 2),
        hud_visible=hud_visible,
        boost_pct=boost_pct,
        boost_delta=boost_delta,
        brightness=brightness,
        saturation=saturation,
        field_motion=field_motion,
    )
    return metrics, field_gray


def sample_video_frames(
    video_path: Path,
    *,
    fps: float = 5.0,
    excerpt_start_sec: float = 0.0,
) -> list[SampledFrame]:
    """Decode a Rocket League video at `fps` samples/sec (default 5 fps) with 3-frame boost
    median smoothing so transient orange explosions never create single-frame spikes.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if native_fps <= 0:
        native_fps = 30.0
    step = max(1, int(round(native_fps / fps)))

    sampled: list[SampledFrame] = []
    prev_field: np.ndarray | None = None
    prev_boost: float | None = None
    raw_idx = 0
    sample_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_idx % step == 0:
                ts = raw_idx / native_fps
                metrics, prev_field = analyze_single_frame(
                    frame,
                    frame_index=sample_idx,
                    timestamp_sec=ts,
                    original_video_sec=excerpt_start_sec + ts,
                    prev_field_gray=prev_field,
                    prev_boost=prev_boost,
                )
                if metrics.hud_visible:
                    prev_boost = metrics.boost_pct
                sampled.append(SampledFrame(metrics=metrics, bgr=frame.copy()))
                sample_idx += 1
            raw_idx += 1
    finally:
        cap.release()

    # Apply 3-frame median filter on live-HUD boost_pct to eliminate 1-frame occlusion glitches
    n = len(sampled)
    if n >= 3:
        smoothed: list[float] = [sf.metrics.boost_pct for sf in sampled]
        for i in range(1, n - 1):
            if (
                sampled[i - 1].metrics.hud_visible
                and sampled[i].metrics.hud_visible
                and sampled[i + 1].metrics.hud_visible
            ):
                smoothed[i] = float(
                    np.median(
                        [
                            sampled[i - 1].metrics.boost_pct,
                            sampled[i].metrics.boost_pct,
                            sampled[i + 1].metrics.boost_pct,
                        ]
                    )
                )
        last_live: float | None = None
        for i, sf in enumerate(sampled):
            if sf.metrics.hud_visible:
                b = round(smoothed[i], 1)
                delta = round(b - last_live, 1) if last_live is not None else 0.0
                sf.metrics = sf.metrics.model_copy(update={"boost_pct": b, "boost_delta": delta})
                last_live = b

    return sampled


def annotate_frame(
    bgr_frame: np.ndarray,
    metrics: FrameMetrics,
    *,
    event_label: str,
) -> np.ndarray:
    """Draw the exact polar Boost Gauge arc overlay + detected reading onto a keyframe."""
    out = bgr_frame.copy()
    h, w = out.shape[:2]
    cx = int(round(GAUGE_CX_NORM * w))
    cy = int(round(GAUGE_CY_NORM * h))
    r_mid = int(round(((GAUGE_R_IN_NORM + GAUGE_R_OUT_NORM) / 2.0) * h))

    # Draw subtle full arc track + bright detected boost arc
    cv2.ellipse(
        out,
        (cx, cy),
        (r_mid, r_mid),
        0,
        GAUGE_START_DEG,
        GAUGE_START_DEG + GAUGE_SPAN_DEG,
        (90, 100, 120),
        2,
        cv2.LINE_AA,
    )
    if metrics.hud_visible and metrics.boost_pct > 0:
        end_deg = GAUGE_START_DEG + (metrics.boost_pct / 100.0) * GAUGE_SPAN_DEG
        col = (60, 220, 100) if metrics.boost_pct >= 30 else (40, 80, 245)
        cv2.ellipse(
            out,
            (cx, cy),
            (r_mid, r_mid),
            0,
            GAUGE_START_DEG,
            end_deg,
            col,
            4,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        f"READ: {metrics.boost_pct:.0f}" if metrics.hud_visible else "REPLAY/NO HUD",
        (max(10, cx - 70), max(22, cy - r_mid - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (60, 240, 255),
        2,
        cv2.LINE_AA,
    )

    bar_h = 34
    cv2.rectangle(out, (0, h - bar_h), (w, h), (15, 23, 42), -1)
    ts_clip = format_timestamp(metrics.timestamp_sec)
    ts_orig = format_timestamp(metrics.original_video_sec)
    banner = (
        f"THE-EYE RL | {event_label.upper()} | Clip {ts_clip} (Orig {ts_orig}) | "
        f"Boost={metrics.boost_pct:.0f}/100 | Motion={metrics.field_motion:.1f} | "
        f"AudioRMS={metrics.audio_rms:.3f}"
    )
    cv2.putText(
        out,
        banner,
        (10, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return out
