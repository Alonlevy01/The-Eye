"""Deterministic OpenCV visual detectors for CS2 gameplay HUD & screen events."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from the_eye.schemas import FrameMetrics, format_timestamp

# Normalized (x1, y1, x2, y2) HUD Regions of Interest for 16:9 CS2 POV recordings
KILLFEED_ROI = (0.68, 0.04, 0.99, 0.28)
CENTER_CROSSHAIR_ROI = (0.35, 0.35, 0.65, 0.65)
CORNER_SIZE = 0.14


@dataclass
class SampledFrame:
    """A decoded BGR frame paired with its extracted CS2 visual metrics."""

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


def detect_flashbang(hsv_frame: np.ndarray) -> tuple[bool, float, float]:
    """Detect CS2 flashbang whiteout: high luminance (V >= 215) + low saturation (S <= 45)."""
    mean_s = float(np.mean(hsv_frame[:, :, 1]))
    mean_v = float(np.mean(hsv_frame[:, :, 2]))
    is_flashed = mean_v >= 215.0 and mean_s <= 45.0
    return is_flashed, round(mean_v, 2), round(mean_s, 2)


def detect_scope_vignette(gray_frame: np.ndarray) -> bool:
    """Detect AWP/SSG08/AUG sniper scope: 4 very dark outer corners with brighter center."""
    h, w = gray_frame.shape[:2]
    ch = max(1, int(h * CORNER_SIZE))
    cw = max(1, int(w * CORNER_SIZE))
    corners = [
        gray_frame[:ch, :cw],
        gray_frame[:ch, w - cw :],
        gray_frame[h - ch :, :cw],
        gray_frame[h - ch :, w - cw :],
    ]
    corner_means = [float(np.mean(c)) for c in corners]
    center = _crop_norm(gray_frame, CENTER_CROSSHAIR_ROI)
    center_mean = float(np.mean(center))
    return all(m < 22.0 for m in corner_means) and center_mean > 35.0


def compute_killfeed_red_ratio(bgr_frame: np.ndarray) -> float:
    """Measure red highlight pixel ratio in CS2's top-right killfeed HUD region."""
    crop = _crop_norm(bgr_frame, KILLFEED_ROI)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    # Vivid CS2 killfeed crimson border (high S & V avoids Mirage/Dust2 sandstone false positives)
    mask_low = cv2.inRange(hsv, np.array([0, 165, 150]), np.array([8, 255, 255]))
    mask_high = cv2.inRange(hsv, np.array([172, 165, 150]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(mask_low, mask_high)
    total_pixels = max(1, red_mask.size)
    return round(float(np.count_nonzero(red_mask)) / total_pixels, 5)


def analyze_single_frame(
    bgr_frame: np.ndarray,
    *,
    frame_index: int,
    timestamp_sec: float,
    original_video_sec: float,
    prev_center_gray: np.ndarray | None = None,
    prev_killfeed_gray: np.ndarray | None = None,
) -> tuple[FrameMetrics, np.ndarray, np.ndarray]:
    """Compute all CS2 OpenCV visual signals for one BGR frame."""
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)

    is_flashed, brightness, saturation = detect_flashbang(hsv)
    is_scoped = False if is_flashed else detect_scope_vignette(gray)
    kf_red_ratio = compute_killfeed_red_ratio(bgr_frame)

    center_gray = _crop_norm(gray, CENTER_CROSSHAIR_ROI)
    kf_gray = _crop_norm(gray, KILLFEED_ROI)

    center_motion = 0.0
    if prev_center_gray is not None and prev_center_gray.shape == center_gray.shape:
        center_motion = round(float(np.mean(cv2.absdiff(center_gray, prev_center_gray))), 2)

    kf_delta = 0.0
    if prev_killfeed_gray is not None and prev_killfeed_gray.shape == kf_gray.shape:
        kf_delta = round(float(np.mean(cv2.absdiff(kf_gray, prev_killfeed_gray))), 2)

    metrics = FrameMetrics(
        frame_index=frame_index,
        timestamp_sec=round(timestamp_sec, 2),
        original_video_sec=round(original_video_sec, 2),
        brightness=brightness,
        saturation=saturation,
        is_flashed=is_flashed,
        is_scoped=is_scoped,
        killfeed_red_ratio=kf_red_ratio,
        killfeed_delta=kf_delta,
        center_motion=center_motion,
    )
    return metrics, center_gray, kf_gray


def sample_video_frames(
    video_path: Path,
    *,
    fps: float = 2.0,
    excerpt_start_sec: float = 0.0,
) -> list[SampledFrame]:
    """Decode a gameplay video at `fps` samples/sec and compute OpenCV metrics per frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if native_fps <= 0:
        native_fps = 30.0
    step = max(1, int(round(native_fps / fps)))

    sampled: list[SampledFrame] = []
    prev_center: np.ndarray | None = None
    prev_kf: np.ndarray | None = None
    raw_idx = 0
    sample_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_idx % step == 0:
                ts = raw_idx / native_fps
                metrics, prev_center, prev_kf = analyze_single_frame(
                    frame,
                    frame_index=sample_idx,
                    timestamp_sec=ts,
                    original_video_sec=excerpt_start_sec + ts,
                    prev_center_gray=prev_center,
                    prev_killfeed_gray=prev_kf,
                )
                sampled.append(SampledFrame(metrics=metrics, bgr=frame.copy()))
                sample_idx += 1
            raw_idx += 1
    finally:
        cap.release()

    return sampled


def annotate_frame(
    bgr_frame: np.ndarray,
    metrics: FrameMetrics,
    *,
    event_label: str,
) -> np.ndarray:
    """Draw color-coded CS2 HUD ROI boxes and visual evidence badges onto a keyframe."""
    out = bgr_frame.copy()
    h, w = out.shape[:2]

    def _rect(roi: tuple[float, float, float, float], color: tuple[int, int, int], label: str):
        x1, y1, x2, y2 = (
            int(roi[0] * w),
            int(roi[1] * h),
            int(roi[2] * w),
            int(roi[3] * h),
        )
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            out,
            label,
            (x1 + 4, max(18, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    _rect(KILLFEED_ROI, (60, 220, 120), f"Killfeed ROI (red={metrics.killfeed_red_ratio:.3f})")
    _rect(CENTER_CROSSHAIR_ROI, (255, 210, 60), f"Center ROI (motion={metrics.center_motion:.1f})")

    if metrics.is_flashed:
        cv2.rectangle(out, (4, 4), (w - 4, h - 4), (0, 210, 255), 5)
    elif metrics.is_scoped:
        cv2.rectangle(out, (4, 4), (w - 4, h - 4), (220, 80, 240), 4)

    # Bottom status banner with exact timestamp & detected event
    bar_h = 34
    cv2.rectangle(out, (0, h - bar_h), (w, h), (18, 22, 30), -1)
    ts_clip = format_timestamp(metrics.timestamp_sec)
    ts_orig = format_timestamp(metrics.original_video_sec)
    banner = (
        f"THE-EYE | {event_label.upper()} | Clip {ts_clip} (Orig {ts_orig}) | "
        f"V={metrics.brightness:.0f} S={metrics.saturation:.0f} "
        f"AudioRMS={metrics.audio_rms:.3f}"
    )
    cv2.putText(
        out,
        banner,
        (10, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (240, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return out
