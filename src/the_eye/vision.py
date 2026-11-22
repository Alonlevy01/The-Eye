"""BTD6 Computer Vision HUD Reader (RapidOCR) for Lives, Cash ($), Round, and Selected Tower."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from the_eye.btd6_wiki import match_tower_name
from the_eye.schemas import FrameMetrics, format_timestamp

# Top HUD strip covering Lives, Cash ($), Round (X/Y), and open Upgrade Panel titles
TOP_HUD_ROI = (0.0, 0.0, 0.85, 0.115)
PLAYABLE_MAP_ROI = (0.02, 0.12, 0.82, 0.94)

_OCR_INSTANCE: Any | None = None


def _get_ocr() -> Any:
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        from rapidocr_onnxruntime import RapidOCR

        _OCR_INSTANCE = RapidOCR()
    return _OCR_INSTANCE


@dataclass
class SampledFrame:
    """A decoded BGR frame paired with its extracted BTD6 HUD & map telemetry."""

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


def parse_btd6_hud_texts(
    items: list[tuple[float, str]] | list[tuple[float, float, str]],
) -> tuple[int | None, int | None, int | None, int | None, str | None]:
    """Parse `(lives, cash, round_num, max_round, selected_tower)` from OCR items."""
    normalized: list[tuple[float, float, str]] = []
    for item in items:
        if len(item) == 3:
            normalized.append((float(item[0]), float(item[1]), str(item[2])))
        else:
            normalized.append((float(item[0]), 0.25, str(item[1])))

    ordered = sorted(normalized, key=lambda entry: entry[0])
    cash: int | None = None
    cash_x: float | None = None
    cash_y: float | None = None
    round_num: int | None = None
    max_round: int | None = None
    lives: int | None = None
    selected_tower: str | None = None

    for x_pos, y_norm, raw in ordered:
        text = raw.strip()
        if not text:
            continue

        # 1. Cash: "$317", "$1,250", "$14,820" (top HUD bar)
        m_cash = re.search(r"\$\s*([\d,]+)", text)
        if m_cash and cash is None and y_norm < 0.65:
            digits = m_cash.group(1).replace(",", "")
            if digits.isdigit():
                cash = int(digits)
                cash_x = x_pos
                cash_y = y_norm
                continue

        # 2. Round: "30/100", "26/80", "59/60", "61/70", "14/40"
        m_round = re.search(r"(\d{1,3})\s*/\s*(\d{2,3})", text)
        if m_round and round_num is None:
            r_cur = int(m_round.group(1))
            r_max = int(m_round.group(2))
            if 30 <= r_max <= 140 and 1 <= r_cur <= r_max + 20:
                round_num = r_cur
                max_round = r_max
                continue

        # 3. Selected tower panel title (e.g. "MONKeYVILLAGe" -> "Monkey Village")
        if selected_tower is None:
            tower = match_tower_name(text)
            if tower is not None:
                selected_tower = tower

    # 4. Lives: strictly in the top heart row (y_norm < 0.58) and directly left of Cash,
    #    ignoring tower pop counts that appear lower inside an open left upgrade panel.
    lives_candidates: list[tuple[float, int]] = []
    for x_pos, y_norm, raw in ordered:
        if y_norm >= 0.58:
            continue
        if cash_x is not None and x_pos >= cash_x:
            continue
        if cash_y is not None and abs(y_norm - cash_y) > 0.25:
            continue
        cleaned = raw.strip().replace(",", "")
        if cleaned.isdigit():
            val = int(cleaned)
            if 0 <= val <= 9999:
                lives_candidates.append((x_pos, val))
        elif cleaned in ("I", "l", "L", "|") and max_round == 100:
            lives_candidates.append((x_pos, 1))

    if lives_candidates:
        # Closest candidate to the left of Cash is the Heart/Lives counter
        lives = lives_candidates[-1][1]

    # In CHIMPS (max_round == 100), if lives is unread because '1' is super narrow, infer 1
    if lives is None and max_round == 100 and cash is not None:
        lives = 1

    return lives, cash, round_num, max_round, selected_tower


def read_btd6_hud(
    bgr_frame: np.ndarray,
    *,
    ocr_engine: Any | None = None,
) -> tuple[int | None, int | None, int | None, int | None, str | None]:
    """Run RapidOCR on BTD6's top HUD strip and parse Lives, Cash, Round, and Selected Tower."""
    engine = ocr_engine or _get_ocr()
    top_strip = _crop_norm(bgr_frame, TOP_HUD_ROI)
    strip_h = max(1.0, float(top_strip.shape[0]))
    res, _ = engine(top_strip)
    items: list[tuple[float, float, str]] = []
    for entry in res or []:
        box, text = entry[0], str(entry[1])
        if box:
            x_left = float(min(pt[0] for pt in box))
            y_center = float(sum(pt[1] for pt in box)) / float(len(box))
            y_norm = y_center / strip_h
        else:
            x_left, y_norm = 0.0, 0.25
        items.append((x_left, y_norm, text))
    return parse_btd6_hud_texts(items)


def analyze_single_frame(
    bgr_frame: np.ndarray,
    *,
    frame_index: int,
    timestamp_sec: float,
    original_video_sec: float,
    prev_map_gray: np.ndarray | None = None,
    ocr_engine: Any | None = None,
) -> tuple[FrameMetrics, np.ndarray]:
    """Extract BTD6 HUD telemetry and static-map activity delta for one BGR frame."""
    lives, cash, round_num, max_round, tower = read_btd6_hud(bgr_frame, ocr_engine=ocr_engine)

    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    map_gray = cv2.resize(_crop_norm(gray, PLAYABLE_MAP_ROI), (320, 180))
    map_activity = 0.0
    if prev_map_gray is not None and prev_map_gray.shape == map_gray.shape:
        map_activity = round(float(np.mean(cv2.absdiff(map_gray, prev_map_gray))), 2)

    metrics = FrameMetrics(
        frame_index=frame_index,
        timestamp_sec=round(timestamp_sec, 2),
        original_video_sec=round(original_video_sec, 2),
        lives=lives,
        cash=cash,
        round_num=round_num,
        max_round=max_round,
        selected_tower=tower,
        map_activity=map_activity,
    )
    return metrics, map_gray


def sample_video_frames(
    video_path: Path,
    *,
    fps: float = 2.0,
    excerpt_start_sec: float = 0.0,
) -> list[SampledFrame]:
    """Decode a BTD6 video at `fps` samples/sec, read HUD via OCR, and smooth transient glitches."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if native_fps <= 0:
        native_fps = 30.0
    step = max(1, int(round(native_fps / fps)))

    ocr = _get_ocr()
    sampled: list[SampledFrame] = []
    prev_map: np.ndarray | None = None
    raw_idx = 0
    sample_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if raw_idx % step == 0:
                ts = raw_idx / native_fps
                metrics, prev_map = analyze_single_frame(
                    frame,
                    frame_index=sample_idx,
                    timestamp_sec=ts,
                    original_video_sec=excerpt_start_sec + ts,
                    prev_map_gray=prev_map,
                    ocr_engine=ocr,
                )
                sampled.append(SampledFrame(metrics=metrics, bgr=frame.copy()))
                sample_idx += 1
            raw_idx += 1
    finally:
        cap.release()

    # Post-process: enforce clean rounds & cash/lives deltas (ignoring non-HUD intro/menu frames)
    raw_lives = [
        sf.metrics.lives
        if (sf.metrics.cash is not None or sf.metrics.round_num is not None)
        else None
        for sf in sampled
    ]
    raw_rounds = [sf.metrics.round_num for sf in sampled]
    valid_lives = [lv for lv in raw_lives if lv is not None]
    valid_max_rounds = [sf.metrics.max_round for sf in sampled if sf.metrics.max_round is not None]
    clip_max_round = (
        max(set(valid_max_rounds), key=valid_max_rounds.count) if valid_max_rounds else None
    )
    is_chimps_like = (
        clip_max_round == 100
        and valid_lives
        and (sum(1 for lv in valid_lives if lv == 1) / len(valid_lives)) >= 0.4
    )

    # 3-frame median filter on raw lives to suppress 1-frame OCR misreads
    smoothed_lives: list[int | None] = []
    for idx, lv in enumerate(raw_lives):
        if is_chimps_like:
            smoothed_lives.append(0 if lv == 0 else 1)
            continue
        window = [
            raw_lives[j]
            for j in range(max(0, idx - 1), min(len(raw_lives), idx + 2))
            if raw_lives[j] is not None
        ]
        if window:
            window.sort()
            smoothed_lives.append(window[len(window) // 2])
        else:
            smoothed_lives.append(lv)

    last_round: int | None = None
    last_max_round: int | None = None
    last_cash: int | None = None
    last_lives: int | None = None

    for idx, sf in enumerate(sampled):
        m = sf.metrics
        r = m.round_num
        mr = m.max_round or last_max_round
        if r is not None:
            next_r = raw_rounds[idx + 1] if idx + 1 < len(raw_rounds) else None
            confirmed_jump = next_r is not None and abs(next_r - r) <= 2
            if (
                last_round is not None
                and (r < last_round or r > last_round + 5)
                and not confirmed_jump
            ):
                r = last_round  # reject single-frame OCR misread
            else:
                last_round = r
        else:
            r = last_round
        if mr is not None:
            last_max_round = mr

        c = m.cash
        cash_delta = 0
        if c is not None:
            # Reject extreme single-frame OCR digit drops/explosions (> $200k jump in 0.5s)
            if last_cash is not None and abs(c - last_cash) > 200_000:
                c = last_cash
            elif last_cash is not None:
                cash_delta = c - last_cash
            last_cash = c
        else:
            c = last_cash

        lv = smoothed_lives[idx]
        lives_delta = 0
        if lv is not None:
            next_lv = smoothed_lives[idx + 1] if idx + 1 < len(smoothed_lives) else None
            confirmed_lv = next_lv is not None and abs(next_lv - lv) <= 5
            if (
                last_lives is not None
                and (lv > last_lives + 25 or abs(lv - last_lives) > 500)
                and not confirmed_lv
            ):
                # Reject 1-frame upward lives spike (e.g. pop-count OCR misread)
                lv = last_lives
            elif last_lives is not None:
                lives_delta = lv - last_lives
            last_lives = lv
        else:
            lv = last_lives

        sf.metrics = m.model_copy(
            update={
                "round_num": r,
                "max_round": mr,
                "cash": c,
                "lives": lv,
                "cash_delta": cash_delta,
                "lives_delta": lives_delta,
            }
        )

    return sampled


def annotate_frame(
    bgr_frame: np.ndarray,
    metrics: FrameMetrics,
    *,
    event_label: str,
) -> np.ndarray:
    """Draw a clean BTD6 HUD verification banner & top-HUD ROI box onto a keyframe."""
    out = bgr_frame.copy()
    h, w = out.shape[:2]
    cv2.rectangle(
        out,
        (0, 0),
        (int(TOP_HUD_ROI[2] * w), int(TOP_HUD_ROI[3] * h)),
        (56, 189, 248),
        2,
    )

    bar_h = 34
    cv2.rectangle(out, (0, h - bar_h), (w, h), (15, 23, 42), -1)
    ts_clip = format_timestamp(metrics.timestamp_sec)
    r_str = (
        f"R{metrics.round_num}/{metrics.max_round}"
        if metrics.round_num and metrics.max_round
        else f"R{metrics.round_num or '?'}"
    )
    c_str = f"${metrics.cash:,}" if metrics.cash is not None else "$?"
    l_str = str(metrics.lives) if metrics.lives is not None else "?"
    t_str = f" | Tower: {metrics.selected_tower}" if metrics.selected_tower else ""
    banner = (
        f"THE-EYE BTD6 | {event_label.upper()} | Clip {ts_clip} | "
        f"{r_str} | Cash {c_str} | Lives {l_str}{t_str}"
    )
    cv2.putText(
        out,
        banner[:120],
        (10, h - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (240, 245, 255),
        1,
        cv2.LINE_AA,
    )
    return out
