"""Rocket League boost discipline scorecard, grouped mistake detection, and 3-frame filmstrips."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.schemas import (
    BoostScorecard,
    DetectedEvent,
    EventType,
    FrameMetrics,
    MistakeItem,
    format_timestamp,
)


def compute_boost_scorecard(
    timeline: list[FrameMetrics],
    events: list[DetectedEvent],
    *,
    fps: float = 2.0,
) -> BoostScorecard:
    """Compute 100% visually ground-truth Rocket League boost discipline metrics & mistake cards."""
    live = [m for m in timeline if m.hud_visible]
    if not live:
        return BoostScorecard()

    step_sec = 1.0 / max(fps, 0.5)
    boosts = np.array([m.boost_pct for m in live], dtype=np.float64)
    avg_boost = round(float(np.mean(boosts)), 1)
    live_sec = round(len(live) * step_sec, 1)

    starved_frames = [m for m in live if m.boost_pct <= 12.0]
    over_capped_frames = [m for m in live if m.boost_pct >= 96.0]
    starved_sec = round(len(starved_frames) * step_sec, 1)
    over_capped_sec = round(len(over_capped_frames) * step_sec, 1)
    starved_pct = round((len(starved_frames) / len(live)) * 100.0, 1)
    over_capped_pct = round((len(over_capped_frames) / len(live)) * 100.0, 1)

    # Count big-pad (+45..+100), small-pad (+8..+30), and total Boost Spent Per Minute (BPM)!
    big_pads = 0
    small_pads = 0
    total_boost_spent = 0.0
    last_pad_ts = -99.0
    for prev, curr in zip(live, live[1:], strict=False):
        if curr.timestamp_sec - prev.timestamp_sec > step_sec * 3.0:
            continue  # skip across goal replay gaps
        delta = curr.boost_pct - prev.boost_pct
        if delta < -1.0:
            total_boost_spent += abs(delta)
        elif delta >= 45.0 and (curr.timestamp_sec - last_pad_ts) >= 0.8:
            big_pads += 1
            last_pad_ts = curr.timestamp_sec
        elif 8.0 <= delta < 35.0 and (curr.timestamp_sec - last_pad_ts) >= 0.6:
            small_pads += 1
            last_pad_ts = curr.timestamp_sec

    bpm = round((total_boost_spent / max(live_sec, 1.0)) * 60.0, 1)
    total_pads = big_pads + small_pads
    small_pad_ratio = round(small_pads / total_pads, 2) if total_pads > 0 else 0.0

    # Average recovery time from boost-starved (<= 15) back to playable (>= 35)
    recovery_times: list[float] = []
    starve_start: float | None = None
    for m in live:
        if m.boost_pct <= 15.0 and starve_start is None:
            starve_start = m.timestamp_sec
        elif m.boost_pct >= 35.0 and starve_start is not None:
            dt = m.timestamp_sec - starve_start
            if 0.5 <= dt <= 25.0:
                recovery_times.append(dt)
            starve_start = None
    avg_recovery = round(float(np.mean(recovery_times)), 2) if recovery_times else None

    # Build grouped, clickable MistakeItems from contiguous bad stretches & panic burnouts
    mistakes: list[MistakeItem] = []

    # 1. Contiguous Boost Starvation stretches (<= 12 boost for >= 3.0s)
    run: list[FrameMetrics] = []
    for m in live:
        if m.boost_pct <= 12.0:
            run.append(m)
        else:
            if len(run) * step_sec >= 3.0:
                dur = round(len(run) * step_sec, 1)
                mistakes.append(
                    MistakeItem(
                        category="Boost Starvation",
                        severity="high" if dur >= 5.0 else "medium",
                        timestamp_sec=max(0.0, round(run[0].timestamp_sec - 0.5, 2)),
                        original_video_sec=max(0.0, round(run[0].original_video_sec - 0.5, 2)),
                        duration_sec=dur,
                        summary=f"Stayed on 0-12 boost for {dur:.1f}s without collecting pads.",
                        how_to_fix=(
                            "Route through midfield 12-boost small pads while rotating so you "
                            "always have 24-36 boost for emergency saves."
                        ),
                    )
                )
            run = []
    if len(run) * step_sec >= 3.0:
        dur = round(len(run) * step_sec, 1)
        mistakes.append(
            MistakeItem(
                category="Boost Starvation",
                severity="high" if dur >= 5.0 else "medium",
                timestamp_sec=max(0.0, round(run[0].timestamp_sec - 0.5, 2)),
                original_video_sec=max(0.0, round(run[0].original_video_sec - 0.5, 2)),
                duration_sec=dur,
                summary=f"Stayed on 0-12 boost for {dur:.1f}s without collecting pads.",
                how_to_fix=(
                    "Route through midfield 12-boost small pads while rotating so you "
                    "always have 24-36 boost for emergency saves."
                ),
            )
        )

    # 2. Panic Burnouts (dropping >= 55 boost straight down to <= 12 within <= 3.5s)
    burnout_events = [e for e in events if e.event_type == EventType.PANIC_BURNOUT]
    for be in burnout_events:
        mistakes.append(
            MistakeItem(
                category="Panic Boost Burnout",
                severity="high",
                timestamp_sec=max(0.0, round(be.timestamp_sec - 1.0, 2)),
                original_video_sec=max(0.0, round(be.original_video_sec - 1.0, 2)),
                duration_sec=be.duration_sec,
                summary=be.visual_evidence,
                how_to_fix=(
                    "Stop holding boost once you hit supersonic — flip for momentum and save "
                    "25-35 boost for recovery."
                ),
            )
        )

    # 3. Over-Capped Passive Stretches (>= 96 boost for >= 6.0s)
    cap_run: list[FrameMetrics] = []
    for m in live:
        if m.boost_pct >= 96.0:
            cap_run.append(m)
        else:
            if len(cap_run) * step_sec >= 6.0:
                dur = round(len(cap_run) * step_sec, 1)
                mistakes.append(
                    MistakeItem(
                        category="Over-Capped / Passive Play",
                        severity="medium",
                        timestamp_sec=max(0.0, round(cap_run[0].timestamp_sec - 0.5, 2)),
                        original_video_sec=max(0.0, round(cap_run[0].original_video_sec - 0.5, 2)),
                        duration_sec=dur,
                        summary=f"Sat at 96-100 boost for {dur:.1f}s (wasting pad pickups).",
                        how_to_fix=(
                            "Use boost proactively to stay close to the play and maintain pressure "
                            "instead of camping at 100."
                        ),
                    )
                )
            cap_run = []
    if len(cap_run) * step_sec >= 6.0:
        dur = round(len(cap_run) * step_sec, 1)
        mistakes.append(
            MistakeItem(
                category="Over-Capped / Passive Play",
                severity="medium",
                timestamp_sec=max(0.0, round(cap_run[0].timestamp_sec - 0.5, 2)),
                original_video_sec=max(0.0, round(cap_run[0].original_video_sec - 0.5, 2)),
                duration_sec=dur,
                summary=f"Sat at 96-100 boost for {dur:.1f}s (wasting pad pickups).",
                how_to_fix=(
                    "Use boost proactively to stay close to the play and maintain pressure "
                    "instead of camping at 100."
                ),
            )
        )

    mistakes.sort(key=lambda item: (0 if item.severity == "high" else 1, item.timestamp_sec))
    mistakes = sorted(mistakes[:8], key=lambda item: item.timestamp_sec)

    # Overall Boost Management Score (0-100): rewards balanced 45-68 avg boost, low starvation,
    # low over-capping, small-pad pathing, and zero panic burnouts.
    avg_penalty = max(0.0, abs(avg_boost - 55.0) - 12.0) * 0.9
    starve_penalty = starved_pct * 0.85
    cap_penalty = max(0.0, over_capped_pct - 10.0) * 0.6
    burnout_penalty = len(burnout_events) * 7.0
    pad_bonus = min(10.0, small_pads * 2.0)
    raw_score = 92.0 - avg_penalty - starve_penalty - cap_penalty - burnout_penalty + pad_bonus
    overall = max(10.0, min(100.0, round(raw_score, 1)))

    drills: list[str] = []
    if starved_pct >= 20.0:
        drills.append(
            f"Boost Starvation ({starved_pct:.0f}% of clip at <=12 boost): Practice small-pad "
            "rotation lanes in Free Play so you pick up 2-3 small pads every rotation."
        )
    if burnout_events:
        drills.append(
            f"Panic Burnouts ({len(burnout_events)}x): Release boost immediately upon hitting "
            "supersonic and use front-flips/wavedashes to preserve 30+ boost for defense."
        )
    if over_capped_pct >= 20.0:
        drills.append(
            f"Over-Capped ({over_capped_pct:.0f}% at 96-100 boost): Push closer to the play "
            "as 2nd man — sitting at 100 boost means you're conceding midfield pressure."
        )
    if small_pads == 0 and big_pads >= 2:
        drills.append(
            "Small-Pad Pathing: You relied exclusively on 100-boost corner pads; grabbing small "
            "pads keeps you in the play without rotating all the way to the corners."
        )
    if not drills:
        drills.append(
            f"Strong boost discipline (avg {avg_boost:.0f}, only {starved_pct:.0f}% starved) — "
            "review the key moment filmstrips to fine-tune challenge timing."
        )

    goal_count = sum(1 for e in events if e.event_type == EventType.GOAL_EXPLOSION)
    avg_speed = round(float(np.mean([m.field_motion for m in live])), 2)

    return BoostScorecard(
        overall_boost_score=overall,
        avg_boost=avg_boost,
        boost_spent_per_min=bpm,
        starved_pct=starved_pct,
        starved_seconds=starved_sec,
        over_capped_pct=over_capped_pct,
        over_capped_seconds=over_capped_sec,
        big_pad_pickups=big_pads,
        small_pad_pickups=small_pads,
        small_pad_ratio=small_pad_ratio,
        panic_burnouts=len(burnout_events),
        avg_recovery_sec=avg_recovery,
        goal_explosions=goal_count,
        avg_field_speed=avg_speed,
        live_hud_seconds=live_sec,
        mistakes=mistakes,
        improvement_drills=drills,
    )


def build_moment_filmstrip(
    frames_bgr: list[np.ndarray],
    timestamps: list[float],
    boosts: list[float],
    *,
    event_idx: int,
    event_label: str,
    fps: float,
    output_path: Path,
) -> Path:
    """Stitch a 3-frame Rocket League strip: BEFORE (-1.5s) | MOMENT (0.0s) | AFTER (+1.0s)."""
    offsets_sec = (-1.5, 0.0, 1.0)
    stage_names = ("1. BEFORE (-1.5s)", "2. MOMENT (0.0s)", "3. AFTER (+1.0s)")
    tiles: list[np.ndarray] = []
    tile_w, tile_h = 426, 240

    for offset, stage_name in zip(offsets_sec, stage_names, strict=True):
        idx = max(0, min(len(frames_bgr) - 1, event_idx + int(round(offset * fps))))
        resized = cv2.resize(frames_bgr[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (tile_w, 26), (15, 23, 42), -1)
        ts_str = format_timestamp(timestamps[idx])
        cv2.putText(
            resized,
            f"{stage_name} | {ts_str} | Boost {boosts[idx]:.0f}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (251, 191, 36),
            1,
            cv2.LINE_AA,
        )
        tiles.append(resized)

    strip = np.hstack(tiles)
    header = np.full((28, strip.shape[1], 3), (2, 6, 23), dtype=np.uint8)
    cv2.putText(
        header,
        f"THE-EYE ROCKET LEAGUE | {event_label.upper()}",
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    full = np.vstack([header, strip])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), full)
    return output_path
