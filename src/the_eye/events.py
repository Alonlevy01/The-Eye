"""Rocket League visual + audio event detection, filmstrips, and BoostScorecard builder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.mechanics import build_moment_filmstrip, compute_boost_scorecard
from the_eye.schemas import BoostScorecard, ClipSummary, DetectedEvent, EventType
from the_eye.vision import SampledFrame, annotate_frame


def detect_fused_events(
    sampled_frames: list[SampledFrame],
    *,
    fps: float = 2.0,
    keyframes_dir: Path | None = None,
    max_keyframes: int = 12,
) -> tuple[list[DetectedEvent], ClipSummary, BoostScorecard]:
    """Detect Rocket League boost & goal events, export filmstrips, and compute BoostScorecard."""
    if not sampled_frames:
        empty_summary = ClipSummary(
            analyzed_duration_sec=0.0,
            total_frames=0,
            live_hud_frames=0,
            total_events=0,
            audio_spike_count=0,
            avg_field_motion=0.0,
        )
        return [], empty_summary, BoostScorecard()

    step_sec = 1.0 / max(fps, 0.5)
    timeline = [sf.metrics for sf in sampled_frames]
    frames_bgr = [sf.bgr for sf in sampled_frames]
    timestamps = [m.timestamp_sec for m in timeline]
    boosts = [m.boost_pct for m in timeline]

    events: list[DetectedEvent] = []
    last_event_ts: dict[EventType, float] = {}
    cooldowns: dict[EventType, float] = {
        EventType.GOAL_EXPLOSION: 8.0,
        EventType.PANIC_BURNOUT: 5.0,
        EventType.BOOST_STARVED: 6.0,
        EventType.BIG_PAD_PICKUP: 4.0,
        EventType.OVER_CAPPED: 8.0,
    }

    def _can_emit(etype: EventType, ts: float) -> bool:
        prev = last_event_ts.get(etype)
        return prev is None or (ts - prev) >= cooldowns.get(etype, 5.0)

    if keyframes_dir is not None:
        keyframes_dir.mkdir(parents=True, exist_ok=True)

    saved_kf = 0
    lookback_burn = max(2, int(round(3.5 * fps)))
    starve_run_frames = 0
    cap_run_frames = 0

    for idx, sf in enumerate(sampled_frames):
        m = sf.metrics
        etype: EventType | None = None
        confidence = 0.0
        visual_ev = ""
        audio_ev = (
            f"Audio peak detected (RMS={m.audio_rms:.3f}, onset={m.audio_onset:.2f})."
            if m.audio_spike
            else f"Ambient game audio (RMS={m.audio_rms:.3f})."
        )

        # Priority 1: Goal explosion (massive brightness/saturation flash + audio burst)
        if (
            m.brightness >= 185.0
            and m.audio_spike
            and _can_emit(EventType.GOAL_EXPLOSION, m.timestamp_sec)
        ):
            etype = EventType.GOAL_EXPLOSION
            confidence = 0.90
            visual_ev = (
                f"Goal explosion flash detected (brightness={m.brightness:.0f}/255, "
                f"motion={m.field_motion:.1f})."
            )

        if m.hud_visible:
            # Track consecutive starved / over-capped frames
            starve_run_frames = starve_run_frames + 1 if m.boost_pct <= 12.0 else 0
            cap_run_frames = cap_run_frames + 1 if m.boost_pct >= 96.0 else 0

            # Priority 2: Panic Boost Burnout (dropped >= 55 boost straight to <= 12 within 3.5s)
            if etype is None and m.boost_pct <= 12.0 and _can_emit(
                EventType.PANIC_BURNOUT, m.timestamp_sec
            ):
                window = [
                    timeline[j]
                    for j in range(max(0, idx - lookback_burn), idx)
                    if timeline[j].hud_visible
                ]
                if window:
                    peak_before = max(w.boost_pct for w in window)
                    if peak_before - m.boost_pct >= 55.0:
                        etype = EventType.PANIC_BURNOUT
                        confidence = 0.94
                        visual_ev = (
                            f"Burned {peak_before:.0f} -> {m.boost_pct:.0f} boost in <=3.5s "
                            "(rapid depletion to near-zero)."
                        )

            # Priority 3: Big-Pad Pickup (+45 or more boost jump)
            if (
                etype is None
                and m.boost_delta >= 45.0
                and _can_emit(EventType.BIG_PAD_PICKUP, m.timestamp_sec)
            ):
                etype = EventType.BIG_PAD_PICKUP
                confidence = 0.96
                visual_ev = (
                    f"100-boost corner/mid pad pickup (+{m.boost_delta:.0f} -> "
                    f"{m.boost_pct:.0f}/100)."
                )

            # Priority 4: Sustained Boost Starvation (>= 3.0s at <= 12 boost)
            if (
                etype is None
                and starve_run_frames == max(2, int(round(3.0 * fps)))
                and _can_emit(EventType.BOOST_STARVED, m.timestamp_sec)
            ):
                etype = EventType.BOOST_STARVED
                confidence = 0.95
                visual_ev = (
                    f"Boost-starved (<=12 boost) for >=3.0s (current reading: "
                    f"{m.boost_pct:.0f}/100)."
                )

            # Priority 5: Over-Capped Passive Play (>= 6.0s at >= 96 boost)
            if (
                etype is None
                and cap_run_frames == max(3, int(round(6.0 * fps)))
                and _can_emit(EventType.OVER_CAPPED, m.timestamp_sec)
            ):
                etype = EventType.OVER_CAPPED
                confidence = 0.92
                visual_ev = (
                    f"Sat at {m.boost_pct:.0f}/100 boost for >=6.0s without spending boost."
                )
        else:
            starve_run_frames = 0
            cap_run_frames = 0

        if etype is not None:
            last_event_ts[etype] = m.timestamp_sec
            event_id = f"E{len(events) + 1}"
            kf_str: str | None = None
            strip_str: str | None = None

            if keyframes_dir is not None and saved_kf < max_keyframes:
                annotated = annotate_frame(sf.bgr, m, event_label=f"{event_id} {etype.value}")
                kf_file = keyframes_dir / f"{event_id}_{etype.value}_{m.timestamp_sec:.1f}s.png"
                cv2.imwrite(str(kf_file), annotated)
                kf_str = str(kf_file)

                strip_file = keyframes_dir / f"{event_id}_filmstrip_{m.timestamp_sec:.1f}s.png"
                build_moment_filmstrip(
                    frames_bgr,
                    timestamps,
                    boosts,
                    event_idx=idx,
                    event_label=f"{event_id} {etype.value}",
                    fps=fps,
                    output_path=strip_file,
                )
                strip_str = str(strip_file)
                saved_kf += 1

            events.append(
                DetectedEvent(
                    event_id=event_id,
                    event_type=etype,
                    timestamp_sec=m.timestamp_sec,
                    original_video_sec=m.original_video_sec,
                    duration_sec=round(step_sec, 2),
                    boost_at_event=m.boost_pct,
                    confidence=round(confidence, 2),
                    visual_evidence=visual_ev,
                    audio_evidence=audio_ev,
                    keyframe_path=kf_str,
                    filmstrip_path=strip_str,
                )
            )

    total_duration = round(len(sampled_frames) * step_sec, 2)
    live_hud_frames = sum(1 for m in timeline if m.hud_visible)
    audio_spikes = sum(1 for m in timeline if m.audio_spike)
    motions = [m.field_motion for m in timeline]

    summary = ClipSummary(
        analyzed_duration_sec=total_duration,
        total_frames=len(sampled_frames),
        live_hud_frames=live_hud_frames,
        total_events=len(events),
        audio_spike_count=audio_spikes,
        avg_field_motion=round(float(np.mean(motions)), 2),
    )
    scorecard = compute_boost_scorecard(timeline, events, fps=fps)
    return events, summary, scorecard
