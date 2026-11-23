"""BTD6 event detection (leaks, tower purchases, wiki killer rounds) & filmstrip export."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.btd6_wiki import KILLER_ROUNDS
from the_eye.mechanics import build_moment_filmstrip, compute_btd6_scorecard
from the_eye.schemas import BTD6Scorecard, ClipSummary, DetectedEvent, EventType
from the_eye.vision import SampledFrame, annotate_frame


def detect_fused_events(
    sampled_frames: list[SampledFrame],
    *,
    fps: float = 2.0,
    keyframes_dir: Path | None = None,
    max_keyframes: int = 12,
    event_prefix: str = "E",
    title: str = "",
) -> tuple[list[DetectedEvent], ClipSummary, BTD6Scorecard]:
    """Detect BTD6 leaks, tower purchases, and wiki killer rounds with prioritized filmstrips."""
    if not sampled_frames:
        empty_summary = ClipSummary(
            analyzed_duration_sec=0.0,
            total_frames=0,
            ocr_valid_frames=0,
            total_events=0,
            audio_spike_count=0,
            avg_map_activity=0.0,
        )
        return [], empty_summary, BTD6Scorecard()

    step_sec = 1.0 / max(fps, 0.5)
    timeline = [sf.metrics for sf in sampled_frames]
    frames_bgr = [sf.bgr for sf in sampled_frames]
    timestamps = [m.timestamp_sec for m in timeline]
    rounds = [m.round_num for m in timeline]
    cashes = [m.cash for m in timeline]

    events: list[DetectedEvent] = []
    event_frame_indices: list[tuple[int, int]] = []  # (frame_idx, spend_amount)
    last_event_ts: dict[EventType, float] = {}
    seen_killer_rounds: set[int] = set()
    cooldowns: dict[EventType, float] = {
        EventType.BLOON_LEAK: 3.0,
        EventType.TOWER_PURCHASE: 2.5,
        EventType.KILLER_ROUND: 1.0,
        EventType.CASH_HOARDING: 12.0,
    }

    def _can_emit(etype: EventType, ts: float) -> bool:
        prev = last_event_ts.get(etype)
        return prev is None or (ts - prev) >= cooldowns.get(etype, 3.0)

    if keyframes_dir is not None:
        keyframes_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: detect all events across the entire game
    for idx, sf in enumerate(sampled_frames):
        m = sf.metrics
        etype: EventType | None = None
        confidence = 0.0
        visual_ev = ""
        spend_amt = 0
        audio_ev = (
            f"Audio spike detected (RMS={m.audio_rms:.3f}, onset={m.audio_onset:.2f})."
            if m.audio_spike
            else f"Game audio (RMS={m.audio_rms:.3f})."
        )

        # Priority 1: Bloon Leak (Lives dropped!)
        if m.lives_delta < 0 and _can_emit(EventType.BLOON_LEAK, m.timestamp_sec):
            etype = EventType.BLOON_LEAK
            confidence = 0.96
            visual_ev = (
                f"Lost {abs(m.lives_delta)} lives on Round {m.round_num or '?'} "
                f"(Lives now {m.lives})."
            )

        # Priority 2: Wiki Killer Round reached (R24, R28, R33, R40, R59, R63, R76, R78, R80, R90+)
        elif (
            m.round_num is not None
            and m.round_num in KILLER_ROUNDS
            and m.round_num not in seen_killer_rounds
            and _can_emit(EventType.KILLER_ROUND, m.timestamp_sec)
        ):
            seen_killer_rounds.add(m.round_num)
            info = KILLER_ROUNDS[m.round_num]
            etype = EventType.KILLER_ROUND
            confidence = 0.98
            c_str = f"${m.cash:,}" if m.cash is not None else "$?"
            visual_ev = f"Entered Round {m.round_num} — {info.name} (Cash: {c_str})."

        # Priority 3: Tower Placement / Upgrade Purchase (Cash dropped >= $150)
        elif m.cash_delta <= -150 and _can_emit(EventType.TOWER_PURCHASE, m.timestamp_sec):
            etype = EventType.TOWER_PURCHASE
            confidence = 0.94
            spend_amt = abs(m.cash_delta)
            t_lbl = f" ({m.selected_tower})" if m.selected_tower else ""
            visual_ev = (
                f"Spent ${spend_amt:,} on Round {m.round_num or '?'}{t_lbl} "
                f"(Remaining cash: ${m.cash:,})."
            )

        if etype is not None:
            last_event_ts[etype] = m.timestamp_sec
            event_id = f"{event_prefix}{len(events) + 1}"
            events.append(
                DetectedEvent(
                    event_id=event_id,
                    event_type=etype,
                    timestamp_sec=m.timestamp_sec,
                    original_video_sec=m.original_video_sec,
                    round_num=m.round_num,
                    cash_at_event=m.cash,
                    lives_at_event=m.lives,
                    tower_name=m.selected_tower,
                    confidence=round(confidence, 2),
                    visual_evidence=visual_ev,
                    audio_evidence=audio_ev,
                )
            )
            event_frame_indices.append((idx, spend_amt))

    # Pass 2: render keyframes & 3-frame filmstrips for the highest-impact moments in this game
    if keyframes_dir is not None and events:
        def _priority_key(pos: int) -> tuple[int, int]:
            ev = events[pos]
            _, spend = event_frame_indices[pos]
            if ev.event_type in (EventType.DEFEAT_OR_END, EventType.BLOON_LEAK):
                return (0, -(ev.round_num or 0))
            if ev.event_type == EventType.KILLER_ROUND:
                return (1, -(ev.round_num or 0))
            return (2, -spend)

        chosen_positions = sorted(
            sorted(range(len(events)), key=_priority_key)[:max_keyframes]
        )
        for pos in chosen_positions:
            ev = events[pos]
            f_idx, _ = event_frame_indices[pos]
            sf = sampled_frames[f_idx]
            annotated = annotate_frame(
                sf.bgr,
                sf.metrics,
                event_label=f"{ev.event_id} {ev.event_type.value}",
            )
            kf_name = f"{ev.event_id}_{ev.event_type.value}_{ev.timestamp_sec:.1f}s.png"
            kf_file = keyframes_dir / kf_name
            cv2.imwrite(str(kf_file), annotated)
            ev.keyframe_path = str(kf_file)

            strip_file = keyframes_dir / f"{ev.event_id}_filmstrip_{ev.timestamp_sec:.1f}s.png"
            build_moment_filmstrip(
                frames_bgr,
                timestamps,
                rounds,
                cashes,
                event_idx=f_idx,
                event_label=f"{ev.event_id} {ev.event_type.value}",
                fps=fps,
                output_path=strip_file,
            )
            ev.filmstrip_path = str(strip_file)

    total_duration = round(len(sampled_frames) * step_sec, 2)
    ocr_valid = sum(1 for m in timeline if m.cash is not None or m.round_num is not None)
    audio_spikes = sum(1 for m in timeline if m.audio_spike)
    acts = [m.map_activity for m in timeline]

    summary = ClipSummary(
        analyzed_duration_sec=total_duration,
        total_frames=len(sampled_frames),
        ocr_valid_frames=ocr_valid,
        total_events=len(events),
        audio_spike_count=audio_spikes,
        avg_map_activity=round(float(np.mean(acts)), 2),
    )
    scorecard = compute_btd6_scorecard(timeline, events, fps=fps, title=title)
    return events, summary, scorecard
