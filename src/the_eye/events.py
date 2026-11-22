"""Visual + audio event fusion, pre-duel filmstrips, annotated keyframes, and CS2 scorecard."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.mechanics import build_pre_duel_filmstrip, compute_mechanics_scorecard
from the_eye.schemas import ClipSummary, DetectedEvent, EventType, MechanicsScorecard
from the_eye.vision import SampledFrame, annotate_frame


def detect_fused_events(
    sampled_frames: list[SampledFrame],
    *,
    fps: float = 4.0,
    keyframes_dir: Path | None = None,
    max_keyframes: int = 12,
    weapon: str = "auto",
) -> tuple[list[DetectedEvent], ClipSummary, MechanicsScorecard]:
    """Fuse CS2 visual + audio signals into events, filmstrips, summary, and mechanics scorecard."""
    if not sampled_frames:
        empty_summary = ClipSummary(
            analyzed_duration_sec=0.0,
            total_frames=0,
            total_events=0,
            engagements_per_min=0.0,
            flash_count=0,
            flashed_seconds=0.0,
            killfeed_events=0,
            scoped_seconds=0.0,
            scoped_ratio=0.0,
            damage_events=0,
            combat_bursts=0,
            audio_spike_count=0,
            avg_center_motion=0.0,
        )
        return [], empty_summary, MechanicsScorecard()

    frame_step_sec = 1.0 / max(fps, 0.5)
    timeline = [sf.metrics for sf in sampled_frames]
    frames_bgr = [sf.bgr for sf in sampled_frames]
    timestamps = [m.original_video_sec for m in timeline]

    motions = np.array([m.center_motion for m in timeline], dtype=np.float64)
    motion_thresh = max(18.0, float(np.mean(motions) + 1.2 * np.std(motions)))

    kf_deltas = np.array([m.killfeed_delta for m in timeline], dtype=np.float64)
    kf_thresh = max(8.0, float(np.mean(kf_deltas) + 1.5 * np.std(kf_deltas)))

    events: list[DetectedEvent] = []
    last_event_ts: dict[EventType, float] = {}
    cooldown_map: dict[EventType, float] = {
        EventType.FLASHBANG: 3.0,
        EventType.KILLFEED_EVENT: 3.5,
        EventType.DAMAGE_TAKEN: 3.0,
        EventType.SCOPED_DUEL: 4.0,
        EventType.COMBAT_BURST: 3.5,
    }

    def _can_emit(etype: EventType, ts: float) -> bool:
        prev = last_event_ts.get(etype)
        return prev is None or (ts - prev) >= cooldown_map.get(etype, 3.0)

    if keyframes_dir is not None:
        keyframes_dir.mkdir(parents=True, exist_ok=True)

    saved_kf_count = 0

    for idx, sf in enumerate(sampled_frames):
        m = sf.metrics
        etype: EventType | None = None
        confidence = 0.0
        visual_ev = ""
        audio_ev = (
            f"Audio spike detected (RMS={m.audio_rms:.3f}, onset={m.audio_onset:.2f})."
            if m.audio_spike
            else f"Ambient audio (RMS={m.audio_rms:.3f})."
        )

        # Priority 1: Flashbang whiteout
        if m.is_flashed and _can_emit(EventType.FLASHBANG, m.timestamp_sec):
            etype = EventType.FLASHBANG
            confidence = min(0.99, 0.75 + (m.brightness - 215.0) / 160.0)
            visual_ev = (
                f"Screen whiteout detected (brightness={m.brightness:.1f}/255, "
                f"saturation={m.saturation:.1f}/255)."
            )

        # Priority 2: Player-involved Killfeed event
        # Red border = player kill/death; otherwise require local combat audio/motion so
        # off-site kills while dead or on the other bombsite are filtered out.
        elif (
            not m.is_flashed
            and (
                m.killfeed_red_ratio >= 0.008
                or (
                    m.killfeed_delta >= kf_thresh
                    and (m.audio_spike or m.center_motion >= motion_thresh * 0.75)
                )
            )
            and _can_emit(EventType.KILLFEED_EVENT, m.timestamp_sec)
        ):
            etype = EventType.KILLFEED_EVENT
            confidence = 0.88 if m.killfeed_red_ratio >= 0.008 else 0.76
            if m.audio_spike:
                confidence = min(0.97, confidence + 0.07)
            visual_ev = (
                f"Top-right Killfeed ROI active (red_ratio={m.killfeed_red_ratio:.4f}, "
                f"delta={m.killfeed_delta:.1f}, flow={m.flow_mag:.1f})."
            )

        # Priority 3: Damage taken (CS2 directional red damage arc)
        elif m.is_damaged and _can_emit(EventType.DAMAGE_TAKEN, m.timestamp_sec):
            etype = EventType.DAMAGE_TAKEN
            confidence = 0.87 if m.audio_spike else 0.78
            visual_ev = (
                f"Directional red damage arc detected (damage_red={m.damage_red_ratio:.4f}, "
                f"flow={m.flow_mag:.1f})."
            )

        # Priority 4: Scoped duel (AWP/SSG08 dark corner vignette)
        elif m.is_scoped and _can_emit(EventType.SCOPED_DUEL, m.timestamp_sec):
            etype = EventType.SCOPED_DUEL
            confidence = 0.90 if m.audio_spike else 0.82
            visual_ev = "Sniper scope vignette detected (4 dark outer corners + bright center)."

        # Priority 5: Combat burst (high center-ROI motion fused with audio gunfire burst)
        elif (
            not m.is_flashed
            and m.center_motion >= motion_thresh
            and m.audio_spike
            and _can_emit(EventType.COMBAT_BURST, m.timestamp_sec)
        ):
            etype = EventType.COMBAT_BURST
            confidence = 0.84
            visual_ev = (
                f"High center-crosshair motion ({m.center_motion:.1f} >= {motion_thresh:.1f}, "
                f"flow={m.flow_mag:.1f}) coinciding with gunfire/explosion audio spike."
            )

        if etype is not None:
            last_event_ts[etype] = m.timestamp_sec
            event_id = f"E{len(events) + 1}"
            kf_path_str: str | None = None
            filmstrip_str: str | None = None

            if keyframes_dir is not None and saved_kf_count < max_keyframes:
                annotated = annotate_frame(sf.bgr, m, event_label=f"{event_id} {etype.value}")
                kf_file = keyframes_dir / f"{event_id}_{etype.value}_{m.timestamp_sec:.1f}s.png"
                cv2.imwrite(str(kf_file), annotated)
                kf_path_str = str(kf_file)

                # Build 3-frame Approach -> Contact -> Outcome filmstrip for duel diagnosis
                strip_file = keyframes_dir / f"{event_id}_filmstrip_{m.timestamp_sec:.1f}s.png"
                build_pre_duel_filmstrip(
                    frames_bgr,
                    timestamps,
                    event_idx=idx,
                    event_label=f"{event_id} {etype.value}",
                    fps=fps,
                    output_path=strip_file,
                )
                filmstrip_str = str(strip_file)
                saved_kf_count += 1

            events.append(
                DetectedEvent(
                    event_id=event_id,
                    event_type=etype,
                    timestamp_sec=m.timestamp_sec,
                    original_video_sec=m.original_video_sec,
                    duration_sec=round(frame_step_sec, 2),
                    confidence=round(confidence, 2),
                    visual_evidence=visual_ev,
                    audio_evidence=audio_ev,
                    keyframe_path=kf_path_str,
                    filmstrip_path=filmstrip_str,
                )
            )

    total_duration = round(len(sampled_frames) * frame_step_sec, 2)
    flashed_frames = sum(1 for m in timeline if m.is_flashed)
    scoped_frames = sum(1 for m in timeline if m.is_scoped)
    audio_spikes = sum(1 for m in timeline if m.audio_spike)

    flash_count = sum(1 for e in events if e.event_type == EventType.FLASHBANG)
    killfeed_count = sum(1 for e in events if e.event_type == EventType.KILLFEED_EVENT)
    damage_count = sum(1 for e in events if e.event_type == EventType.DAMAGE_TAKEN)
    combat_count = sum(1 for e in events if e.event_type == EventType.COMBAT_BURST)
    engagements_per_min = (
        round((len(events) / total_duration) * 60.0, 2) if total_duration > 0 else 0.0
    )

    summary = ClipSummary(
        analyzed_duration_sec=total_duration,
        total_frames=len(sampled_frames),
        total_events=len(events),
        engagements_per_min=engagements_per_min,
        flash_count=flash_count,
        flashed_seconds=round(flashed_frames * frame_step_sec, 2),
        killfeed_events=killfeed_count,
        scoped_seconds=round(scoped_frames * frame_step_sec, 2),
        scoped_ratio=round(scoped_frames / max(1, len(sampled_frames)), 3),
        damage_events=damage_count,
        combat_bursts=combat_count,
        audio_spike_count=audio_spikes,
        avg_center_motion=round(float(np.mean(motions)), 2),
    )
    mechanics = compute_mechanics_scorecard(timeline, events, fps=fps, weapon=weapon)
    return events, summary, mechanics
