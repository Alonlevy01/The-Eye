"""CS2 mechanics analysis: optical-flow crosshair discipline, counter-strafing, reaction time,
team trade/utility timing, and 3-frame pre-duel filmstrip stitching.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.schemas import (
    DetectedEvent,
    EventType,
    FrameMetrics,
    MechanicsScorecard,
    format_timestamp,
)


def compute_optical_flow_metrics(
    prev_small_gray: np.ndarray | None,
    curr_small_gray: np.ndarray,
) -> tuple[float, float, float, float]:
    """Compute Farneback optical flow on a downscaled viewport (dx, dy, magnitude, v_jitter)."""
    if prev_small_gray is None or prev_small_gray.shape != curr_small_gray.shape:
        return 0.0, 0.0, 0.0, 0.0

    flow = cv2.calcOpticalFlowFarneback(
        prev_small_gray,
        curr_small_gray,
        None,
        pyr_scale=0.5,
        levels=2,
        winsize=11,
        iterations=2,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )
    dx = float(np.median(flow[:, :, 0]))
    dy = float(np.median(flow[:, :, 1]))
    mag = float(np.hypot(dx, dy))
    denom = abs(dx) + abs(dy)
    vertical_jitter = abs(dy) / denom if denom > 0.25 else 0.0
    return round(dx, 3), round(dy, 3), round(mag, 3), round(vertical_jitter, 3)


def compute_mechanics_scorecard(
    timeline: list[FrameMetrics],
    events: list[DetectedEvent],
    *,
    fps: float = 4.0,
) -> MechanicsScorecard:
    """Derive CS2-specific mechanical and team-coordination scores from frame + audio telemetry."""
    if not timeline:
        return MechanicsScorecard()

    # 1. Crosshair Discipline (head-level horizontal stability vs vertical bounce)
    active_flows = [m for m in timeline if not m.is_flashed and m.flow_mag >= 0.4]
    if active_flows:
        avg_v_jitter = float(np.mean([m.vertical_jitter for m in active_flows]))
        # Ideal horizontal CS2 clearing has v_jitter ~0.10-0.25; frantic vertical jitter > 0.55
        crosshair_score = max(15.0, min(100.0, 100.0 - max(0.0, avg_v_jitter - 0.12) * 145.0))
    else:
        avg_v_jitter = 0.0
        crosshair_score = 75.0

    # 2. Counter-Strafe / Movement-to-Shooting Coordination
    # Track both stabilized shots AND exact timestamps of uncoordinated moving shots!
    shot_indices = [i for i, m in enumerate(timeline) if m.audio_spike and not m.is_flashed]
    coordinated_shots = 0
    uncoordinated_ts: list[str] = []
    for idx in shot_indices:
        curr_flow = timeline[idx].flow_mag
        prev_flow = timeline[idx - 1].flow_mag if idx > 0 else curr_flow
        # Stabilized at shot (<= 1.8 px/frame) OR decelerated by >= 25% into the shot
        if curr_flow <= 1.8 or (prev_flow > 0.5 and curr_flow <= prev_flow * 0.75):
            coordinated_shots += 1
        else:
            uncoordinated_ts.append(format_timestamp(timeline[idx].original_video_sec))
    if shot_indices:
        coord_ratio = coordinated_shots / len(shot_indices)
        counter_strafe_score = round(coord_ratio * 100.0, 1)
    else:
        coord_ratio = 0.0
        counter_strafe_score = 70.0

    # Track worst vertical-jitter timestamps for timestamped crosshair drills
    worst_jitter_frames = sorted(active_flows, key=lambda m: m.vertical_jitter, reverse=True)[:3]
    worst_jitter_ts = [format_timestamp(m.original_video_sec) for m in worst_jitter_frames]

    # 3. Sound-Cue Awareness & Reaction Time (Audio cue -> Aim flick within 0.15s..1.25s)
    # Filter to non-peak-motion cues so self-recoil bursts don't inflate reaction stats
    flow_mags = np.array([m.flow_mag for m in timeline], dtype=np.float64)
    flick_thresh = max(1.2, float(np.mean(flow_mags) + 0.8 * np.std(flow_mags)))
    max_lookahead_frames = max(2, int(round(1.25 * fps)))
    reaction_times_ms: list[float] = []
    responded_cues = 0
    missed_cue_ts: list[str] = []

    for idx in shot_indices:
        t0 = timeline[idx].timestamp_sec
        found_dt: float | None = None
        for look in range(idx + 1, min(len(timeline), idx + 1 + max_lookahead_frames)):
            if timeline[look].flow_mag >= flick_thresh or timeline[look].center_motion >= 25.0:
                dt = timeline[look].timestamp_sec - t0
                if 0.05 <= dt <= 1.25:
                    found_dt = dt
                    break
        if found_dt is not None:
            responded_cues += 1
            reaction_times_ms.append(found_dt * 1000.0)
        else:
            missed_cue_ts.append(format_timestamp(timeline[idx].original_video_sec))

    if shot_indices:
        response_rate = responded_cues / len(shot_indices)
        avg_rx_ms = round(float(np.mean(reaction_times_ms)), 1) if reaction_times_ms else None
        speed_bonus = 0.0
        if avg_rx_ms is not None:
            speed_bonus = max(0.0, min(25.0, (900.0 - avg_rx_ms) / 22.0))
        sound_score = min(100.0, round(response_rate * 75.0 + speed_bonus, 1))
    else:
        avg_rx_ms = None
        sound_score = 65.0

    # 4. Team Coordination: Kill-Trade Clusters (<= 4.0s, capped to avoid chaotic DM inflation)
    kf_times = [e.timestamp_sec for e in events if e.event_type == EventType.KILLFEED_EVENT]
    trade_windows = sum(
        1 for a, b in zip(kf_times, kf_times[1:], strict=False) if 0.0 < (b - a) <= 4.0
    )

    flash_times = [e.timestamp_sec for e in events if e.event_type == EventType.FLASHBANG]
    combat_or_kf_times = [
        e.timestamp_sec
        for e in events
        if e.event_type in (EventType.KILLFEED_EVENT, EventType.COMBAT_BURST)
    ]
    utility_entry_combos = 0
    for ft in flash_times:
        if any(0.0 < (ct - ft) <= 4.0 for ct in combat_or_kf_times):
            utility_entry_combos += 1

    radar_activity = round(float(np.mean([m.radar_motion for m in timeline])), 2)
    capped_trades = min(3, trade_windows)
    raw_team = (
        40.0
        + capped_trades * 10.0
        + min(2, utility_entry_combos) * 15.0
        + min(15.0, radar_activity)
    )
    team_score = min(100.0, round(raw_team, 1))

    damage_events = sum(1 for e in events if e.event_type == EventType.DAMAGE_TAKEN)

    overall = round(
        0.35 * crosshair_score
        + 0.35 * counter_strafe_score
        + 0.15 * sound_score
        + 0.15 * team_score,
        1,
    )

    # Generate timestamped CS2 improvement drills pointing to exact moments in the video!
    drills: list[str] = []
    if crosshair_score < 72.0:
        ts_ref = f" (check {', '.join(worst_jitter_ts)})" if worst_jitter_ts else ""
        drills.append(
            f"Crosshair Placement{ts_ref} — vertical jitter={avg_v_jitter:.2f}: Keep crosshair "
            "locked at head level on horizontal pre-aim lines instead of dipping vertically."
        )
    if counter_strafe_score < 70.0:
        ts_ref = f" (check {', '.join(uncoordinated_ts[:3])})" if uncoordinated_ts else ""
        drills.append(
            f"Counter-Strafing{ts_ref} — only {coord_ratio * 100:.0f}% stabilized shots: Tap "
            "the opposite A/D key to zero velocity right before firing rifles."
        )
    if sound_score < 65.0 or (avg_rx_ms is not None and avg_rx_ms > 550.0):
        ts_ref = f" (check {', '.join(missed_cue_ts[:3])})" if missed_cue_ts else ""
        drills.append(
            f"Sound-Cue Reaction{ts_ref}: Pre-aim toward audio cues (footsteps, utility pins, "
            "off-angle shots) immediately before swinging."
        )
    if trade_windows == 0 and utility_entry_combos == 0:
        drills.append(
            "Team & Utility Timing: Swing within 1-3s of teammate flashes or contact so fights "
            "convert into immediate kill trades rather than isolated 1v1s."
        )
    if not drills:
        drills.append(
            "Strong mechanical fundamentals in this excerpt — review the 3-frame pre-duel "
            "filmstrips to refine off-angle spacing and utility timing."
        )

    return MechanicsScorecard(
        crosshair_discipline_score=round(crosshair_score, 1),
        avg_vertical_jitter=round(avg_v_jitter, 3),
        counter_strafe_score=round(counter_strafe_score, 1),
        coordinated_shot_ratio=round(coord_ratio, 3),
        sound_awareness_score=round(sound_score, 1),
        avg_reaction_time_ms=avg_rx_ms,
        team_coordination_score=round(team_score, 1),
        trade_window_count=trade_windows,
        utility_entry_combos=utility_entry_combos,
        radar_rotation_activity=radar_activity,
        damage_events=damage_events,
        overall_mechanics_score=overall,
        improvement_drills=drills,
    )


def build_pre_duel_filmstrip(
    frames_bgr: list[np.ndarray],
    timestamps: list[float],
    *,
    event_idx: int,
    event_label: str,
    fps: float,
    output_path: Path,
) -> Path:
    """Stitch a 3-frame CS2 duel strip: Approach (-1.5s) -> Contact (-0.5s) -> Outcome (0s)."""
    offsets_sec = (-1.5, -0.5, 0.0)
    stage_names = ("1. APPROACH (-1.5s)", "2. CONTACT (-0.5s)", "3. OUTCOME (0.0s)")
    tiles: list[np.ndarray] = []
    tile_w, tile_h = 426, 240

    for offset, stage_name in zip(offsets_sec, stage_names, strict=True):
        idx = max(0, min(len(frames_bgr) - 1, event_idx + int(round(offset * fps))))
        resized = cv2.resize(frames_bgr[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (tile_w, 26), (15, 23, 42), -1)
        ts_str = format_timestamp(timestamps[idx])
        cv2.putText(
            resized,
            f"{stage_name} | {ts_str}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (56, 189, 248),
            1,
            cv2.LINE_AA,
        )
        tiles.append(resized)

    strip = np.hstack(tiles)
    # Add compact header banner identifying the duel event
    header = np.full((28, strip.shape[1], 3), (2, 6, 23), dtype=np.uint8)
    cv2.putText(
        header,
        f"THE-EYE PRE-DUEL SEQUENCE | {event_label.upper()}",
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
