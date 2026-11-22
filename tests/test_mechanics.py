"""Zero-network unit tests for CS2 optical-flow mechanics, counter-strafing, reaction time,
trade windows, and 3-frame pre-duel filmstrips.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.mechanics import (
    build_pre_duel_filmstrip,
    compute_mechanics_scorecard,
    compute_optical_flow_metrics,
)
from the_eye.schemas import DetectedEvent, EventType, FrameMetrics


def test_optical_flow_distinguishes_horizontal_preaim_from_vertical_jitter() -> None:
    # Textured synthetic scene shifted horizontally (clean CS2 corner clear) vs vertically (jitter)
    rng = np.random.default_rng(42)
    base = rng.integers(40, 210, size=(90, 160), dtype=np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)

    horiz_shift = np.roll(base, shift=4, axis=1)
    vert_shift = np.roll(base, shift=4, axis=0)

    dx_h, dy_h, mag_h, vjit_h = compute_optical_flow_metrics(base, horiz_shift)
    dx_v, dy_v, mag_v, vjit_v = compute_optical_flow_metrics(base, vert_shift)

    assert mag_h > 1.0 and abs(dx_h) > abs(dy_h)
    assert vjit_h < 0.25  # clean horizontal pre-aim
    assert mag_v > 1.0 and abs(dy_v) > abs(dx_v)
    assert vjit_v > 0.70  # vertical bounce


def test_mechanics_scorecard_counterstrafe_reaction_and_trades() -> None:
    # Build synthetic 4 fps timeline:
    # - Low vertical jitter (good crosshair discipline)
    # - Audio spike at t=1.0s while moving -> decelerates cleanly at t=1.0s (counter-strafe!)
    # - Aim flick 250ms later at t=1.25s (fast reaction time!)
    timeline = [
        FrameMetrics(
            frame_index=0, timestamp_sec=0.0, original_video_sec=0.0,
            brightness=120, saturation=80, flow_mag=3.0, vertical_jitter=0.15, radar_motion=5.0,
        ),
        FrameMetrics(
            frame_index=1, timestamp_sec=0.25, original_video_sec=0.25,
            brightness=120, saturation=80, flow_mag=3.2, vertical_jitter=0.15, radar_motion=5.0,
        ),
        FrameMetrics(
            frame_index=2, timestamp_sec=0.5, original_video_sec=0.5,
            brightness=120, saturation=80, flow_mag=0.8, vertical_jitter=0.12,
            audio_rms=0.25, audio_spike=True, radar_motion=5.0,
        ),
        FrameMetrics(
            frame_index=3, timestamp_sec=0.75, original_video_sec=0.75,
            brightness=120, saturation=80, flow_mag=4.5, vertical_jitter=0.18, radar_motion=6.0,
        ),
    ]
    events = [
        DetectedEvent(
            event_id="E1", event_type=EventType.FLASHBANG, timestamp_sec=0.25,
            original_video_sec=0.25, confidence=0.9, visual_evidence="flash", audio_evidence="pop",
        ),
        DetectedEvent(
            event_id="E2", event_type=EventType.KILLFEED_EVENT, timestamp_sec=1.5,
            original_video_sec=1.5, confidence=0.9, visual_evidence="kf1", audio_evidence="shot",
        ),
        DetectedEvent(
            event_id="E3", event_type=EventType.KILLFEED_EVENT, timestamp_sec=3.0,
            original_video_sec=3.0, confidence=0.9, visual_evidence="kf2", audio_evidence="shot",
        ),
    ]

    sc = compute_mechanics_scorecard(timeline, events, fps=4.0)
    assert sc.crosshair_discipline_score >= 90.0
    assert sc.counter_strafe_score == 100.0
    assert sc.avg_reaction_time_ms == 250.0
    assert sc.trade_window_count == 1
    assert sc.utility_entry_combos == 1
    assert sc.overall_mechanics_score >= 80.0
    assert len(sc.improvement_drills) >= 1


def test_pre_duel_filmstrip_stitching(tmp_path: Path) -> None:
    frames = [np.full((180, 320, 3), 40 + i * 15, dtype=np.uint8) for i in range(10)]
    timestamps = [i * 0.25 for i in range(10)]
    out_png = tmp_path / "strip.png"
    build_pre_duel_filmstrip(
        frames,
        timestamps,
        event_idx=8,
        event_label="E1 killfeed_event",
        fps=4.0,
        output_path=out_png,
    )
    assert out_png.exists()
    img = cv2.imread(str(out_png))
    assert img is not None
    # 3 horizontal tiles of width 426 -> 1278px wide, 240 + 28 header -> 268px tall
    assert img.shape[1] == 1278
    assert img.shape[0] == 268


def test_spray_control_static_turret_and_flank_awareness() -> None:
    # Static turret player: low flow before & during sustained spray with wild vertical kick
    # plus sudden flank damage after an audio/radar cue with zero horizontal turn
    timeline = [
        FrameMetrics(
            frame_index=0, timestamp_sec=0.0, original_video_sec=10.0,
            brightness=120, saturation=80, flow_dx=0.1, flow_dy=0.0, flow_mag=0.1,
            audio_onset=2.5, radar_motion=8.0,
        ),
        FrameMetrics(
            frame_index=1, timestamp_sec=0.25, original_video_sec=10.25,
            brightness=120, saturation=80, flow_dx=0.0, flow_dy=0.0, flow_mag=0.0,
            radar_motion=8.0,
        ),
        FrameMetrics(
            frame_index=2, timestamp_sec=0.5, original_video_sec=10.5,
            brightness=120, saturation=80, flow_dx=0.1, flow_dy=2.8, flow_mag=0.9,
            is_damaged=True, damage_red_ratio=0.04, audio_rms=0.25, audio_spike=True,
        ),
        FrameMetrics(
            frame_index=3, timestamp_sec=0.75, original_video_sec=10.75,
            brightness=120, saturation=80, flow_dx=-0.4, flow_dy=3.4, flow_mag=0.9,
            audio_rms=0.26, audio_spike=True,
        ),
        FrameMetrics(
            frame_index=4, timestamp_sec=1.0, original_video_sec=11.0,
            brightness=120, saturation=80, flow_dx=0.3, flow_dy=3.1, flow_mag=0.8,
            audio_rms=0.24, audio_spike=True,
        ),
    ]
    events = [
        DetectedEvent(
            event_id="E1", event_type=EventType.DAMAGE_TAKEN, timestamp_sec=0.5,
            original_video_sec=10.5, confidence=0.88, visual_evidence="dmg", audio_evidence="shot",
        )
    ]

    sc = compute_mechanics_scorecard(timeline, events, fps=4.0)
    assert sc.active_counter_strafe_count == 0
    assert sc.stationary_shot_count == 3
    assert sc.counter_strafe_score == 60.0  # Static turret capped at 60, NOT 100!
    assert sc.spray_burst_count == 1
    assert sc.spray_control_score < 40.0  # Wild vertical kick caught!
    assert sc.flank_awareness_misses == 1
    assert any("Static Turret Shooting" in d for d in sc.improvement_drills)
    assert any("Spray Control" in d for d in sc.improvement_drills)
    assert any("Flank & Radar Awareness" in d for d in sc.improvement_drills)
    assert len(sc.mistakes) >= 1

    # Single-shot weapon (AWP) should not be spray-penalized
    sc_awp = compute_mechanics_scorecard(timeline, events, fps=4.0, weapon="awp")
    assert sc_awp.weapon_profile == "awp"
    assert sc_awp.spray_control_score == 85.0


def test_mistakes_cluster_into_single_duel_card() -> None:
    # 5 moving shots within 1.5 seconds should collapse into 1 Duel card saying "5 moving shots"
    timeline = [
        FrameMetrics(
            frame_index=i,
            timestamp_sec=10.0 + i * 0.25,
            original_video_sec=10.0 + i * 0.25,
            brightness=120,
            saturation=80,
            flow_dx=3.5,
            flow_dy=0.2,
            flow_mag=3.5,
            audio_rms=0.25,
            audio_spike=True,
        )
        for i in range(5)
    ]
    sc = compute_mechanics_scorecard(timeline, [], fps=4.0, weapon="ak47")
    assert len(sc.mistakes) == 1
    assert "5 moving shots" in sc.mistakes[0].summary
    assert sc.mistakes[0].timestamp_sec == 9.5  # 0.5s lead-in before duel start



