"""Zero-network unit tests for CS2 OpenCV detectors, librosa audio fusion, and keyframe overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import soundfile as sf

from the_eye.audio import analyze_audio_cues
from the_eye.events import detect_fused_events
from the_eye.schemas import EventType
from the_eye.vision import SampledFrame, analyze_single_frame


def _make_normal_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(60, 140, size=(360, 640, 3), dtype=np.uint8)
    return frame


def _make_flashbang_frame() -> np.ndarray:
    return np.full((360, 640, 3), 245, dtype=np.uint8)


def _make_scoped_awp_frame() -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 180), 120, (130, 150, 160), -1)
    # Crosshair lines inside the scope
    cv2.line(frame, (200, 180), (440, 180), (10, 10, 10), 2)
    cv2.line(frame, (320, 60), (320, 300), (10, 10, 10), 2)
    return frame


def _make_killfeed_frame() -> np.ndarray:
    frame = _make_normal_frame(seed=7)
    # Draw a bright red player kill-highlight box inside the top-right Killfeed ROI
    cv2.rectangle(frame, (460, 25), (620, 60), (20, 20, 235), -1)
    return frame


def test_opencv_cs2_detectors_identify_flash_scope_and_killfeed() -> None:
    normal_m, *_ = analyze_single_frame(
        _make_normal_frame(1),
        frame_index=0,
        timestamp_sec=0.0,
        original_video_sec=60.0,
    )
    assert normal_m.is_flashed is False
    assert normal_m.is_scoped is False
    assert normal_m.killfeed_red_ratio < 0.005

    flash_m, *_ = analyze_single_frame(
        _make_flashbang_frame(),
        frame_index=1,
        timestamp_sec=0.5,
        original_video_sec=60.5,
    )
    assert flash_m.is_flashed is True
    assert flash_m.is_scoped is False
    assert flash_m.brightness >= 240.0

    scope_m, *_ = analyze_single_frame(
        _make_scoped_awp_frame(),
        frame_index=2,
        timestamp_sec=1.0,
        original_video_sec=61.0,
    )
    assert scope_m.is_scoped is True
    assert scope_m.is_flashed is False

    kf_m, *_ = analyze_single_frame(
        _make_killfeed_frame(),
        frame_index=3,
        timestamp_sec=1.5,
        original_video_sec=61.5,
    )
    assert kf_m.killfeed_red_ratio > 0.05


def test_audio_fusion_and_annotated_keyframe_export(tmp_path: Path) -> None:
    # Create a 6-second synthetic timeline at 2 fps (12 frames)
    frames_bgr = [
        _make_normal_frame(0),  # 0.0s normal
        _make_normal_frame(1),  # 0.5s normal
        _make_flashbang_frame(),  # 1.0s flashbang
        _make_normal_frame(2),  # 1.5s normal
        _make_normal_frame(3),  # 2.0s normal
        _make_normal_frame(4),  # 2.5s normal
        _make_scoped_awp_frame(),  # 3.0s scoped AWP
        _make_normal_frame(5),  # 3.5s normal
        _make_normal_frame(6),  # 4.0s normal
        _make_killfeed_frame(),  # 4.5s killfeed red box
        _make_normal_frame(7),  # 5.0s normal
        _make_normal_frame(8),  # 5.5s normal
    ]

    sampled: list[SampledFrame] = []
    prev_c = None
    prev_kf = None
    prev_r = None
    prev_f = None
    for idx, bgr in enumerate(frames_bgr):
        ts = idx * 0.5
        m, prev_c, prev_kf, prev_r, prev_f = analyze_single_frame(
            bgr,
            frame_index=idx,
            timestamp_sec=ts,
            original_video_sec=30.0 + ts,
            prev_center_gray=prev_c,
            prev_killfeed_gray=prev_kf,
            prev_radar_gray=prev_r,
            prev_flow_gray=prev_f,
        )
        sampled.append(SampledFrame(metrics=m, bgr=bgr))

    # Synthesize 6s of 16 kHz audio with a loud gunfire burst at 3.0s
    sr = 16000
    audio = np.zeros(sr * 6, dtype=np.float32) + 0.002
    burst_start, burst_end = int(2.9 * sr), int(3.2 * sr)
    t = np.linspace(0, 0.3, burst_end - burst_start, endpoint=False)
    audio[burst_start:burst_end] = 0.8 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "synthetic_cs2.wav"
    sf.write(str(wav_path), audio, sr)

    enriched = analyze_audio_cues(wav_path, [s.metrics for s in sampled], fps=2.0)
    for sf_item, new_m in zip(sampled, enriched, strict=True):
        sf_item.metrics = new_m

    assert enriched[6].audio_spike is True  # t=3.0s gunfire burst detected!

    kf_dir = tmp_path / "keyframes"
    events, summary, mechanics = detect_fused_events(sampled, fps=2.0, keyframes_dir=kf_dir)
    assert 0.0 <= mechanics.overall_mechanics_score <= 100.0

    event_types = {e.event_type for e in events}
    assert EventType.FLASHBANG in event_types
    assert EventType.SCOPED_DUEL in event_types
    assert EventType.KILLFEED_EVENT in event_types
    assert summary.flash_count >= 1
    assert summary.scoped_seconds >= 0.5
    assert summary.killfeed_events >= 1

    # Verify annotated keyframe PNGs were written to disk
    pngs = list(kf_dir.glob("*.png"))
    assert len(pngs) >= 3
