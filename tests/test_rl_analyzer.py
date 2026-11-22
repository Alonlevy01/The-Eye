"""Zero-network unit tests for Rocket League Boost Gauge reader, scorecard, filmstrips & reports."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from typer.testing import CliRunner

from the_eye.audio import analyze_audio_cues
from the_eye.cli import app
from the_eye.events import detect_fused_events
from the_eye.mechanics import compute_boost_scorecard
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import (
    AnalysisReport,
    ClipSummary,
    DetectedEvent,
    EventType,
    FrameMetrics,
)
from the_eye.vision import (
    GAUGE_CX_NORM,
    GAUGE_CY_NORM,
    GAUGE_R_IN_NORM,
    GAUGE_R_OUT_NORM,
    GAUGE_SPAN_DEG,
    GAUGE_START_DEG,
    SampledFrame,
    analyze_single_frame,
    read_boost_gauge,
)
from the_eye.vlm import parse_vlm_json


def _make_rl_frame(boost_pct: float, *, hud_visible: bool = True) -> np.ndarray:
    """Synthesize a 16:9 Rocket League frame with a calibrated bottom-right Boost Gauge."""
    h, w = 360, 640
    frame = np.full((h, w, 3), (60, 95, 55), dtype=np.uint8)  # pitch green
    if not hud_visible:
        return np.full((h, w, 3), 210, dtype=np.uint8)  # bright replay without dark gauge

    cx = int(round(GAUGE_CX_NORM * w))
    cy = int(round(GAUGE_CY_NORM * h))
    r_bg = int(round(0.080 * h))
    cv2.circle(frame, (cx, cy), r_bg, (20, 22, 26), -1)

    if boost_pct > 0:
        r_mid = int(round(((GAUGE_R_IN_NORM + GAUGE_R_OUT_NORM) / 2.0) * h))
        thickness = max(3, int(round((GAUGE_R_OUT_NORM - GAUGE_R_IN_NORM) * h)))
        end_deg = GAUGE_START_DEG + (boost_pct / 100.0) * GAUGE_SPAN_DEG
        # Glowing orange-yellow in BGR (B=30, G=180, R=250 -> HSV H~20, S~224, V~250)
        cv2.ellipse(
            frame,
            (cx, cy),
            (r_mid, r_mid),
            0,
            GAUGE_START_DEG,
            end_deg,
            (30, 180, 250),
            thickness,
        )
    return frame


def test_boost_gauge_polar_reader_accuracy() -> None:
    for target in (0.0, 25.0, 50.0, 79.0, 100.0):
        hud, read = read_boost_gauge(_make_rl_frame(target))
        assert hud is True
        assert abs(read - target) <= 5.0, f"Expected ~{target}, got {read}"

    hud_replay, read_replay = read_boost_gauge(_make_rl_frame(0.0, hud_visible=False))
    assert hud_replay is False
    assert read_replay == 0.0


def test_end_to_end_events_scorecard_audio_and_filmstrips(tmp_path: Path) -> None:
    # 10-second synthetic sequence at 2 fps (20 frames):
    # - Starts at 80 boost, burns rapidly down to 5 -> PANIC_BURNOUT
    # - Stays <= 10 for 4 seconds -> BOOST_STARVED
    # - Picks up big pad (+90 -> 100) -> BIG_PAD_PICKUP
    boost_seq = [
        80, 75, 55, 30, 10, 5, 5, 5, 5, 5, 5, 5, 5, 100, 98, 98, 70, 60, 50, 50
    ]
    sampled: list[SampledFrame] = []
    prev_field = None
    prev_boost = None
    for idx, b in enumerate(boost_seq):
        bgr = _make_rl_frame(float(b))
        ts = idx * 0.5
        m, prev_field = analyze_single_frame(
            bgr,
            frame_index=idx,
            timestamp_sec=ts,
            original_video_sec=60.0 + ts,
            prev_field_gray=prev_field,
            prev_boost=prev_boost,
        )
        prev_boost = m.boost_pct
        sampled.append(SampledFrame(metrics=m, bgr=bgr))

    # Synthetic 10s WAV
    sr = 16000
    wav_path = tmp_path / "rl.wav"
    sf.write(str(wav_path), np.zeros(sr * 10, dtype=np.float32) + 0.005, sr)
    enriched = analyze_audio_cues(wav_path, [s.metrics for s in sampled], fps=2.0)
    for sf_item, new_m in zip(sampled, enriched, strict=True):
        sf_item.metrics = new_m

    kf_dir = tmp_path / "keyframes"
    events, summary, scorecard = detect_fused_events(sampled, fps=2.0, keyframes_dir=kf_dir)
    etypes = {e.event_type for e in events}
    assert EventType.PANIC_BURNOUT in etypes
    assert EventType.BOOST_STARVED in etypes
    assert EventType.BIG_PAD_PICKUP in etypes
    assert scorecard.big_pad_pickups >= 1
    assert scorecard.panic_burnouts >= 1
    assert scorecard.starved_seconds >= 3.0
    assert len(scorecard.mistakes) >= 1
    assert len(list(kf_dir.glob("*filmstrip*.png"))) >= 2


def test_html_report_comparison_and_cli_compare(tmp_path: Path) -> None:
    tl = [
        FrameMetrics(frame_index=0, timestamp_sec=0.0, original_video_sec=60.0, boost_pct=60.0),
        FrameMetrics(frame_index=1, timestamp_sec=0.5, original_video_sec=60.5, boost_pct=55.0),
    ]
    ev = [
        DetectedEvent(
            event_id="E1",
            event_type=EventType.BIG_PAD_PICKUP,
            timestamp_sec=0.5,
            original_video_sec=60.5,
            boost_at_event=100.0,
            confidence=0.96,
            visual_evidence="100-boost pad pickup.",
            audio_evidence="Ambient.",
        )
    ]
    sc = compute_boost_scorecard(tl, ev, fps=2.0)
    rep_a = AnalysisReport(
        title="SSL 2v2 POV",
        source="https://www.youtube.com/watch?v=5X77jb5DR6I",
        is_url=True,
        excerpt_start_sec=60.0,
        excerpt_duration_sec=120.0,
        fps=2.0,
        summary=ClipSummary(
            analyzed_duration_sec=1.0,
            total_frames=2,
            live_hud_frames=2,
            total_events=1,
            audio_spike_count=0,
            avg_field_motion=10.0,
        ),
        scorecard=sc,
        events=ev,
        timeline=tl,
        local_clip_path="clip.mp4",
    )
    rep_b = rep_a.model_copy(update={"title": "Gold 3 POV"})

    html_out = write_html_report(rep_a, tmp_path / "report.html")
    html_text = html_out.read_text(encoding="utf-8")
    assert "<video id=\"localPlayer\"" in html_text
    assert "Player View (Simple)" in html_text
    assert "Coach View (Advanced)" in html_text

    cmp_out = write_comparison_html(rep_a, rep_b, tmp_path / "compare.html")
    assert "Gold 3 POV" in cmp_out.read_text(encoding="utf-8")

    ja, jb = tmp_path / "a.json", tmp_path / "b.json"
    ja.write_text(rep_a.model_dump_json(), encoding="utf-8")
    jb.write_text(rep_b.model_dump_json(), encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(app, ["compare", str(ja), str(jb), "--out", str(tmp_path / "c.html")])
    assert res.exit_code == 0, res.output

    raw = '{"event_observations": {"E1": "Good pad"}, "tactical_summary": "Nice"}'
    parsed = parse_vlm_json(raw)
    assert parsed["tactical_summary"] == "Nice"
