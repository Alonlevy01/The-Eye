"""Zero-network unit tests for VLM payload/parser, HTML reports, and CLI compare command."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from typer.testing import CliRunner

from the_eye.cli import app
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import AnalysisReport, ClipSummary, DetectedEvent, EventType, FrameMetrics
from the_eye.vlm import build_vlm_messages, parse_vlm_json


def _sample_report(tmp_path: Path, title: str, events_per_min: float) -> AnalysisReport:
    kf = tmp_path / f"{title.replace(' ', '_')}_kf.png"
    cv2.imwrite(str(kf), np.full((180, 320, 3), 90, dtype=np.uint8))

    return AnalysisReport(
        title=title,
        source="https://www.youtube.com/watch?v=example",
        is_url=True,
        excerpt_start_sec=60.0,
        excerpt_duration_sec=30.0,
        fps=2.0,
        summary=ClipSummary(
            analyzed_duration_sec=30.0,
            total_frames=60,
            total_events=2,
            engagements_per_min=events_per_min,
            flash_count=1,
            flashed_seconds=1.5,
            killfeed_events=1,
            scoped_seconds=2.0,
            scoped_ratio=0.067,
            combat_bursts=0,
            audio_spike_count=4,
            avg_center_motion=14.2,
        ),
        events=[
            DetectedEvent(
                event_id="E1",
                event_type=EventType.FLASHBANG,
                timestamp_sec=5.0,
                original_video_sec=65.0,
                confidence=0.92,
                visual_evidence="Screen whiteout detected.",
                audio_evidence="Audio spike detected.",
                keyframe_path=str(kf),
                vlm_observation="Player caught full-blind pushing mid.",
            )
        ],
        timeline=[
            FrameMetrics(
                frame_index=0,
                timestamp_sec=0.0,
                original_video_sec=60.0,
                brightness=100.0,
                saturation=60.0,
                center_motion=12.0,
                audio_rms=0.04,
            ),
            FrameMetrics(
                frame_index=1,
                timestamp_sec=5.0,
                original_video_sec=65.0,
                brightness=245.0,
                saturation=5.0,
                is_flashed=True,
                center_motion=28.0,
                audio_rms=0.22,
                audio_spike=True,
            ),
        ],
        vlm_model="ollama:qwen3.5:9b",
        vlm_tactical_summary="Tighten flash turns and hold tighter angles.",
    )


def test_vlm_message_builder_and_json_parser(tmp_path: Path) -> None:
    rep = _sample_report(tmp_path, "Pro Major POV", 4.0)
    msgs = build_vlm_messages(title=rep.title, summary=rep.summary, events=rep.events, max_frames=2)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    user_content = msgs[1]["content"]
    assert any(p.get("type") == "image_url" for p in user_content)

    raw_json = (
        '```json\n{"event_observations": {"E1": "Full blind"}, '
        '"tactical_summary": "Turn away."}\n```'
    )
    parsed = parse_vlm_json(raw_json)
    assert parsed["event_observations"]["E1"] == "Full blind"
    assert parsed["tactical_summary"] == "Turn away."


def test_html_report_and_cli_compare(tmp_path: Path) -> None:
    rep_a = _sample_report(tmp_path, "Amateur Pug POV", 6.0)
    rep_b = _sample_report(tmp_path, "Pro Major POV", 10.0)

    html_a = write_html_report(rep_a, tmp_path / "report_a.html")
    content_a = html_a.read_text(encoding="utf-8")
    assert "data:image/png;base64," in content_a
    assert "timeline-svg" in content_a
    assert "Amateur Pug POV" in content_a

    cmp_html = write_comparison_html(rep_a, rep_b, tmp_path / "compare.html")
    assert "Pro Major POV" in cmp_html.read_text(encoding="utf-8")

    # Test CLI compare end-to-end
    json_a = tmp_path / "a.json"
    json_b = tmp_path / "b.json"
    json_a.write_text(rep_a.model_dump_json(), encoding="utf-8")
    json_b.write_text(rep_b.model_dump_json(), encoding="utf-8")

    runner = CliRunner()
    out_cmp = tmp_path / "cli_compare.html"
    res = runner.invoke(app, ["compare", str(json_a), str(json_b), "--out", str(out_cmp)])
    assert res.exit_code == 0, res.output
    assert out_cmp.exists()
