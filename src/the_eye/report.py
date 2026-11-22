"""Standalone interactive HTML visual report & side-by-side comparison generator."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from the_eye.schemas import AnalysisReport, EventType, format_timestamp

EVENT_COLORS: dict[EventType, str] = {
    EventType.FLASHBANG: "#f59e0b",
    EventType.KILLFEED_EVENT: "#ef4444",
    EventType.SCOPED_DUEL: "#a855f7",
    EventType.COMBAT_BURST: "#38bdf8",
}


def _img_to_data_uri(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _build_svg_timeline(report: AnalysisReport) -> str:
    """Render a clean inline SVG chart of center motion, audio RMS, and detected event markers."""
    width, height = 920, 210
    pad_l, pad_r, pad_t, pad_b = 44, 20, 22, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    timeline = report.timeline
    if not timeline:
        return "<p class='muted'>No frames sampled.</p>"

    max_t = max(timeline[-1].timestamp_sec, 1.0)
    max_motion = max((m.center_motion for m in timeline), default=1.0) or 1.0
    max_rms = max((m.audio_rms for m in timeline), default=0.05) or 0.05

    def _x(t: float) -> float:
        return pad_l + (t / max_t) * plot_w

    def _y(norm: float) -> float:
        clamped = max(0.0, min(1.0, norm))
        return pad_t + (1.0 - clamped) * plot_h

    motion_pts = " ".join(
        f"{_x(m.timestamp_sec):.1f},{_y(m.center_motion / max_motion):.1f}" for m in timeline
    )
    rms_pts = " ".join(
        f"{_x(m.timestamp_sec):.1f},{_y(m.audio_rms / max_rms):.1f}" for m in timeline
    )

    markers = []
    for e in report.events:
        cx = _x(e.timestamp_sec)
        color = EVENT_COLORS.get(e.event_type, "#e2e8f0")
        markers.append(
            f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="4 3" opacity="0.85" />'
            f'<circle cx="{cx:.1f}" cy="{pad_t + 8}" r="5" fill="{color}" />'
            f'<text x="{cx + 6:.1f}" y="{pad_t + 12}" fill="{color}" font-size="11" '
            f'font-family="monospace">{html.escape(e.event_id)}</text>'
        )

    return f"""
    <svg viewBox="0 0 {width} {height}" class="timeline-svg" role="img" aria-label="CS2 Timeline">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0f172a" rx="6" />
      <polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{motion_pts}" />
      <polyline fill="none" stroke="#22c55e" stroke-width="1.8" opacity="0.85" points="{rms_pts}" />
      {"".join(markers)}
      <text x="{pad_l}" y="{height - 10}" fill="#94a3b8" font-size="11">00:00.0</text>
      <text x="{width - 62}" y="{height - 10}" fill="#94a3b8" font-size="11">{format_timestamp(max_t)}</text>
      <text x="{pad_l + 6}" y="14" fill="#38bdf8" font-size="11">■ Center Motion</text>
      <text x="{pad_l + 125}" y="14" fill="#22c55e" font-size="11">■ Audio RMS</text>
    </svg>
    """


def write_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write a self-contained HTML visual report with embedded annotated keyframes."""
    s = report.summary
    cards_html = []
    for e in report.events:
        color = EVENT_COLORS.get(e.event_type, "#cbd5e1")
        uri = _img_to_data_uri(e.keyframe_path)
        img_block = (
            f'<img src="{uri}" alt="{html.escape(e.event_id)}" class="kf-img" />'
            if uri
            else '<div class="kf-missing">No keyframe saved</div>'
        )
        vlm_block = (
            f'<p class="vlm-note"><strong>VLM Coach:</strong> {html.escape(e.vlm_observation)}</p>'
            if e.vlm_observation
            else ""
        )
        cards_html.append(
            f"""
            <article class="event-card">
              <div class="event-head">
                <span class="badge" style="background:{color}">{html.escape(e.event_id)} · {html.escape(e.event_type.value)}</span>
                <span class="ts">Clip {format_timestamp(e.timestamp_sec)} · Orig {format_timestamp(e.original_video_sec)} · conf {e.confidence:.2f}</span>
              </div>
              {img_block}
              <p><strong>Visual:</strong> {html.escape(e.visual_evidence)}</p>
              <p><strong>Audio:</strong> {html.escape(e.audio_evidence)}</p>
              {vlm_block}
            </article>
            """
        )

    vlm_section = ""
    if report.vlm_tactical_summary:
        vlm_section = f"""
        <section class="panel">
          <h2>VLM Tactical Coaching ({html.escape(report.vlm_model or "VLM")})</h2>
          <p class="coaching-text">{html.escape(report.vlm_tactical_summary)}</p>
        </section>
        """

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye Report — {html.escape(report.title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; }}
    .wrap {{ max-width: 1120px; margin: 0 auto; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; color: #f8fafc; }}
    .sub {{ color: #94a3b8; margin: 0 0 22px; font-size: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .kpi {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .kpi .val {{ font-size: 24px; font-weight: 700; color: #38bdf8; }}
    .kpi .lbl {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 22px; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .timeline-svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
    .event-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }}
    .event-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
    .badge {{ color: #020617; font-weight: 700; font-size: 12px; padding: 3px 9px; border-radius: 999px; }}
    .ts {{ font-family: monospace; font-size: 12px; color: #cbd5e1; }}
    .kf-img {{ width: 100%; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px; }}
    .kf-missing {{ padding: 24px; text-align: center; background: #1e293b; border-radius: 8px; color: #94a3b8; margin-bottom: 10px; }}
    .vlm-note {{ background: #172554; border-left: 3px solid #38bdf8; padding: 8px 10px; border-radius: 6px; font-size: 13px; }}
    .coaching-text {{ white-space: pre-line; line-height: 1.6; color: #e2e8f0; }}
    p {{ margin: 6px 0; font-size: 13px; line-height: 1.5; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>The-Eye · CS2 Gameplay Analysis</h1>
    <p class="sub">{html.escape(report.title)} · Source: {html.escape(report.source)} · Start {format_timestamp(report.excerpt_start_sec)} · Duration {s.analyzed_duration_sec:.1f}s ({s.total_frames} frames @ {report.fps:.1f} fps)</p>

    <section class="kpis">
      <div class="kpi"><div class="val">{s.engagements_per_min:.1f}</div><div class="lbl">Events / Min</div></div>
      <div class="kpi"><div class="val">{s.killfeed_events}</div><div class="lbl">Killfeed Events</div></div>
      <div class="kpi"><div class="val">{s.flash_count} ({s.flashed_seconds:.1f}s)</div><div class="lbl">Flashbangs (Blind)</div></div>
      <div class="kpi"><div class="val">{s.scoped_ratio * 100:.1f}%</div><div class="lbl">Scoped Time ({s.scoped_seconds:.1f}s)</div></div>
      <div class="kpi"><div class="val">{s.combat_bursts}</div><div class="lbl">Combat Bursts</div></div>
      <div class="kpi"><div class="val">{s.audio_spike_count}</div><div class="lbl">Audio Spikes</div></div>
    </section>

    <section class="panel">
      <h2>Visual &amp; Audio Activity Timeline</h2>
      {_build_svg_timeline(report)}
    </section>

    {vlm_section}

    <section>
      <h2>Detected Moments ({len(report.events)})</h2>
      <div class="grid">
        {"".join(cards_html) or "<p>No high-confidence combat or HUD events detected in this excerpt.</p>"}
      </div>
    </section>
  </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path


def write_comparison_html(
    report_a: AnalysisReport,
    report_b: AnalysisReport,
    output_path: Path,
) -> Path:
    """Write a side-by-side HTML comparison report for two CS2 recordings."""
    sa, sb = report_a.summary, report_b.summary
    rows = [
        ("Analyzed Duration (s)", f"{sa.analyzed_duration_sec:.1f}", f"{sb.analyzed_duration_sec:.1f}"),
        ("Total Detected Events", str(sa.total_events), str(sb.total_events)),
        ("Events / Minute", f"{sa.engagements_per_min:.2f}", f"{sb.engagements_per_min:.2f}"),
        ("Killfeed Events", str(sa.killfeed_events), str(sb.killfeed_events)),
        ("Flashbangs (Blind sec)", f"{sa.flash_count} ({sa.flashed_seconds:.1f}s)", f"{sb.flash_count} ({sb.flashed_seconds:.1f}s)"),
        ("Scoped Time (%)", f"{sa.scoped_seconds:.1f}s ({sa.scoped_ratio * 100:.1f}%)", f"{sb.scoped_seconds:.1f}s ({sb.scoped_ratio * 100:.1f}%)"),
        ("Combat Bursts (Visual+Audio)", str(sa.combat_bursts), str(sb.combat_bursts)),
        ("Audio Spikes", str(sa.audio_spike_count), str(sb.audio_spike_count)),
        ("Avg Center Motion", f"{sa.avg_center_motion:.2f}", f"{sb.avg_center_motion:.2f}"),
    ]
    rows_html = "".join(
        f"<tr><td>{html.escape(lbl)}</td><td>{html.escape(va)}</td><td>{html.escape(vb)}</td></tr>"
        for lbl, va, vb in rows
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye Side-by-Side Comparison</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; }}
    .wrap {{ max-width: 1020px; margin: 0 auto; }}
    h1 {{ margin: 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #1e293b; text-align: left; font-size: 14px; }}
    th {{ background: #1e293b; color: #38bdf8; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 16px; color: #38bdf8; }}
    p {{ font-size: 13px; line-height: 1.6; white-space: pre-line; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>The-Eye · CS2 Side-by-Side Comparison</h1>
    <table>
      <thead>
        <tr><th>Metric</th><th>{html.escape(report_a.title)}</th><th>{html.escape(report_b.title)}</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="cols">
      <div class="panel">
        <h2>{html.escape(report_a.title)} — Tactical Takeaways</h2>
        <p>{html.escape(report_a.vlm_tactical_summary or "Run without --no-vlm for VLM tactical takeaways.")}</p>
      </div>
      <div class="panel">
        <h2>{html.escape(report_b.title)} — Tactical Takeaways</h2>
        <p>{html.escape(report_b.vlm_tactical_summary or "Run without --no-vlm for VLM tactical takeaways.")}</p>
      </div>
    </div>
  </div>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
