"""Standalone interactive HTML visual report & side-by-side CS2 mechanics comparison generator."""

from __future__ import annotations

import base64
import html
from pathlib import Path

from the_eye.schemas import AnalysisReport, EventType, format_timestamp

EVENT_COLORS: dict[EventType, str] = {
    EventType.FLASHBANG: "#f59e0b",
    EventType.KILLFEED_EVENT: "#ef4444",
    EventType.DAMAGE_TAKEN: "#fb7185",
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


def _score_color(score: float) -> str:
    if score >= 75.0:
        return "#22c55e"
    if score >= 55.0:
        return "#f59e0b"
    return "#ef4444"


def _build_svg_timeline(report: AnalysisReport) -> str:
    """Render a clean inline SVG chart of optical flow, audio RMS, and detected event markers."""
    width, height = 960, 215
    pad_l, pad_r, pad_t, pad_b = 44, 20, 22, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    timeline = report.timeline
    if not timeline:
        return "<p class='muted'>No frames sampled.</p>"

    max_t = max(timeline[-1].timestamp_sec, 1.0)
    max_flow = max((m.flow_mag for m in timeline), default=1.0) or 1.0
    max_rms = max((m.audio_rms for m in timeline), default=0.05) or 0.05

    def _x(t: float) -> float:
        return pad_l + (t / max_t) * plot_w

    def _y(norm: float) -> float:
        clamped = max(0.0, min(1.0, norm))
        return pad_t + (1.0 - clamped) * plot_h

    flow_pts = " ".join(
        f"{_x(m.timestamp_sec):.1f},{_y(m.flow_mag / max_flow):.1f}" for m in timeline
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
      <polyline fill="none" stroke="#38bdf8" stroke-width="2" points="{flow_pts}" />
      <polyline fill="none" stroke="#22c55e" stroke-width="1.8" opacity="0.85" points="{rms_pts}" />
      {"".join(markers)}
      <text x="{pad_l}" y="{height - 10}" fill="#94a3b8" font-size="11">00:00.0</text>
      <text x="{width - 62}" y="{height - 10}" fill="#94a3b8" font-size="11">{format_timestamp(max_t)}</text>
      <text x="{pad_l + 6}" y="14" fill="#38bdf8" font-size="11">■ Optical Flow (Aim/Movement)</text>
      <text x="{pad_l + 205}" y="14" fill="#22c55e" font-size="11">■ Audio RMS (Gunfire/Utility)</text>
    </svg>
    """


def write_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write a self-contained HTML visual report with CS2 Mechanics Scorecard & pre-duel strips."""
    s = report.summary
    mc = report.mechanics
    rx_display = f"{mc.avg_reaction_time_ms:.0f} ms" if mc.avg_reaction_time_ms else "n/a"

    def _bar(label: str, val: float, detail: str) -> str:
        col = _score_color(val)
        return f"""
        <div class="meter">
          <div class="meter-head"><span>{html.escape(label)}</span><strong style="color:{col}">{val:.1f}/100</strong></div>
          <div class="meter-track"><div class="meter-fill" style="width:{min(100.0, max(0.0, val)):.1f}%;background:{col}"></div></div>
          <div class="meter-sub">{html.escape(detail)}</div>
        </div>
        """

    scorecard_html = "".join(
        [
            _bar("Overall CS2 Mechanics", mc.overall_mechanics_score, "Weighted composite across crosshair, active counter-strafe, spray control, sound & team timing"),
            _bar("Crosshair Discipline", mc.crosshair_discipline_score, f"Avg vertical jitter: {mc.avg_vertical_jitter:.2f} (lower = cleaner head-level clearing)"),
            _bar("Active Counter-Strafe", mc.counter_strafe_score, f"Active strafe-brakes: {mc.active_counter_strafe_count} · Static shots: {mc.stationary_shot_count} · Moving shots: {mc.moving_shot_count}"),
            _bar("Spray Control", mc.spray_control_score, f"Measured across {mc.spray_burst_count} sustained gunfire bursts (vertical recoil pull-down stability)"),
            _bar("Sound & Flank Awareness", mc.sound_awareness_score, f"Avg audio-to-aim reaction: {rx_display} · Flank/radar misses: {mc.flank_awareness_misses}"),
            _bar("Team & Utility Coordination", mc.team_coordination_score, f"Trade windows (<=4s): {mc.trade_window_count} · Flash-entry combos: {mc.utility_entry_combos} · Radar activity: {mc.radar_rotation_activity:.1f}"),
        ]
    )

    drills_html = "".join(f"<li>{html.escape(d)}</li>" for d in mc.improvement_drills)

    cards_html = []
    for e in report.events:
        color = EVENT_COLORS.get(e.event_type, "#cbd5e1")
        strip_uri = _img_to_data_uri(e.filmstrip_path)
        kf_uri = _img_to_data_uri(e.keyframe_path)
        media_blocks = []
        if strip_uri:
            media_blocks.append(f'<img src="{strip_uri}" alt="{html.escape(e.event_id)} filmstrip" class="kf-img" />')
        if kf_uri:
            media_blocks.append(f'<img src="{kf_uri}" alt="{html.escape(e.event_id)} HUD keyframe" class="kf-img" />')
        img_block = "".join(media_blocks) or '<div class="kf-missing">No keyframe saved</div>'
        vlm_block = (
            f'<p class="vlm-note"><strong>CS2 Coach Diagnosis:</strong> {html.escape(e.vlm_observation)}</p>'
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
  <title>The-Eye CS2 Coaching Report — {html.escape(report.title)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    h1 {{ margin: 0 0 6px; font-size: 26px; color: #f8fafc; }}
    .sub {{ color: #94a3b8; margin: 0 0 22px; font-size: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .kpi {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .kpi .val {{ font-size: 22px; font-weight: 700; color: #38bdf8; }}
    .kpi .lbl {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 22px; }}
    .panel h2 {{ margin: 0 0 12px; font-size: 18px; color: #f8fafc; }}
    .meters {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; }}
    .meter {{ background: #020617; border: 1px solid #1e293b; border-radius: 8px; padding: 12px; }}
    .meter-head {{ display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 6px; }}
    .meter-track {{ background: #1e293b; height: 8px; border-radius: 999px; overflow: hidden; }}
    .meter-fill {{ height: 100%; border-radius: 999px; }}
    .meter-sub {{ font-size: 11px; color: #94a3b8; margin-top: 6px; }}
    .drills {{ margin: 8px 0 0; padding-left: 20px; line-height: 1.65; font-size: 13px; color: #cbd5e1; }}
    .timeline-svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; }}
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
    <h1>The-Eye · CS2 Mechanics &amp; Gameplay Coaching Report</h1>
    <p class="sub">{html.escape(report.title)} · Source: {html.escape(report.source)} · Start {format_timestamp(report.excerpt_start_sec)} · Duration {s.analyzed_duration_sec:.1f}s ({s.total_frames} frames @ {report.fps:.1f} fps)</p>

    <section class="panel">
      <h2>CS2 Mechanics Scorecard</h2>
      <div class="meters">{scorecard_html}</div>
      <h3 style="margin:16px 0 6px;font-size:15px;color:#38bdf8;">Targeted Practice Drills</h3>
      <ul class="drills">{drills_html}</ul>
    </section>

    <section class="kpis">
      <div class="kpi"><div class="val">{s.engagements_per_min:.1f}</div><div class="lbl">Events / Min</div></div>
      <div class="kpi"><div class="val">{s.killfeed_events}</div><div class="lbl">Killfeed Events</div></div>
      <div class="kpi"><div class="val">{s.damage_events}</div><div class="lbl">Damage Flashes</div></div>
      <div class="kpi"><div class="val">{s.flash_count} ({s.flashed_seconds:.1f}s)</div><div class="lbl">Flashbangs (Blind)</div></div>
      <div class="kpi"><div class="val">{s.scoped_ratio * 100:.1f}%</div><div class="lbl">Scoped Time ({s.scoped_seconds:.1f}s)</div></div>
      <div class="kpi"><div class="val">{s.audio_spike_count}</div><div class="lbl">Audio Spikes</div></div>
    </section>

    <section class="panel">
      <h2>Optical Flow &amp; Audio Activity Timeline</h2>
      {_build_svg_timeline(report)}
    </section>

    {vlm_section}

    <section>
      <h2>Detected Moments &amp; 3-Frame Pre-Duel Filmstrips ({len(report.events)})</h2>
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
    ma, mb = report_a.mechanics, report_b.mechanics
    rx_a = f"{ma.avg_reaction_time_ms:.0f} ms" if ma.avg_reaction_time_ms else "n/a"
    rx_b = f"{mb.avg_reaction_time_ms:.0f} ms" if mb.avg_reaction_time_ms else "n/a"

    rows = [
        ("Overall CS2 Mechanics Score", f"{ma.overall_mechanics_score:.1f} / 100", f"{mb.overall_mechanics_score:.1f} / 100"),
        ("Crosshair Discipline (v-jitter)", f"{ma.crosshair_discipline_score:.1f} ({ma.avg_vertical_jitter:.2f})", f"{mb.crosshair_discipline_score:.1f} ({mb.avg_vertical_jitter:.2f})"),
        ("Counter-Strafe (Active/Static/Moving)", f"{ma.counter_strafe_score:.1f} ({ma.active_counter_strafe_count}/{ma.stationary_shot_count}/{ma.moving_shot_count})", f"{mb.counter_strafe_score:.1f} ({mb.active_counter_strafe_count}/{mb.stationary_shot_count}/{mb.moving_shot_count})"),
        ("Spray Control (Bursts)", f"{ma.spray_control_score:.1f} ({ma.spray_burst_count} sprays)", f"{mb.spray_control_score:.1f} ({mb.spray_burst_count} sprays)"),
        ("Sound & Flank Awareness", f"{ma.sound_awareness_score:.1f} ({rx_a}, flank_miss={ma.flank_awareness_misses})", f"{mb.sound_awareness_score:.1f} ({rx_b}, flank_miss={mb.flank_awareness_misses})"),
        ("Team & Utility Coordination", f"{ma.team_coordination_score:.1f} (trades={ma.trade_window_count}, util={ma.utility_entry_combos})", f"{mb.team_coordination_score:.1f} (trades={mb.trade_window_count}, util={mb.utility_entry_combos})"),
        ("Radar Rotation Activity", f"{ma.radar_rotation_activity:.1f}", f"{mb.radar_rotation_activity:.1f}"),
        ("Events / Minute", f"{sa.engagements_per_min:.2f}", f"{sb.engagements_per_min:.2f}"),
        ("Killfeed Events", str(sa.killfeed_events), str(sb.killfeed_events)),
        ("Damage Taken Flashes", str(sa.damage_events), str(sb.damage_events)),
        ("Flashbangs (Blind sec)", f"{sa.flash_count} ({sa.flashed_seconds:.1f}s)", f"{sb.flash_count} ({sb.flashed_seconds:.1f}s)"),
        ("Scoped Time (%)", f"{sa.scoped_seconds:.1f}s ({sa.scoped_ratio * 100:.1f}%)", f"{sb.scoped_seconds:.1f}s ({sb.scoped_ratio * 100:.1f}%)"),
        ("Audio Spikes", str(sa.audio_spike_count), str(sb.audio_spike_count)),
    ]
    rows_html = "".join(
        f"<tr><td>{html.escape(lbl)}</td><td>{html.escape(va)}</td><td>{html.escape(vb)}</td></tr>"
        for lbl, va, vb in rows
    )

    drills_a = "".join(f"<li>{html.escape(d)}</li>" for d in ma.improvement_drills)
    drills_b = "".join(f"<li>{html.escape(d)}</li>" for d in mb.improvement_drills)

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye CS2 Side-by-Side Comparison</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    h1 {{ margin: 0 0 18px; }}
    table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
    th, td {{ padding: 11px 14px; border-bottom: 1px solid #1e293b; text-align: left; font-size: 14px; }}
    th {{ background: #1e293b; color: #38bdf8; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 16px; color: #38bdf8; }}
    p, li {{ font-size: 13px; line-height: 1.6; white-space: pre-line; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>The-Eye · CS2 Side-by-Side Mechanics Comparison</h1>
    <table>
      <thead>
        <tr><th>CS2 Metric</th><th>{html.escape(report_a.title)}</th><th>{html.escape(report_b.title)}</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="cols">
      <div class="panel">
        <h2>{html.escape(report_a.title)} — Coaching &amp; Drills</h2>
        <ul>{drills_a}</ul>
        <p>{html.escape(report_a.vlm_tactical_summary or "Run without --no-vlm for VLM tactical takeaways.")}</p>
      </div>
      <div class="panel">
        <h2>{html.escape(report_b.title)} — Coaching &amp; Drills</h2>
        <ul>{drills_b}</ul>
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
