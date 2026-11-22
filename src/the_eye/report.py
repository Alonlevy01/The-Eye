"""Player-first interactive HTML CS2 coaching report & side-by-side comparison generator."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

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


def _extract_youtube_id(url: str) -> str | None:
    parsed = urlparse(url.strip())
    if "youtu.be" in parsed.netloc:
        vid = parsed.path.strip("/").split("/")[0]
        return vid or None
    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]
        m = re.search(r"/(?:embed|shorts|live)/([A-Za-z0-9_-]{6,})", parsed.path)
        if m:
            return m.group(1)
    return None


def _score_color(score: float) -> str:
    if score >= 80.0:
        return "#22c55e"
    if score >= 62.0:
        return "#38bdf8"
    if score >= 45.0:
        return "#f59e0b"
    return "#ef4444"


def _letter_grade(score: float) -> str:
    if score >= 85.0:
        return "S"
    if score >= 75.0:
        return "A"
    if score >= 62.0:
        return "B"
    if score >= 48.0:
        return "C"
    return "D"


def _build_svg_timeline(report: AnalysisReport) -> str:
    """Render an inline SVG chart of optical flow, audio RMS, and clickable event markers."""
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
            f'<g style="cursor:pointer" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">'
            f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="4 3" opacity="0.85" />'
            f'<circle cx="{cx:.1f}" cy="{pad_t + 8}" r="5" fill="{color}" />'
            f'<text x="{cx + 6:.1f}" y="{pad_t + 12}" fill="{color}" font-size="11" '
            f'font-family="monospace">{html.escape(e.event_id)}</text></g>'
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


def _build_video_player_html(report: AnalysisReport) -> tuple[str, str]:
    """Build native HTML5 <video> player (works offline from file:/// with instant seek) + YouTube link."""
    yt_id = _extract_youtube_id(report.source) if report.is_url else None
    start_int = max(0, int(report.excerpt_start_sec))
    yt_link_html = ""
    yt_link_js = ""
    if yt_id:
        watch_url = f"https://www.youtube.com/watch?v={yt_id}&t={start_int}s"
        yt_link_html = (
            f'<a id="ytFallback" href="{watch_url}" target="_blank" rel="noopener" '
            f'class="ext-link">Open on YouTube ↗</a>'
        )
        yt_link_js = (
            f"const link = document.getElementById('ytFallback'); "
            f"if (link) link.href = 'https://www.youtube.com/watch?v={yt_id}&t=' + Math.max(0, Math.floor(origSec)) + 's';"
        )

    if report.local_clip_path:
        player_html = f"""
        <div class="player-wrap" id="videoSection">
          <video id="localPlayer" class="video-frame" controls preload="metadata" src="{html.escape(report.local_clip_path)}"></video>
          <div class="player-bar">
            <span id="playerStatus" class="muted">Click any timestamp below to jump &amp; play that exact second.</span>
            {yt_link_html}
          </div>
        </div>
        """
        jump_js = f"""
        function jumpTo(origSec, clipSec) {{
          const v = document.getElementById('localPlayer');
          if (v) {{
            v.currentTime = Math.max(0, clipSec);
            v.play().catch(() => {{}});
            document.getElementById('videoSection').scrollIntoView({{ behavior: 'smooth', block: 'center' }});
          }}
          {yt_link_js}
          const st = document.getElementById('playerStatus');
          if (st) st.textContent = 'Playing clip at ' + fmtSec(clipSec) + ' (orig ' + fmtSec(origSec) + ')';
        }}
        """
        return player_html, jump_js

    if yt_id:
        watch_url = f"https://www.youtube.com/watch?v={yt_id}&t={start_int}s"
        player_html = f"""
        <div class="player-wrap" id="videoSection" style="padding:18px;">
          <p style="margin:0 0 10px;">Re-run <code>the-eye analyze</code> to embed the local cut clip directly here, or watch at timestamp on YouTube:</p>
          <a id="ytFallback" href="{watch_url}" target="_blank" rel="noopener" class="jump-btn" style="text-decoration:none;display:inline-block;">▶ Open on YouTube at {format_timestamp(report.excerpt_start_sec)} ↗</a>
        </div>
        """
        jump_js = f"""
        function jumpTo(origSec, clipSec) {{
          const s = Math.max(0, Math.floor(origSec));
          window.open('https://www.youtube.com/watch?v={yt_id}&t=' + s + 's', '_blank', 'noopener');
        }}
        """
        return player_html, jump_js

    return (
        "<div class='player-wrap'><p class='muted' style='padding:16px;'>No video attached to this report.</p></div>",
        "function jumpTo(origSec, clipSec) {}",
    )


def write_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write a player-first interactive HTML report with embedded video & Simple/Advanced toggle."""
    s = report.summary
    mc = report.mechanics
    rx_display = f"{mc.avg_reaction_time_ms:.0f} ms" if mc.avg_reaction_time_ms else "n/a"
    overall_grade = _letter_grade(mc.overall_mechanics_score)
    overall_color = _score_color(mc.overall_mechanics_score)

    player_html, jump_js = _build_video_player_html(report)

    def _bar(label: str, val: float, simple_sub: str, adv_sub: str) -> str:
        col = _score_color(val)
        grade = _letter_grade(val)
        return f"""
        <div class="meter">
          <div class="meter-head">
            <span>{html.escape(label)}</span>
            <span class="grade-pill" style="background:{col}">{grade} · {val:.0f}</span>
          </div>
          <div class="meter-track"><div class="meter-fill" style="width:{min(100.0, max(0.0, val)):.1f}%;background:{col}"></div></div>
          <div class="meter-sub">{html.escape(simple_sub)}</div>
          <div class="meter-sub advanced-only adv-tag">{html.escape(adv_sub)}</div>
        </div>
        """

    scorecard_html = "".join(
        [
            _bar(
                "Crosshair Discipline",
                mc.crosshair_discipline_score,
                "How steady your aim stays at head level between angles",
                f"Optical-flow vertical jitter = {mc.avg_vertical_jitter:.2f} (lower is cleaner)",
            ),
            _bar(
                "Counter-Strafing",
                mc.counter_strafe_score,
                "Stopping movement cleanly right before firing",
                f"Active strafe-brakes: {mc.active_counter_strafe_count} · Static: {mc.stationary_shot_count} · Moving: {mc.moving_shot_count}",
            ),
            _bar(
                f"Spray Control ({mc.weapon_display_name})",
                mc.spray_control_score,
                mc.weapon_recoil_tip or "Recoil control during sustained bursts",
                f"Measured across {mc.spray_burst_count} sustained gunfire bursts (profile: {mc.weapon_profile})",
            ),
            _bar(
                "Sound & Flank Awareness",
                mc.sound_awareness_score,
                "Reacting to audio cues & avoiding surprise flanks",
                f"Avg audio-to-aim reaction: {rx_display} · Flank/radar misses: {mc.flank_awareness_misses}",
            ),
            _bar(
                "Team & Trade Timing",
                mc.team_coordination_score,
                "Trading kills & swinging off utility with teammates",
                f"Trade windows (<=4s): {mc.trade_window_count} · Flash-entry combos: {mc.utility_entry_combos} · Radar rot: {mc.radar_rotation_activity:.1f}",
            ),
        ]
    )

    mistakes_html_items = []
    for mk in mc.mistakes:
        sev_col = "#ef4444" if mk.severity == "high" else "#f59e0b"
        ts_lbl = format_timestamp(mk.original_video_sec)
        mistakes_html_items.append(
            f"""
            <div class="mistake-card">
              <div class="mistake-top">
                <div>
                  <span class="sev" style="background:{sev_col}">{html.escape(mk.severity.upper())}</span>
                  <strong>{html.escape(mk.category)}</strong>
                </div>
                <button class="jump-btn" onclick="jumpTo({mk.original_video_sec:.2f}, {mk.timestamp_sec:.2f})">▶ Jump to {ts_lbl}</button>
              </div>
              <p class="mistake-what">{html.escape(mk.summary)}</p>
              <p class="mistake-fix"><strong>How to fix:</strong> {html.escape(mk.how_to_fix)}</p>
            </div>
            """
        )

    if not mistakes_html_items:
        drills_fallback = "".join(f"<li>{html.escape(d)}</li>" for d in mc.improvement_drills)
        mistakes_section_body = f"<ul class='drills'>{drills_fallback}</ul>"
    else:
        mistakes_section_body = f"<div class='mistake-grid'>{''.join(mistakes_html_items)}</div>"

    cards_html = []
    for e in report.events:
        color = EVENT_COLORS.get(e.event_type, "#cbd5e1")
        strip_uri = _img_to_data_uri(e.filmstrip_path)
        kf_uri = _img_to_data_uri(e.keyframe_path)
        if not strip_uri and not kf_uri:
            # Skip events without visual frames so Player View never shows empty placeholder cards
            continue
        ts_lbl = format_timestamp(e.original_video_sec)
        media_blocks = []
        if strip_uri:
            media_blocks.append(f'<img src="{strip_uri}" alt="{html.escape(e.event_id)} filmstrip" class="kf-img" />')
        if kf_uri:
            media_blocks.append(f'<img src="{kf_uri}" alt="{html.escape(e.event_id)} HUD keyframe" class="kf-img advanced-only" />')
        img_block = "".join(media_blocks)
        vlm_block = (
            f'<p class="vlm-note"><strong>Coach Note:</strong> {html.escape(e.vlm_observation)}</p>'
            if e.vlm_observation
            else ""
        )
        cards_html.append(
            f"""
            <article class="event-card">
              <div class="event-head">
                <span class="badge" style="background:{color}">{html.escape(e.event_id)} · {html.escape(e.event_type.value)}</span>
                <button class="jump-btn" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">▶ Watch {ts_lbl}</button>
              </div>
              {img_block}
              {vlm_block}
              <p class="advanced-only"><strong>Visual CV:</strong> {html.escape(e.visual_evidence)}</p>
              <p class="advanced-only"><strong>Audio Signal:</strong> {html.escape(e.audio_evidence)} (conf {e.confidence:.2f})</p>
            </article>
            """
        )

    vlm_section = ""
    if report.vlm_tactical_summary:
        vlm_section = f"""
        <section class="panel">
          <h2>AI Coach Summary <span class="muted advanced-only">({html.escape(report.vlm_model or "VLM")})</span></h2>
          <p class="coaching-text">{html.escape(report.vlm_tactical_summary)}</p>
        </section>
        """

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye CS2 Coach — {html.escape(report.title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 24px; }}
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 26px; color: #f8fafc; }}
    .sub {{ color: #94a3b8; margin: 4px 0 0; font-size: 13px; }}
    .toggle-wrap {{ display: flex; align-items: center; gap: 8px; background: #0f172a; border: 1px solid #1e293b; padding: 5px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #94a3b8; padding: 7px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; cursor: pointer; }}
    .mode-btn.active {{ background: #38bdf8; color: #020617; }}
    .hero {{ display: grid; grid-template-columns: 1.55fr 1fr; gap: 18px; margin-bottom: 22px; }}
    @media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; }} }}
    .player-wrap {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; }}
    .video-frame {{ width: 100%; aspect-ratio: 16 / 9; border: 0; display: block; background: #000; }}
    .player-bar {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; font-size: 12px; gap: 10px; flex-wrap: wrap; }}
    .ext-link {{ color: #38bdf8; text-decoration: none; font-weight: 600; }}
    .grade-hero {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; }}
    .overall-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }}
    .big-grade {{ width: 64px; height: 64px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 32px; font-weight: 800; color: #020617; flex-shrink: 0; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 22px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; color: #f8fafc; }}
    .meters {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .meter {{ background: #020617; border: 1px solid #1e293b; border-radius: 10px; padding: 12px; }}
    .meter-head {{ display: flex; justify-content: space-between; align-items: center; font-size: 13px; margin-bottom: 8px; gap: 8px; }}
    .grade-pill {{ color: #020617; font-weight: 800; font-size: 12px; padding: 2px 8px; border-radius: 999px; }}
    .meter-track {{ background: #1e293b; height: 8px; border-radius: 999px; overflow: hidden; }}
    .meter-fill {{ height: 100%; border-radius: 999px; }}
    .meter-sub {{ font-size: 12px; color: #94a3b8; margin-top: 6px; line-height: 1.4; }}
    .adv-tag {{ color: #64748b; font-family: monospace; font-size: 11px; }}
    .mistake-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .mistake-card {{ background: #020617; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .mistake-top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
    .sev {{ color: #020617; font-weight: 800; font-size: 10px; padding: 2px 7px; border-radius: 4px; margin-right: 6px; }}
    .mistake-what {{ font-size: 13px; color: #e2e8f0; margin: 4px 0; }}
    .mistake-fix {{ font-size: 12px; color: #93c5fd; margin: 6px 0 0; }}
    .jump-btn {{ background: #1e293b; color: #38bdf8; border: 1px solid #334155; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer; font-family: monospace; }}
    .jump-btn:hover {{ background: #38bdf8; color: #020617; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .kpi {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .kpi .val {{ font-size: 22px; font-weight: 700; color: #38bdf8; }}
    .kpi .lbl {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .timeline-svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; }}
    .event-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }}
    .event-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
    .badge {{ color: #020617; font-weight: 700; font-size: 12px; padding: 3px 9px; border-radius: 999px; }}
    .kf-img {{ width: 100%; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px; }}
    .kf-missing {{ padding: 24px; text-align: center; background: #1e293b; border-radius: 8px; color: #94a3b8; margin-bottom: 10px; }}
    .vlm-note {{ background: #172554; border-left: 3px solid #38bdf8; padding: 8px 10px; border-radius: 6px; font-size: 13px; }}
    .coaching-text {{ white-space: pre-line; line-height: 1.6; color: #e2e8f0; }}
    .drills {{ margin: 6px 0 0; padding-left: 20px; line-height: 1.65; font-size: 13px; color: #cbd5e1; }}
    .muted {{ color: #94a3b8; }}
    p {{ margin: 6px 0; font-size: 13px; line-height: 1.5; }}
  </style>
</head>
<body class="simple-mode">
  <div class="wrap">
    <header class="topbar">
      <div>
        <h1>The-Eye · CS2 Player Coach</h1>
        <p class="sub">{html.escape(report.title)} · Weapon Profile: <strong>{html.escape(mc.weapon_display_name)}</strong> · {s.analyzed_duration_sec:.0f}s clip</p>
      </div>
      <div class="toggle-wrap">
        <button id="btnSimple" class="mode-btn active" onclick="setMode('simple')">Player View (Simple)</button>
        <button id="btnAdv" class="mode-btn" onclick="setMode('advanced')">Coach View (Advanced)</button>
      </div>
    </header>

    <section class="hero">
      {player_html}
      <div class="grade-hero">
        <div>
          <div class="overall-row">
            <div class="big-grade" style="background:{overall_color}">{overall_grade}</div>
            <div>
              <div style="font-size:13px;color:#94a3b8;">OVERALL MECHANICS</div>
              <div style="font-size:28px;font-weight:800;color:#f8fafc;">{mc.overall_mechanics_score:.1f} <span style="font-size:15px;color:#94a3b8;">/ 100</span></div>
            </div>
          </div>
          <p class="muted" style="margin-bottom:10px;">Weapon Tip: {html.escape(mc.weapon_recoil_tip)}</p>
        </div>
        <div>
          <div style="font-size:12px;color:#38bdf8;font-weight:700;margin-bottom:6px;">TOP FOCUS AREAS</div>
          <ul class="drills">{"".join(f"<li>{html.escape(d)}</li>" for d in mc.improvement_drills[:3])}</ul>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Your CS2 Skill Breakdown</h2>
      <div class="meters">{scorecard_html}</div>
    </section>

    <section class="panel">
      <h2>Mistakes to Fix (Click Any Timestamp to Watch)</h2>
      {mistakes_section_body}
    </section>

    {vlm_section}

    <section class="kpis advanced-only">
      <div class="kpi"><div class="val">{s.engagements_per_min:.1f}</div><div class="lbl">Events / Min</div></div>
      <div class="kpi"><div class="val">{s.killfeed_events}</div><div class="lbl">Killfeed Events</div></div>
      <div class="kpi"><div class="val">{s.damage_events}</div><div class="lbl">Damage Flashes</div></div>
      <div class="kpi"><div class="val">{s.flash_count} ({s.flashed_seconds:.1f}s)</div><div class="lbl">Flashbangs (Blind)</div></div>
      <div class="kpi"><div class="val">{s.scoped_ratio * 100:.1f}%</div><div class="lbl">Scoped Time</div></div>
      <div class="kpi"><div class="val">{s.audio_spike_count}</div><div class="lbl">Audio Spikes</div></div>
    </section>

    <section class="panel advanced-only">
      <h2>Optical Flow &amp; Audio Activity Timeline (Click Markers to Jump)</h2>
      {_build_svg_timeline(report)}
    </section>

    <section>
      <h2 style="margin-bottom:12px;">Key Duel Filmstrips ({len(report.events)})</h2>
      <div class="grid">
        {"".join(cards_html) or "<p>No high-confidence combat or HUD events detected in this excerpt.</p>"}
      </div>
    </section>
  </div>

  <script>
    function fmtSec(sec) {{
      const s = Math.max(0, sec);
      const m = Math.floor(s / 60);
      const r = (s - m * 60).toFixed(1).padStart(4, '0');
      return String(m).padStart(2, '0') + ':' + r;
    }}
    function setMode(mode) {{
      const simple = mode === 'simple';
      document.body.classList.toggle('simple-mode', simple);
      document.getElementById('btnSimple').classList.toggle('active', simple);
      document.getElementById('btnAdv').classList.toggle('active', !simple);
    }}
    {jump_js}
  </script>
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
    """Write a side-by-side HTML comparison report with Simple/Advanced view toggle."""
    sa, sb = report_a.summary, report_b.summary
    ma, mb = report_a.mechanics, report_b.mechanics
    rx_a = f"{ma.avg_reaction_time_ms:.0f} ms" if ma.avg_reaction_time_ms else "n/a"
    rx_b = f"{mb.avg_reaction_time_ms:.0f} ms" if mb.avg_reaction_time_ms else "n/a"

    core_rows = [
        ("Overall CS2 Mechanics", f"{_letter_grade(ma.overall_mechanics_score)} · {ma.overall_mechanics_score:.1f}/100", f"{_letter_grade(mb.overall_mechanics_score)} · {mb.overall_mechanics_score:.1f}/100"),
        ("Crosshair Discipline", f"{ma.crosshair_discipline_score:.1f}/100", f"{mb.crosshair_discipline_score:.1f}/100"),
        ("Active Counter-Strafe", f"{ma.counter_strafe_score:.1f}/100", f"{mb.counter_strafe_score:.1f}/100"),
        ("Spray Control", f"{ma.spray_control_score:.1f}/100 ({ma.weapon_display_name})", f"{mb.spray_control_score:.1f}/100 ({mb.weapon_display_name})"),
        ("Sound & Flank Awareness", f"{ma.sound_awareness_score:.1f}/100", f"{mb.sound_awareness_score:.1f}/100"),
        ("Team & Trade Timing", f"{ma.team_coordination_score:.1f}/100", f"{mb.team_coordination_score:.1f}/100"),
    ]
    adv_rows = [
        ("Vertical Jitter (Optical Flow)", f"{ma.avg_vertical_jitter:.2f}", f"{mb.avg_vertical_jitter:.2f}"),
        ("Shots (Active / Static / Moving)", f"{ma.active_counter_strafe_count} / {ma.stationary_shot_count} / {ma.moving_shot_count}", f"{mb.active_counter_strafe_count} / {mb.stationary_shot_count} / {mb.moving_shot_count}"),
        ("Sustained Spray Bursts", str(ma.spray_burst_count), str(mb.spray_burst_count)),
        ("Avg Audio-to-Aim Reaction", rx_a, rx_b),
        ("Trade Windows / Radar Activity", f"{ma.trade_window_count} / {ma.radar_rotation_activity:.1f}", f"{mb.trade_window_count} / {mb.radar_rotation_activity:.1f}"),
        ("Events / Min", f"{sa.engagements_per_min:.2f}", f"{sb.engagements_per_min:.2f}"),
        ("Killfeed / Damage / Audio Spikes", f"{sa.killfeed_events} / {sa.damage_events} / {sa.audio_spike_count}", f"{sb.killfeed_events} / {sb.damage_events} / {sb.audio_spike_count}"),
    ]

    rows_html = "".join(
        f"<tr><td>{html.escape(lbl)}</td><td><strong>{html.escape(va)}</strong></td><td><strong>{html.escape(vb)}</strong></td></tr>"
        for lbl, va, vb in core_rows
    ) + "".join(
        f"<tr class='advanced-only'><td>{html.escape(lbl)}</td><td>{html.escape(va)}</td><td>{html.escape(vb)}</td></tr>"
        for lbl, va, vb in adv_rows
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
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .toggle-wrap {{ display: flex; gap: 6px; background: #0f172a; border: 1px solid #1e293b; padding: 4px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #94a3b8; padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; cursor: pointer; }}
    .mode-btn.active {{ background: #38bdf8; color: #020617; }}
    table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #1e293b; text-align: left; font-size: 14px; }}
    th {{ background: #1e293b; color: #38bdf8; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 16px; color: #38bdf8; }}
    p, li {{ font-size: 13px; line-height: 1.6; white-space: pre-line; }}
  </style>
</head>
<body class="simple-mode">
  <div class="wrap">
    <div class="topbar">
      <h1>The-Eye · CS2 Side-by-Side Comparison</h1>
      <div class="toggle-wrap">
        <button id="btnSimple" class="mode-btn active" onclick="setMode('simple')">Player View (Simple)</button>
        <button id="btnAdv" class="mode-btn" onclick="setMode('advanced')">Coach View (Advanced)</button>
      </div>
    </div>
    <table>
      <thead>
        <tr><th>CS2 Skill</th><th>{html.escape(report_a.title)}</th><th>{html.escape(report_b.title)}</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <div class="cols">
      <div class="panel">
        <h2>{html.escape(report_a.title)} — How to Improve</h2>
        <ul>{drills_a}</ul>
        <p>{html.escape(report_a.vlm_tactical_summary or "")}</p>
      </div>
      <div class="panel">
        <h2>{html.escape(report_b.title)} — How to Improve</h2>
        <ul>{drills_b}</ul>
        <p>{html.escape(report_b.vlm_tactical_summary or "")}</p>
      </div>
    </div>
  </div>
  <script>
    function setMode(mode) {{
      const simple = mode === 'simple';
      document.body.classList.toggle('simple-mode', simple);
      document.getElementById('btnSimple').classList.toggle('active', simple);
      document.getElementById('btnAdv').classList.toggle('active', !simple);
    }}
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
