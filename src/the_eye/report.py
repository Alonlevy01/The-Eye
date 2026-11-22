"""Player-first interactive HTML Rocket League Boost Coach & side-by-side comparison generator."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from the_eye.schemas import AnalysisReport, EventType, format_timestamp

EVENT_COLORS: dict[EventType, str] = {
    EventType.BOOST_STARVED: "#ef4444",
    EventType.PANIC_BURNOUT: "#f97316",
    EventType.BIG_PAD_PICKUP: "#22c55e",
    EventType.OVER_CAPPED: "#a855f7",
    EventType.GOAL_EXPLOSION: "#38bdf8",
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


def _build_boost_svg(report: AnalysisReport) -> str:
    """Render an interactive 0-100 Boost-Over-Time SVG with Starved zones & clickable events."""
    width, height = 960, 225
    pad_l, pad_r, pad_t, pad_b = 46, 20, 24, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    timeline = report.timeline
    if not timeline:
        return "<p class='muted'>No frames sampled.</p>"

    max_t = max(timeline[-1].timestamp_sec, 1.0)

    def _x(t: float) -> float:
        return pad_l + (t / max_t) * plot_w

    def _y(boost: float) -> float:
        clamped = max(0.0, min(100.0, boost))
        return pad_t + (1.0 - clamped / 100.0) * plot_h

    # Shaded starved zone (0-12 boost) and over-capped zone (96-100)
    y_12 = _y(12.0)
    y_0 = _y(0.0)
    y_96 = _y(96.0)
    y_100 = _y(100.0)

    # Break polyline across replay/no-HUD gaps so graph stays honest
    segments: list[str] = []
    curr_pts: list[str] = []
    for m in timeline:
        if m.hud_visible:
            curr_pts.append(f"{_x(m.timestamp_sec):.1f},{_y(m.boost_pct):.1f}")
        elif curr_pts:
            segments.append(" ".join(curr_pts))
            curr_pts = []
    if curr_pts:
        segments.append(" ".join(curr_pts))

    lines_svg = "".join(
        f'<polyline fill="none" stroke="#fb923c" stroke-width="2.4" points="{seg}" />'
        for seg in segments
    )

    markers = []
    for e in report.events:
        cx = _x(e.timestamp_sec)
        color = EVENT_COLORS.get(e.event_type, "#e2e8f0")
        markers.append(
            f'<g style="cursor:pointer" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">'
            f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.85" />'
            f'<circle cx="{cx:.1f}" cy="{_y(e.boost_at_event):.1f}" r="5" fill="{color}" />'
            f'<text x="{cx + 5:.1f}" y="{pad_t + 12}" fill="{color}" font-size="11" '
            f'font-family="monospace">{html.escape(e.event_id)}</text></g>'
        )

    return f"""
    <svg id="boostSvg" viewBox="0 0 {width} {height}" class="timeline-svg" role="img" aria-label="Rocket League Boost Timeline"
         style="cursor:crosshair" onclick="scrubSvg(event, {pad_l}, {plot_w}, {max_t:.2f}, {report.excerpt_start_sec:.2f})">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0b1220" rx="8" />
      <rect x="{pad_l}" y="{y_12:.1f}" width="{plot_w}" height="{y_0 - y_12:.1f}" fill="#ef4444" opacity="0.18" />
      <rect x="{pad_l}" y="{y_100:.1f}" width="{plot_w}" height="{y_96 - y_100:.1f}" fill="#a855f7" opacity="0.18" />
      <line x1="{pad_l}" y1="{_y(50.0):.1f}" x2="{pad_l + plot_w}" y2="{_y(50.0):.1f}" stroke="#1e293b" stroke-dasharray="3 3" />
      {lines_svg}
      {"".join(markers)}
      <text x="8" y="{pad_t + 6}" fill="#94a3b8" font-size="11">100</text>
      <text x="14" y="{_y(50.0) + 4:.1f}" fill="#94a3b8" font-size="11">50</text>
      <text x="20" y="{y_0:.1f}" fill="#ef4444" font-size="11">0</text>
      <text x="{pad_l}" y="{height - 10}" fill="#94a3b8" font-size="11">00:00.0</text>
      <text x="{width - 62}" y="{height - 10}" fill="#94a3b8" font-size="11">{format_timestamp(max_t)}</text>
      <text x="{pad_l + 6}" y="15" fill="#fb923c" font-size="11">■ Boost (0-100) — click anywhere on graph to jump video</text>
      <text x="{pad_l + 325}" y="15" fill="#ef4444" font-size="11">■ Starved (&lt;=12)</text>
      <text x="{pad_l + 440}" y="15" fill="#a855f7" font-size="11">■ Over-Capped (&gt;=96)</text>
    </svg>
    """


def _build_video_player_html(report: AnalysisReport) -> tuple[str, str]:
    """Build offline HTML5 <video> player with instant seek + fallback YouTube link."""
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

    return (
        "<div class='player-wrap'><p class='muted' style='padding:16px;'>Run `the-eye analyze` to attach clip.mp4 for instant playback.</p></div>",
        "function jumpTo(origSec, clipSec) {}",
    )


def write_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write a player-first Rocket League Boost Coaching HTML report."""
    s = report.summary
    sc = report.scorecard
    overall_grade = _letter_grade(sc.overall_boost_score)
    overall_color = _score_color(sc.overall_boost_score)
    rec_str = f"{sc.avg_recovery_sec:.1f}s" if sc.avg_recovery_sec is not None else "n/a"

    player_html, jump_js = _build_video_player_html(report)

    mistakes_items = []
    for mk in sc.mistakes:
        sev_col = "#ef4444" if mk.severity == "high" else "#f59e0b"
        clip_lbl = format_timestamp(mk.timestamp_sec)
        orig_lbl = format_timestamp(mk.original_video_sec)
        btn_lbl = clip_lbl if report.local_clip_path else orig_lbl
        mistakes_items.append(
            f"""
            <div class="mistake-card">
              <div class="mistake-top">
                <div>
                  <span class="sev" style="background:{sev_col}">{html.escape(mk.severity.upper())}</span>
                  <strong>{html.escape(mk.category)}</strong>
                </div>
                <button class="jump-btn" title="Original video: {orig_lbl}" onclick="jumpTo({mk.original_video_sec:.2f}, {mk.timestamp_sec:.2f})">▶ Jump {btn_lbl}</button>
              </div>
              <p class="mistake-what">{html.escape(mk.summary)}</p>
              <p class="mistake-fix"><strong>How to fix:</strong> {html.escape(mk.how_to_fix)}</p>
            </div>
            """
        )

    if not mistakes_items:
        mistakes_body = "<p class='muted'>No major boost starvation, panic burnout, or over-capped stretches detected in this clip!</p>"
    else:
        mistakes_body = f"<div class='mistake-grid'>{''.join(mistakes_items)}</div>"

    cards_html = []
    for e in report.events:
        color = EVENT_COLORS.get(e.event_type, "#cbd5e1")
        strip_uri = _img_to_data_uri(e.filmstrip_path)
        kf_uri = _img_to_data_uri(e.keyframe_path)
        if not strip_uri and not kf_uri:
            continue
        clip_lbl = format_timestamp(e.timestamp_sec)
        orig_lbl = format_timestamp(e.original_video_sec)
        btn_lbl = clip_lbl if report.local_clip_path else orig_lbl
        media_blocks = []
        if strip_uri:
            media_blocks.append(f'<img src="{strip_uri}" alt="{html.escape(e.event_id)} filmstrip" class="kf-img" />')
        if kf_uri:
            media_blocks.append(f'<img src="{kf_uri}" alt="{html.escape(e.event_id)} gauge verification" class="kf-img advanced-only" />')
        vlm_block = (
            f'<p class="vlm-note"><strong>Coach Note:</strong> {html.escape(e.vlm_observation)}</p>'
            if e.vlm_observation
            else ""
        )
        cards_html.append(
            f"""
            <article class="event-card">
              <div class="event-head">
                <span class="badge" style="background:{color}">{html.escape(e.event_id)} · {html.escape(e.event_type.value)} (Boost {e.boost_at_event:.0f})</span>
                <button class="jump-btn" title="Original video: {orig_lbl}" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">▶ Watch {btn_lbl}</button>
              </div>
              {"".join(media_blocks)}
              {vlm_block}
              <p class="advanced-only"><strong>Visual Proof:</strong> {html.escape(e.visual_evidence)}</p>
              <p class="advanced-only"><strong>Audio Signal:</strong> {html.escape(e.audio_evidence)}</p>
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
  <title>The-Eye Rocket League Coach — {html.escape(report.title)}</title>
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
    .mode-btn.active {{ background: #fb923c; color: #020617; }}
    .hero {{ display: grid; grid-template-columns: 1.55fr 1fr; gap: 18px; margin-bottom: 22px; }}
    @media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; }} }}
    .player-wrap {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; overflow: hidden; }}
    .video-frame {{ width: 100%; aspect-ratio: 16 / 9; border: 0; display: block; background: #000; }}
    .player-bar {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; font-size: 12px; gap: 10px; flex-wrap: wrap; }}
    .ext-link {{ color: #fb923c; text-decoration: none; font-weight: 600; }}
    .grade-hero {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; }}
    .overall-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }}
    .big-grade {{ width: 64px; height: 64px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 32px; font-weight: 800; color: #020617; flex-shrink: 0; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 18px; margin-bottom: 22px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; color: #f8fafc; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .kpi {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .kpi .val {{ font-size: 22px; font-weight: 800; color: #fb923c; }}
    .kpi .lbl {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .mistake-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .mistake-card {{ background: #020617; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .mistake-top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
    .sev {{ color: #020617; font-weight: 800; font-size: 10px; padding: 2px 7px; border-radius: 4px; margin-right: 6px; }}
    .mistake-what {{ font-size: 13px; color: #e2e8f0; margin: 4px 0; }}
    .mistake-fix {{ font-size: 12px; color: #93c5fd; margin: 6px 0 0; }}
    .jump-btn {{ background: #1e293b; color: #fb923c; border: 1px solid #334155; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer; font-family: monospace; }}
    .jump-btn:hover {{ background: #fb923c; color: #020617; }}
    .timeline-svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 16px; }}
    .event-card {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }}
    .event-head {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }}
    .badge {{ color: #020617; font-weight: 700; font-size: 12px; padding: 3px 9px; border-radius: 999px; }}
    .kf-img {{ width: 100%; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px; }}
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
        <h1>The-Eye · Rocket League Boost Coach</h1>
        <p class="sub">{html.escape(report.title)} · {sc.live_hud_seconds:.0f}s live gameplay analyzed ({s.analyzed_duration_sec:.0f}s total clip)</p>
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
              <div style="font-size:13px;color:#94a3b8;">BOOST DISCIPLINE SCORE</div>
              <div style="font-size:28px;font-weight:800;color:#f8fafc;">{sc.overall_boost_score:.1f} <span style="font-size:15px;color:#94a3b8;">/ 100</span></div>
            </div>
          </div>
          <p class="muted" style="margin-bottom:10px;">100% read directly from the on-screen 0-100 Boost Gauge arc (replays automatically excluded).</p>
        </div>
        <div>
          <div style="font-size:12px;color:#fb923c;font-weight:700;margin-bottom:6px;">TOP FOCUS AREAS</div>
          <ul class="drills">{"".join(f"<li>{html.escape(d)}</li>" for d in sc.improvement_drills[:3])}</ul>
        </div>
      </div>
    </section>

    <section class="kpis">
      <div class="kpi"><div class="val">{sc.avg_boost:.0f} <span style="font-size:13px;color:#94a3b8">/ 100</span></div><div class="lbl">Avg Boost Held</div></div>
      <div class="kpi"><div class="val" style="color:#38bdf8">{sc.boost_spent_per_min:.0f} <span style="font-size:13px;color:#94a3b8">BPM</span></div><div class="lbl">Boost Spent / Min</div></div>
      <div class="kpi"><div class="val" style="color:#ef4444">{sc.starved_pct:.0f}% <span style="font-size:13px;color:#94a3b8">({sc.starved_seconds:.0f}s)</span></div><div class="lbl">Boost-Starved (&lt;=12)</div></div>
      <div class="kpi"><div class="val" style="color:#22c55e">{sc.small_pad_pickups} / {sc.big_pad_pickups}</div><div class="lbl">Small Pads / Big Pads</div></div>
      <div class="kpi"><div class="val" style="color:#f97316">{sc.panic_burnouts}</div><div class="lbl">Panic Burnouts</div></div>
      <div class="kpi advanced-only"><div class="val" style="color:#a855f7">{sc.over_capped_pct:.0f}% ({sc.over_capped_seconds:.0f}s)</div><div class="lbl">Over-Capped (&gt;=96)</div></div>
      <div class="kpi advanced-only"><div class="val">{rec_str}</div><div class="lbl">Avg Recovery Time</div></div>
      <div class="kpi advanced-only"><div class="val">{sc.avg_field_speed:.1f}</div><div class="lbl">Avg Field Speed</div></div>
    </section>

    <section class="panel">
      <h2>Boost Over Time (Click Any Marker to Jump)</h2>
      {_build_boost_svg(report)}
    </section>

    <section class="panel">
      <h2>Boost Mistakes to Fix (Click Any Timestamp to Watch)</h2>
      {mistakes_body}
    </section>

    {vlm_section}

    <section>
      <h2 style="margin-bottom:12px;">Key Moment Filmstrips ({len(cards_html)})</h2>
      <div class="grid">
        {"".join(cards_html) or "<p>No boost events detected in this clip.</p>"}
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
    function scrubSvg(evt, padL, plotW, maxT, startOff) {{
      const svg = document.getElementById('boostSvg');
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const viewX = ((evt.clientX - rect.left) / rect.width) * 960;
      const norm = Math.max(0, Math.min(1, (viewX - padL) / plotW));
      const clipSec = norm * maxT;
      jumpTo(startOff + clipSec, clipSec);
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
    """Write a side-by-side HTML comparison report for two Rocket League recordings."""
    sa, sb = report_a.scorecard, report_b.scorecard
    rec_a = f"{sa.avg_recovery_sec:.1f}s" if sa.avg_recovery_sec is not None else "n/a"
    rec_b = f"{sb.avg_recovery_sec:.1f}s" if sb.avg_recovery_sec is not None else "n/a"

    core_rows = [
        ("Overall Boost Discipline", f"{_letter_grade(sa.overall_boost_score)} · {sa.overall_boost_score:.1f}/100", f"{_letter_grade(sb.overall_boost_score)} · {sb.overall_boost_score:.1f}/100"),
        ("Average Boost Held", f"{sa.avg_boost:.1f} / 100", f"{sb.avg_boost:.1f} / 100"),
        ("Boost Spent / Min (BPM)", f"{sa.boost_spent_per_min:.0f} BPM", f"{sb.boost_spent_per_min:.0f} BPM"),
        ("Time Boost-Starved (<=12)", f"{sa.starved_pct:.1f}% ({sa.starved_seconds:.1f}s)", f"{sb.starved_pct:.1f}% ({sb.starved_seconds:.1f}s)"),
        ("Small Pads / Big Pads", f"{sa.small_pad_pickups} small / {sa.big_pad_pickups} big ({sa.small_pad_ratio * 100:.0f}% small)", f"{sb.small_pad_pickups} small / {sb.big_pad_pickups} big ({sb.small_pad_ratio * 100:.0f}% small)"),
        ("Panic Burnouts", str(sa.panic_burnouts), str(sb.panic_burnouts)),
    ]
    adv_rows = [
        ("Avg Starved Recovery Time", rec_a, rec_b),
        ("Avg Field Speed (Motion)", f"{sa.avg_field_speed:.1f}", f"{sb.avg_field_speed:.1f}"),
        ("Live Gameplay Analyzed", f"{sa.live_hud_seconds:.1f}s", f"{sb.live_hud_seconds:.1f}s"),
        ("Goal Explosions", str(sa.goal_explosions), str(sb.goal_explosions)),
    ]

    rows_html = "".join(
        f"<tr><td>{html.escape(lbl)}</td><td><strong>{html.escape(va)}</strong></td><td><strong>{html.escape(vb)}</strong></td></tr>"
        for lbl, va, vb in core_rows
    ) + "".join(
        f"<tr class='advanced-only'><td>{html.escape(lbl)}</td><td>{html.escape(va)}</td><td>{html.escape(vb)}</td></tr>"
        for lbl, va, vb in adv_rows
    )

    drills_a = "".join(f"<li>{html.escape(d)}</li>" for d in sa.improvement_drills)
    drills_b = "".join(f"<li>{html.escape(d)}</li>" for d in sb.improvement_drills)

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye Rocket League Comparison</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; }}
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .toggle-wrap {{ display: flex; gap: 6px; background: #0f172a; border: 1px solid #1e293b; padding: 4px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #94a3b8; padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; cursor: pointer; }}
    .mode-btn.active {{ background: #fb923c; color: #020617; }}
    table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #1e293b; text-align: left; font-size: 14px; }}
    th {{ background: #1e293b; color: #fb923c; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 16px; color: #fb923c; }}
    p, li {{ font-size: 13px; line-height: 1.6; white-space: pre-line; }}
  </style>
</head>
<body class="simple-mode">
  <div class="wrap">
    <div class="topbar">
      <h1>The-Eye · Rocket League Side-by-Side Comparison</h1>
      <div class="toggle-wrap">
        <button id="btnSimple" class="mode-btn active" onclick="setMode('simple')">Player View (Simple)</button>
        <button id="btnAdv" class="mode-btn" onclick="setMode('advanced')">Coach View (Advanced)</button>
      </div>
    </div>
    <table>
      <thead>
        <tr><th>Rocket League Boost Metric</th><th>{html.escape(report_a.title)}</th><th>{html.escape(report_b.title)}</th></tr>
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
