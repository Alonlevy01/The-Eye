"""Player-first interactive HTML BTD6 Strategy & Economy Coach + side-by-side comparison."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from the_eye.schemas import AnalysisReport, EventType, format_timestamp

EVENT_COLORS: dict[EventType, str] = {
    EventType.BLOON_LEAK: "#ef4444",
    EventType.KILLER_ROUND: "#f59e0b",
    EventType.TOWER_PURCHASE: "#22c55e",
    EventType.CASH_HOARDING: "#a855f7",
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
    if score >= 85.0:
        return "#22c55e"
    if score >= 70.0:
        return "#38bdf8"
    if score >= 50.0:
        return "#f59e0b"
    return "#ef4444"


def _letter_grade(score: float) -> str:
    if score >= 90.0:
        return "S"
    if score >= 80.0:
        return "A"
    if score >= 68.0:
        return "B"
    if score >= 52.0:
        return "C"
    return "D"


def _build_cash_svg(report: AnalysisReport) -> str:
    """Render an interactive Cash ($) Over Time SVG with Wiki Killer Round & Leak markers."""
    width, height = 960, 225
    pad_l, pad_r, pad_t, pad_b = 58, 20, 24, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    valid = [m for m in report.timeline if m.cash is not None]
    if not valid:
        return "<p class='muted'>No HUD cash readings detected.</p>"

    max_t = max(report.timeline[-1].timestamp_sec, 1.0)
    max_cash = max(max(m.cash or 0 for m in valid), 1000)

    def _x(t: float) -> float:
        return pad_l + (t / max_t) * plot_w

    def _y(cash_val: float) -> float:
        clamped = max(0.0, min(float(max_cash), cash_val))
        return pad_t + (1.0 - clamped / float(max_cash)) * plot_h

    pts = " ".join(f"{_x(m.timestamp_sec):.1f},{_y(float(m.cash or 0)):.1f}" for m in valid)

    markers = []
    for e in report.events:
        cx = _x(e.timestamp_sec)
        color = EVENT_COLORS.get(e.event_type, "#e2e8f0")
        cy = _y(float(e.cash_at_event or 0))
        lbl = f"R{e.round_num}" if e.event_type == EventType.KILLER_ROUND and e.round_num else e.event_id
        markers.append(
            f'<g style="cursor:pointer" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">'
            f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="{color}" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.8" />'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" fill="{color}" />'
            f'<text x="{cx + 5:.1f}" y="{pad_t + 12}" fill="{color}" font-size="11" '
            f'font-family="monospace">{html.escape(lbl)}</text></g>'
        )

    return f"""
    <svg id="cashSvg" viewBox="0 0 {width} {height}" class="timeline-svg" role="img" aria-label="BTD6 Cash Economy Timeline"
         style="cursor:crosshair" onclick="scrubSvg(event, {pad_l}, {plot_w}, {max_t:.2f}, {report.excerpt_start_sec:.2f})">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0b1220" rx="8" />
      <line x1="{pad_l}" y1="{_y(max_cash * 0.5):.1f}" x2="{pad_l + plot_w}" y2="{_y(max_cash * 0.5):.1f}" stroke="#1e293b" stroke-dasharray="3 3" />
      <polyline fill="none" stroke="#4ade80" stroke-width="2.4" points="{pts}" />
      {"".join(markers)}
      <text x="6" y="{pad_t + 6}" fill="#94a3b8" font-size="11">${max_cash:,}</text>
      <text x="6" y="{_y(max_cash * 0.5) + 4:.1f}" fill="#94a3b8" font-size="11">${int(max_cash * 0.5):,}</text>
      <text x="26" y="{pad_t + plot_h:.1f}" fill="#94a3b8" font-size="11">$0</text>
      <text x="{pad_l}" y="{height - 10}" fill="#94a3b8" font-size="11">00:00.0</text>
      <text x="{width - 62}" y="{height - 10}" fill="#94a3b8" font-size="11">{format_timestamp(max_t)}</text>
      <text x="{pad_l + 6}" y="15" fill="#4ade80" font-size="11">■ Cash ($) — click anywhere on graph to jump video</text>
      <text x="{pad_l + 335}" y="15" fill="#f59e0b" font-size="11">■ Wiki Killer Round</text>
      <text x="{pad_l + 470}" y="15" fill="#ef4444" font-size="11">■ Bloon Leak</text>
    </svg>
    """


def _build_video_player_html(report: AnalysisReport) -> tuple[str, str]:
    """Build offline HTML5 <video> player pointing directly to clip.mp4 + fallback YouTube link."""
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
            <span id="playerStatus" class="muted">Click any timestamp or graph point to jump the video on this page.</span>
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
          if (st) st.textContent = 'Playing video at ' + fmtSec(clipSec);
        }}
        """
        return player_html, jump_js

    return (
        "<div class='player-wrap'><p class='muted' style='padding:16px;'>Run `the-eye analyze` to attach clip.mp4 for instant playback.</p></div>",
        "function jumpTo(origSec, clipSec) {}",
    )


def write_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write a sleek, simple player-first BTD6 Strategy & Economy HTML report."""
    s = report.summary
    sc = report.scorecard
    overall_grade = _letter_grade(sc.overall_strategy_score)
    overall_color = _score_color(sc.overall_strategy_score)
    round_span = (
        f"R{sc.start_round} → R{sc.end_round}"
        if sc.start_round is not None and sc.end_round is not None
        else "n/a"
    )

    player_html, jump_js = _build_video_player_html(report)

    tower_pills = "".join(
        f'<span class="tower-pill">{html.escape(t)}</span>' for t in sc.towers_observed
    ) or '<span class="muted">No tower upgrade panels opened in this excerpt.</span>'

    killer_pills = "".join(
        f'<span class="killer-pill">{html.escape(k)}</span>' for k in sc.killer_rounds_seen
    ) or '<span class="muted">None in this round range.</span>'

    mistakes_items = []
    for mk in sc.mistakes:
        sev_col = "#ef4444" if mk.severity == "high" else "#f59e0b"
        clip_lbl = format_timestamp(mk.timestamp_sec)
        mistakes_items.append(
            f"""
            <div class="mistake-card">
              <div class="mistake-top">
                <div>
                  <span class="sev" style="background:{sev_col}">{html.escape(mk.severity.upper())}</span>
                  <strong>{html.escape(mk.category)}</strong>
                </div>
                <button class="jump-btn" onclick="jumpTo({mk.original_video_sec:.2f}, {mk.timestamp_sec:.2f})">▶ Jump {clip_lbl}</button>
              </div>
              <p class="mistake-what">{html.escape(mk.summary)}</p>
              <p class="mistake-fix"><strong>Bloons Wiki Tip:</strong> {html.escape(mk.how_to_fix)}</p>
            </div>
            """
        )

    mistakes_body = (
        f"<div class='mistake-grid'>{''.join(mistakes_items)}</div>"
        if mistakes_items
        else "<p class='muted'>Zero leaks or cash-hoarding issues detected in this clip!</p>"
    )

    cards_html = []
    for e in report.events:
        color = EVENT_COLORS.get(e.event_type, "#cbd5e1")
        strip_uri = _img_to_data_uri(e.filmstrip_path)
        kf_uri = _img_to_data_uri(e.keyframe_path)
        if not strip_uri and not kf_uri:
            continue
        clip_lbl = format_timestamp(e.timestamp_sec)
        r_tag = f"R{e.round_num}" if e.round_num else ""
        media_blocks = []
        if strip_uri:
            media_blocks.append(f'<img src="{strip_uri}" alt="{html.escape(e.event_id)} filmstrip" class="kf-img" />')
        if kf_uri:
            media_blocks.append(f'<img src="{kf_uri}" alt="{html.escape(e.event_id)} HUD check" class="kf-img advanced-only" />')
        vlm_block = (
            f'<p class="vlm-note"><strong>Coach Note:</strong> {html.escape(e.vlm_observation)}</p>'
            if e.vlm_observation
            else ""
        )
        cards_html.append(
            f"""
            <article class="event-card">
              <div class="event-head">
                <span class="badge" style="background:{color}">{html.escape(e.event_id)} · {html.escape(e.event_type.value)} {r_tag}</span>
                <button class="jump-btn" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">▶ Watch {clip_lbl}</button>
              </div>
              {"".join(media_blocks)}
              <p>{html.escape(e.visual_evidence)}</p>
              {vlm_block}
              <p class="advanced-only muted">Audio: {html.escape(e.audio_evidence)}</p>
            </article>
            """
        )

    vlm_section = ""
    if report.vlm_tactical_summary:
        vlm_section = f"""
        <section class="panel">
          <h2>AI Strategy Summary <span class="muted advanced-only">({html.escape(report.vlm_model or "VLM")})</span></h2>
          <p class="coaching-text">{html.escape(report.vlm_tactical_summary)}</p>
        </section>
        """

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye BTD6 Strategy Coach — {html.escape(report.title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: radial-gradient(circle at top, #0f172a 0%, #020617 70%); color: #e2e8f0; margin: 0; padding: 24px; min-height: 100vh; }}
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 20px; }}
    h1 {{ margin: 0; font-size: 26px; color: #f8fafc; letter-spacing: -0.02em; }}
    .sub {{ color: #94a3b8; margin: 4px 0 0; font-size: 13px; }}
    .toggle-wrap {{ display: flex; align-items: center; gap: 6px; background: #0f172a; border: 1px solid #1e293b; padding: 5px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #94a3b8; padding: 7px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; cursor: pointer; }}
    .mode-btn.active {{ background: #4ade80; color: #020617; }}
    .hero {{ display: grid; grid-template-columns: 1.55fr 1fr; gap: 18px; margin-bottom: 22px; }}
    @media (max-width: 900px) {{ .hero {{ grid-template-columns: 1fr; }} }}
    .player-wrap {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 14px; overflow: hidden; box-shadow: 0 12px 32px rgba(0,0,0,0.4); }}
    .video-frame {{ width: 100%; aspect-ratio: 16 / 9; border: 0; display: block; background: #000; }}
    .player-bar {{ display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; font-size: 12px; gap: 10px; flex-wrap: wrap; }}
    .ext-link {{ color: #4ade80; text-decoration: none; font-weight: 600; }}
    .grade-hero {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 14px; padding: 18px; display: flex; flex-direction: column; justify-content: space-between; }}
    .overall-row {{ display: flex; align-items: center; gap: 16px; margin-bottom: 14px; }}
    .big-grade {{ width: 64px; height: 64px; border-radius: 14px; display: flex; align-items: center; justify-content: center; font-size: 32px; font-weight: 800; color: #020617; flex-shrink: 0; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 14px; padding: 18px; margin-bottom: 22px; }}
    .panel h2 {{ margin: 0 0 14px; font-size: 18px; color: #f8fafc; }}
    .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(155px, 1fr)); gap: 12px; margin-bottom: 22px; }}
    .kpi {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }}
    .kpi .val {{ font-size: 22px; font-weight: 800; color: #4ade80; }}
    .kpi .lbl {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .tower-pill {{ background: #1e293b; color: #38bdf8; border: 1px solid #334155; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .killer-pill {{ background: #422006; color: #fbbf24; border: 1px solid #713f12; padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }}
    .mistake-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    .mistake-card {{ background: #020617; border: 1px solid #1e293b; border-radius: 10px; padding: 14px; }}
    .mistake-top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
    .sev {{ color: #020617; font-weight: 800; font-size: 10px; padding: 2px 7px; border-radius: 4px; margin-right: 6px; }}
    .mistake-what {{ font-size: 13px; color: #e2e8f0; margin: 4px 0; }}
    .mistake-fix {{ font-size: 12px; color: #93c5fd; margin: 6px 0 0; }}
    .jump-btn {{ background: #1e293b; color: #4ade80; border: 1px solid #334155; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 600; cursor: pointer; font-family: monospace; }}
    .jump-btn:hover {{ background: #4ade80; color: #020617; }}
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
        <h1>The-Eye · BTD6 Strategy &amp; Economy Coach</h1>
        <p class="sub">{html.escape(report.title)} · <strong>{html.escape(sc.game_mode)}</strong> · {round_span} ({s.analyzed_duration_sec:.0f}s clip)</p>
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
              <div style="font-size:12px;color:#94a3b8;">STRATEGY &amp; ECONOMY SCORE</div>
              <div style="font-size:28px;font-weight:800;color:#f8fafc;">{sc.overall_strategy_score:.1f} <span style="font-size:15px;color:#94a3b8;">/ 100</span></div>
            </div>
          </div>
          <div style="margin-bottom:10px;">
            <div style="font-size:11px;color:#94a3b8;font-weight:700;margin-bottom:5px;">TOWERS DETECTED ON HUD</div>
            <div class="pill-row">{tower_pills}</div>
          </div>
          <div style="margin-bottom:10px;">
            <div style="font-size:11px;color:#fbbf24;font-weight:700;margin-bottom:5px;">WIKI CHECKPOINT ROUNDS</div>
            <div class="pill-row">{killer_pills}</div>
          </div>
        </div>
        <div>
          <div style="font-size:12px;color:#4ade80;font-weight:700;margin-bottom:4px;">COACHING FOCUS</div>
          <ul class="drills">{"".join(f"<li>{html.escape(d)}</li>" for d in sc.improvement_drills[:3])}</ul>
        </div>
      </div>
    </section>

    <section class="kpis">
      <div class="kpi"><div class="val">{round_span}</div><div class="lbl">Rounds Covered</div></div>
      <div class="kpi"><div class="val" style="color:{'#ef4444' if sc.lives_lost > 0 else '#4ade80'}">{sc.lives_lost}</div><div class="lbl">Lives Lost (Leaks)</div></div>
      <div class="kpi"><div class="val">${sc.total_cash_spent:,}</div><div class="lbl">Cash Invested</div></div>
      <div class="kpi"><div class="val" style="color:#38bdf8">{sc.purchase_count}</div><div class="lbl">Upgrades / Placements</div></div>
      <div class="kpi"><div class="val" style="color:#fbbf24">${sc.peak_unspent_cash:,}</div><div class="lbl">Peak Unspent Cash</div></div>
      <div class="kpi advanced-only"><div class="val">${sc.total_cash_earned:,}</div><div class="lbl">Total Cash Earned</div></div>
      <div class="kpi advanced-only"><div class="val">${sc.avg_unspent_cash:,.0f}</div><div class="lbl">Avg Unspent Cash</div></div>
      <div class="kpi advanced-only"><div class="val">{s.ocr_valid_frames}/{s.total_frames}</div><div class="lbl">HUD OCR Frames</div></div>
      <div class="kpi advanced-only"><div class="val" style="color:#c084fc">{report.visual_matched_towers}</div><div class="lbl">Embedding-Matched Towers</div></div>
    </section>

    <section class="panel">
      <h2>Cash Economy &amp; Round Timeline (Click Anywhere to Jump Video)</h2>
      {_build_cash_svg(report)}
    </section>

    <section class="panel">
      <h2>Mistakes &amp; Wiki Checkpoint Warnings</h2>
      {mistakes_body}
    </section>

    {vlm_section}

    <section>
      <h2 style="margin-bottom:12px;">Build Order &amp; Key Moment Filmstrips ({len(cards_html)})</h2>
      <div class="grid">
        {"".join(cards_html) or "<p>No major events detected in this clip.</p>"}
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
      const svg = document.getElementById('cashSvg');
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
    """Write a side-by-side HTML comparison report for two BTD6 gameplay recordings."""
    from the_eye.embeddings import compare_board_embeddings

    sa, sb = report_a.scorecard, report_b.scorecard
    ra_span = f"R{sa.start_round}-R{sa.end_round}" if sa.start_round else "n/a"
    rb_span = f"R{sb.start_round}-R{sb.end_round}" if sb.start_round else "n/a"
    emb_cmp = compare_board_embeddings(
        report_a.map_embedding,
        report_a.round_embeddings,
        report_b.map_embedding,
        report_b.round_embeddings,
    )
    emb_panel_html = ""
    if emb_cmp["map_similarity"] is not None:
        sim_pct = emb_cmp["map_similarity"] * 100.0
        same_badge = (
            "<span style='background:#4ade80;color:#020617;padding:3px 10px;"
            "border-radius:999px;font-weight:700;font-size:12px;'>SAME MAP VERIFIED</span>"
            if emb_cmp["same_map_detected"]
            else "<span style='background:#f59e0b;color:#020617;padding:3px 10px;"
            "border-radius:999px;font-weight:700;font-size:12px;'>DIFFERENT MAPS</span>"
        )
        round_pills = "".join(
            f"<span style='background:#1e293b;color:#38bdf8;border:1px solid #334155;"
            f"padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;'>"
            f"R{item['round']}: {item['similarity'] * 100:.1f}% board match</span>"
            for item in emb_cmp["shared_round_similarities"]
        )
        backend_lbl = html.escape(report_a.embedding_backend or "vision-embeddings")
        emb_panel_html = f"""
        <div class="panel" style="margin-bottom:22px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
            <h2 style="margin:0;">Visual Embedding Similarity ({sim_pct:.1f}% Map Match)</h2>
            {same_badge}
          </div>
          <p class="advanced-only" style="color:#94a3b8;margin:0 0 8px;">Embedding model: {backend_lbl}</p>
          <div style="display:flex;flex-wrap:wrap;gap:8px;">{round_pills or "<span style='color:#94a3b8;font-size:13px;'>No overlapping checkpoint rounds to compare board layouts.</span>"}</div>
        </div>
        """

    core_rows = [
        ("Game Mode", sa.game_mode, sb.game_mode),
        ("Overall Strategy Score", f"{_letter_grade(sa.overall_strategy_score)} · {sa.overall_strategy_score:.1f}/100", f"{_letter_grade(sb.overall_strategy_score)} · {sb.overall_strategy_score:.1f}/100"),
        ("Rounds Covered", ra_span, rb_span),
        ("Lives Lost (Leaks)", str(sa.lives_lost), str(sb.lives_lost)),
        ("Total Cash Invested", f"${sa.total_cash_spent:,}", f"${sb.total_cash_spent:,}"),
        ("Upgrades / Placements", str(sa.purchase_count), str(sb.purchase_count)),
        ("Towers Observed", ", ".join(sa.towers_observed) or "n/a", ", ".join(sb.towers_observed) or "n/a"),
        ("Wiki Killer Rounds", ", ".join(sa.killer_rounds_seen) or "none", ", ".join(sb.killer_rounds_seen) or "none"),
    ]
    adv_rows = [
        ("Total Cash Earned", f"${sa.total_cash_earned:,}", f"${sb.total_cash_earned:,}"),
        ("Avg Unspent Cash", f"${sa.avg_unspent_cash:,.0f}", f"${sb.avg_unspent_cash:,.0f}"),
        ("Peak Unspent Cash", f"${sa.peak_unspent_cash:,}", f"${sb.peak_unspent_cash:,}"),
        ("Eco Farming Used", "Yes" if sa.has_eco_farming else "No", "Yes" if sb.has_eco_farming else "No"),
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
  <title>The-Eye BTD6 Side-by-Side Comparison</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #020617; color: #e2e8f0; margin: 0; padding: 28px; }}
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1080px; margin: 0 auto; }}
    .topbar {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 18px; }}
    h1 {{ margin: 0; font-size: 24px; }}
    .toggle-wrap {{ display: flex; gap: 6px; background: #0f172a; border: 1px solid #1e293b; padding: 4px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #94a3b8; padding: 6px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; cursor: pointer; }}
    .mode-btn.active {{ background: #4ade80; color: #020617; }}
    table {{ width: 100%; border-collapse: collapse; background: #0f172a; border-radius: 10px; overflow: hidden; margin-bottom: 24px; }}
    th, td {{ padding: 12px 14px; border-bottom: 1px solid #1e293b; text-align: left; font-size: 14px; }}
    th {{ background: #1e293b; color: #4ade80; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: #0f172a; border: 1px solid #1e293b; border-radius: 10px; padding: 16px; }}
    .panel h2 {{ margin: 0 0 8px; font-size: 16px; color: #4ade80; }}
    p, li {{ font-size: 13px; line-height: 1.6; white-space: pre-line; }}
  </style>
</head>
<body class="simple-mode">
  <div class="wrap">
    <div class="topbar">
      <h1>The-Eye · BTD6 Side-by-Side Strategy Comparison</h1>
      <div class="toggle-wrap">
        <button id="btnSimple" class="mode-btn active" onclick="setMode('simple')">Player View (Simple)</button>
        <button id="btnAdv" class="mode-btn" onclick="setMode('advanced')">Coach View (Advanced)</button>
      </div>
    </div>
    <table>
      <thead>
        <tr><th>BTD6 Metric</th><th>{html.escape(report_a.title)}</th><th>{html.escape(report_b.title)}</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    {emb_panel_html}
    <div class="cols">
      <div class="panel">
        <h2>{html.escape(report_a.title)} — Takeaways</h2>
        <ul>{drills_a}</ul>
        <p>{html.escape(report_a.vlm_tactical_summary or "")}</p>
      </div>
      <div class="panel">
        <h2>{html.escape(report_b.title)} — Takeaways</h2>
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
