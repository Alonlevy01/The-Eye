"""BTD6-themed compact interactive HTML Strategy & Economy Coach + side-by-side comparison."""

from __future__ import annotations

import base64
import html
import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from the_eye.embeddings import cosine_similarity
from the_eye.schemas import (
    AnalysisReport,
    DetectedEvent,
    EventType,
    FrameMetrics,
    GameSession,
    format_timestamp,
)

EVENT_COLORS: dict[EventType, str] = {
    EventType.BLOON_LEAK: "#ff4d4f",
    EventType.KILLER_ROUND: "#fbbf24",
    EventType.TOWER_PURCHASE: "#57cc2a",
    EventType.CASH_HOARDING: "#c084fc",
    EventType.DEFEAT_OR_END: "#ff4d4f",
}


SPRITES_DIR = Path(__file__).resolve().parent / "assets" / "sprites"


def _sprite_data_uri(name: str) -> str | None:
    """Load a bundled official BTD6 PNG sprite by tower/hero/bloon name as a base64 data URI."""
    key = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")
    while "__" in key:
        key = key.replace("__", "_")
    p = SPRITES_DIR / f"{key}.png"
    if not p.exists():
        return None
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _sprite_img(name: str, *, size: int = 22, alt: str = "") -> str:
    uri = _sprite_data_uri(name)
    if not uri:
        return ""
    return (
        f'<img src="{uri}" alt="{html.escape(alt or name)}" '
        f'style="width:{size}px;height:{size}px;object-fit:contain;vertical-align:middle;" />'
    )


def _img_to_data_uri(path_str: str | None) -> str | None:
    if not path_str:
        return None
    p = Path(path_str)
    if not p.exists():
        return None
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


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
        return "#57cc2a"
    if score >= 70.0:
        return "#38bdf8"
    if score >= 50.0:
        return "#fbbf24"
    return "#ff4d4f"


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


def _build_cash_svg(
    timeline: list[FrameMetrics],
    events: list[DetectedEvent],
    *,
    excerpt_start_sec: float,
    svg_id: str = "cashSvg",
) -> str:
    """Render a compact interactive Cash ($) Over Time SVG scoped to one game's timestamp range."""
    width, height = 760, 195
    pad_l, pad_r, pad_t, pad_b = 54, 16, 22, 28
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    valid = [m for m in timeline if m.cash is not None]
    if not valid:
        return "<p class='muted'>No HUD cash readings detected in this game.</p>"

    min_t = timeline[0].timestamp_sec
    max_t = max(timeline[-1].timestamp_sec, min_t + 1.0)
    span_t = max(1.0, max_t - min_t)
    max_cash = max(max(m.cash or 0 for m in valid), 1000)

    def _x(t: float) -> float:
        return pad_l + ((t - min_t) / span_t) * plot_w

    def _y(cash_val: float) -> float:
        clamped = max(0.0, min(float(max_cash), cash_val))
        return pad_t + (1.0 - clamped / float(max_cash)) * plot_h

    pts = " ".join(f"{_x(m.timestamp_sec):.1f},{_y(float(m.cash or 0)):.1f}" for m in valid)

    markers = []
    for e in events:
        if e.event_type == EventType.TOWER_PURCHASE and not e.filmstrip_path:
            continue
        cx = _x(e.timestamp_sec)
        color = EVENT_COLORS.get(e.event_type, "#e2e8f0")
        cy = _y(float(e.cash_at_event or 0))
        lbl = (
            f"R{e.round_num}"
            if e.event_type == EventType.KILLER_ROUND and e.round_num
            else e.event_id
        )
        markers.append(
            f'<g style="cursor:pointer" '
            f'onclick="event.stopPropagation();jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">'
            f'<line x1="{cx:.1f}" y1="{pad_t}" x2="{cx:.1f}" y2="{pad_t + plot_h}" '
            f'stroke="{color}" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.85" />'
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.5" fill="{color}" />'
            f'<text x="{cx + 4:.1f}" y="{pad_t + 11}" fill="{color}" font-size="10" '
            f'font-family="monospace">{html.escape(lbl)}</text></g>'
        )

    return f"""
    <svg id="{html.escape(svg_id)}" viewBox="0 0 {width} {height}" class="timeline-svg"
         role="img" aria-label="BTD6 Cash Economy Timeline" style="cursor:crosshair"
         onclick="scrubSvg(event, '{html.escape(svg_id)}', {width}, {pad_l}, {plot_w}, {min_t:.2f}, {span_t:.2f}, {excerpt_start_sec:.2f})">
      <rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" fill="#0c1a30" rx="8" stroke="#2c4e7c" />
      <line x1="{pad_l}" y1="{_y(max_cash * 0.5):.1f}" x2="{pad_l + plot_w}" y2="{_y(max_cash * 0.5):.1f}" stroke="#1f3b63" stroke-dasharray="3 3" />
      <polyline fill="none" stroke="#ffd43b" stroke-width="2.4" points="{pts}" />
      {"".join(markers)}
      <text x="5" y="{pad_t + 6}" fill="#cbd5e1" font-size="10">${max_cash:,}</text>
      <text x="5" y="{_y(max_cash * 0.5) + 4:.1f}" fill="#cbd5e1" font-size="10">${int(max_cash * 0.5):,}</text>
      <text x="24" y="{pad_t + plot_h:.1f}" fill="#cbd5e1" font-size="10">$0</text>
      <text x="{pad_l}" y="{height - 8}" fill="#94a3b8" font-size="10">{format_timestamp(min_t)}</text>
      <text x="{width - 54}" y="{height - 8}" fill="#94a3b8" font-size="10">{format_timestamp(max_t)}</text>
      <text x="{pad_l + 4}" y="14" fill="#ffd43b" font-size="10" font-weight="700">■ Cash ($) — click to jump</text>
      <text x="{pad_l + 185}" y="14" fill="#fbbf24" font-size="10">■ Killer Round</text>
      <text x="{pad_l + 280}" y="14" fill="#ff4d4f" font-size="10">■ Leak</text>
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
            f'class="ext-link">YouTube ↗</a>'
        )
        yt_link_js = (
            f"const link = document.getElementById('ytFallback'); "
            f"if (link) link.href = 'https://www.youtube.com/watch?v={yt_id}&t=' "
            "+ Math.max(0, Math.floor(origSec)) + 's';"
        )

    if report.local_clip_path:
        player_html = f"""
        <div class="player-wrap" id="videoSection">
          <video id="localPlayer" class="video-frame" controls preload="metadata" src="{html.escape(report.local_clip_path)}"></video>
          <div class="player-bar">
            <span id="playerStatus" class="muted">Click any ▶ Jump button or cash graph point to seek.</span>
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
            document.getElementById('videoSection').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
          }}
          {yt_link_js}
          const st = document.getElementById('playerStatus');
          if (st) st.textContent = 'Playing at ' + fmtSec(clipSec);
        }}
        """
        return player_html, jump_js

    return (
        "<div class='player-wrap'><p class='muted' style='padding:14px;'>"
        "Run `the-eye analyze` to attach clip.mp4 for playback.</p></div>",
        "function jumpTo(origSec, clipSec) {}",
    )


def _render_game_hero_card(sess: GameSession, excerpt_start_sec: float) -> str:
    sc = sess.scorecard
    s = sess.summary
    grade = _letter_grade(sc.overall_strategy_score)
    color = _score_color(sc.overall_strategy_score)
    outcome_bg = (
        "#57cc2a"
        if sc.outcome == "Victory"
        else "#ff4d4f"
        if "Defeat" in sc.outcome
        else "#38bdf8"
    )
    outcome_badge = (
        f'<span class="outcome-pill" style="background:{outcome_bg};">'
        f"{html.escape(sc.outcome.upper())}</span>"
    )
    round_span = (
        f"R{sc.start_round}→R{sc.end_round}"
        if sc.start_round is not None and sc.end_round is not None
        else "n/a"
    )
    tower_pills = "".join(
        f'<span class="tower-pill">{_sprite_img(t, size=18)}<span>{html.escape(t)}</span></span>'
        for t in sc.towers_observed
    ) or '<span class="muted">None opened</span>'
    upgrade_pills = "".join(
        f'<span class="upgrade-pill">{html.escape(u)}</span>' for u in sc.upgrades_detected
    )
    upgrade_block = (
        f'<div class="pill-group"><span class="micro-lbl" style="color:#d8b4fe;">UPGRADES</span>'
        f'<div class="pill-row">{upgrade_pills}</div></div>'
        if upgrade_pills
        else ""
    )

    def _killer_sprite(k_str: str) -> str:
        low_k = k_str.lower()
        if "ddt" in low_k:
            return _sprite_img("ddt", size=16)
        if "bad" in low_k:
            return _sprite_img("bad", size=16)
        if "moab" in low_k or "zomg" in low_k:
            return _sprite_img("moab", size=16)
        if "ceramic" in low_k:
            return _sprite_img("ceramic", size=16)
        return ""

    killer_pills = "".join(
        f'<span class="killer-pill">{_killer_sprite(k)}<span>{html.escape(k.split(":")[0])}</span></span>'
        for k in sc.killer_rounds_seen
    ) or '<span class="muted">None</span>'
    time_range = f"{format_timestamp(sess.start_sec)}–{format_timestamp(sess.end_sec)}"

    hero_badge = (
        f'<span class="hero-pill">{_sprite_img(sc.hero_used, size=20)}'
        f"<span>Hero: {html.escape(sc.hero_used)}</span></span>"
        if sc.hero_used
        else ""
    )
    map_badge = (
        f'<span class="map-pill">Map: {html.escape(sc.map_name)}</span>'
        if sc.map_name
        else ""
    )
    rules_html = (
        f'<div class="mode-rules-line">{html.escape(sc.mode_rules_note)}</div>'
        if sc.mode_rules_note
        else ""
    )

    inv_border = (
        "#ff4d4f"
        if "Defeat" in sc.outcome
        else "#fbbf24"
        if sc.lives_lost > 0
        else "#57cc2a"
    )

    return f"""
    <div class="btd-panel grade-hero">
      <div class="overall-row">
        <div class="big-grade" style="background:{color}">{grade}</div>
        <div style="flex:1;min-width:0;">
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span class="mode-tag">{html.escape(sc.game_mode)}</span>
            {outcome_badge}
            {map_badge}
            {hero_badge}
            <button class="jump-btn" style="margin-left:auto;" onclick="jumpTo({excerpt_start_sec + sess.start_sec:.2f}, {sess.start_sec:.2f})">▶ {time_range}</button>
          </div>
          <div class="score-big">{sc.overall_strategy_score:.1f}<span> / 100</span></div>
        </div>
      </div>

      {rules_html}

      <div class="mini-kpis">
        <div class="mkpi"><strong>{round_span}</strong><span>Rounds</span></div>
        <div class="mkpi"><strong style="color:{'#ff4d4f' if sc.lives_lost > 0 else '#57cc2a'}">❤ {sc.lives_lost}</strong><span>Leaks</span></div>
        <div class="mkpi"><strong style="color:#ffd43b">${sc.total_cash_spent:,}</strong><span>Invested</span></div>
        <div class="mkpi"><strong style="color:#38bdf8">{sc.purchase_count}</strong><span>Buys</span></div>
        <div class="mkpi"><strong style="color:#fbbf24">${sc.peak_unspent_cash:,}</strong><span>Peak Bank</span></div>
        <div class="mkpi advanced-only"><strong>{s.ocr_valid_frames}/{s.total_frames}</strong><span>OCR</span></div>
        <div class="mkpi advanced-only"><strong style="color:#d8b4fe">{sess.visual_matched_towers}</strong><span>Emb Match</span></div>
      </div>

      <div class="verdict-box" style="border-left:4px solid {inv_border};">
        <div class="verdict-head">{html.escape(sc.root_cause_headline or "Match audit complete.")}</div>
        <div class="verdict-early"><strong>Early Game (R1–R25):</strong> {html.escape(sc.early_game_summary or "n/a")}</div>
      </div>

      <div class="pill-group">
        <span class="micro-lbl">TOWERS</span>
        <div class="pill-row">{tower_pills}</div>
      </div>
      {upgrade_block}
      <div class="pill-group" style="margin-bottom:0;">
        <span class="micro-lbl" style="color:#fbbf24;">KILLER ROUNDS</span>
        <div class="pill-row">{killer_pills}</div>
      </div>
    </div>
    """


def _render_game_body(sess: GameSession, report: AnalysisReport) -> str:
    sc = sess.scorecard

    mistakes_items = []
    for mk in sc.mistakes:
        sev_col = "#ff4d4f" if mk.severity == "high" else "#fbbf24"
        clip_lbl = format_timestamp(mk.timestamp_sec)
        root_html = (
            f'<div class="cause-box"><strong>Why:</strong> {html.escape(mk.root_cause)}</div>'
            if mk.root_cause
            else ""
        )
        sol_html = (
            f'<div class="solution-box"><strong>Fix:</strong> '
            f"{html.escape(mk.affordable_solution)}</div>"
            if mk.affordable_solution
            else ""
        )
        mistakes_items.append(
            f"""
            <div class="mistake-card">
              <div class="mistake-top">
                <div>
                  <span class="sev" style="background:{sev_col}">{html.escape(mk.severity.upper())}</span>
                  <strong>{html.escape(mk.category)}</strong>
                </div>
                <button class="jump-btn" onclick="jumpTo({mk.original_video_sec:.2f}, {mk.timestamp_sec:.2f})">▶ {clip_lbl}</button>
              </div>
              <p class="mistake-what">{html.escape(mk.summary)}</p>
              {root_html}
              {sol_html}
              <p class="mistake-fix"><strong>Wiki Tip:</strong> {html.escape(mk.how_to_fix)}</p>
            </div>
            """
        )
    mistakes_body = (
        f'<div class="mistake-list">{"".join(mistakes_items)}</div>'
        if mistakes_items
        else "<p style='color:#57cc2a;font-weight:700;'>Zero leaks and clean economy — no major mistakes flagged!</p>"
    )

    cards_html = []
    for e in sess.events:
        img_uri = _img_to_data_uri(e.filmstrip_path) or _img_to_data_uri(e.keyframe_path)
        if not img_uri:
            continue
        color = EVENT_COLORS.get(e.event_type, "#57cc2a")
        clip_lbl = format_timestamp(e.timestamp_sec)
        r_badge = f" · R{e.round_num}" if e.round_num else ""
        vlm_block = (
            f'<p class="vlm-note"><strong>Coach:</strong> {html.escape(e.vlm_observation)}</p>'
            if e.vlm_observation
            else ""
        )
        cards_html.append(
            f"""
            <article class="event-card">
              <div class="event-head">
                <span class="badge" style="background:{color}">{html.escape(e.event_id)} · {html.escape(e.event_type.value.replace("_", " ").upper())}{r_badge}</span>
                <button class="jump-btn" onclick="jumpTo({e.original_video_sec:.2f}, {e.timestamp_sec:.2f})">▶ {clip_lbl}</button>
              </div>
              <img src="{img_uri}" alt="{html.escape(e.event_id)}" class="kf-img" />
              <p>{html.escape(e.visual_evidence)}</p>
              {vlm_block}
            </article>
            """
        )

    vlm_section = ""
    if sess.vlm_tactical_summary:
        vlm_section = f"""
        <div class="btd-panel" style="margin-top:12px;">
          <h2>AI Coach Summary <span class="muted advanced-only">({html.escape(report.vlm_model or "VLM")})</span></h2>
          <p class="coaching-text">{html.escape(sess.vlm_tactical_summary)}</p>
        </div>
        """

    svg_html = _build_cash_svg(
        sess.timeline,
        sess.events,
        excerpt_start_sec=report.excerpt_start_sec,
        svg_id=f"cashSvg_{sess.game_index}",
    )

    filmstrips_drawer = ""
    if cards_html:
        filmstrips_drawer = f"""
        <details class="btd-panel advanced-only" style="margin-top:12px;">
          <summary style="cursor:pointer;font-weight:800;color:#ffd43b;font-size:14px;">
            Key Moment Filmstrips ({len(cards_html)}) — Click to expand visual evidence
          </summary>
          <div class="grid" style="margin-top:12px;">
            {"".join(cards_html)}
          </div>
        </details>
        """

    return f"""
    <section class="coach-deck">
      <div class="btd-panel">
        <h2>Mistakes → Root Cause → Exact Affordable Fixes</h2>
        {mistakes_body}
      </div>

      <div>
        <div class="btd-panel">
          <h2>Cash ($) Economy &amp; Round Timeline</h2>
          {svg_html}
        </div>
        <div class="btd-panel" style="margin-top:12px;">
          <h2>Next-Game Coaching Drills</h2>
          <ul class="drills">{"".join(f"<li>{html.escape(d)}</li>" for d in sc.improvement_drills[:3])}</ul>
        </div>
        {vlm_section}
      </div>
    </section>

    {filmstrips_drawer}
    """


def write_html_report(report: AnalysisReport, output_path: Path) -> Path:
    """Write a compact, BTD6-themed interactive HTML report with multi-game switcher tabs."""
    games = report.games
    if not games:
        t0 = report.timeline[0].timestamp_sec if report.timeline else 0.0
        t1 = (
            report.timeline[-1].timestamp_sec
            if report.timeline
            else report.summary.analyzed_duration_sec
        )
        games = [
            GameSession(
                game_index=1,
                label="Game 1",
                start_sec=t0,
                end_sec=t1,
                summary=report.summary,
                scorecard=report.scorecard,
                events=report.events,
                timeline=report.timeline,
                map_embedding=report.map_embedding,
                round_embeddings=report.round_embeddings,
                visual_matched_towers=report.visual_matched_towers,
                end_screen_detected=report.end_screen_detected,
                vlm_tactical_summary=report.vlm_tactical_summary,
            )
        ]

    player_html, jump_js = _build_video_player_html(report)

    game_tabs_html = ""
    if len(games) > 1:
        map_note = ""
        if games[0].map_embedding and games[1].map_embedding:
            sim = cosine_similarity(games[0].map_embedding, games[1].map_embedding)
            map_lbl = "Different Maps" if sim < 0.85 else "Same Map"
            map_note = f'<span class="map-sim-pill">{map_lbl} ({sim * 100:.0f}%)</span>'
        tab_btns = []
        for idx, g in enumerate(games):
            g_grade = _letter_grade(g.scorecard.overall_strategy_score)
            active_cls = " active" if idx == 0 else ""
            tab_btns.append(
                f'<button id="tabBtn_{g.game_index}" class="game-tab{active_cls}" '
                f'onclick="selectGame({g.game_index}, {report.excerpt_start_sec + g.start_sec:.2f}, {g.start_sec:.2f})">'
                f"{html.escape(g.label)} · <strong>{g_grade}</strong></button>"
            )
        game_tabs_html = f"""
        <div class="game-switcher">
          <div class="game-tabs">{"".join(tab_btns)}</div>
          {map_note}
        </div>
        """

    hero_cards_html = "".join(
        f'<div class="game-hero-slot" data-game="{g.game_index}" '
        f'style="display:{"block" if idx == 0 else "none"};">'
        f"{_render_game_hero_card(g, report.excerpt_start_sec)}</div>"
        for idx, g in enumerate(games)
    )

    body_panes_html = "".join(
        f'<div class="game-body-pane" data-game="{g.game_index}" '
        f'style="display:{"block" if idx == 0 else "none"};">'
        f"{_render_game_body(g, report)}</div>"
        for idx, g in enumerate(games)
    )

    games_subtitle = (
        f"{len(games)} Games Detected"
        if len(games) > 1
        else html.escape(games[0].scorecard.game_mode)
    )
    map_bg_uri = _img_to_data_uri(games[0].map_thumbnail_path) if games else None
    topbar_bg_style = (
        f"background: linear-gradient(90deg, rgba(16,37,66,0.92) 0%, rgba(22,48,89,0.78) 65%, "
        f"rgba(16,37,66,0.92) 100%), url('{map_bg_uri}') center/cover no-repeat;"
        if map_bg_uri
        else ""
    )
    brand_sprites = (
        f"{_sprite_img('dart_monkey', size=34, alt='Dart Monkey')}"
        f"{_sprite_img('moab', size=34, alt='MOAB')}"
    )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>The-Eye BTD6 Coach — {html.escape(report.title)}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Trebuchet MS", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at top, #1f4884 0%, #102542 52%, #091528 100%);
      color: #f1f5f9;
      margin: 0;
      padding: 12px 16px 24px;
      min-height: 100vh;
    }}
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1360px; margin: 0 auto; }}
    .topbar {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
      background: linear-gradient(180deg, #21457a 0%, #163059 100%);
      border: 2px solid #d49b35; border-radius: 14px; padding: 10px 16px; margin-bottom: 12px;
      box-shadow: 0 4px 0 rgba(0,0,0,0.4);
    }}
    h1 {{ margin: 0; font-size: 21px; color: #ffd43b; text-shadow: 0 2px 0 #091528; letter-spacing: 0.01em; }}
    .sub {{ color: #cbd5e1; margin: 2px 0 0; font-size: 12px; }}
    .toggle-wrap {{ display: flex; gap: 4px; background: #0c1a30; border: 2px solid #355c91; padding: 3px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #cbd5e1; padding: 5px 12px; border-radius: 999px; font-weight: 700; font-size: 12px; cursor: pointer; }}
    .mode-btn.active {{ background: linear-gradient(180deg, #6de23d 0%, #44ad1d 100%); color: #052e16; box-shadow: 0 2px 0 #1e5c0a; }}
    .game-switcher {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
      background: #122647; border: 2px solid #355c91; border-radius: 12px; padding: 7px 12px; margin-bottom: 12px;
    }}
    .game-tabs {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .game-tab {{
      background: #0c1a30; color: #e2e8f0; border: 2px solid #355c91; padding: 6px 13px;
      border-radius: 9px; font-size: 12px; font-weight: 700; cursor: pointer;
    }}
    .game-tab.active {{
      background: linear-gradient(180deg, #ffd43b 0%, #e09f1f 100%); color: #1c1002;
      border-color: #fef08a; box-shadow: 0 2px 0 #854d0e;
    }}
    .map-sim-pill {{ background: #1e3a5f; color: #7dd3fc; border: 1px solid #38bdf8; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .hero {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 12px; margin-bottom: 12px; align-items: stretch; }}
    @media (max-width: 960px) {{ .hero {{ grid-template-columns: 1fr; }} }}
    .player-wrap {{
      background: #0c1a30; border: 2px solid #355c91; border-radius: 14px; overflow: hidden;
      box-shadow: 0 4px 0 rgba(0,0,0,0.4); display: flex; flex-direction: column;
    }}
    .video-frame {{ width: 100%; aspect-ratio: 16 / 9; border: 0; display: block; background: #000; }}
    .player-bar {{ display: flex; justify-content: space-between; align-items: center; padding: 7px 12px; font-size: 12px; gap: 8px; background: #122647; }}
    .ext-link {{ color: #ffd43b; text-decoration: none; font-weight: 700; }}
    .btd-panel {{
      background: linear-gradient(180deg, #193561 0%, #122647 100%);
      border: 2px solid #355c91; border-radius: 14px; padding: 12px 14px;
      box-shadow: 0 4px 0 rgba(0,0,0,0.35);
    }}
    .btd-panel h2 {{ margin: 0 0 10px; font-size: 15px; color: #ffd43b; text-shadow: 0 1px 0 #091528; }}
    .grade-hero {{ display: flex; flex-direction: column; gap: 8px; height: 100%; border-color: #d49b35; }}
    .overall-row {{ display: flex; align-items: center; gap: 12px; }}
    .big-grade {{
      width: 54px; height: 54px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
      font-size: 28px; font-weight: 900; color: #091528; flex-shrink: 0; border: 2px solid rgba(255,255,255,0.4);
      box-shadow: 0 3px 0 rgba(0,0,0,0.35);
    }}
    .score-big {{ font-size: 24px; font-weight: 900; color: #ffffff; line-height: 1.1; margin-top: 2px; }}
    .score-big span {{ font-size: 13px; color: #94a3b8; font-weight: 600; }}
    .mode-tag {{ font-size: 11px; font-weight: 700; color: #cbd5e1; text-transform: uppercase; }}
    .outcome-pill {{ color: #091528; padding: 2px 9px; border-radius: 999px; font-weight: 900; font-size: 10px; }}
    .mini-kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(78px, 1fr)); gap: 6px; }}
    .mkpi {{ background: #0c1a30; border: 1px solid #2c4e7c; border-radius: 8px; padding: 6px 8px; text-align: center; }}
    .mkpi strong {{ display: block; font-size: 14px; color: #57cc2a; }}
    .mkpi span {{ font-size: 10px; color: #94a3b8; }}
    .verdict-box {{ background: #0c1a30; border-radius: 8px; padding: 8px 10px; }}
    .verdict-head {{ font-size: 13px; font-weight: 700; color: #f8fafc; line-height: 1.35; margin-bottom: 4px; }}
    .verdict-early {{ font-size: 11.5px; color: #cbd5e1; line-height: 1.35; }}
    .pill-group {{ margin-bottom: 4px; }}
    .micro-lbl {{ display: block; font-size: 10px; font-weight: 800; color: #94a3b8; margin-bottom: 3px; letter-spacing: 0.04em; }}
    .pill-row {{ display: flex; flex-wrap: wrap; gap: 5px; }}
    .tower-pill {{ display: inline-flex; align-items: center; gap: 5px; background: #0c1a30; color: #7dd3fc; border: 1px solid #2c4e7c; padding: 2px 9px 2px 5px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .hero-pill {{ display: inline-flex; align-items: center; gap: 4px; background: #14532d; color: #bbf7d0; border: 1px solid #57cc2a; padding: 2px 8px 2px 4px; border-radius: 999px; font-size: 10.5px; font-weight: 800; }}
    .map-pill {{ background: #1e3a5f; color: #fef08a; border: 1px solid #d49b35; padding: 2px 8px; border-radius: 999px; font-size: 10.5px; font-weight: 800; }}
    .mode-rules-line {{ background: #10203a; border: 1px dashed #3b6998; border-radius: 7px; padding: 5px 9px; font-size: 11px; color: #93c5fd; line-height: 1.35; }}
    .upgrade-pill {{ background: #2e1065; color: #e9d5ff; border: 1px solid #7e22ce; padding: 2px 8px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .killer-pill {{ display: inline-flex; align-items: center; gap: 4px; background: #451a03; color: #ffd43b; border: 1px solid #b45309; padding: 2px 8px 2px 5px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
    .coach-deck {{ display: grid; grid-template-columns: 1.25fr 1fr; gap: 12px; align-items: start; }}
    @media (max-width: 960px) {{ .coach-deck {{ grid-template-columns: 1fr; }} }}
    .mistake-list {{ display: grid; grid-template-columns: 1fr; gap: 8px; }}
    .mistake-card {{ background: #0c1a30; border: 1px solid #2c4e7c; border-radius: 10px; padding: 10px 12px; }}
    .mistake-top {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-bottom: 5px; flex-wrap: wrap; }}
    .sev {{ color: #091528; font-weight: 900; font-size: 10px; padding: 2px 6px; border-radius: 4px; margin-right: 5px; }}
    .mistake-what {{ font-size: 12.5px; color: #f1f5f9; margin: 2px 0 6px; }}
    .cause-box {{ background: #231542; border-left: 3px solid #c084fc; padding: 5px 8px; border-radius: 5px; font-size: 11.5px; color: #f3e8ff; margin-bottom: 5px; line-height: 1.4; }}
    .solution-box {{ background: #0b3319; border-left: 3px solid #57cc2a; padding: 5px 8px; border-radius: 5px; font-size: 11.5px; color: #dcfce7; margin-bottom: 5px; line-height: 1.4; }}
    .mistake-fix {{ font-size: 11.5px; color: #93c5fd; margin: 2px 0 0; }}
    .jump-btn {{
      background: linear-gradient(180deg, #6de23d 0%, #44ad1d 100%); color: #052e16;
      border: 1px solid #86efac; border-radius: 6px; padding: 3px 9px; font-size: 11px;
      font-weight: 800; cursor: pointer; font-family: monospace; box-shadow: 0 2px 0 #1e5c0a;
    }}
    .jump-btn:hover {{ filter: brightness(1.08); }}
    .timeline-svg {{ width: 100%; height: auto; display: block; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 12px; }}
    .event-card {{ background: #0c1a30; border: 1px solid #2c4e7c; border-radius: 10px; padding: 10px; }}
    .event-head {{ display: flex; justify-content: space-between; align-items: center; gap: 6px; margin-bottom: 8px; flex-wrap: wrap; }}
    .badge {{ color: #091528; font-weight: 800; font-size: 11px; padding: 2px 8px; border-radius: 999px; }}
    .kf-img {{ width: 100%; border-radius: 6px; border: 1px solid #355c91; margin-bottom: 6px; }}
    .vlm-note {{ background: #172554; border-left: 3px solid #38bdf8; padding: 6px 8px; border-radius: 5px; font-size: 12px; }}
    .coaching-text {{ white-space: pre-line; line-height: 1.5; color: #e2e8f0; font-size: 12.5px; }}
    .drills {{ margin: 0; padding-left: 18px; line-height: 1.55; font-size: 12.5px; color: #e2e8f0; }}
    .muted {{ color: #94a3b8; }}
    p {{ margin: 4px 0; font-size: 12.5px; line-height: 1.45; }}
  </style>
</head>
<body class="simple-mode">
  <div class="wrap">
    <header class="topbar" style="{topbar_bg_style}">
      <div style="display:flex;align-items:center;gap:10px;">
        <div style="display:flex;align-items:center;gap:4px;">{brand_sprites}</div>
        <div>
          <h1>The-Eye · Bloons TD 6 Strategy Coach</h1>
          <p class="sub">{html.escape(report.title)} · <strong>{games_subtitle}</strong> ({report.excerpt_duration_sec:.0f}s)</p>
        </div>
      </div>
      <div class="toggle-wrap">
        <button id="btnSimple" class="mode-btn active" onclick="setMode('simple')">Player View (Simple)</button>
        <button id="btnAdv" class="mode-btn" onclick="setMode('advanced')">Coach View (Advanced)</button>
      </div>
    </header>

    {game_tabs_html}

    <section class="hero">
      {player_html}
      <div>{hero_cards_html}</div>
    </section>

    {body_panes_html}
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
    function selectGame(gameIdx, origStart, clipStart) {{
      document.querySelectorAll('.game-tab').forEach(b => b.classList.remove('active'));
      const btn = document.getElementById('tabBtn_' + gameIdx);
      if (btn) btn.classList.add('active');
      document.querySelectorAll('.game-hero-slot').forEach(el => {{
        el.style.display = el.getAttribute('data-game') === String(gameIdx) ? 'block' : 'none';
      }});
      document.querySelectorAll('.game-body-pane').forEach(el => {{
        el.style.display = el.getAttribute('data-game') === String(gameIdx) ? 'block' : 'none';
      }});
      const v = document.getElementById('localPlayer');
      if (v) v.currentTime = Math.max(0, clipStart);
    }}
    function scrubSvg(evt, svgId, svgW, padL, plotW, minT, spanT, startOff) {{
      const svg = document.getElementById(svgId);
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const viewX = ((evt.clientX - rect.left) / rect.width) * svgW;
      const norm = Math.max(0, Math.min(1, (viewX - padL) / plotW));
      const clipSec = minT + norm * spanT;
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
    """Write a BTD6-themed side-by-side HTML comparison report for two BTD6 recordings."""
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
            "<span style='background:#57cc2a;color:#052e16;padding:3px 10px;"
            "border-radius:999px;font-weight:800;font-size:11px;'>SAME MAP VERIFIED</span>"
            if emb_cmp["same_map_detected"]
            else "<span style='background:#fbbf24;color:#1c1002;padding:3px 10px;"
            "border-radius:999px;font-weight:800;font-size:11px;'>DIFFERENT MAPS</span>"
        )
        round_pills = "".join(
            f"<span style='background:#0c1a30;color:#7dd3fc;border:1px solid #2c4e7c;"
            f"padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;'>"
            f"R{item['round']}: {item['similarity'] * 100:.1f}%</span>"
            for item in emb_cmp["shared_round_similarities"]
        )
        backend_lbl = html.escape(report_a.embedding_backend or "vision-embeddings")
        emb_panel_html = f"""
        <div class="panel" style="margin-bottom:14px;">
          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:6px;">
            <h2 style="margin:0;">Visual Embedding Similarity ({sim_pct:.1f}% Map Match)</h2>
            {same_badge}
          </div>
          <p class="advanced-only" style="color:#94a3b8;margin:0 0 6px;">Model: {backend_lbl}</p>
          <div style="display:flex;flex-wrap:wrap;gap:6px;">{round_pills or "<span style='color:#94a3b8;font-size:12px;'>No overlapping checkpoint rounds.</span>"}</div>
        </div>
        """

    core_rows = [
        ("Game Mode", sa.game_mode, sb.game_mode),
        ("Match Outcome", sa.outcome, sb.outcome),
        (
            "Overall Strategy Score",
            f"{_letter_grade(sa.overall_strategy_score)} · {sa.overall_strategy_score:.1f}/100",
            f"{_letter_grade(sb.overall_strategy_score)} · {sb.overall_strategy_score:.1f}/100",
        ),
        ("Rounds Covered", ra_span, rb_span),
        ("Lives Lost (Leaks)", str(sa.lives_lost), str(sb.lives_lost)),
        ("Total Cash Invested", f"${sa.total_cash_spent:,}", f"${sb.total_cash_spent:,}"),
        ("Upgrades / Placements", str(sa.purchase_count), str(sb.purchase_count)),
        (
            "Towers Observed",
            ", ".join(sa.towers_observed) or "n/a",
            ", ".join(sb.towers_observed) or "n/a",
        ),
        (
            "Notable Upgrades",
            ", ".join(sa.upgrades_detected) or "n/a",
            ", ".join(sb.upgrades_detected) or "n/a",
        ),
        (
            "Wiki Killer Rounds",
            ", ".join(sa.killer_rounds_seen) or "none",
            ", ".join(sb.killer_rounds_seen) or "none",
        ),
    ]
    adv_rows = [
        ("Total Cash Earned", f"${sa.total_cash_earned:,}", f"${sb.total_cash_earned:,}"),
        ("Avg Unspent Cash", f"${sa.avg_unspent_cash:,.0f}", f"${sb.avg_unspent_cash:,.0f}"),
        ("Peak Unspent Cash", f"${sa.peak_unspent_cash:,}", f"${sb.peak_unspent_cash:,}"),
        (
            "Eco Farming Used",
            "Yes" if sa.has_eco_farming else "No",
            "Yes" if sb.has_eco_farming else "No",
        ),
    ]

    rows_html = "".join(
        f"<tr><td>{html.escape(lbl)}</td><td><strong>{html.escape(va)}</strong></td>"
        f"<td><strong>{html.escape(vb)}</strong></td></tr>"
        for lbl, va, vb in core_rows
    ) + "".join(
        f"<tr class='advanced-only'><td>{html.escape(lbl)}</td><td>{html.escape(va)}</td>"
        f"<td>{html.escape(vb)}</td></tr>"
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
    body {{
      font-family: "Trebuchet MS", "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at top, #1f4884 0%, #102542 52%, #091528 100%);
      color: #f1f5f9; margin: 0; padding: 18px; min-height: 100vh;
    }}
    body.simple-mode .advanced-only {{ display: none !important; }}
    .wrap {{ max-width: 1180px; margin: 0 auto; }}
    .topbar {{
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
      background: linear-gradient(180deg, #21457a 0%, #163059 100%);
      border: 2px solid #d49b35; border-radius: 14px; padding: 10px 16px; margin-bottom: 14px;
    }}
    h1 {{ margin: 0; font-size: 21px; color: #ffd43b; text-shadow: 0 2px 0 #091528; }}
    .toggle-wrap {{ display: flex; gap: 4px; background: #0c1a30; border: 2px solid #355c91; padding: 3px; border-radius: 999px; }}
    .mode-btn {{ border: none; background: transparent; color: #cbd5e1; padding: 5px 12px; border-radius: 999px; font-weight: 700; font-size: 12px; cursor: pointer; }}
    .mode-btn.active {{ background: linear-gradient(180deg, #6de23d 0%, #44ad1d 100%); color: #052e16; }}
    table {{ width: 100%; border-collapse: collapse; background: #122647; border: 2px solid #355c91; border-radius: 12px; overflow: hidden; margin-bottom: 14px; }}
    th, td {{ padding: 9px 12px; border-bottom: 1px solid #24406b; text-align: left; font-size: 13px; }}
    th {{ background: #1c3a69; color: #ffd43b; }}
    .cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    .panel {{ background: linear-gradient(180deg, #193561 0%, #122647 100%); border: 2px solid #355c91; border-radius: 12px; padding: 12px 14px; }}
    .panel h2 {{ margin: 0 0 6px; font-size: 15px; color: #ffd43b; }}
    p, li {{ font-size: 12.5px; line-height: 1.5; white-space: pre-line; }}
  </style>
</head>
<body class="simple-mode">
  <div class="wrap">
    <div class="topbar">
      <h1>🎈 The-Eye · BTD6 Side-by-Side Comparison</h1>
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
        <h2>{html.escape(report_a.title)} — Investigation</h2>
        <p><strong>{html.escape(sa.root_cause_headline)}</strong></p>
        <ul>{drills_a}</ul>
        <p>{html.escape(report_a.vlm_tactical_summary or "")}</p>
      </div>
      <div class="panel">
        <h2>{html.escape(report_b.title)} — Investigation</h2>
        <p><strong>{html.escape(sb.root_cause_headline)}</strong></p>
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
