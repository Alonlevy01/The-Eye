"""Typer + Rich CLI for The-Eye Bloons TD 6 (BTD6) Strategy & Economy Analyzer."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from the_eye.audio import analyze_audio_cues
from the_eye.btd6_wiki import KILLER_ROUNDS
from the_eye.config import Settings
from the_eye.embeddings import compare_board_embeddings, run_visual_embedding_analysis
from the_eye.events import detect_fused_events
from the_eye.ingest import prepare_media
from the_eye.mechanics import (
    _format_affordable_fix,
    _trace_preceding_spend_drought,
    build_early_game_and_root_cause,
)
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import AnalysisReport, GameSession, MistakeItem, format_timestamp
from the_eye.vision import (
    detect_end_screen_outcome,
    extract_panel_upgrades,
    sample_video_frames,
    segment_games,
)
from the_eye.vlm import run_vlm_coaching

app = typer.Typer(
    name="the-eye",
    help=(
        "The-Eye: Bloons TD 6 (BTD6) Strategy, Economy & Wiki Checkpoint Analyzer. "
        "Accepts local video files (.mp4/.mkv/.webm/.mov/.avi) or YouTube URLs."
    ),
    no_args_is_help=True,
)
console = Console()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_").lower()
    return cleaned or "btd6_run"


@app.command()
def analyze(
    source: Annotated[
        str,
        typer.Argument(help="Local gameplay video path (.mp4/.mkv/.webm) OR YouTube URL."),
    ],
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Readable title for this BTD6 recording."),
    ] = "BTD6 Gameplay",
    start: Annotated[
        float,
        typer.Option("--start", "-s", help="Start timestamp in seconds within the source video."),
    ] = 0.0,
    duration: Annotated[
        float,
        typer.Option(
            "--duration",
            "-d",
            help="Excerpt duration in seconds (0 = analyze the FULL video from start to end).",
        ),
    ] = 0.0,
    fps: Annotated[
        float,
        typer.Option(
            "--fps",
            help="Frame sampling rate (0 = auto: 1.0 fps for <=6m clips, 0.5 fps for full games).",
        ),
    ] = 0.0,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output folder for report.json, report.html & clip.mp4."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override VLM model (e.g. ollama:qwen3.5:9b)."),
    ] = None,
    no_vlm: Annotated[
        bool,
        typer.Option("--no-vlm", help="Skip VLM coaching and run pure OpenCV + RapidOCR + Wiki."),
    ] = False,
    no_embeddings: Annotated[
        bool,
        typer.Option("--no-embeddings", help="Skip neural image-similarity embeddings pass."),
    ] = False,
) -> None:
    """Analyze a BTD6 video (local file or YouTube URL) and build interactive coaching reports."""
    target_dir = out_dir or (Path("output") / _slugify(title))
    target_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = target_dir / "keyframes"

    console.print(Panel.fit(f"[bold green]The-Eye BTD6[/] analyzing: [bold]{title}[/]"))
    cfg = Settings()

    with tempfile.TemporaryDirectory(prefix="the-eye-") as tmpdir:
        with console.status("Preparing video & extracting 16 kHz audio via FFmpeg..."):
            media = prepare_media(
                source,
                work_dir=Path(tmpdir),
                start_sec=start,
                duration_sec=duration,
            )

        effective_fps = fps
        if effective_fps <= 0.0:
            effective_fps = 0.5 if media.duration_sec > 360.0 else 1.0

        with console.status(
            f"Reading BTD6 HUD (Lives, Cash, Round & Active Tower) via RapidOCR @ "
            f"{effective_fps:.1f} fps ({media.duration_sec:.0f}s video)..."
        ):
            sampled = sample_video_frames(
                media.clip_path,
                fps=effective_fps,
                excerpt_start_sec=start,
            )

        with console.status("Extracting librosa audio cues..."):
            enriched = analyze_audio_cues(
                media.audio_path,
                [sf.metrics for sf in sampled],
                fps=effective_fps,
            )
            for sf_item, new_m in zip(sampled, enriched, strict=True):
                sf_item.metrics = new_m

        with console.status("Segmenting individual BTD6 games & running per-game analysis..."):
            game_chunks = segment_games(sampled)

        sessions: list[GameSession] = []
        embed_backend_used: str | None = None
        vlm_model_used: str | None = None

        for g_idx, chunk in enumerate(game_chunks, start=1):
            prefix = f"G{g_idx}_E" if len(game_chunks) > 1 else "E"
            with console.status(
                f"Game {g_idx}/{len(game_chunks)}: detecting leaks, upgrades & filmstrips..."
            ):
                g_events, g_summary, g_scorecard = detect_fused_events(
                    chunk,
                    fps=effective_fps,
                    keyframes_dir=keyframes_dir,
                    event_prefix=prefix,
                    title=title,
                )

            g_map_emb: list[float] = []
            g_round_embs: dict[str, list[float]] = {}
            g_visual_matched = 0
            g_end_detected = False
            if not no_embeddings and chunk:
                with console.status(
                    f"Game {g_idx}/{len(game_chunks)}: computing image-similarity embeddings..."
                ):
                    emb = run_visual_embedding_analysis(
                        chunk,
                        g_events,
                        backend=cfg.embed_backend,
                        settings=cfg,
                    )
                    embed_backend_used = emb.backend
                    g_map_emb = emb.map_embedding
                    g_round_embs = emb.round_embeddings
                    g_visual_matched = emb.visual_matched_towers
                    g_end_detected = emb.end_screen_detected

            with console.status(
                f"Game {g_idx}/{len(game_chunks)}: checking Victory/Defeat & upgrade paths..."
            ):
                g_outcome, _ = detect_end_screen_outcome(chunk)
                if g_outcome == "In Progress":
                    if (
                        g_scorecard.end_round is not None
                        and chunk[-1].metrics.max_round is not None
                        and g_scorecard.end_round >= chunk[-1].metrics.max_round
                    ):
                        g_outcome = "Victory"
                    elif (
                        g_end_detected
                        and g_scorecard.end_round is not None
                        and chunk[-1].metrics.max_round is not None
                        and g_scorecard.end_round < chunk[-1].metrics.max_round
                    ):
                        g_outcome = "Defeat"
                g_upgrades = extract_panel_upgrades(chunk)
                g_score = g_scorecard.overall_strategy_score
                g_mistakes = list(g_scorecard.mistakes)
                if "Defeat" in g_outcome:
                    g_score = round(min(48.0, max(15.0, g_score - 38.0)), 1)
                    end_r = g_scorecard.end_round
                    last_valid = next(
                        (sf.metrics for sf in reversed(chunk) if sf.metrics.cash is not None),
                        chunk[-1].metrics,
                    )
                    c_str = f"${last_valid.cash:,}" if last_valid.cash is not None else "$?"
                    r_info = KILLER_ROUNDS.get(end_r or -1)
                    fix_tip = (
                        r_info.wiki_advice
                        if r_info
                        else f"Spend your remaining {c_str} on defense upgrades before this wave."
                    )
                    purchase_evs = [e for e in g_events if e.event_type.value == "tower_purchase"]
                    root_txt = _trace_preceding_spend_drought(
                        purchase_evs, end_r, last_valid.timestamp_sec
                    )
                    aff_sol = _format_affordable_fix(
                        r_info.threat_tags if r_info else ("ceramic", "moab"),
                        last_valid.cash,
                    )
                    g_mistakes.insert(
                        0,
                        MistakeItem(
                            category=f"Defeat on Round {end_r or '?'}",
                            severity="high",
                            timestamp_sec=max(0.0, round(last_valid.timestamp_sec - 2.0, 2)),
                            original_video_sec=max(
                                0.0, round(last_valid.original_video_sec - 2.0, 2)
                            ),
                            round_num=end_r,
                            summary=(
                                f"Match ended in Defeat on Round {end_r or '?'} with {c_str} "
                                "unspent cash still in the bank."
                            ),
                            root_cause=root_txt,
                            how_to_fix=fix_tip,
                            affordable_solution=aff_sol,
                        ),
                    )
                g_scorecard = g_scorecard.model_copy(
                    update={
                        "outcome": g_outcome,
                        "upgrades_detected": g_upgrades,
                        "overall_strategy_score": g_score,
                        "mistakes": g_mistakes,
                    }
                )
                early_sum, root_head = build_early_game_and_root_cause(
                    [sf.metrics for sf in chunk], g_events, g_scorecard
                )
                g_scorecard = g_scorecard.model_copy(
                    update={
                        "early_game_summary": early_sum,
                        "root_cause_headline": root_head,
                    }
                )

            g_vlm_summary: str | None = None
            if not no_vlm and g_events:
                chosen_model = model or cfg.vlm_model
                with console.status(
                    f"Game {g_idx}/{len(game_chunks)}: running VLM review ({chosen_model})..."
                ):
                    try:
                        obs_map, g_vlm_summary, vlm_model_used = run_vlm_coaching(
                            title=f"{title} — Game {g_idx}",
                            summary=g_summary,
                            events=g_events,
                            scorecard=g_scorecard,
                            settings=cfg,
                            model_override=model,
                        )
                        for e in g_events:
                            if e.event_id in obs_map:
                                e.vlm_observation = obs_map[e.event_id]
                    except Exception as exc:
                        console.print(f"[yellow]VLM coaching skipped (Game {g_idx}):[/] {exc}")

            map_thumb_str: str | None = None
            if chunk:
                import cv2

                mid_idx = min(len(chunk) - 1, max(0, len(chunk) // 5))
                thumb_bgr = cv2.resize(chunk[mid_idx].bgr, (640, 360), interpolation=cv2.INTER_AREA)
                keyframes_dir.mkdir(parents=True, exist_ok=True)
                thumb_file = keyframes_dir / f"game_{g_idx}_map.jpg"
                cv2.imwrite(str(thumb_file), thumb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
                map_thumb_str = str(thumb_file)

            r_lbl = (
                f"R{g_scorecard.start_round}->R{g_scorecard.end_round}"
                if g_scorecard.start_round is not None
                else "Game"
            )
            sessions.append(
                GameSession(
                    game_index=g_idx,
                    label=f"Game {g_idx} ({r_lbl} · {g_scorecard.outcome})",
                    start_sec=chunk[0].metrics.timestamp_sec,
                    end_sec=chunk[-1].metrics.timestamp_sec,
                    summary=g_summary,
                    scorecard=g_scorecard,
                    events=g_events,
                    timeline=[sf.metrics for sf in chunk],
                    map_embedding=g_map_emb,
                    round_embeddings=g_round_embs,
                    visual_matched_towers=g_visual_matched,
                    end_screen_detected=g_end_detected,
                    map_thumbnail_path=map_thumb_str,
                    vlm_tactical_summary=g_vlm_summary,
                )
            )

        local_clip_rel: str | None = None
        if media.clip_path.exists():
            shutil.copy2(media.clip_path, target_dir / "clip.mp4")
            local_clip_rel = "clip.mp4"

    # Primary/featured game for top-level backward compatibility (latest game if multi-game)
    primary = sessions[-1] if sessions else None
    report = AnalysisReport(
        title=title,
        source=source,
        is_url=media.is_url,
        excerpt_start_sec=start,
        excerpt_duration_sec=media.duration_sec,
        fps=effective_fps,
        summary=primary.summary if primary else detect_fused_events([], fps=effective_fps)[1],
        scorecard=primary.scorecard if primary else detect_fused_events([], fps=effective_fps)[2],
        events=primary.events if primary else [],
        timeline=primary.timeline if primary else [],
        games=sessions,
        vlm_model=vlm_model_used,
        vlm_tactical_summary=primary.vlm_tactical_summary if primary else None,
        local_clip_path=local_clip_rel,
        embedding_backend=embed_backend_used,
        map_embedding=primary.map_embedding if primary else [],
        round_embeddings=primary.round_embeddings if primary else {},
        visual_matched_towers=primary.visual_matched_towers if primary else 0,
        end_screen_detected=primary.end_screen_detected if primary else False,
    )

    json_path = target_dir / "report.json"
    html_path = target_dir / "report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    write_html_report(report, html_path)

    if len(sessions) > 1:
        console.print(
            f"[bold cyan]Detected {len(sessions)} separate BTD6 games in this recording![/]"
        )

    for sess in sessions:
        sc = sess.scorecard
        sc_table = Table(
            title=f"BTD6 Scorecard — {title} [{sess.label}] "
            f"({format_timestamp(sess.start_sec)} - {format_timestamp(sess.end_sec)})"
        )
        sc_table.add_column("Metric", style="bold")
        sc_table.add_column("Value", style="bold green")
        r_span = f"R{sc.start_round} -> R{sc.end_round}" if sc.start_round else "?"
        sc_table.add_row("Game Mode", sc.game_mode)
        sc_table.add_row("Outcome", sc.outcome)
        sc_table.add_row("Overall Strategy Score", f"{sc.overall_strategy_score:.1f} / 100")
        sc_table.add_row("Rounds Covered", r_span)
        sc_table.add_row("Lives Lost (Leaks)", str(sc.lives_lost))
        sc_table.add_row("Total Cash Invested", f"${sc.total_cash_spent:,}")
        sc_table.add_row("Upgrades / Placements", str(sc.purchase_count))
        sc_table.add_row("Peak Unspent Cash", f"${sc.peak_unspent_cash:,}")
        sc_table.add_row("Towers Observed", ", ".join(sc.towers_observed) or "n/a")
        sc_table.add_row("Notable Upgrades", ", ".join(sc.upgrades_detected) or "n/a")
        sc_table.add_row("Wiki Killer Rounds", ", ".join(sc.killer_rounds_seen) or "none")
        sc_table.add_row(
            "Saved Filmstrips",
            str(sum(1 for e in sess.events if e.filmstrip_path)),
        )
        console.print(sc_table)

    console.print(f"[green]Saved:[/] {json_path}  |  [bold green]{html_path}[/]")


@app.command()
def compare(
    report_a: Annotated[Path, typer.Argument(help="First report.json from `the-eye analyze`.")],
    report_b: Annotated[Path, typer.Argument(help="Second report.json from `the-eye analyze`.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output path for side-by-side compare.html."),
    ] = Path("output/compare_btd6.html"),
) -> None:
    """Compare two analyzed BTD6 gameplay recordings side-by-side."""
    ra = AnalysisReport.model_validate(json.loads(report_a.read_text(encoding="utf-8")))
    rb = AnalysisReport.model_validate(json.loads(report_b.read_text(encoding="utf-8")))
    sa, sb = ra.scorecard, rb.scorecard

    table = Table(title="The-Eye · BTD6 Side-by-Side Strategy Comparison")
    table.add_column("Metric", style="bold")
    table.add_column(ra.title, style="cyan")
    table.add_column(rb.title, style="green")
    table.add_row("Game Mode", sa.game_mode, sb.game_mode)
    table.add_row("Outcome", sa.outcome, sb.outcome)
    table.add_row(
        "Overall Strategy Score",
        f"{sa.overall_strategy_score:.1f}/100",
        f"{sb.overall_strategy_score:.1f}/100",
    )
    table.add_row("Lives Lost", str(sa.lives_lost), str(sb.lives_lost))
    table.add_row("Cash Invested", f"${sa.total_cash_spent:,}", f"${sb.total_cash_spent:,}")
    table.add_row("Upgrades / Placements", str(sa.purchase_count), str(sb.purchase_count))
    table.add_row(
        "Towers Observed",
        ", ".join(sa.towers_observed) or "n/a",
        ", ".join(sb.towers_observed) or "n/a",
    )
    table.add_row(
        "Wiki Killer Rounds",
        ", ".join(sa.killer_rounds_seen) or "none",
        ", ".join(sb.killer_rounds_seen) or "none",
    )
    emb_cmp = compare_board_embeddings(
        ra.map_embedding,
        ra.round_embeddings,
        rb.map_embedding,
        rb.round_embeddings,
    )
    if emb_cmp["map_similarity"] is not None:
        sim_pct = f"{emb_cmp['map_similarity'] * 100:.1f}%"
        verdict = "Same Map" if emb_cmp["same_map_detected"] else "Different Maps"
        table.add_row("Visual Map Similarity", f"{sim_pct} ({verdict})", f"{sim_pct} ({verdict})")
    console.print(table)

    write_comparison_html(ra, rb, out)
    console.print(f"[bold green]Saved comparison report:[/] {out}")


if __name__ == "__main__":
    app()
