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
from the_eye.config import Settings
from the_eye.embeddings import compare_board_embeddings, run_visual_embedding_analysis
from the_eye.events import detect_fused_events
from the_eye.ingest import prepare_media
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import AnalysisReport, format_timestamp
from the_eye.vision import sample_video_frames
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

        with console.status(
            "Cross-referencing Bloons Wiki checkpoints & exporting 3-frame filmstrips..."
        ):
            events, summary, scorecard = detect_fused_events(
                sampled,
                fps=effective_fps,
                keyframes_dir=keyframes_dir,
            )

        embed_backend_used: str | None = None
        map_embedding: list[float] = []
        round_embeddings: dict[str, list[float]] = {}
        visual_matched_towers = 0
        end_screen_detected = False
        if not no_embeddings and sampled:
            with console.status(
                "Computing image-similarity embeddings (map fingerprint & unclicked towers)..."
            ):
                emb = run_visual_embedding_analysis(
                    sampled,
                    events,
                    backend=cfg.embed_backend,
                    settings=cfg,
                )
                embed_backend_used = emb.backend
                map_embedding = emb.map_embedding
                round_embeddings = emb.round_embeddings
                visual_matched_towers = emb.visual_matched_towers
                end_screen_detected = emb.end_screen_detected

        local_clip_rel: str | None = None
        if media.clip_path.exists():
            shutil.copy2(media.clip_path, target_dir / "clip.mp4")
            local_clip_rel = "clip.mp4"

    vlm_model_used: str | None = None
    vlm_summary: str | None = None
    if not no_vlm and events:
        chosen_model = model or cfg.vlm_model
        with console.status(f"Running VLM BTD6 strategy review ({chosen_model})..."):
            try:
                obs_map, vlm_summary, vlm_model_used = run_vlm_coaching(
                    title=title,
                    summary=summary,
                    events=events,
                    scorecard=scorecard,
                    settings=cfg,
                    model_override=model,
                )
                for e in events:
                    if e.event_id in obs_map:
                        e.vlm_observation = obs_map[e.event_id]
            except Exception as exc:
                console.print(f"[yellow]VLM coaching skipped:[/] {exc}")

    report = AnalysisReport(
        title=title,
        source=source,
        is_url=media.is_url,
        excerpt_start_sec=start,
        excerpt_duration_sec=media.duration_sec,
        fps=effective_fps,
        summary=summary,
        scorecard=scorecard,
        events=events,
        timeline=[sf.metrics for sf in sampled],
        vlm_model=vlm_model_used,
        vlm_tactical_summary=vlm_summary,
        local_clip_path=local_clip_rel,
        embedding_backend=embed_backend_used,
        map_embedding=map_embedding,
        round_embeddings=round_embeddings,
        visual_matched_towers=visual_matched_towers,
        end_screen_detected=end_screen_detected,
    )

    json_path = target_dir / "report.json"
    html_path = target_dir / "report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    write_html_report(report, html_path)

    sc_table = Table(title=f"BTD6 Strategy & Economy Scorecard — {title}")
    sc_table.add_column("Metric", style="bold")
    sc_table.add_column("Value", style="bold green")
    r_span = f"R{scorecard.start_round} -> R{scorecard.end_round}" if scorecard.start_round else "?"
    sc_table.add_row("Game Mode", scorecard.game_mode)
    sc_table.add_row("Overall Strategy Score", f"{scorecard.overall_strategy_score:.1f} / 100")
    sc_table.add_row("Rounds Covered", r_span)
    sc_table.add_row("Lives Lost (Leaks)", str(scorecard.lives_lost))
    sc_table.add_row("Total Cash Invested", f"${scorecard.total_cash_spent:,}")
    sc_table.add_row("Upgrades / Placements", str(scorecard.purchase_count))
    sc_table.add_row("Peak Unspent Cash", f"${scorecard.peak_unspent_cash:,}")
    sc_table.add_row("Towers Observed", ", ".join(scorecard.towers_observed) or "n/a")
    sc_table.add_row("Wiki Killer Rounds", ", ".join(scorecard.killer_rounds_seen) or "none")
    console.print(sc_table)

    ev_table = Table(title=f"Detected Build Order & Wiki Events ({len(events)})")
    ev_table.add_column("ID", style="bold cyan")
    ev_table.add_column("Type", style="bold")
    ev_table.add_column("Clip Time")
    ev_table.add_column("Round")
    ev_table.add_column("Visual Evidence")
    for e in events:
        ev_table.add_row(
            e.event_id,
            e.event_type.value,
            format_timestamp(e.timestamp_sec),
            f"R{e.round_num}" if e.round_num else "?",
            e.visual_evidence,
        )
    console.print(ev_table)
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
