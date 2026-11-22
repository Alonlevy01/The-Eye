"""Typer + Rich CLI for The-Eye CS2 gameplay mechanics & coaching analyzer."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from the_eye.audio import analyze_audio_cues
from the_eye.config import Settings
from the_eye.events import detect_fused_events
from the_eye.ingest import prepare_media
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import AnalysisReport, format_timestamp
from the_eye.vision import sample_video_frames
from the_eye.vlm import run_vlm_coaching

app = typer.Typer(
    name="the-eye",
    help=(
        "The-Eye: CS2 gameplay visual, audio & mechanics coaching analyzer. "
        "Accepts local video files (.mp4/.mkv/.webm/.mov/.avi) or YouTube URLs."
    ),
    no_args_is_help=True,
)
console = Console()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_").lower()
    return cleaned or "cs2_run"


@app.command()
def analyze(
    source: Annotated[
        str,
        typer.Argument(help="Local gameplay video path (.mp4/.mkv/.webm) OR YouTube URL."),
    ],
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Readable title for this recording."),
    ] = "CS2 Gameplay Excerpt",
    start: Annotated[
        float,
        typer.Option("--start", "-s", help="Start timestamp in seconds within the source video."),
    ] = 0.0,
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Excerpt duration in seconds to analyze."),
    ] = 40.0,
    fps: Annotated[
        float,
        typer.Option("--fps", help="Frame sampling rate (default 4.0 fps = 250ms resolution)."),
    ] = 4.0,
    out_dir: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Output folder for report.json, report.html & keyframes."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Override VLM model (e.g. ollama:qwen3.5:9b)."),
    ] = None,
    no_vlm: Annotated[
        bool,
        typer.Option("--no-vlm", help="Skip VLM coaching and run pure OpenCV + librosa analysis."),
    ] = False,
) -> None:
    """Analyze a CS2 gameplay video (local file or YouTube URL) and build coaching reports."""
    target_dir = out_dir or (Path("output") / _slugify(title))
    target_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = target_dir / "keyframes"

    console.print(Panel.fit(f"[bold cyan]The-Eye[/] analyzing: [bold]{title}[/]"))

    with tempfile.TemporaryDirectory(prefix="the-eye-") as tmpdir:
        with console.status("Preparing video & extracting 16 kHz audio via FFmpeg..."):
            media = prepare_media(
                source,
                work_dir=Path(tmpdir),
                start_sec=start,
                duration_sec=duration,
            )

        with console.status(
            f"Running OpenCV CS2 HUD, radar, damage & optical-flow analysis @ {fps:.1f} fps..."
        ):
            sampled = sample_video_frames(
                media.clip_path,
                fps=fps,
                excerpt_start_sec=start,
            )

        with console.status("Extracting librosa audio combat cues..."):
            enriched = analyze_audio_cues(
                media.audio_path,
                [sf.metrics for sf in sampled],
                fps=fps,
            )
            for sf_item, new_m in zip(sampled, enriched, strict=True):
                sf_item.metrics = new_m

        with console.status(
            "Fusing events, computing CS2 Mechanics Scorecard & stitching 3-frame filmstrips..."
        ):
            events, summary, mechanics = detect_fused_events(
                sampled,
                fps=fps,
                keyframes_dir=keyframes_dir,
            )

    vlm_model_used: str | None = None
    vlm_summary: str | None = None
    if not no_vlm and events:
        cfg = Settings()
        chosen_model = model or cfg.vlm_model
        with console.status(f"Running VLM pre-duel diagnosis ({chosen_model})..."):
            try:
                obs_map, vlm_summary, vlm_model_used = run_vlm_coaching(
                    title=title,
                    summary=summary,
                    events=events,
                    mechanics=mechanics,
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
        excerpt_duration_sec=duration,
        fps=fps,
        summary=summary,
        mechanics=mechanics,
        events=events,
        timeline=[sf.metrics for sf in sampled],
        vlm_model=vlm_model_used,
        vlm_tactical_summary=vlm_summary,
    )

    json_path = target_dir / "report.json"
    html_path = target_dir / "report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    write_html_report(report, html_path)

    # Print CS2 Mechanics Scorecard
    mc_table = Table(title=f"CS2 Mechanics Scorecard — {title}")
    mc_table.add_column("Mechanic", style="bold")
    mc_table.add_column("Score", style="bold cyan")
    mc_table.add_column("Measured Signal")
    rx_str = f"{mechanics.avg_reaction_time_ms:.0f} ms" if mechanics.avg_reaction_time_ms else "n/a"
    mc_table.add_row(
        "Overall CS2 Mechanics",
        f"{mechanics.overall_mechanics_score:.1f}/100",
        "Weighted composite score",
    )
    mc_table.add_row(
        "Crosshair Discipline",
        f"{mechanics.crosshair_discipline_score:.1f}/100",
        f"Avg vertical jitter = {mechanics.avg_vertical_jitter:.2f} (lower is steadier)",
    )
    mc_table.add_row(
        "Active Counter-Strafe",
        f"{mechanics.counter_strafe_score:.1f}/100",
        f"Active={mechanics.active_counter_strafe_count} | "
        f"Static={mechanics.stationary_shot_count} | Moving={mechanics.moving_shot_count}",
    )
    mc_table.add_row(
        "Spray Control",
        f"{mechanics.spray_control_score:.1f}/100",
        f"Measured across {mechanics.spray_burst_count} sustained gunfire bursts",
    )
    mc_table.add_row(
        "Sound & Flank Aware.",
        f"{mechanics.sound_awareness_score:.1f}/100",
        f"Avg reaction = {rx_str} | Flank misses = {mechanics.flank_awareness_misses}",
    )
    mc_table.add_row(
        "Team & Utility Timing",
        f"{mechanics.team_coordination_score:.1f}/100",
        f"Trades (<=4s): {mechanics.trade_window_count} | "
        f"Flash-entry combos: {mechanics.utility_entry_combos} | "
        f"Radar rot: {mechanics.radar_rotation_activity:.1f}",
    )
    console.print(mc_table)

    # Print detected events
    table = Table(title=f"Detected Events — {title}")
    table.add_column("ID", style="bold cyan")
    table.add_column("Type", style="bold")
    table.add_column("Clip Time")
    table.add_column("Orig Video")
    table.add_column("Conf")
    table.add_column("Visual Evidence")
    for e in events:
        table.add_row(
            e.event_id,
            e.event_type.value,
            format_timestamp(e.timestamp_sec),
            format_timestamp(e.original_video_sec),
            f"{e.confidence:.2f}",
            e.visual_evidence,
        )
    console.print(table)
    console.print(
        f"[green]Saved:[/] {json_path}  |  [bold green]{html_path}[/]  "
        f"|  Keyframes & Filmstrips: {keyframes_dir}"
    )


@app.command()
def compare(
    report_a: Annotated[Path, typer.Argument(help="First report.json from `the-eye analyze`.")],
    report_b: Annotated[Path, typer.Argument(help="Second report.json from `the-eye analyze`.")],
    out: Annotated[
        Path,
        typer.Option("--out", "-o", help="Output path for side-by-side compare.html."),
    ] = Path("output/compare.html"),
) -> None:
    """Compare two analyzed CS2 gameplay recordings side-by-side."""
    ra = AnalysisReport.model_validate(json.loads(report_a.read_text(encoding="utf-8")))
    rb = AnalysisReport.model_validate(json.loads(report_b.read_text(encoding="utf-8")))
    sa, sb = ra.summary, rb.summary
    ma, mb = ra.mechanics, rb.mechanics
    rx_a = f"{ma.avg_reaction_time_ms:.0f}ms" if ma.avg_reaction_time_ms else "n/a"
    rx_b = f"{mb.avg_reaction_time_ms:.0f}ms" if mb.avg_reaction_time_ms else "n/a"

    table = Table(title="The-Eye · Side-by-Side CS2 Mechanics Comparison")
    table.add_column("CS2 Metric", style="bold")
    table.add_column(ra.title, style="cyan")
    table.add_column(rb.title, style="magenta")
    table.add_row(
        "Overall Mechanics",
        f"{ma.overall_mechanics_score:.1f}/100",
        f"{mb.overall_mechanics_score:.1f}/100",
    )
    table.add_row(
        "Crosshair Discipline",
        f"{ma.crosshair_discipline_score:.1f} (vJit={ma.avg_vertical_jitter:.2f})",
        f"{mb.crosshair_discipline_score:.1f} (vJit={mb.avg_vertical_jitter:.2f})",
    )
    table.add_row(
        "Active Counter-Strafe",
        f"{ma.counter_strafe_score:.1f} "
        f"({ma.active_counter_strafe_count}/{ma.stationary_shot_count}/{ma.moving_shot_count})",
        f"{mb.counter_strafe_score:.1f} "
        f"({mb.active_counter_strafe_count}/{mb.stationary_shot_count}/{mb.moving_shot_count})",
    )
    table.add_row(
        "Spray Control",
        f"{ma.spray_control_score:.1f} ({ma.spray_burst_count} sprays)",
        f"{mb.spray_control_score:.1f} ({mb.spray_burst_count} sprays)",
    )
    table.add_row(
        "Sound Awareness (Rx)",
        f"{ma.sound_awareness_score:.1f} ({rx_a})",
        f"{mb.sound_awareness_score:.1f} ({rx_b})",
    )
    table.add_row(
        "Team & Utility Timing",
        f"{ma.team_coordination_score:.1f} (trades={ma.trade_window_count})",
        f"{mb.team_coordination_score:.1f} (trades={mb.trade_window_count})",
    )
    table.add_row("Events / Min", f"{sa.engagements_per_min:.2f}", f"{sb.engagements_per_min:.2f}")
    table.add_row("Killfeed Events", str(sa.killfeed_events), str(sb.killfeed_events))
    table.add_row("Damage Flashes", str(sa.damage_events), str(sb.damage_events))
    table.add_row("Audio Spikes", str(sa.audio_spike_count), str(sb.audio_spike_count))

    console.print(table)
    write_comparison_html(ra, rb, out)
    console.print(f"[bold green]Saved comparison report:[/] {out}")


if __name__ == "__main__":
    app()
