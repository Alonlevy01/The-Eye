"""Typer + Rich CLI for The-Eye Rocket League Boost & Gameplay Analyzer."""

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
from the_eye.events import detect_fused_events
from the_eye.ingest import prepare_media
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import AnalysisReport, format_timestamp
from the_eye.vision import sample_video_frames
from the_eye.vlm import run_vlm_coaching

app = typer.Typer(
    name="the-eye",
    help=(
        "The-Eye: Rocket League Boost Discipline & Gameplay Analyzer. "
        "Accepts local video files (.mp4/.mkv/.webm/.mov/.avi) or YouTube URLs."
    ),
    no_args_is_help=True,
)
console = Console()


def _slugify(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip()).strip("_").lower()
    return cleaned or "rl_run"


@app.command()
def analyze(
    source: Annotated[
        str,
        typer.Argument(help="Local gameplay video path (.mp4/.mkv/.webm) OR YouTube URL."),
    ],
    title: Annotated[
        str,
        typer.Option("--title", "-t", help="Readable title for this Rocket League recording."),
    ] = "Rocket League Gameplay",
    start: Annotated[
        float,
        typer.Option("--start", "-s", help="Start timestamp in seconds within the source video."),
    ] = 0.0,
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Excerpt duration in seconds to analyze."),
    ] = 90.0,
    fps: Annotated[
        float,
        typer.Option("--fps", help="Frame sampling rate (default 5.0 fps = 200ms resolution)."),
    ] = 5.0,
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
        typer.Option("--no-vlm", help="Skip VLM coaching and run pure OpenCV + librosa analysis."),
    ] = False,
) -> None:
    """Analyze a Rocket League video (local file or YouTube URL) and build coaching reports."""
    target_dir = out_dir or (Path("output") / _slugify(title))
    target_dir.mkdir(parents=True, exist_ok=True)
    keyframes_dir = target_dir / "keyframes"

    console.print(Panel.fit(f"[bold orange1]The-Eye[/] analyzing: [bold]{title}[/]"))

    with tempfile.TemporaryDirectory(prefix="the-eye-") as tmpdir:
        with console.status("Preparing video & extracting 16 kHz audio via FFmpeg..."):
            media = prepare_media(
                source,
                work_dir=Path(tmpdir),
                start_sec=start,
                duration_sec=duration,
            )

        with console.status(f"Reading Rocket League Boost Gauge (0-100) @ {fps:.1f} fps..."):
            sampled = sample_video_frames(
                media.clip_path,
                fps=fps,
                excerpt_start_sec=start,
            )

        with console.status("Extracting librosa audio cues..."):
            enriched = analyze_audio_cues(
                media.audio_path,
                [sf.metrics for sf in sampled],
                fps=fps,
            )
            for sf_item, new_m in zip(sampled, enriched, strict=True):
                sf_item.metrics = new_m

        with console.status("Computing Boost Scorecard & exporting annotated filmstrips..."):
            events, summary, scorecard = detect_fused_events(
                sampled,
                fps=fps,
                keyframes_dir=keyframes_dir,
            )

        local_clip_rel: str | None = None
        if media.clip_path.exists():
            shutil.copy2(media.clip_path, target_dir / "clip.mp4")
            local_clip_rel = "clip.mp4"

    vlm_model_used: str | None = None
    vlm_summary: str | None = None
    if not no_vlm and events:
        cfg = Settings()
        chosen_model = model or cfg.vlm_model
        with console.status(f"Running VLM coaching ({chosen_model})..."):
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
        excerpt_duration_sec=duration,
        fps=fps,
        summary=summary,
        scorecard=scorecard,
        events=events,
        timeline=[sf.metrics for sf in sampled],
        vlm_model=vlm_model_used,
        vlm_tactical_summary=vlm_summary,
        local_clip_path=local_clip_rel,
    )

    json_path = target_dir / "report.json"
    html_path = target_dir / "report.html"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    write_html_report(report, html_path)

    sc_table = Table(title=f"Rocket League Boost Scorecard — {title}")
    sc_table.add_column("Metric", style="bold")
    sc_table.add_column("Value", style="bold orange1")
    rec_s = f"{scorecard.avg_recovery_sec:.1f}s" if scorecard.avg_recovery_sec else "n/a"
    sc_table.add_row("Overall Boost Score", f"{scorecard.overall_boost_score:.1f} / 100")
    sc_table.add_row("Average Boost Held", f"{scorecard.avg_boost:.1f} / 100")
    sc_table.add_row("Boost Spent / Min (BPM)", f"{scorecard.boost_spent_per_min:.0f} BPM")
    sc_table.add_row(
        "Boost-Starved (<=12)",
        f"{scorecard.starved_pct:.1f}% ({scorecard.starved_seconds:.1f}s)",
    )
    sc_table.add_row(
        "Over-Capped (>=96)",
        f"{scorecard.over_capped_pct:.1f}% ({scorecard.over_capped_seconds:.1f}s)",
    )
    sc_table.add_row(
        "Small Pads / Big Pads",
        f"{scorecard.small_pad_pickups} small | {scorecard.big_pad_pickups} big",
    )
    sc_table.add_row("Panic Burnouts", str(scorecard.panic_burnouts))
    sc_table.add_row("Avg Starved Recovery", rec_s)
    console.print(sc_table)

    ev_table = Table(title=f"Detected Boost & Gameplay Events ({len(events)})")
    ev_table.add_column("ID", style="bold cyan")
    ev_table.add_column("Type", style="bold")
    ev_table.add_column("Clip Time")
    ev_table.add_column("Boost")
    ev_table.add_column("Visual Evidence")
    for e in events:
        ev_table.add_row(
            e.event_id,
            e.event_type.value,
            format_timestamp(e.timestamp_sec),
            f"{e.boost_at_event:.0f}",
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
    ] = Path("output/compare_rl.html"),
) -> None:
    """Compare two analyzed Rocket League gameplay recordings side-by-side."""
    ra = AnalysisReport.model_validate(json.loads(report_a.read_text(encoding="utf-8")))
    rb = AnalysisReport.model_validate(json.loads(report_b.read_text(encoding="utf-8")))
    sa, sb = ra.scorecard, rb.scorecard

    table = Table(title="The-Eye · Rocket League Side-by-Side Comparison")
    table.add_column("Metric", style="bold")
    table.add_column(ra.title, style="cyan")
    table.add_column(rb.title, style="orange1")
    table.add_row(
        "Overall Boost Score",
        f"{sa.overall_boost_score:.1f}/100",
        f"{sb.overall_boost_score:.1f}/100",
    )
    table.add_row("Average Boost Held", f"{sa.avg_boost:.1f}", f"{sb.avg_boost:.1f}")
    table.add_row(
        "Boost-Starved (<=12)",
        f"{sa.starved_pct:.1f}% ({sa.starved_seconds:.1f}s)",
        f"{sb.starved_pct:.1f}% ({sb.starved_seconds:.1f}s)",
    )
    table.add_row(
        "Over-Capped (>=96)",
        f"{sa.over_capped_pct:.1f}% ({sa.over_capped_seconds:.1f}s)",
        f"{sb.over_capped_pct:.1f}% ({sb.over_capped_seconds:.1f}s)",
    )
    table.add_row(
        "Small / Big Pads",
        f"{sa.small_pad_pickups} / {sa.big_pad_pickups}",
        f"{sb.small_pad_pickups} / {sb.big_pad_pickups}",
    )
    table.add_row("Panic Burnouts", str(sa.panic_burnouts), str(sb.panic_burnouts))
    console.print(table)

    write_comparison_html(ra, rb, out)
    console.print(f"[bold green]Saved comparison report:[/] {out}")


if __name__ == "__main__":
    app()
