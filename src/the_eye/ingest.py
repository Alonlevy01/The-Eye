"""Video & audio ingestion for local gameplay files and online YouTube links."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mkv", ".webm", ".mov", ".avi")


@dataclass(frozen=True)
class PreparedMedia:
    """Paths to the cut gameplay clip and its extracted 16 kHz mono WAV track."""

    clip_path: Path
    audio_path: Path
    is_url: bool
    source: str
    start_sec: float
    duration_sec: float


def is_url(source: str) -> bool:
    parsed = urlparse(source.strip())
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def _require_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "FFmpeg was not found on PATH. Install FFmpeg to extract frames and audio."
        )
    return path


def _run_cmd(cmd: list[str], *, label: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-400:]
        raise RuntimeError(f"{label} failed (exit {proc.returncode}): {detail}")


def prepare_media(
    source: str,
    *,
    work_dir: Path,
    start_sec: float = 0.0,
    duration_sec: float = 45.0,
) -> PreparedMedia:
    """Resolve a local file or YouTube URL into a cut MP4 clip and 16 kHz mono WAV audio."""
    ffmpeg_bin = _require_ffmpeg()
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_path = work_dir / "excerpt.mp4"
    audio_path = work_dir / "audio.wav"
    end_sec = start_sec + duration_sec
    source_is_url = is_url(source)

    if source_is_url:
        raw_download = work_dir / "downloaded.mp4"
        yt_dlp_bin = shutil.which("yt-dlp")
        yt_cmd_prefix = [yt_dlp_bin] if yt_dlp_bin else ["python", "-m", "yt_dlp"]
        _run_cmd(
            [
                *yt_cmd_prefix,
                "--no-playlist",
                "--quiet",
                "--no-warnings",
                "-f",
                "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720][ext=mp4]/b[height<=720]/best",
                "--merge-output-format",
                "mp4",
                "--download-sections",
                f"*{start_sec:.2f}-{end_sec:.2f}",
                "--force-keyframes-at-cuts",
                "-o",
                str(raw_download),
                source.strip(),
            ],
            label="yt-dlp download",
        )
        if not raw_download.exists():
            matches = sorted(work_dir.glob("downloaded.*"))
            if not matches:
                raise RuntimeError("yt-dlp completed but no video file was produced.")
            raw_download = matches[0]

        # Normalize to exact duration & clean MP4 container
        _run_cmd(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(raw_download),
                "-t",
                f"{duration_sec:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                str(clip_path),
            ],
            label="ffmpeg normalize",
        )
    else:
        local_path = Path(source).expanduser().resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Gameplay video not found: {local_path}")
        if local_path.suffix.lower() not in SUPPORTED_VIDEO_EXTS:
            raise ValueError(
                f"Unsupported video format '{local_path.suffix}'. "
                f"Supported: {', '.join(SUPPORTED_VIDEO_EXTS)}"
            )
        _run_cmd(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{start_sec:.3f}",
                "-i",
                str(local_path),
                "-t",
                f"{duration_sec:.3f}",
                "-c",
                "copy",
                str(clip_path),
            ],
            label="ffmpeg cut",
        )

    # Extract 16 kHz mono WAV for librosa combat audio analysis
    _run_cmd(
        [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(audio_path),
        ],
        label="ffmpeg audio extract",
    )

    return PreparedMedia(
        clip_path=clip_path,
        audio_path=audio_path,
        is_url=source_is_url,
        source=source.strip(),
        start_sec=start_sec,
        duration_sec=duration_sec,
    )
