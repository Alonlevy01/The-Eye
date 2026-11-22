# The-Eye

Counter-Strike 2 (CS2) gameplay visual & audio analyzer that processes recorded POV footage (local files or YouTube URLs) using **FFmpeg**, **OpenCV**, **librosa**, and optional **Vision-Language Models (local Ollama `qwen3.5:9b` or OpenRouter)** to detect key engagements, killfeed events, flashbangs, and scoped duels—and compare two recordings side-by-side.

---

## Architecture

```mermaid
flowchart LR
    Input["Local Video (.mp4/.mkv/.webm)<br/>or YouTube URL"] --> Ingest["ingest.py<br/>(yt-dlp + FFmpeg)"]
    Ingest -->|"Cut MP4 Clip"| CV["vision.py<br/>(OpenCV CS2 HUD & Screen ROIs)"]
    Ingest -->|"16 kHz Mono WAV"| Audio["audio.py<br/>(librosa RMS & Onset Spikes)"]
    CV --> Fuse["events.py<br/>(Visual + Audio Fusion)"]
    Audio --> Fuse
    Fuse -->|"Annotated Keyframe PNGs"| VLM["vlm.py<br/>(Ollama qwen3.5:9b / OpenRouter)"]
    Fuse --> Report["report.py<br/>(Standalone HTML + SVG Timeline)"]
    VLM --> Report
    Report --> Compare["the-eye compare<br/>(Side-by-Side HTML + Rich Table)"]
```

### What It Detects On Screen & In Audio
- **Killfeed Events (`killfeed_event`):** Monitors CS2's top-right Killfeed ROI (`x: 0.74..0.99`, `y: 0.04..0.28` — offset past the spectator top-bar player cards) for vivid crimson player-highlight borders (`HSV S >= 165, V >= 150` to avoid Mirage/Dust2 sandstone false positives) and sharp pixel deltas.
- **Flashbang Whiteouts (`flashbang`):** High full-frame luminance (`V >= 215`) combined with low saturation (`S <= 45`) to separate true flashbang blinds from bright sky/sand areas.
- **Sniper Scope Vignette (`scoped_duel`):** Four dark outer screen corners (`mean < 22`) paired with a bright center reticle region (`mean > 35`) for AWP/SSG08/AUG zoom windows.
- **Visual + Audio Combat Bursts (`combat_burst`):** Fuses center-crosshair motion spikes with `librosa` RMS loudness and onset bursts (gunfire, utility explosions, headshots).
- **Annotated Evidence & VLM Coaching:** Draws color-coded HUD ROI overlays and exact `MM:SS.s` clip + original video timestamps onto keyframe PNGs, then optionally sends top keyframes to a Vision-Language Model for actionable CS2 tactical coaching.

---

## Quick Start

### Prerequisites
- Python `3.12+` and [`uv`](https://docs.astral.sh/uv/)
- `ffmpeg` available on `PATH`
- Optional: [Ollama](https://ollama.com/) (`ollama pull qwen3.5:9b`) or an `OPENROUTER_API_KEY` in `.env` for VLM tactical coaching (or pass `--no-vlm` for pure OpenCV + `librosa`).

### Install
```bash
uv sync --extra dev
cp .env.example .env
```

### Analyze a CS2 Video (YouTube URL or Local File)
Supported local formats: `.mp4`, `.mkv`, `.webm`, `.mov`, `.avi` — or any public YouTube URL.

```bash
# Example 1: Pro Major CS2 POV (donk — Spirit vs MOUZ on Mirage)
uv run the-eye analyze "https://www.youtube.com/watch?v=sZBzpFwhBNY" \
  --title "Pro Major POV - donk Mirage" \
  --start 120 --duration 35 --fps 2.0 \
  --model "ollama:qwen3.5:9b" \
  --out output/pro_major_mirage

# Example 2: Amateur / Premier CS2 POV (Mirage, No Commentary)
uv run the-eye analyze "https://www.youtube.com/watch?v=ODPp5e-7vGo" \
  --title "Premier POV - Mirage" \
  --start 120 --duration 35 --fps 2.0 \
  --model "ollama:qwen3.5:9b" \
  --out output/premier_mirage

# Pure CV + Audio mode (no VLM required)
uv run the-eye analyze path/to/my_cs2_clip.mp4 --title "My Mirage Clip" --no-vlm
```

Each run writes:
- `report.json` — structured metrics, timeline, and detected events
- `report.html` — standalone interactive visual report (SVG timeline + embedded annotated keyframes + VLM coaching)
- `keyframes/*.png` — annotated OpenCV frames showing the exact HUD regions used

### Compare Two Recordings Side-by-Side
```bash
uv run the-eye compare \
  output/pro_major_mirage/report.json \
  output/premier_mirage/report.json \
  --out output/compare_mirage.html
```

---

## Included Demo Catalog ([catalog/cs2_videos.json](catalog/cs2_videos.json))

| Recording | Skill Tier | Map | Source URL | Excerpt Start | Duration |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Pro Major POV — donk (Spirit vs MOUZ)** | Pro Major | Mirage | `https://www.youtube.com/watch?v=sZBzpFwhBNY` | `02:00` (`120s`) | `35s` |
| **Premier POV — Matchmaking Mirage** | Amateur / Premier | Mirage | `https://www.youtube.com/watch?v=ODPp5e-7vGo` | `02:00` (`120s`) | `35s` |

---

## Testing & Linting

```bash
uv run ruff check src tests
uv run pytest -v
```
