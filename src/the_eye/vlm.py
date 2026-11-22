"""Vision-Language Model (VLM) tactical coach supporting local Ollama and OpenRouter."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2
import httpx

from the_eye.config import Settings
from the_eye.schemas import ClipSummary, DetectedEvent, format_timestamp

VLM_SYSTEM_PROMPT = """You are a Counter-Strike 2 (CS2) tactical VOD coach reviewing POV frames.
Each image comes from a timestamp detected by our OpenCV + audio pipeline (killfeed,
flashbang, sniper scope, or combat burst).

Return STRICT JSON ONLY (no markdown fences, no extra prose):
{
  "event_observations": {
    "E1": "1 sentence on what is visible (weapon, position, crosshair/utility) & detection.",
    "E2": "..."
  },
  "tactical_summary": "3-4 concise sentences on what a CS2 player can learn from this clip."
}

RULES:
1. Only describe details visible in the frames (never hallucinate unseen kills or maps).
2. If an image is heavily flashed or unclear, say so honestly.
3. Keep every observation grounded and actionable for CS2 players."""


def encode_image_base64_jpeg(image_path: Path, *, max_width: int = 768) -> str:
    """Resize and JPEG-encode a keyframe to keep VLM token usage compact and fast."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read keyframe image: {image_path}")
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode keyframe: {image_path}")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def parse_vlm_json(content: str) -> dict[str, Any]:
    """Extract a JSON object cleanly from a VLM response."""
    text = content.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    return dict(json.loads(text))


def build_vlm_messages(
    *,
    title: str,
    summary: ClipSummary,
    events: list[DetectedEvent],
    max_frames: int = 4,
) -> list[dict[str, Any]]:
    """Build OpenAI-compatible multimodal chat messages containing top keyframe images."""
    chosen = [e for e in events if e.keyframe_path and Path(e.keyframe_path).exists()][:max_frames]

    meta_lines = [
        f"Recording: {title}",
        (
            f"Clip stats: duration={summary.analyzed_duration_sec:.1f}s, "
            f"events={summary.total_events} ({summary.engagements_per_min:.1f}/min), "
            f"killfeed={summary.killfeed_events}, flashes={summary.flash_count} "
            f"({summary.flashed_seconds:.1f}s blind), scoped={summary.scoped_seconds:.1f}s "
            f"({summary.scoped_ratio * 100:.1f}%), audio_spikes={summary.audio_spike_count}."
        ),
        "Selected keyframes to inspect:",
    ]
    for e in chosen:
        meta_lines.append(
            f"- {e.event_id} [{e.event_type.value}] at clip {format_timestamp(e.timestamp_sec)} "
            f"(orig {format_timestamp(e.original_video_sec)}): "
            f"{e.visual_evidence} {e.audio_evidence}"
        )

    user_parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(meta_lines)}]
    for e in chosen:
        assert e.keyframe_path is not None
        data_uri = encode_image_base64_jpeg(Path(e.keyframe_path))
        user_parts.append({"type": "text", "text": f"Keyframe for {e.event_id}:"})
        user_parts.append({"type": "image_url", "image_url": {"url": data_uri}})

    return [
        {"role": "system", "content": VLM_SYSTEM_PROMPT},
        {"role": "user", "content": user_parts},
    ]


def run_vlm_coaching(
    *,
    title: str,
    summary: ClipSummary,
    events: list[DetectedEvent],
    settings: Settings | None = None,
    model_override: str | None = None,
    max_frames: int = 4,
) -> tuple[dict[str, str], str, str]:
    """Run VLM visual coaching via Ollama (`ollama:<model>`) or OpenRouter."""
    cfg = settings or Settings()
    model_spec = (model_override or cfg.vlm_model).strip()
    messages = build_vlm_messages(
        title=title,
        summary=summary,
        events=events,
        max_frames=max_frames,
    )

    if model_spec.startswith("ollama:"):
        actual_model = model_spec.removeprefix("ollama:").strip()
        base_url = cfg.ollama_base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
    else:
        actual_model = model_spec
        base_url = cfg.openrouter_base_url.rstrip("/")
        api_key = cfg.openrouter_api_key.get_secret_value().strip()
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured. Set it in `.env`, use "
                "`--model ollama:qwen3.5:9b`, or pass `--no-vlm`."
            )
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Alonlevy01/The-Eye",
            "X-Title": "The-Eye",
        }

    payload = {
        "model": actual_model,
        "temperature": 0.2,
        "messages": messages,
    }

    with httpx.Client(timeout=180.0) as client:
        resp = client.post(f"{base_url}/chat/completions", json=payload, headers=headers)
        if resp.status_code >= 400:
            raise RuntimeError(f"VLM endpoint returned HTTP {resp.status_code}.")
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("VLM endpoint returned no choices.")
        content = str(choices[0].get("message", {}).get("content") or "")

    parsed = parse_vlm_json(content)
    raw_obs = parsed.get("event_observations") or {}
    observations = {str(k): str(v).strip() for k, v in raw_obs.items() if v}
    tactical_summary = str(parsed.get("tactical_summary") or "").strip()
    return observations, tactical_summary, model_spec
