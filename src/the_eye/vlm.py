"""Vision-Language Model (VLM) CS2 tactical coach using 3-frame pre-duel filmstrips."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2
import httpx

from the_eye.config import Settings
from the_eye.schemas import ClipSummary, DetectedEvent, MechanicsScorecard, format_timestamp

VLM_SYSTEM_PROMPT = """You are a high-Elo Counter-Strike 2 (CS2) mechanics coach reviewing 3-frame
pre-duel filmstrips (1. APPROACH -1.5s | 2. CONTACT -0.5s | 3. OUTCOME 0.0s) paired with measured
CS2 mechanics (crosshair discipline, counter-strafing, sound reaction time, trade timing).

Return STRICT JSON ONLY (no markdown fences, no extra prose):
{
  "event_observations": {
    "E1": "1-2 sentences on observable mechanics (crosshair, peek, counter-strafe, cover).",
    "E2": "..."
  },
  "tactical_summary": "3-4 actionable CS2 takeaways citing event IDs (E1, E2) & drills."
}

STRICT RULES:
1. DO NOT guess or invent map callout names unless you can literally read the callout text
   printed under the top-left radar. Focus purely on mechanics and geometry.
2. Never narrate obvious filler ('player holds gun'). Diagnose WHY the duel went well or poorly.
3. If a frame is unclear or flashed, state that honestly."""


def encode_image_base64_jpeg(image_path: Path, *, max_width: int = 960) -> str:
    """Resize and JPEG-encode a filmstrip/keyframe for compact VLM vision inference."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        img = cv2.resize(img, (max_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode image: {image_path}")
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
    mechanics: MechanicsScorecard | None = None,
    max_frames: int = 3,
) -> list[dict[str, Any]]:
    """Build multimodal messages preferring 3-frame pre-duel filmstrips over single frames."""
    chosen: list[tuple[DetectedEvent, Path]] = []
    for e in events:
        if len(chosen) >= max_frames:
            break
        if e.filmstrip_path and Path(e.filmstrip_path).exists():
            chosen.append((e, Path(e.filmstrip_path)))
        elif e.keyframe_path and Path(e.keyframe_path).exists():
            chosen.append((e, Path(e.keyframe_path)))

    meta_lines = [
        f"Recording: {title}",
        (
            f"Clip stats: duration={summary.analyzed_duration_sec:.1f}s, "
            f"events={summary.total_events} ({summary.engagements_per_min:.1f}/min), "
            f"killfeed={summary.killfeed_events}, damage={summary.damage_events}, "
            f"flashes={summary.flash_count} ({summary.flashed_seconds:.1f}s blind), "
            f"scoped={summary.scoped_seconds:.1f}s ({summary.scoped_ratio * 100:.1f}%)."
        ),
    ]
    if mechanics is not None:
        rx_ms = mechanics.avg_reaction_time_ms
        rx_str = f"{rx_ms:.0f}ms" if rx_ms else "n/a"
        meta_lines.append(
            f"Measured CS2 Mechanics: overall={mechanics.overall_mechanics_score:.1f}/100, "
            f"crosshair_discipline={mechanics.crosshair_discipline_score:.1f}/100 "
            f"(v_jitter={mechanics.avg_vertical_jitter:.2f}), "
            f"counter_strafe={mechanics.counter_strafe_score:.1f}/100, "
            f"sound_awareness={mechanics.sound_awareness_score:.1f}/100 (avg_rx={rx_str}), "
            f"trade_windows={mechanics.trade_window_count}, "
            f"utility_entry_combos={mechanics.utility_entry_combos}."
        )

    meta_lines.append("Selected pre-duel sequences to diagnose:")
    for e, _ in chosen:
        meta_lines.append(
            f"- {e.event_id} [{e.event_type.value}] at clip {format_timestamp(e.timestamp_sec)} "
            f"(orig {format_timestamp(e.original_video_sec)}): "
            f"{e.visual_evidence} {e.audio_evidence}"
        )

    user_parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(meta_lines)}]
    for e, img_path in chosen:
        data_uri = encode_image_base64_jpeg(img_path)
        user_parts.append({"type": "text", "text": f"3-frame pre-duel sequence for {e.event_id}:"})
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
    mechanics: MechanicsScorecard | None = None,
    settings: Settings | None = None,
    model_override: str | None = None,
    max_frames: int = 3,
) -> tuple[dict[str, str], str, str]:
    """Run VLM tactical coaching via Ollama (`ollama:<model>`) or OpenRouter."""
    cfg = settings or Settings()
    model_spec = (model_override or cfg.vlm_model).strip()
    messages = build_vlm_messages(
        title=title,
        summary=summary,
        events=events,
        mechanics=mechanics,
        max_frames=max_frames,
    )

    if model_spec.startswith("ollama:"):
        actual_model = model_spec.removeprefix("ollama:").strip()
        ollama_root = cfg.ollama_base_url.rstrip("/")
        if ollama_root.endswith("/v1"):
            ollama_root = ollama_root[:-3]
        native_images = [
            p["image_url"]["url"].split(",", 1)[1]
            for p in messages[1]["content"]
            if p.get("type") == "image_url"
        ]
        native_text = "\n".join(
            p["text"] for p in messages[1]["content"] if p.get("type") == "text"
        )
        ollama_payload = {
            "model": actual_model,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.2},
            "messages": [
                {"role": "system", "content": messages[0]["content"]},
                {"role": "user", "content": native_text, "images": native_images},
            ],
        }
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(f"{ollama_root}/api/chat", json=ollama_payload)
            if resp.status_code >= 400:
                raise RuntimeError(f"Ollama VLM returned HTTP {resp.status_code}.")
            data = resp.json()
            content = str((data.get("message") or {}).get("content") or "").strip()
            if not content:
                raise RuntimeError("Ollama VLM returned empty content.")
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
        payload: dict[str, Any] = {
            "model": actual_model,
            "temperature": 0.2,
            "max_tokens": 1200,
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
            msg = choices[0].get("message") or {}
            content = str(msg.get("content") or msg.get("reasoning") or "").strip()
            if not content:
                raise RuntimeError("VLM returned empty content.")

    parsed = parse_vlm_json(content)
    raw_obs = parsed.get("event_observations") or {}
    observations = {str(k): str(v).strip() for k, v in raw_obs.items() if v}
    tactical_summary = str(parsed.get("tactical_summary") or "").strip()
    return observations, tactical_summary, model_spec
