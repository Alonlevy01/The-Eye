"""Vision-Language Model (VLM) BTD6 Strategy Coach reviewing 3-frame build-order filmstrips."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import cv2
import httpx

from the_eye.config import Settings
from the_eye.schemas import BTD6Scorecard, ClipSummary, DetectedEvent, format_timestamp

VLM_SYSTEM_PROMPT = """You are a veteran Bloons TD 6 (BTD6) CHIMPS strategist reviewing 3-frame
gameplay filmstrips (1. BEFORE -1.5s | 2. MOMENT 0.0s | 3. AFTER +1.0s) paired with HUD-verified
Round, Cash ($), Lives, and Observed Towers.

Return STRICT JSON ONLY (no markdown fences, no extra prose):
{
  "event_observations": {
    "E1": "1-2 concise sentences evaluating the tower placement, upgrade timing, or round threat.",
    "E2": "..."
  },
  "tactical_summary": "3-4 actionable BTD6 strategy takeaways (placement synergy, eco vs defense,
   and preparing for upcoming killer rounds like R40/R63/R90)."
}

RULES:
1. Reference ONLY towers, upgrades, and HUD numbers actually visible in the frames or stats.
2. Focus on real BTD6 strategy (tower range/synergy, Alch-buff placement, Camo/Lead/MOAB prep).
3. Keep every tip clear and useful for a BTD6 player."""


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
    scorecard: BTD6Scorecard | None = None,
    max_frames: int = 3,
) -> list[dict[str, Any]]:
    """Build multimodal messages preferring 3-frame moment filmstrips over single frames."""
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
        f"Clip duration: {summary.analyzed_duration_sec:.1f}s | Events: {summary.total_events}.",
    ]
    if scorecard is not None:
        meta_lines.append(
            f"HUD-Verified BTD6 Stats: mode={scorecard.game_mode}, "
            f"rounds=R{scorecard.start_round}-R{scorecard.end_round}, "
            f"lives_lost={scorecard.lives_lost}, spent=${scorecard.total_cash_spent:,}, "
            f"towers={', '.join(scorecard.towers_observed) or 'n/a'}, "
            f"killer_rounds={', '.join(scorecard.killer_rounds_seen) or 'none'}."
        )

    meta_lines.append("Selected BTD6 moments to review:")
    for e, _ in chosen:
        meta_lines.append(
            f"- {e.event_id} [{e.event_type.value}] at clip {format_timestamp(e.timestamp_sec)} "
            f"(R{e.round_num or '?'}): {e.visual_evidence}"
        )

    user_parts: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(meta_lines)}]
    for e, img_path in chosen:
        data_uri = encode_image_base64_jpeg(img_path)
        user_parts.append({"type": "text", "text": f"3-frame sequence for {e.event_id}:"})
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
    scorecard: BTD6Scorecard | None = None,
    settings: Settings | None = None,
    model_override: str | None = None,
    max_frames: int = 3,
) -> tuple[dict[str, str], str, str]:
    """Run VLM BTD6 coaching via Ollama (`ollama:<model>`) or OpenRouter."""
    cfg = settings or Settings()
    model_spec = (model_override or cfg.vlm_model).strip()
    messages = build_vlm_messages(
        title=title,
        summary=summary,
        events=events,
        scorecard=scorecard,
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
