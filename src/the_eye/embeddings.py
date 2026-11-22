"""Multimodal image-similarity embeddings for BTD6 map matching, tower clustering & state drift.

Supports:
1. Local ONNX Vision Embeddings (`fastembed` -> `Qdrant/clip-ViT-B-32-vision`, zero API key).
2. Google `gemini-embedding-2-preview` (EmbeddingGemma 2 / Gemini Embedding 2) when
   `GEMINI_API_KEY` is configured.
3. Fast deterministic perceptual descriptor (`backend="perceptual"`) for zero-network unit tests.
"""

from __future__ import annotations

import base64
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import httpx
import numpy as np

from the_eye.config import Settings
from the_eye.schemas import DetectedEvent, EventType
from the_eye.vision import PLAYABLE_MAP_ROI, SampledFrame, _crop_norm

CHECKPOINT_ROUNDS: tuple[int, ...] = (
    1,
    10,
    20,
    24,
    28,
    33,
    40,
    50,
    59,
    63,
    70,
    76,
    80,
    90,
    95,
    100,
)

_FASTEMBED_MODEL: Any | None = None


def cosine_similarity(vec_a: list[float] | np.ndarray, vec_b: list[float] | np.ndarray) -> float:
    """Compute cosine similarity in [-1.0, 1.0] between two embedding vectors."""
    a = np.asarray(vec_a, dtype=np.float32)
    b = np.asarray(vec_b, dtype=np.float32)
    if a.size == 0 or b.size == 0 or a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-8:
        return 0.0
    return round(float(np.dot(a, b) / denom), 4)


def _l2_normalize(vec: np.ndarray) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr = arr / norm
    return [round(float(x), 5) for x in arr.tolist()]


def _perceptual_embed(bgr: np.ndarray) -> list[float]:
    """Deterministic 256-dim L2-normalized spatial HSV+edge descriptor for offline tests."""
    small = cv2.resize(bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    parts: list[np.ndarray] = []
    # 2x2 spatial grid x 16-bin H, S, V histograms = 192 dims + 64-dim grayscale thumbnail
    for gy in range(2):
        for gx in range(2):
            cell = hsv[gy * 32 : (gy + 1) * 32, gx * 32 : (gx + 1) * 32]
            for ch, upper in ((0, 180), (1, 256), (2, 256)):
                hist = cv2.calcHist([cell], [ch], None, [16], [0, upper]).reshape(-1)
                parts.append(hist.astype(np.float32))
    thumb = cv2.resize(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (8, 8)).reshape(-1)
    parts.append(thumb.astype(np.float32) / 255.0)
    return _l2_normalize(np.concatenate(parts))


def _get_fastembed_model() -> Any:
    global _FASTEMBED_MODEL
    if _FASTEMBED_MODEL is None:
        from fastembed import ImageEmbedding

        _FASTEMBED_MODEL = ImageEmbedding(model_name="Qdrant/clip-ViT-B-32-vision")
    return _FASTEMBED_MODEL


class ImageEmbedder:
    """Unified image embedding interface (Gemini Embedding 2, local ONNX CLIP, or perceptual)."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        settings: Settings | None = None,
    ) -> None:
        cfg = settings or Settings()
        self._gemini_key = cfg.gemini_api_key.get_secret_value().strip()
        if backend == "auto":
            if self._gemini_key:
                self.backend = "gemini-embedding-2-preview"
            else:
                self.backend = "fastembed:clip-ViT-B-32-vision"
        else:
            self.backend = backend

    def embed_images(self, images_bgr: list[np.ndarray]) -> list[list[float]]:
        """Embed a batch of BGR images into L2-normalized float vectors."""
        if not images_bgr:
            return []

        if self.backend == "perceptual":
            return [_perceptual_embed(img) for img in images_bgr]

        if self.backend.startswith("gemini") and self._gemini_key:
            try:
                return [self._embed_gemini_single(img) for img in images_bgr]
            except Exception:
                # Fall back seamlessly to local ONNX vision embeddings
                self.backend = "fastembed:clip-ViT-B-32-vision"

        if self.backend.startswith("fastembed"):
            try:
                return self._embed_fastembed_batch(images_bgr)
            except Exception:
                self.backend = "perceptual"
                return [_perceptual_embed(img) for img in images_bgr]

        return [_perceptual_embed(img) for img in images_bgr]

    def _embed_gemini_single(self, bgr: np.ndarray) -> list[float]:
        resized = cv2.resize(bgr, (384, 216), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return _perceptual_embed(bgr)
        b64 = base64.b64encode(buf.tobytes()).decode("ascii")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"gemini-embedding-2-preview:embedContent?key={self._gemini_key}"
        )
        payload = {
            "model": "models/gemini-embedding-2-preview",
            "content": {
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                ]
            },
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        values = data.get("embedding", {}).get("values", [])
        if not values:
            raise RuntimeError("Empty embedding returned by Gemini Embedding 2.")
        return _l2_normalize(np.asarray(values, dtype=np.float32))

    def _embed_fastembed_batch(self, images_bgr: list[np.ndarray]) -> list[list[float]]:
        model = _get_fastembed_model()
        with tempfile.TemporaryDirectory(prefix="the-eye-emb-") as tmpdir:
            paths: list[str] = []
            for idx, img in enumerate(images_bgr):
                p = Path(tmpdir) / f"img_{idx}.jpg"
                cv2.imwrite(str(p), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
                paths.append(str(p))
            raw_vecs = list(model.embed(paths))
        return [_l2_normalize(np.asarray(v, dtype=np.float32)) for v in raw_vecs]


@dataclass
class VisualEmbeddingSummary:
    """Outputs of the embedding pass for one BTD6 video."""

    backend: str
    map_embedding: list[float]
    round_embeddings: dict[str, list[float]]
    visual_matched_towers: int
    end_screen_detected: bool
    end_screen_similarity: float


def _crop_changed_tower_patch(
    before_bgr: np.ndarray,
    after_bgr: np.ndarray,
    patch_size: int = 128,
) -> np.ndarray:
    """Locate the playable-map tile with maximum pixel change after a tower purchase."""
    before_map = _crop_norm(before_bgr, PLAYABLE_MAP_ROI)
    after_map = _crop_norm(after_bgr, PLAYABLE_MAP_ROI)
    diff = cv2.absdiff(
        cv2.GaussianBlur(before_map, (5, 5), 0),
        cv2.GaussianBlur(after_map, (5, 5), 0),
    )
    mag = np.mean(diff, axis=2).astype(np.float32)
    blurred = cv2.blur(mag, (31, 31))
    _, _, _, max_loc = cv2.minMaxLoc(blurred)
    cx, cy = max_loc
    h, w = after_map.shape[:2]
    half = patch_size // 2
    x1 = max(0, min(w - patch_size, cx - half))
    y1 = max(0, min(h - patch_size, cy - half))
    x2 = min(w, x1 + patch_size)
    y2 = min(h, y1 + patch_size)
    crop = after_map[y1:y2, x1:x2]
    if crop.size == 0:
        return after_map
    return crop


def run_visual_embedding_analysis(
    sampled_frames: list[SampledFrame],
    events: list[DetectedEvent],
    *,
    backend: str = "auto",
    settings: Settings | None = None,
) -> VisualEmbeddingSummary:
    """Compute map/round embeddings, detect end-screen drift, and match unclicked towers."""
    if not sampled_frames:
        return VisualEmbeddingSummary(
            backend=backend,
            map_embedding=[],
            round_embeddings={},
            visual_matched_towers=0,
            end_screen_detected=False,
            end_screen_similarity=1.0,
        )

    embedder = ImageEmbedder(backend=backend, settings=settings)

    # 1. Pick representative playable-map crops: early gameplay anchor + key checkpoint rounds
    total = len(sampled_frames)
    anchor_indices = sorted(
        {
            min(total - 1, max(0, int(total * frac)))
            for frac in (0.05, 0.12, 0.20)
        }
    )
    round_to_idx: dict[int, int] = {}
    for idx, sf in enumerate(sampled_frames):
        r = sf.metrics.round_num
        if r is not None and r not in round_to_idx and (r in CHECKPOINT_ROUNDS or r % 10 == 0):
            round_to_idx[r] = idx
    # Always include the final playable frame to detect Victory/Defeat overlay transitions
    tail_idx = total - 1

    unique_map_indices = sorted(set(anchor_indices) | set(round_to_idx.values()) | {tail_idx})
    map_crops = [_crop_norm(sampled_frames[i].bgr, PLAYABLE_MAP_ROI) for i in unique_map_indices]
    map_vecs = embedder.embed_images(map_crops)
    idx_to_vec = dict(zip(unique_map_indices, map_vecs, strict=True))

    anchor_stack = np.mean([idx_to_vec[i] for i in anchor_indices], axis=0)
    map_embedding = _l2_normalize(anchor_stack)
    round_embeddings = {str(r): idx_to_vec[i] for r, i in sorted(round_to_idx.items())}

    tail_sim = cosine_similarity(map_embedding, idx_to_vec[tail_idx])
    end_screen_detected = tail_sim < 0.78

    # 2. Unclicked tower matching via changed-patch embeddings
    ts_to_frame_idx = {sf.metrics.timestamp_sec: i for i, sf in enumerate(sampled_frames)}
    purchase_events = [e for e in events if e.event_type == EventType.TOWER_PURCHASE]
    patch_crops: list[np.ndarray] = []
    patch_events: list[DetectedEvent] = []

    for ev in purchase_events:
        f_idx = ts_to_frame_idx.get(ev.timestamp_sec)
        if f_idx is None:
            continue
        before_idx = max(0, f_idx - 1)
        crop = _crop_changed_tower_patch(sampled_frames[before_idx].bgr, sampled_frames[f_idx].bgr)
        patch_crops.append(crop)
        patch_events.append(ev)

    matched_count = 0
    if patch_crops:
        patch_vecs = embedder.embed_images(patch_crops)
        known_gallery: list[tuple[str, list[float]]] = []
        for ev, vec in zip(patch_events, patch_vecs, strict=True):
            if ev.tower_name:
                known_gallery.append((ev.tower_name, vec))

        if known_gallery:
            for ev, vec in zip(patch_events, patch_vecs, strict=True):
                if ev.tower_name:
                    continue
                best_name = ""
                best_sim = 0.0
                for cand_name, cand_vec in known_gallery:
                    sim = cosine_similarity(vec, cand_vec)
                    if sim > best_sim:
                        best_sim = sim
                        best_name = cand_name
                if best_name and best_sim >= 0.78:
                    ev.tower_name = best_name
                    ev.visual_evidence = (
                        f"{ev.visual_evidence} "
                        f"[Visual embedding matched: {best_name}, sim={best_sim:.2f}]"
                    )
                    matched_count += 1

    return VisualEmbeddingSummary(
        backend=embedder.backend,
        map_embedding=map_embedding,
        round_embeddings=round_embeddings,
        visual_matched_towers=matched_count,
        end_screen_detected=end_screen_detected,
        end_screen_similarity=tail_sim,
    )


def compare_board_embeddings(
    map_a: list[float],
    rounds_a: dict[str, list[float]],
    map_b: list[float],
    rounds_b: dict[str, list[float]],
) -> dict[str, Any]:
    """Compare two runs' visual map fingerprints and shared-round defense layouts."""
    map_sim = cosine_similarity(map_a, map_b) if (map_a and map_b) else None
    shared_rounds = sorted(
        set(rounds_a.keys()) & set(rounds_b.keys()),
        key=lambda k: int(k) if k.isdigit() else 0,
    )
    per_round = [
        {
            "round": int(r) if r.isdigit() else r,
            "similarity": cosine_similarity(rounds_a[r], rounds_b[r]),
        }
        for r in shared_rounds
    ]
    same_map = bool(map_sim is not None and map_sim >= 0.82)
    return {
        "map_similarity": map_sim,
        "same_map_detected": same_map,
        "shared_round_similarities": per_round,
    }
