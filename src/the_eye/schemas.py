"""Pydantic v2 data contracts for CS2 frame metrics, detected events, and analysis reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Visual/audio CS2 gameplay event categories detected by The-Eye."""

    FLASHBANG = "flashbang"
    KILLFEED_EVENT = "killfeed_event"
    SCOPED_DUEL = "scoped_duel"
    COMBAT_BURST = "combat_burst"


class FrameMetrics(BaseModel):
    """Per-sample visual + audio telemetry extracted from one gameplay frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    brightness: float = Field(ge=0.0, le=255.0)
    saturation: float = Field(ge=0.0, le=255.0)
    is_flashed: bool = False
    is_scoped: bool = False
    killfeed_red_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    killfeed_delta: float = Field(default=0.0, ge=0.0)
    center_motion: float = Field(default=0.0, ge=0.0)
    audio_rms: float = Field(default=0.0, ge=0.0)
    audio_onset: float = Field(default=0.0, ge=0.0)
    audio_spike: bool = False


class DetectedEvent(BaseModel):
    """One fused CS2 gameplay moment grounded in visual + audio signals."""

    event_id: str
    event_type: EventType
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    duration_sec: float = Field(default=0.5, ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: str
    audio_evidence: str
    keyframe_path: str | None = None
    vlm_observation: str | None = None


class ClipSummary(BaseModel):
    """Aggregate tactical & pacing metrics for comparing two CS2 recordings."""

    analyzed_duration_sec: float = Field(ge=0.0)
    total_frames: int = Field(ge=0)
    total_events: int = Field(ge=0)
    engagements_per_min: float = Field(ge=0.0)
    flash_count: int = Field(ge=0)
    flashed_seconds: float = Field(ge=0.0)
    killfeed_events: int = Field(ge=0)
    scoped_seconds: float = Field(ge=0.0)
    scoped_ratio: float = Field(ge=0.0, le=1.0)
    combat_bursts: int = Field(ge=0)
    audio_spike_count: int = Field(ge=0)
    avg_center_motion: float = Field(ge=0.0)


class AnalysisReport(BaseModel):
    """Complete saved analysis report for one CS2 gameplay recording."""

    title: str
    source: str
    is_url: bool = False
    excerpt_start_sec: float = Field(default=0.0, ge=0.0)
    excerpt_duration_sec: float = Field(default=0.0, ge=0.0)
    fps: float = Field(default=2.0, gt=0.0)
    summary: ClipSummary
    events: list[DetectedEvent] = Field(default_factory=list)
    timeline: list[FrameMetrics] = Field(default_factory=list)
    vlm_model: str | None = None
    vlm_tactical_summary: str | None = None


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.s for readable CLI & report display."""
    sec = max(0.0, seconds)
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:04.1f}"
