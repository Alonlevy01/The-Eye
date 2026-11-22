"""Pydantic v2 data contracts for Rocket League boost, gameplay events, and coaching reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Visual/audio Rocket League gameplay events detected by The-Eye."""

    BOOST_STARVED = "boost_starved"
    PANIC_BURNOUT = "panic_burnout"
    BIG_PAD_PICKUP = "big_pad_pickup"
    OVER_CAPPED = "over_capped"
    GOAL_EXPLOSION = "goal_explosion"


class FrameMetrics(BaseModel):
    """Per-sample Rocket League visual + audio telemetry extracted from one frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    hud_visible: bool = True
    boost_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    boost_delta: float = 0.0
    brightness: float = Field(default=0.0, ge=0.0, le=255.0)
    saturation: float = Field(default=0.0, ge=0.0, le=255.0)
    field_motion: float = Field(default=0.0, ge=0.0)
    audio_rms: float = Field(default=0.0, ge=0.0)
    audio_onset: float = Field(default=0.0, ge=0.0)
    audio_spike: bool = False


class DetectedEvent(BaseModel):
    """One timestamped Rocket League gameplay moment grounded in visual + audio signals."""

    event_id: str
    event_type: EventType
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    duration_sec: float = Field(default=0.5, ge=0.0)
    boost_at_event: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: str
    audio_evidence: str
    keyframe_path: str | None = None
    filmstrip_path: str | None = None
    vlm_observation: str | None = None


class MistakeItem(BaseModel):
    """One grouped, timestamped Rocket League boost/positioning mistake with an actionable fix."""

    category: str
    severity: str = "medium"  # "high" | "medium" | "low"
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    duration_sec: float = Field(default=0.0, ge=0.0)
    summary: str
    how_to_fix: str


class BoostScorecard(BaseModel):
    """100% visually ground-truth Rocket League boost discipline & pacing scorecard."""

    overall_boost_score: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_boost: float = Field(default=0.0, ge=0.0, le=100.0)
    boost_spent_per_min: float = Field(default=0.0, ge=0.0)
    starved_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    starved_seconds: float = Field(default=0.0, ge=0.0)
    over_capped_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    over_capped_seconds: float = Field(default=0.0, ge=0.0)
    big_pad_pickups: int = Field(default=0, ge=0)
    small_pad_pickups: int = Field(default=0, ge=0)
    small_pad_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    panic_burnouts: int = Field(default=0, ge=0)
    avg_recovery_sec: float | None = None
    goal_explosions: int = Field(default=0, ge=0)
    avg_field_speed: float = Field(default=0.0, ge=0.0)
    live_hud_seconds: float = Field(default=0.0, ge=0.0)
    mistakes: list[MistakeItem] = Field(default_factory=list)
    improvement_drills: list[str] = Field(default_factory=list)


class ClipSummary(BaseModel):
    """Aggregate clip metadata & event counts."""

    analyzed_duration_sec: float = Field(ge=0.0)
    total_frames: int = Field(ge=0)
    live_hud_frames: int = Field(ge=0)
    total_events: int = Field(ge=0)
    audio_spike_count: int = Field(ge=0)
    avg_field_motion: float = Field(ge=0.0)


class AnalysisReport(BaseModel):
    """Complete saved analysis report for one Rocket League gameplay recording."""

    title: str
    source: str
    is_url: bool = False
    excerpt_start_sec: float = Field(default=0.0, ge=0.0)
    excerpt_duration_sec: float = Field(default=0.0, ge=0.0)
    fps: float = Field(default=2.0, gt=0.0)
    summary: ClipSummary
    scorecard: BoostScorecard = Field(default_factory=BoostScorecard)
    events: list[DetectedEvent] = Field(default_factory=list)
    timeline: list[FrameMetrics] = Field(default_factory=list)
    vlm_model: str | None = None
    vlm_tactical_summary: str | None = None
    local_clip_path: str | None = None


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.s for readable CLI & report display."""
    sec = max(0.0, seconds)
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:04.1f}"
