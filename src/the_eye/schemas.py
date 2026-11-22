"""Pydantic v2 data contracts for CS2 frame metrics, mechanics scorecard, events, and reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Visual/audio CS2 gameplay event categories detected by The-Eye."""

    FLASHBANG = "flashbang"
    KILLFEED_EVENT = "killfeed_event"
    SCOPED_DUEL = "scoped_duel"
    DAMAGE_TAKEN = "damage_taken"
    COMBAT_BURST = "combat_burst"


class FrameMetrics(BaseModel):
    """Per-sample visual, optical-flow, radar, and audio telemetry from one gameplay frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    brightness: float = Field(ge=0.0, le=255.0)
    saturation: float = Field(ge=0.0, le=255.0)
    is_flashed: bool = False
    is_scoped: bool = False
    is_damaged: bool = False
    damage_red_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    killfeed_red_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    killfeed_delta: float = Field(default=0.0, ge=0.0)
    center_motion: float = Field(default=0.0, ge=0.0)
    radar_motion: float = Field(default=0.0, ge=0.0)
    flow_dx: float = 0.0
    flow_dy: float = 0.0
    flow_mag: float = Field(default=0.0, ge=0.0)
    vertical_jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    audio_rms: float = Field(default=0.0, ge=0.0)
    audio_onset: float = Field(default=0.0, ge=0.0)
    audio_spike: bool = False


class DetectedEvent(BaseModel):
    """One fused CS2 gameplay moment grounded in visual + audio signals."""

    event_id: str
    event_type: EventType
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    duration_sec: float = Field(default=0.25, ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: str
    audio_evidence: str
    keyframe_path: str | None = None
    filmstrip_path: str | None = None
    vlm_observation: str | None = None


class MechanicsScorecard(BaseModel):
    """CS2-specific mechanical & tactical coaching metrics derived from video + audio."""

    crosshair_discipline_score: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_vertical_jitter: float = Field(default=0.0, ge=0.0, le=1.0)
    counter_strafe_score: float = Field(default=0.0, ge=0.0, le=100.0)
    coordinated_shot_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    sound_awareness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    avg_reaction_time_ms: float | None = None
    team_coordination_score: float = Field(default=0.0, ge=0.0, le=100.0)
    trade_window_count: int = Field(default=0, ge=0)
    utility_entry_combos: int = Field(default=0, ge=0)
    radar_rotation_activity: float = Field(default=0.0, ge=0.0)
    damage_events: int = Field(default=0, ge=0)
    overall_mechanics_score: float = Field(default=0.0, ge=0.0, le=100.0)
    improvement_drills: list[str] = Field(default_factory=list)


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
    damage_events: int = Field(default=0, ge=0)
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
    fps: float = Field(default=4.0, gt=0.0)
    summary: ClipSummary
    mechanics: MechanicsScorecard = Field(default_factory=MechanicsScorecard)
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
