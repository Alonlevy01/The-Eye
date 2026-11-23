"""Pydantic v2 data contracts for BTD6 HUD telemetry, build order, wiki checks, and reports."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class EventType(StrEnum):
    """Visual/audio BTD6 gameplay events detected by The-Eye."""

    BLOON_LEAK = "bloon_leak"
    TOWER_PURCHASE = "tower_purchase"
    KILLER_ROUND = "killer_round"
    CASH_HOARDING = "cash_hoarding"
    DEFEAT_OR_END = "defeat_or_end"


class FrameMetrics(BaseModel):
    """Per-sample BTD6 HUD & map telemetry extracted from one video frame."""

    frame_index: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    lives: int | None = None
    cash: int | None = None
    round_num: int | None = None
    max_round: int | None = None
    selected_tower: str | None = None
    cash_delta: int = 0
    lives_delta: int = 0
    map_activity: float = Field(default=0.0, ge=0.0)
    audio_rms: float = Field(default=0.0, ge=0.0)
    audio_onset: float = Field(default=0.0, ge=0.0)
    audio_spike: bool = False


class DetectedEvent(BaseModel):
    """One timestamped BTD6 economy, leak, or killer-round moment."""

    event_id: str
    event_type: EventType
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    round_num: int | None = None
    cash_at_event: int | None = None
    lives_at_event: int | None = None
    tower_name: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    visual_evidence: str
    audio_evidence: str
    keyframe_path: str | None = None
    filmstrip_path: str | None = None
    vlm_observation: str | None = None


class MistakeItem(BaseModel):
    """One grouped, timestamped BTD6 mistake paired with root-cause investigation & Wiki fix."""

    category: str
    severity: str = "medium"  # "high" | "medium" | "low"
    timestamp_sec: float = Field(ge=0.0)
    original_video_sec: float = Field(ge=0.0)
    round_num: int | None = None
    summary: str
    root_cause: str = ""
    how_to_fix: str
    affordable_solution: str = ""


class BTD6Scorecard(BaseModel):
    """100% HUD-verified BTD6 economy, build-order & wiki checkpoint scorecard."""

    overall_strategy_score: float = Field(default=0.0, ge=0.0, le=100.0)
    game_mode: str = "Standard"
    map_name: str | None = None
    hero_used: str | None = None
    mode_rules_note: str = ""
    outcome: str = "In Progress"  # "Victory" | "Defeat" | "In Progress"
    start_round: int | None = None
    end_round: int | None = None
    rounds_progressed: int = Field(default=0, ge=0)
    lives_lost: int = Field(default=0, ge=0)
    total_cash_spent: int = Field(default=0, ge=0)
    total_cash_earned: int = Field(default=0, ge=0)
    avg_unspent_cash: float = Field(default=0.0, ge=0.0)
    peak_unspent_cash: int = Field(default=0, ge=0)
    purchase_count: int = Field(default=0, ge=0)
    has_eco_farming: bool = False
    towers_observed: list[str] = Field(default_factory=list)
    upgrades_detected: list[str] = Field(default_factory=list)
    killer_rounds_seen: list[str] = Field(default_factory=list)
    early_game_summary: str = ""
    root_cause_headline: str = ""
    mistakes: list[MistakeItem] = Field(default_factory=list)
    improvement_drills: list[str] = Field(default_factory=list)


class ClipSummary(BaseModel):
    """Aggregate clip metadata & event counts."""

    analyzed_duration_sec: float = Field(ge=0.0)
    total_frames: int = Field(ge=0)
    ocr_valid_frames: int = Field(ge=0)
    total_events: int = Field(ge=0)
    audio_spike_count: int = Field(ge=0)
    avg_map_activity: float = Field(default=0.0, ge=0.0)


class GameSession(BaseModel):
    """One individual BTD6 match segmented out of a recording."""

    game_index: int = Field(ge=1)
    label: str
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(ge=0.0)
    summary: ClipSummary
    scorecard: BTD6Scorecard = Field(default_factory=BTD6Scorecard)
    events: list[DetectedEvent] = Field(default_factory=list)
    timeline: list[FrameMetrics] = Field(default_factory=list)
    map_embedding: list[float] = Field(default_factory=list)
    round_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    visual_matched_towers: int = 0
    end_screen_detected: bool = False
    map_thumbnail_path: str | None = None
    vlm_tactical_summary: str | None = None


class AnalysisReport(BaseModel):
    """Complete saved analysis report for one BTD6 gameplay recording."""

    title: str
    source: str
    is_url: bool = False
    excerpt_start_sec: float = Field(default=0.0, ge=0.0)
    excerpt_duration_sec: float = Field(default=0.0, ge=0.0)
    fps: float = Field(default=2.0, gt=0.0)
    summary: ClipSummary
    scorecard: BTD6Scorecard = Field(default_factory=BTD6Scorecard)
    events: list[DetectedEvent] = Field(default_factory=list)
    timeline: list[FrameMetrics] = Field(default_factory=list)
    games: list[GameSession] = Field(default_factory=list)
    vlm_model: str | None = None
    vlm_tactical_summary: str | None = None
    local_clip_path: str | None = None
    embedding_backend: str | None = None
    map_embedding: list[float] = Field(default_factory=list)
    round_embeddings: dict[str, list[float]] = Field(default_factory=dict)
    visual_matched_towers: int = 0
    end_screen_detected: bool = False


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS.s for readable CLI & report display."""
    sec = max(0.0, seconds)
    mins = int(sec // 60)
    rem = sec - mins * 60
    return f"{mins:02d}:{rem:04.1f}"
