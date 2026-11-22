"""BTD6 Economy, Build Order, Wiki Killer-Round Readiness & 3-Frame Filmstrip Builder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.btd6_wiki import KILLER_ROUNDS, TOWER_ROLES
from the_eye.schemas import (
    BTD6Scorecard,
    DetectedEvent,
    EventType,
    FrameMetrics,
    MistakeItem,
    format_timestamp,
)


def compute_btd6_scorecard(
    timeline: list[FrameMetrics],
    events: list[DetectedEvent],
    *,
    fps: float = 2.0,
) -> BTD6Scorecard:
    """Compute HUD-verified BTD6 economy, leak, tower coverage, and wiki checkpoint scores."""
    valid = [m for m in timeline if m.cash is not None or m.round_num is not None]
    if not valid:
        return BTD6Scorecard()

    step_sec = 1.0 / max(fps, 0.5)
    rounds = [m.round_num for m in valid if m.round_num is not None]
    max_rounds = [m.max_round for m in valid if m.max_round is not None]
    cashes = [m.cash for m in valid if m.cash is not None]
    lives_list = [m.lives for m in valid if m.lives is not None]

    start_round = rounds[0] if rounds else None
    end_round = rounds[-1] if rounds else None
    rounds_progressed = (
        max(0, end_round - start_round + 1)
        if start_round is not None and end_round is not None
        else 0
    )

    max_r = max_rounds[-1] if max_rounds else None
    if max_r == 100 and lives_list and min(lives_list) == 1 and max(lives_list) == 1:
        game_mode = "CHIMPS / Impoppable"
    elif max_r == 80:
        game_mode = "Hard (80 Rounds)"
    elif max_r == 60:
        game_mode = "Medium (60 Rounds)"
    elif max_r == 40:
        game_mode = "Easy (40 Rounds)"
    else:
        game_mode = "Standard"

    total_spent = sum(abs(m.cash_delta) for m in valid if m.cash_delta <= -100)
    total_earned = sum(m.cash_delta for m in valid if m.cash_delta > 0)
    lives_lost = sum(abs(m.lives_delta) for m in valid if m.lives_delta < 0)
    avg_cash = round(float(np.mean(cashes)), 1) if cashes else 0.0
    peak_cash = max(cashes) if cashes else 0

    towers_seen: list[str] = []
    for m in valid:
        if m.selected_tower and m.selected_tower not in towers_seen:
            towers_seen.append(m.selected_tower)

    has_eco = any("eco" in TOWER_ROLES.get(t, ()) for t in towers_seen)
    purchase_events = [e for e in events if e.event_type == EventType.TOWER_PURCHASE]
    leak_events = [e for e in events if e.event_type == EventType.BLOON_LEAK]

    # Wiki killer rounds encountered in this clip
    killer_seen: list[str] = []
    seen_r_set = sorted(set(rounds))
    for r in seen_r_set:
        if r in KILLER_ROUNDS:
            info = KILLER_ROUNDS[r]
            killer_seen.append(f"R{r}: {info.name}")

    # Build grouped, clickable MistakeItems pointing to exact clip timestamps
    mistakes: list[MistakeItem] = []

    # 1. Bloon Leaks (especially leaks while hoarding unspent cash!)
    for le in leak_events:
        cash_str = f"${le.cash_at_event:,}" if le.cash_at_event is not None else "$?"
        r_info = KILLER_ROUNDS.get(le.round_num or -1)
        fix = (
            r_info.wiki_advice
            if r_info
            else f"Spend your unspent cash ({cash_str}) on defense upgrades before this wave hits."
        )
        mistakes.append(
            MistakeItem(
                category=f"Bloon Leak (Round {le.round_num or '?'})",
                severity="high",
                timestamp_sec=max(0.0, round(le.timestamp_sec - 1.5, 2)),
                original_video_sec=max(0.0, round(le.original_video_sec - 1.5, 2)),
                round_num=le.round_num,
                summary=f"{le.visual_evidence} (Unspent cash at leak: {cash_str}).",
                how_to_fix=fix,
            )
        )

    # 2. Cash Hoarding stretches (>= $2,500 unspent for >= 10s without spending)
    hoard_run: list[FrameMetrics] = []
    for m in valid:
        if m.cash is not None and m.cash >= 2500:
            hoard_run.append(m)
        else:
            if len(hoard_run) * step_sec >= 10.0:
                dur = round(len(hoard_run) * step_sec, 1)
                peak = max(f.cash or 0 for f in hoard_run)
                r0 = hoard_run[0].round_num
                mistakes.append(
                    MistakeItem(
                        category=f"Cash Hoarding (Round {r0 or '?'})",
                        severity="medium",
                        timestamp_sec=max(0.0, round(hoard_run[0].timestamp_sec - 0.5, 2)),
                        original_video_sec=max(
                            0.0, round(hoard_run[0].original_video_sec - 0.5, 2)
                        ),
                        round_num=r0,
                        summary=f"Held ${peak:,} unspent cash for {dur:.0f}s.",
                        how_to_fix=(
                            "Unless saving for a specific Tier 4/5 upgrade, reinvest idle cash "
                            "into Eco (Farms) or upcoming killer-round counters."
                        ),
                    )
                )
            hoard_run = []

    # 3. Upcoming Killer Round Wiki Check (if we saw a killer round without a clear counter tower)
    observed_roles = {role for t in towers_seen for role in TOWER_ROLES.get(t, ())}
    for r in seen_r_set:
        if r in KILLER_ROUNDS:
            info = KILLER_ROUNDS[r]
            missing = [tag for tag in info.threat_tags if tag not in observed_roles]
            if missing and towers_seen:
                first_m = next((m for m in valid if m.round_num == r), valid[0])
                mistakes.append(
                    MistakeItem(
                        category=f"Wiki Checkpoint: R{r} {info.name}",
                        severity="medium",
                        timestamp_sec=max(0.0, round(first_m.timestamp_sec - 0.5, 2)),
                        original_video_sec=max(0.0, round(first_m.original_video_sec - 0.5, 2)),
                        round_num=r,
                        summary=(
                            f"Reached Round {r} ({info.name}); observed towers "
                            f"({', '.join(towers_seen[:4])}) didn't show explicit "
                            f"{'/'.join(missing)} counter panel."
                        ),
                        how_to_fix=info.wiki_advice,
                    )
                )

    mistakes.sort(key=lambda item: (0 if item.severity == "high" else 1, item.timestamp_sec))
    mistakes = sorted(mistakes[:8], key=lambda item: item.timestamp_sec)

    # Overall Strategy Score (0-100): zero leaks + active cash reinvestment + clean killer rounds
    leak_pen = min(45.0, lives_lost * 2.5 + len(leak_events) * 10.0)
    hoard_pen = min(20.0, sum(1 for mk in mistakes if "Hoarding" in mk.category) * 8.0)
    spend_ratio = total_spent / max(1, total_earned)
    efficiency_bonus = min(10.0, spend_ratio * 10.0)
    overall = max(15.0, min(100.0, round(88.0 - leak_pen - hoard_pen + efficiency_bonus, 1)))

    drills: list[str] = []
    if lives_lost > 0:
        drills.append(
            f"Zero-Leak Defense ({lives_lost} lives lost): Spend unspent cash before key spike "
            "rounds (R24 Camo, R28 Lead, R40 MOAB, R63 Ceramics)."
        )
    if killer_seen:
        drills.append(
            f"Wiki Checkpoints Covered ({', '.join(killer_seen[:3])}): Verify Camo + Lead + "
            "Ceramic pierce counters are online 1-2 rounds early."
        )
    if has_eco:
        drills.append(
            "Eco / Farming Detected: Balance greed vs. defense — stop farming 2 rounds before "
            "Round 40 (MOAB) and Round 63 (Ceramic Rush)."
        )
    if not drills:
        drills.append(
            f"Clean zero-leak economy (${total_spent:,} invested across {len(purchase_events)} "
            f"upgrades from Round {start_round or '?'} to {end_round or '?'})."
        )

    return BTD6Scorecard(
        overall_strategy_score=overall,
        game_mode=game_mode,
        start_round=start_round,
        end_round=end_round,
        rounds_progressed=rounds_progressed,
        lives_lost=lives_lost,
        total_cash_spent=total_spent,
        total_cash_earned=total_earned,
        avg_unspent_cash=avg_cash,
        peak_unspent_cash=peak_cash,
        purchase_count=len(purchase_events),
        has_eco_farming=has_eco,
        towers_observed=towers_seen,
        killer_rounds_seen=killer_seen,
        mistakes=mistakes,
        improvement_drills=drills,
    )


def build_moment_filmstrip(
    frames_bgr: list[np.ndarray],
    timestamps: list[float],
    rounds: list[int | None],
    cashes: list[int | None],
    *,
    event_idx: int,
    event_label: str,
    fps: float,
    output_path: Path,
) -> Path:
    """Stitch a 3-frame BTD6 strip: BEFORE (-1.5s) | MOMENT (0.0s) | AFTER (+1.0s)."""
    offsets_sec = (-1.5, 0.0, 1.0)
    stage_names = ("1. BEFORE (-1.5s)", "2. MOMENT (0.0s)", "3. AFTER (+1.0s)")
    tiles: list[np.ndarray] = []
    tile_w, tile_h = 426, 240

    for offset, stage_name in zip(offsets_sec, stage_names, strict=True):
        idx = max(0, min(len(frames_bgr) - 1, event_idx + int(round(offset * fps))))
        resized = cv2.resize(frames_bgr[idx], (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        cv2.rectangle(resized, (0, 0), (tile_w, 26), (15, 23, 42), -1)
        ts_str = format_timestamp(timestamps[idx])
        r_lbl = f"R{rounds[idx]}" if rounds[idx] is not None else "R?"
        c_lbl = f"${cashes[idx]:,}" if cashes[idx] is not None else "$?"
        cv2.putText(
            resized,
            f"{stage_name} | {ts_str} | {r_lbl} {c_lbl}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (74, 222, 128),
            1,
            cv2.LINE_AA,
        )
        tiles.append(resized)

    strip = np.hstack(tiles)
    header = np.full((28, strip.shape[1], 3), (2, 6, 23), dtype=np.uint8)
    cv2.putText(
        header,
        f"THE-EYE BTD6 | {event_label.upper()}",
        (10, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (248, 250, 252),
        1,
        cv2.LINE_AA,
    )
    full = np.vstack([header, strip])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), full)
    return output_path
