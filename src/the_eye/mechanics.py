"""BTD6 Economy, Build Order, Wiki Killer-Round Readiness & 3-Frame Filmstrip Builder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from the_eye.btd6_wiki import (
    HERO_NAMES,
    KILLER_ROUNDS,
    TOWER_ROLES,
    detect_map_name,
    mode_rules_summary,
    recommend_affordable_counters,
)
from the_eye.schemas import (
    BTD6Scorecard,
    DetectedEvent,
    EventType,
    FrameMetrics,
    MistakeItem,
    format_timestamp,
)


def _format_affordable_fix(threat_tags: tuple[str, ...], cash: int | None) -> str:
    """Format exact Bloons Wiki counter towers & costs matched to the player's actual cash."""
    sols = recommend_affordable_counters(threat_tags, cash, limit=2)
    parts = [f"{s.tower_path} ({s.upgrade_name}, ~${s.approx_cost:,})" for s in sols]
    if cash is not None and sols and sols[0].approx_cost <= cash:
        return f"Affordable with your ${cash:,} bank: " + " OR ".join(parts) + "."
    return "Target Wiki counters: " + " OR ".join(parts) + "."


def _trace_preceding_spend_drought(
    purchase_events: list[DetectedEvent],
    target_round: int | None,
    target_ts: float,
) -> str:
    """Investigate how long before a leak/defeat the player stopped investing cash."""
    prior = [e for e in purchase_events if e.timestamp_sec < target_ts]
    if not prior:
        return "No tower upgrades were purchased earlier in this segment before this wave hit."
    last_buy = prior[-1]
    gap_sec = max(0.0, target_ts - last_buy.timestamp_sec)
    if target_round is not None and last_buy.round_num is not None:
        r_gap = max(0, target_round - last_buy.round_num)
        if r_gap >= 4:
            return (
                f"Stopped buying upgrades for {r_gap} rounds (last spend on R{last_buy.round_num}, "
                f"{gap_sec:.0f}s earlier), leaving your defense stagnant into R{target_round}."
            )
    return (
        f"Last upgrade was on R{last_buy.round_num or '?'} ({gap_sec:.0f}s earlier); "
        "recent spend did not add enough pierce/type coverage for this wave."
    )


def build_early_game_and_root_cause(
    timeline: list[FrameMetrics],
    events: list[DetectedEvent],
    scorecard: BTD6Scorecard,
) -> tuple[str, str]:
    """Investigate R1-R25 opening decisions and connect them to the final match outcome."""
    valid = [m for m in timeline if m.cash is not None or m.round_num is not None]
    if not valid:
        return "", ""

    early_frames = [m for m in valid if m.round_num is not None and m.round_num <= 25]
    purchases = [e for e in events if e.event_type == EventType.TOWER_PURCHASE]
    early_buys = [e for e in purchases if e.round_num is not None and e.round_num <= 25]
    early_towers: list[str] = []
    for m in early_frames:
        if m.selected_tower and m.selected_tower not in early_towers:
            early_towers.append(m.selected_tower)

    early_spent = sum(abs(m.cash_delta) for m in early_frames if m.cash_delta <= -100)
    early_cashes = [m.cash for m in early_frames if m.cash is not None]
    avg_early_cash = float(np.mean(early_cashes)) if early_cashes else 0.0
    early_leaks = sum(abs(m.lives_delta) for m in early_frames if m.lives_delta < 0)

    heroes = {
        "Sauda",
        "Quincy",
        "Obyn",
        "Benjamin",
        "Etienne",
        "Geraldo",
        "Psi",
        "Adora",
        "Pat Fusty",
    }
    hero_early = next((t for t in early_towers if t in heroes), None)
    farm_early = "Banana Farm" in early_towers

    early_parts: list[str] = []
    if early_frames:
        r0 = early_frames[0].round_num or 1
        r1 = early_frames[-1].round_num or 25
        early_parts.append(
            f"R{r0}–R{r1}: Invested ${early_spent:,} across {len(early_buys)} early purchases "
            f"(avg bank ${avg_early_cash:,.0f})."
        )
        if hero_early:
            early_parts.append(f"Early Hero spotted ({hero_early}) — good XP snowball.")
        else:
            early_parts.append(
                "No early Hero panel spotted in R1–R25 (delays late-game hero levels)."
            )
        if farm_early:
            early_parts.append("Early Banana Farm eco started before R25.")
        if early_leaks > 0:
            early_parts.append(f"Leaked {early_leaks} lives before R25 (early defense too greedy).")
    else:
        early_parts.append(
            f"Clip begins at R{scorecard.start_round or '?'} (post-opening); "
            f"midgame spend was ${scorecard.total_cash_spent:,}."
        )
    early_summary = " ".join(early_parts)

    # Connect early/midgame economy directly to final outcome
    if "Defeat" in scorecard.outcome:
        last_m = next((m for m in reversed(valid) if m.cash is not None), valid[-1])
        drought = _trace_preceding_spend_drought(
            purchases, scorecard.end_round, last_m.timestamp_sec
        )
        c_str = f"${last_m.cash:,}" if last_m.cash is not None else "$?"
        end_r = scorecard.end_round or "?"
        headline = (
            f"Why You Lost (R{end_r}): {drought} "
            f"You died holding {c_str} unspent cash instead of buying a cheap R{end_r} counter."
        )
    elif scorecard.lives_lost > 0:
        first_leak = next((e for e in events if e.event_type == EventType.BLOON_LEAK), None)
        if first_leak:
            drought = _trace_preceding_spend_drought(
                purchases, first_leak.round_num, first_leak.timestamp_sec
            )
            headline = (
                f"Survived with {scorecard.lives_lost} Lives Lost: First leak hit on "
                f"R{first_leak.round_num or '?'}. {drought}"
            )
        else:
            headline = (
                f"Victory with {scorecard.lives_lost} lives leaked — tighten midgame upgrades."
            )
    else:
        headline = (
            f"Clean Zero-Leak {scorecard.outcome}: Invested ${scorecard.total_cash_spent:,} across "
            f"{scorecard.purchase_count} upgrades with solid checkpoint timing."
        )

    return early_summary, headline


def compute_btd6_scorecard(
    timeline: list[FrameMetrics],
    events: list[DetectedEvent],
    *,
    fps: float = 2.0,
    title: str = "",
) -> BTD6Scorecard:
    """Compute mode-aware, HUD-verified BTD6 economy, leak, hero, map & wiki scorecard."""
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

    total_spent = sum(abs(m.cash_delta) for m in valid if m.cash_delta <= -100)
    total_earned = sum(m.cash_delta for m in valid if m.cash_delta > 0)
    lives_lost = sum(abs(m.lives_delta) for m in valid if m.lives_delta < 0)
    avg_cash = round(float(np.mean(cashes)), 1) if cashes else 0.0
    peak_cash = max(cashes) if cashes else 0

    low_title = title.lower()
    max_r = max_rounds[-1] if max_rounds else None
    is_deflation = (
        "deflation" in low_title
        or (
            start_round is not None
            and 30 <= start_round <= 32
            and max_r == 60
            and peak_cash >= 18_000
        )
    )
    is_abr = "alternate" in low_title or " abr" in low_title or "[abr]" in low_title
    is_chimps = (
        "chimps" in low_title
        or (max_r == 100 and bool(lives_list) and min(lives_list) <= 1 and max(lives_list) <= 1)
    )

    if is_deflation:
        game_mode = "Deflation (R31–R60, Fixed $20k)"
    elif is_abr:
        game_mode = "Alternate Bloons Rounds (ABR)"
    elif is_chimps:
        game_mode = "CHIMPS / Impoppable"
    elif max_r == 80:
        game_mode = "Hard (80 Rounds)"
    elif max_r == 70:
        game_mode = "Odyssey / Custom (70 Rounds)"
    elif max_r == 60:
        game_mode = "Medium (60 Rounds)"
    elif max_r == 40:
        game_mode = "Easy (40 Rounds)"
    else:
        game_mode = "Standard"

    no_income_mode = is_deflation or is_chimps
    map_name = detect_map_name(title)
    rules_note = mode_rules_summary(game_mode)

    towers_seen: list[str] = []
    for m in valid:
        if m.selected_tower and m.selected_tower not in towers_seen:
            towers_seen.append(m.selected_tower)

    hero_used = next((t for t in towers_seen if t in HERO_NAMES), None)
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
        threat_tags = r_info.threat_tags if r_info else ("early", "ceramic")
        fix = (
            r_info.wiki_advice
            if r_info
            else f"Spend your unspent cash ({cash_str}) on defense upgrades before this wave hits."
        )
        root = _trace_preceding_spend_drought(purchase_events, le.round_num, le.timestamp_sec)
        sol = _format_affordable_fix(threat_tags, le.cash_at_event)
        mistakes.append(
            MistakeItem(
                category=f"Bloon Leak (Round {le.round_num or '?'})",
                severity="high",
                timestamp_sec=max(0.0, round(le.timestamp_sec - 1.5, 2)),
                original_video_sec=max(0.0, round(le.original_video_sec - 1.5, 2)),
                round_num=le.round_num,
                summary=f"{le.visual_evidence} (Unspent cash at leak: {cash_str}).",
                root_cause=root,
                how_to_fix=fix,
                affordable_solution=sol,
            )
        )

    # 2. Smart Cash Hoarding check:
    #    - Disabled in Deflation (fixed $20k budget at R31, $0 income).
    #    - Never penalize a deliberate Tier 4/5 save-up where the player spends >= 55% of
    #      the saved peak immediately afterward with zero leaks!
    if not is_deflation:
        hoard_indices: list[int] = []

        def _evaluate_hoard_stretch(indices: list[int]) -> None:
            if len(indices) * step_sec < 14.0:
                return
            stretch = [valid[i] for i in indices]
            dur = round(len(stretch) * step_sec, 1)
            peak = max(f.cash or 0 for f in stretch)
            t_start = stretch[0].timestamp_sec
            t_end = stretch[-1].timestamp_sec
            leaked_during = any(t_start <= le.timestamp_sec <= t_end + 3.0 for le in leak_events)

            # Check if the player immediately converted this save-up into a big upgrade!
            last_idx = indices[-1]
            ahead_end = min(len(valid), last_idx + int(round(12.0 / step_sec)) + 2)
            lookahead = valid[last_idx:ahead_end]
            min_after = min((f.cash for f in lookahead if f.cash is not None), default=peak)
            spent_after = max(0, peak - min_after)
            if not leaked_during and spent_after >= int(peak * 0.55):
                return  # Disciplined save-up for a Tier 4/5 upgrade — NOT cash hoarding!

            r0 = stretch[0].round_num
            upcoming = next(
                (kr for r_k, kr in sorted(KILLER_ROUNDS.items()) if r0 is not None and r_k >= r0),
                None,
            )
            u_tags = upcoming.threat_tags if upcoming else ("ceramic", "moab")
            up_lbl = f"R{upcoming.round_num} ({upcoming.name})" if upcoming else "midgame waves"
            fix_msg = (
                "In CHIMPS, reinvest idle cash into key support/DPS before upcoming killer rounds."
                if no_income_mode
                else "Reinvest idle cash into Eco (Banana Farms) or upcoming killer-round counters."
            )
            mistakes.append(
                MistakeItem(
                    category=f"Unspent Cash Hoarding (Round {r0 or '?'})",
                    severity="high" if leaked_during else "medium",
                    timestamp_sec=max(0.0, round(t_start - 0.5, 2)),
                    original_video_sec=max(0.0, round(stretch[0].original_video_sec - 0.5, 2)),
                    round_num=r0,
                    summary=(
                        f"Held ${peak:,} unspent for {dur:.0f}s without converting it into "
                        "a major upgrade."
                    ),
                    root_cause=(
                        f"Idle bank peaked at ${peak:,} before {up_lbl} without a Tier 4/5 "
                        "payoff purchase."
                    ),
                    how_to_fix=fix_msg,
                    affordable_solution=_format_affordable_fix(u_tags, peak),
                )
            )

        for idx_v, m in enumerate(valid):
            if m.cash is not None and m.cash >= 3000:
                hoard_indices.append(idx_v)
            else:
                _evaluate_hoard_stretch(hoard_indices)
                hoard_indices = []
        _evaluate_hoard_stretch(hoard_indices)

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
                        root_cause=(
                            f"Early/midgame build focused on {', '.join(towers_seen[:3])} without "
                            f"dedicated {'/'.join(missing)} coverage before R{r}."
                        ),
                        how_to_fix=info.wiki_advice,
                        affordable_solution=_format_affordable_fix(info.threat_tags, first_m.cash),
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
        map_name=map_name,
        hero_used=hero_used,
        mode_rules_note=rules_note,
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
