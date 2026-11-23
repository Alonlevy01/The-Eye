"""Zero-network unit tests for BTD6 HUD parser, Bloons Wiki matcher, scorecard, and HTML reports."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from the_eye.btd6_wiki import KILLER_ROUNDS, match_tower_name
from the_eye.cli import app
from the_eye.events import detect_fused_events
from the_eye.mechanics import compute_btd6_scorecard
from the_eye.report import write_comparison_html, write_html_report
from the_eye.schemas import (
    AnalysisReport,
    ClipSummary,
    EventType,
    FrameMetrics,
)
from the_eye.vision import SampledFrame, parse_btd6_hud_texts
from the_eye.vlm import parse_vlm_json


def test_btd6_hud_parser_and_wiki_tower_matcher() -> None:
    # Real OCR output shape from Downstream [Hard] (left upgrade panel open)
    lives, cash, r_cur, r_max, tower = parse_btd6_hud_texts(
        [(280.0, "100"), (440.0, "$574"), (760.0, "26/80"), (80.0, "BANANAFARm")]
    )
    assert lives == 100
    assert cash == 574
    assert r_cur == 26 and r_max == 80
    assert tower == "Banana Farm"

    # Real OCR output shape from Downstream [CHIMPS] (right upgrade panel open, stylized 'L' = 1)
    lives2, cash2, r2, r2_max, tower2 = parse_btd6_hud_texts(
        [(60.0, "L"), (210.0, "$317"), (660.0, "30/100"), (890.0, "MONKeYVILLAGe")]
    )
    assert lives2 == 1
    assert cash2 == 317
    assert r2 == 30 and r2_max == 100
    assert tower2 == "Monkey Village"

    # Regression: lower-row tower pop counter (y_norm=0.78, "140") must NOT overwrite Lives ("100")
    lives3, cash3, r3, r3_max, tower3 = parse_btd6_hud_texts(
        [
            (80.0, 0.35, "BANANAFARm"),
            (110.0, 0.78, "140"),
            (280.0, 0.28, "100"),
            (440.0, 0.28, "$574"),
            (760.0, 0.28, "34/80"),
        ]
    )
    assert lives3 == 100
    assert cash3 == 574
    assert r3 == 34 and r3_max == 80
    assert tower3 == "Banana Farm"

    assert match_tower_name("ALCheMIST") == "Alchemist"
    assert 40 in KILLER_ROUNDS and "MOAB" in KILLER_ROUNDS[40].name


def test_btd6_events_leaks_purchases_wiki_and_filmstrips(tmp_path: Path) -> None:
    # Simulate Rounds 27 -> 28 (Lead) -> 40 (MOAB) with a Banana Farm purchase and a Round 40 leak
    dummy_bgr = np.full((360, 640, 3), (70, 140, 80), dtype=np.uint8)
    timeline = [
        FrameMetrics(
            frame_index=0, timestamp_sec=0.0, original_video_sec=100.0,
            lives=100, cash=1200, round_num=27, max_round=80, selected_tower="Banana Farm",
        ),
        FrameMetrics(
            frame_index=1, timestamp_sec=0.5, original_video_sec=100.5,
            lives=100, cash=550, cash_delta=-650, round_num=27, max_round=80,
            selected_tower="Banana Farm",
        ),
        FrameMetrics(
            frame_index=2, timestamp_sec=4.0, original_video_sec=104.0,
            lives=100, cash=1800, cash_delta=1250, round_num=28, max_round=80,
        ),
        FrameMetrics(
            frame_index=3, timestamp_sec=8.0, original_video_sec=108.0,
            lives=100, cash=3200, cash_delta=1400, round_num=40, max_round=80,
        ),
        FrameMetrics(
            frame_index=4, timestamp_sec=11.5, original_video_sec=111.5,
            lives=74, lives_delta=-26, cash=3200, round_num=40, max_round=80,
        ),
    ]
    sampled = [SampledFrame(metrics=m, bgr=dummy_bgr) for m in timeline]

    kf_dir = tmp_path / "keyframes"
    events, summary, sc = detect_fused_events(sampled, fps=2.0, keyframes_dir=kf_dir)
    etypes = {e.event_type for e in events}

    assert EventType.TOWER_PURCHASE in etypes
    assert EventType.KILLER_ROUND in etypes
    assert EventType.BLOON_LEAK in etypes
    assert sc.lives_lost == 26
    assert sc.total_cash_spent == 650
    assert sc.has_eco_farming is True
    assert "Banana Farm" in sc.towers_observed
    assert any("R40" in k for k in sc.killer_rounds_seen)
    assert any("Bloon Leak" in mk.category for mk in sc.mistakes)
    assert len(list(kf_dir.glob("*filmstrip*.png"))) >= 3


def test_btd6_html_report_and_cli_compare(tmp_path: Path) -> None:
    tl = [
        FrameMetrics(
            frame_index=0, timestamp_sec=0.0, original_video_sec=150.0,
            lives=1, cash=650, round_num=24, max_round=100, selected_tower="Monkey Village",
        ),
        FrameMetrics(
            frame_index=1, timestamp_sec=1.0, original_video_sec=151.0,
            lives=1, cash=320, cash_delta=-330, round_num=28, max_round=100,
        ),
    ]
    sc = compute_btd6_scorecard(tl, [], fps=1.0)
    rep_a = AnalysisReport(
        title="Downstream [CHIMPS]",
        source="https://www.youtube.com/watch?v=nhcA10kWtkA",
        is_url=True,
        excerpt_start_sec=150.0,
        excerpt_duration_sec=150.0,
        fps=2.0,
        summary=ClipSummary(
            analyzed_duration_sec=2.0,
            total_frames=2,
            ocr_valid_frames=2,
            total_events=0,
            audio_spike_count=0,
        ),
        scorecard=sc,
        events=[],
        timeline=tl,
        local_clip_path="clip.mp4",
    )
    rep_b = rep_a.model_copy(update={"title": "Downstream [Hard]"})

    html_out = write_html_report(rep_a, tmp_path / "report.html")
    html_text = html_out.read_text(encoding="utf-8")
    assert '<video id="localPlayer"' in html_text
    assert "Player View (Simple)" in html_text
    assert "Monkey Village" in html_text
    assert "R24" in html_text

    cmp_out = write_comparison_html(rep_a, rep_b, tmp_path / "compare.html")
    assert "Downstream [Hard]" in cmp_out.read_text(encoding="utf-8")

    ja, jb = tmp_path / "a.json", tmp_path / "b.json"
    ja.write_text(rep_a.model_dump_json(), encoding="utf-8")
    jb.write_text(rep_b.model_dump_json(), encoding="utf-8")
    runner = CliRunner()
    res = runner.invoke(app, ["compare", str(ja), str(jb), "--out", str(tmp_path / "c.html")])
    assert res.exit_code == 0, res.output

    parsed = parse_vlm_json('{"event_observations": {"E1": "Strong"}, "tactical_summary": "Good"}')
    assert parsed["tactical_summary"] == "Good"


def test_visual_embeddings_map_match_and_unclicked_tower() -> None:
    from the_eye.embeddings import compare_board_embeddings, run_visual_embedding_analysis

    base_map = np.full((360, 640, 3), (60, 145, 75), dtype=np.uint8)
    # Place an identical orange tower sprite twice: first clicked (known), second unclicked
    after_known = base_map.copy()
    after_known[140:200, 180:240] = (30, 180, 240)
    after_unclicked = after_known.copy()
    after_unclicked[140:200, 300:360] = (30, 180, 240)

    timeline = [
        FrameMetrics(
            frame_index=0, timestamp_sec=0.0, original_video_sec=0.0,
            lives=100, cash=1500, round_num=24, max_round=80,
        ),
        FrameMetrics(
            frame_index=1, timestamp_sec=1.0, original_video_sec=1.0,
            lives=100, cash=900, cash_delta=-600, round_num=24, max_round=80,
            selected_tower="Banana Farm",
        ),
        FrameMetrics(
            frame_index=2, timestamp_sec=5.0, original_video_sec=5.0,
            lives=100, cash=300, cash_delta=-600, round_num=29, max_round=80,
            selected_tower=None,
        ),
    ]
    sampled = [
        SampledFrame(metrics=timeline[0], bgr=base_map),
        SampledFrame(metrics=timeline[1], bgr=after_known),
        SampledFrame(metrics=timeline[2], bgr=after_unclicked),
    ]
    events, _, _ = detect_fused_events(sampled, fps=1.0, keyframes_dir=None)
    emb = run_visual_embedding_analysis(sampled, events, backend="perceptual")

    assert len(emb.map_embedding) > 0
    assert "24" in emb.round_embeddings
    assert emb.visual_matched_towers >= 1

    cmp_same = compare_board_embeddings(
        emb.map_embedding, emb.round_embeddings, emb.map_embedding, emb.round_embeddings
    )
    assert cmp_same["same_map_detected"] is True
    assert cmp_same["map_similarity"] >= 0.99


def test_end_banner_and_upgrade_name_matchers() -> None:
    from the_eye.btd6_wiki import match_upgrade_name
    from the_eye.vision import parse_end_banner_texts

    assert (
        parse_end_banner_texts(["VIGTORY", "Congratulations!Youcompleted thistrackon:Hard"])
        == "Victory"
    )
    assert parse_end_banner_texts(["DEFEAT", "TRY AGAIN"]) == "Defeat"
    assert parse_end_banner_texts(["$1,200", "30/80"]) is None

    assert match_upgrade_name("BLOONJITSU") == "Bloonjitsu (Ninja 4-x-x)"
    assert match_upgrade_name("counTeR- ESPIoNAGe") == "Counter-Espionage (Ninja x-2-x)"
    assert match_upgrade_name("Stronger Stimulant") == "Stronger Stimulant (Alch 4-x-x)"


def test_multi_game_segmentation_separates_two_matches() -> None:
    from the_eye.vision import segment_games

    dummy = np.zeros((180, 320, 3), dtype=np.uint8)
    # Game 1: R1 -> R60 (60-round Medium), then Game 2: R1 -> R63 (70-round Odyssey)
    specs = [
        (0.0, 1, 60, 150, 850),
        (10.0, 2, 60, 150, 600),
        (50.0, 40, 60, 120, 1800),
        (60.0, 59, 60, 95, 2900),
        (70.0, 60, 60, 95, 3200),
        # New game starts on a second map!
        (90.0, 1, 70, 150, 850),
        (100.0, 2, 70, 150, 400),
        (150.0, 62, 70, 150, 25000),
        (160.0, 63, 70, 150, 26680),
    ]
    sampled = [
        SampledFrame(
            metrics=FrameMetrics(
                frame_index=i,
                timestamp_sec=ts,
                original_video_sec=ts,
                round_num=r,
                max_round=mr,
                lives=lv,
                cash=c,
            ),
            bgr=dummy,
        )
        for i, (ts, r, mr, lv, c) in enumerate(specs)
    ]

    games = segment_games(sampled)
    assert len(games) == 2
    assert games[0][0].metrics.round_num == 1
    assert games[0][-1].metrics.round_num == 60
    assert games[1][0].metrics.round_num == 1
    assert games[1][-1].metrics.round_num == 63
    # Transition from Game 1 ($3200) to Game 2 ($850) must NOT create a fake negative cash_delta
    assert games[1][0].metrics.cash_delta == 0
    assert games[1][0].metrics.lives_delta == 0


def test_root_cause_investigator_and_affordable_wiki_solutions() -> None:
    from the_eye.btd6_wiki import recommend_affordable_counters
    from the_eye.mechanics import build_early_game_and_root_cause

    sols = recommend_affordable_counters(("ceramic",), available_cash=6000)
    assert sols
    assert all(s.approx_cost <= 6000 for s in sols)

    tl = [
        FrameMetrics(
            frame_index=0, timestamp_sec=0.0, original_video_sec=0.0,
            round_num=5, max_round=70, lives=150, cash=900, selected_tower="Obyn",
        ),
        FrameMetrics(
            frame_index=1, timestamp_sec=20.0, original_video_sec=20.0,
            round_num=51, max_round=70, lives=150, cash=4200, cash_delta=-1000,
        ),
        FrameMetrics(
            frame_index=2, timestamp_sec=140.0, original_video_sec=140.0,
            round_num=63, max_round=70, lives=150, cash=26680,
        ),
    ]
    sc = compute_btd6_scorecard(tl, [], fps=1.0).model_copy(update={"outcome": "Defeat"})
    early_sum, headline = build_early_game_and_root_cause(tl, [], sc)
    assert "Obyn" in early_sum
    assert "R63" in headline and "$26,680" in headline


def test_deflation_and_deliberate_saveup_not_flagged_as_hoarding() -> None:
    # 1. Deflation starts at R31 with $20,000 fixed cash -> must NOT flag Cash Hoarding
    defl_tl = [
        FrameMetrics(
            frame_index=i, timestamp_sec=float(i * 2), original_video_sec=float(i * 2),
            round_num=31 + (i // 3), max_round=60, lives=150, cash=20000 if i < 10 else 4200,
        )
        for i in range(20)
    ]
    sc_defl = compute_btd6_scorecard(defl_tl, [], fps=0.5, title="Deflation Mode on Logs")
    assert "Deflation" in sc_defl.game_mode
    assert sc_defl.map_name == "Logs"
    assert not any("Hoarding" in mk.category for mk in sc_defl.mistakes)

    # 2. Deliberate save-up ($5,000 saved for 20s then spent down to $400) -> NOT hoarding
    save_tl = [
        FrameMetrics(
            frame_index=i, timestamp_sec=float(i * 2), original_video_sec=float(i * 2),
            round_num=45, max_round=80, lives=100, cash=5000 if i < 12 else 400,
            selected_tower="Obyn" if i == 0 else None,
        )
        for i in range(16)
    ]
    sc_save = compute_btd6_scorecard(save_tl, [], fps=0.5, title="Downstream [Hard]")
    assert sc_save.hero_used == "Obyn"
    assert sc_save.map_name == "Downstream"
    assert not any("Hoarding" in mk.category for mk in sc_save.mistakes)


def test_compact_wiki_reference_grounds_vlm_prompt() -> None:
    from the_eye.btd6_wiki import build_compact_wiki_reference
    from the_eye.vlm import build_vlm_messages

    ref_std = build_compact_wiki_reference("Hard (80 Rounds)")
    assert "Crossbow Master" in ref_std
    assert "Recursive Cluster" in ref_std
    assert "Banana Farm" in ref_std
    assert len(ref_std) < 6000  # compact (~1k tokens for all 24 towers + counters)

    ref_chimps = build_compact_wiki_reference("CHIMPS / Impoppable")
    assert "Banana Farm" not in ref_chimps

    msgs = build_vlm_messages(
        title="Test",
        summary=ClipSummary(
            analyzed_duration_sec=10.0, total_frames=10, ocr_valid_frames=10,
            total_events=0, audio_spike_count=0,
        ),
        events=[],
        scorecard=compute_btd6_scorecard([], []),
    )
    user_text = msgs[1]["content"][0]["text"]
    assert "BLOONS TD 6 WIKI REFERENCE" in user_text
    assert "MOAB Mauler" in user_text
