"""Bloons TD 6 (BTD6) Wiki Knowledge Base: Killer Rounds & Tower Counter Roles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KillerRoundInfo:
    """Wiki-verified BTD6 checkpoint round threat and counter advice."""

    round_num: int
    name: str
    threat_tags: tuple[str, ...]
    wiki_advice: str


# Iconic BTD6 checkpoint rounds from the Bloons Wiki
KILLER_ROUNDS: dict[int, KillerRoundInfo] = {
    24: KillerRoundInfo(
        24,
        "First Camo Bloon",
        ("camo",),
        "Round 24 sends the first Camo Green. Place Camo detection (Ninja, Sauda, 0-2-0 Village, "
        "or 3-0-0 Sub) before Round 24.",
    ),
    28: KillerRoundInfo(
        28,
        "First Lead Bloons",
        ("lead",),
        "Round 28 sends 6 Lead Bloons. Prepare Lead popping (Alchemist, Bomb, 1-0-0 Sniper, "
        "1-0-0 Druid, or 0-0-0 Wizard with 0-2-0 Wall of Fire).",
    ),
    33: KillerRoundInfo(
        33,
        "Camo Rush",
        ("camo",),
        "Round 33 sends 20 Camo Reds + 13 Camo Yellows — requires reliable Camo DPS.",
    ),
    40: KillerRoundInfo(
        40,
        "First MOAB",
        ("moab",),
        "Round 40 sends the first MOAB (200 HP + 4 Ceramics). Ensure single-target MOAB DPS "
        "(MOAB Mauler, Alch-buffed Ninja/Sub, or Sticky Bomb) + Ceramic cleanup.",
    ),
    59: KillerRoundInfo(
        59,
        "First Camo Leads",
        ("camo", "lead"),
        "Round 59 sends 25 Camo Leads — a classic run-killer. Requires combined Camo + Lead "
        "(0-2-0 Village + Alch, 0-2-2 Wizard, 1-1-0 Sniper, or 0-3-0 MIB).",
    ),
    63: KillerRoundInfo(
        63,
        "The Ceramic Rush",
        ("ceramic", "lead"),
        "Round 63 sends 3 dense waves of Ceramics + Leads (#1 midgame run-killer). Needs high "
        "pierce / AOE (Recursive Cluster, Tack Zone, Ice/Glue stall, or Sauda abilities).",
    ),
    76: KillerRoundInfo(
        76,
        "Regrow Ceramic Rush",
        ("ceramic", "regrow"),
        "Round 76 sends 60 tightly-packed Regrow Ceramics — use Grow Blocker (0-1-0 Village) or "
        "high-pierce burst to avoid a regrow farm.",
    ),
    78: KillerRoundInfo(
        78,
        "Double Ceramic Waves (Camo)",
        ("ceramic", "camo"),
        "Round 78 sends two massive Ceramic waves; the second wave is Camo Ceramics!",
    ),
    80: KillerRoundInfo(
        80,
        "First ZOMG",
        ("moab",),
        "Round 80 sends the first ZOMG (4,000 HP + 4 BFBs) — Hard mode final boss.",
    ),
    90: KillerRoundInfo(
        90,
        "First DDTs",
        ("ddt", "camo", "lead"),
        "Round 90 introduces DDTs (Camo + Lead + Black + high speed). Requires 0-3-0 MIB, "
        "Archmage, Prince of Darkness, or Icicle Impale.",
    ),
    95: KillerRoundInfo(
        95,
        "DDT Swarm (30 DDTs)",
        ("ddt", "camo", "lead"),
        "Round 95 sends 30 DDTs + Fortified MOABs — needs sustained DDT DPS + Sabo/Glue slow.",
    ),
    98: KillerRoundInfo(
        98,
        "Highest RBE Round",
        ("moab", "ceramic"),
        "Round 98 has the highest total RBE in CHIMPS (Fortified BFBs + ZOMGs). Needs Shattering "
        "Shells (0-0-4 Mortar) or massive AOE + slow.",
    ),
    99: KillerRoundInfo(
        99,
        "Fortified DDTs",
        ("ddt", "camo", "lead"),
        "Round 99 sends 9 Fortified DDTs — keep Sabo / Bloon Sabotage ready.",
    ),
    100: KillerRoundInfo(
        100,
        "The BAD (CHIMPS Boss)",
        ("moab",),
        "Round 100 sends the BAD (20,000 HP, immune to slows/stuns) — needs pure single-target "
        "DPS (First Strike, Spike Storm, MAD, or Avatar of Wrath).",
    ),
}


# Canonical BTD6 towers & heroes mapped to their primary strategic roles
TOWER_ROLES: dict[str, tuple[str, ...]] = {
    "Dart Monkey": ("early", "camo"),
    "Boomerang Monkey": ("cleanup", "moab"),
    "Bomb Shooter": ("lead", "moab", "cleanup"),
    "Tack Shooter": ("cleanup", "moab"),
    "Ice Monkey": ("support", "ceramic", "ddt"),
    "Glue Gunner": ("support", "ceramic"),
    "Sniper Monkey": ("lead", "camo", "moab"),
    "Monkey Sub": ("early", "camo", "moab"),
    "Monkey Buccaneer": ("early", "camo", "eco"),
    "Monkey Ace": ("moab", "cleanup"),
    "Heli Pilot": ("moab", "support"),
    "Mortar Monkey": ("lead", "cleanup"),
    "Dartling Gunner": ("moab", "cleanup"),
    "Wizard Monkey": ("camo", "lead", "ddt"),
    "Super Monkey": ("moab", "camo"),
    "Ninja Monkey": ("early", "camo", "moab"),
    "Alchemist": ("lead", "support"),
    "Druid": ("lead", "eco", "cleanup"),
    "Mermonkey": ("support", "camo"),
    "Banana Farm": ("eco",),
    "Spike Factory": ("cleanup", "lead", "moab"),
    "Monkey Village": ("support", "camo", "lead"),
    "Engineer Monkey": ("support", "eco"),
    "Beast Handler": ("moab", "cleanup"),
    "Sauda": ("early", "camo"),
    "Quincy": ("early", "camo"),
    "Obyn": ("support", "lead"),
    "Benjamin": ("eco",),
    "Etienne": ("camo",),
    "Geraldo": ("support", "camo", "lead"),
    "Psi": ("camo", "lead"),
    "Adora": ("moab", "lead"),
    "Pat Fusty": ("support", "moab"),
}


def match_tower_name(raw_ocr_text: str) -> str | None:
    """Fuzzy-match noisy BTD6 stylized OCR text (e.g. 'MONKeYVILLAGe') to a canonical tower."""
    cleaned = "".join(ch for ch in raw_ocr_text.lower() if ch.isalpha())
    if len(cleaned) < 4:
        return None
    for canonical in TOWER_ROLES:
        canon_key = "".join(ch for ch in canonical.lower() if ch.isalpha())
        if canon_key in cleaned or cleaned in canon_key:
            return canonical
        # Also match distinct first word for multi-word towers (e.g. 'banana', 'village')
        words = [w.lower() for w in canonical.split() if len(w) >= 5 and w.lower() != "monkey"]
        if any(w in cleaned for w in words):
            return canonical
    return None


# Iconic BTD6 upgrade names mapped to their canonical path notation & tactical role
NOTABLE_UPGRADES: dict[str, str] = {
    "bloonjitsu": "Bloonjitsu (Ninja 4-x-x)",
    "grandmaster": "Grandmaster Ninja (Ninja 5-x-x)",
    "counterespionage": "Counter-Espionage (Ninja x-2-x)",
    "shinobi": "Shinobi Tactics (Ninja x-3-x)",
    "sticky": "Sticky Bomb (Ninja x-x-4)",
    "strongerstimulant": "Stronger Stimulant (Alch 4-x-x)",
    "berserkerbrew": "Berserker Brew (Alch 3-x-x)",
    "acidicmixture": "Acidic Mixture Dip (Alch 2-x-x)",
    "radarscanner": "Radar Scanner (Village x-2-x)",
    "intelligencebureau": "MIB (Village x-3-x)",
    "jungledrums": "Jungle Drums (Village 2-x-x)",
    "primarytraining": "Primary Training (Village 3-x-x)",
    "recursivecluster": "Recursive Cluster (Bomb x-x-4)",
    "clusterbombs": "Cluster Bombs (Bomb x-x-3)",
    "moabmauler": "MOAB Mauler (Bomb x-3-x)",
    "marketplace": "Marketplace (Farm x-x-3)",
    "centralmarket": "Central Market (Farm x-x-4)",
    "bananaplantation": "Banana Plantation (Farm 3-x-x)",
    "researchfacility": "Banana Research Facility (Farm 4-x-x)",
    "monkeybank": "Monkey Bank (Farm x-3-x)",
    "necromancer": "Necromancer (Wizard x-x-4)",
    "princeofdarkness": "Prince of Darkness (Wizard x-x-5)",
    "dragonsbreath": "Dragon's Breath (Wizard x-3-x)",
    "arcanespike": "Arcane Spike (Wizard 4-x-x)",
    "shimmer": "Shimmer (Wizard x-x-3)",
    "moabglue": "MOAB Glue (Glue x-x-3)",
    "relentlessglue": "Relentless Glue (Glue x-x-4)",
    "bloonliquefier": "Bloon Liquefier (Glue 4-x-x)",
    "spikedballs": "Spiked Balls (Spike 3-x-x)",
    "spikedmines": "Spiked Mines (Spike 4-x-x)",
    "permaspike": "Perma-Spike (Spike x-x-5)",
    "armorpiercing": "Armor Piercing Darts (Sub x-x-4)",
    "subcommander": "Sub Commander (Sub x-x-5)",
    "submerge": "Submerge & Support (Sub 3-x-x)",
    "aircraftcarrier": "Aircraft Carrier (Buccaneer 4-x-x)",
    "destroyer": "Destroyer (Buccaneer 3-x-x)",
    "merchantmen": "Merchantmen (Buccaneer x-x-3)",
    "moabpress": "MOAB Press (Boomerang x-x-4)",
    "ringoffire": "Ring of Fire (Tack 4-x-x)",
    "overdrive": "Overdrive (Tack x-x-4)",
    "tackzone": "The Tack Zone (Tack x-x-5)",
    "crossbowmaster": "Crossbow Master (Dart x-x-5)",
    "sharpshooter": "Sharp Shooter (Dart x-x-4)",
    "juggernaut": "Juggernaut (Dart 4-x-x)",
    "nevamiss": "Neva-Miss Targeting (Ace x-x-3)",
    "spectre": "Spectre (Ace x-x-4)",
    "apachedartship": "Apache Dartship (Heli 4-x-x)",
    "sunavatar": "Sun Avatar (Super 3-x-x)",
    "darkknight": "Dark Knight (Super x-x-3)",
    "overclock": "Overclock (Engineer x-4-x)",
    "bloontrap": "Bloon Trap (Engineer x-x-4)",
    "fullauto": "Full Auto Rifle (Sniper x-x-4)",
    "elitedefender": "Elite Defender (Sniper x-x-5)",
    "maimmoab": "Maim MOAB (Sniper 4-x-x)",
    "poplust": "Poplust (Druid x-x-4)",
    "avatarofwrath": "Avatar of Wrath (Druid x-x-5)",
    "spiritoftheforest": "Spirit of the Forest (Druid x-5-x)",
    "icicleimpale": "Icicle Impale (Ice x-x-5)",
}


def match_upgrade_name(raw_ocr_text: str) -> str | None:
    """Fuzzy-match stylized BTD6 upgrade panel OCR text to a canonical upgrade + path."""
    cleaned = "".join(ch for ch in raw_ocr_text.lower() if ch.isalpha())
    if len(cleaned) < 6:
        return None
    for key, label in NOTABLE_UPGRADES.items():
        if key in cleaned:
            return label
    return None


@dataclass(frozen=True)
class WikiCounterSolution:
    """A proven Bloons Wiki counter tower/upgrade with its approximate Medium/Hard total cost."""

    tower_path: str
    upgrade_name: str
    approx_cost: int
    threat_tags: tuple[str, ...]
    why_it_works: str


# Proven Bloons Wiki counter solutions with standard Medium/Hard total build costs
WIKI_COUNTERS: tuple[WikiCounterSolution, ...] = (
    WikiCounterSolution(
        tower_path="0-0-2 Dart Monkey",
        upgrade_name="Enhanced Eyesight",
        approx_cost=650,
        threat_tags=("camo", "early"),
        why_it_works="Cheapest early Camo detection for Round 24 & Round 33.",
    ),
    WikiCounterSolution(
        tower_path="1-1-0 Sniper Monkey",
        upgrade_name="Full Metal Jacket + Night Vision",
        approx_cost=1250,
        threat_tags=("camo", "lead", "early"),
        why_it_works="Global-range Lead & Camo popping for R24, R28, and R33.",
    ),
    WikiCounterSolution(
        tower_path="2-0-0 Alchemist",
        upgrade_name="Acidic Mixture Dip",
        approx_cost=2450,
        threat_tags=("lead", "early"),
        why_it_works="Grants Lead & bonus MOAB damage to nearby carry towers.",
    ),
    WikiCounterSolution(
        tower_path="0-2-0 Monkey Village",
        upgrade_name="Radar Scanner",
        approx_cost=3650,
        threat_tags=("camo",),
        why_it_works="Grants Camo detection to every tower in radius (solves R33, R59, R78).",
    ),
    WikiCounterSolution(
        tower_path="4-0-1 Ninja + 4-2-0 Alchemist",
        upgrade_name="Bloonjitsu + Stronger Stimulant",
        approx_cost=11500,
        threat_tags=("camo", "lead", "moab", "ceramic"),
        why_it_works="Classic BTD6 midgame core that handles R40 MOAB and R59 Camo Leads.",
    ),
    WikiCounterSolution(
        tower_path="2-0-4 Bomb Shooter",
        upgrade_name="Recursive Cluster",
        approx_cost=4850,
        threat_tags=("ceramic", "lead", "cleanup"),
        why_it_works="Massive AoE pierce that deletes all 3 Ceramic waves on R63, R76 & R78.",
    ),
    WikiCounterSolution(
        tower_path="4-0-2 Boomerang Monkey",
        upgrade_name="M.O.A.R. Glaives",
        approx_cost=5200,
        threat_tags=("ceramic", "cleanup"),
        why_it_works="Ricocheting glaives shred dense Ceramic rushes (R63/R76) on tight bends.",
    ),
    WikiCounterSolution(
        tower_path="0-2-4 Ice Monkey",
        upgrade_name="Icicles",
        approx_cost=4200,
        threat_tags=("ceramic", "cleanup"),
        why_it_works="Freezes and chains Ceramic clumps on R63/R76 for under $4.5k.",
    ),
    WikiCounterSolution(
        tower_path="1-3-0 Bomb Shooter",
        upgrade_name="MOAB Mauler",
        approx_cost=2500,
        threat_tags=("moab", "lead"),
        why_it_works="High single-target MOAB layer damage to crack Round 40 cheaply.",
    ),
    WikiCounterSolution(
        tower_path="0-2-3 Glue Gunner",
        upgrade_name="MOAB Glue",
        approx_cost=4300,
        threat_tags=("moab", "ddt", "ceramic"),
        why_it_works="Slows MOABs, BFBs, ZOMGs, and DDTs so your DPS towers have double time.",
    ),
    WikiCounterSolution(
        tower_path="2-3-0 Monkey Village",
        upgrade_name="Monkey Intelligence Bureau (MIB)",
        approx_cost=12500,
        threat_tags=("camo", "lead", "ddt"),
        why_it_works="Lets all nearby towers pop Camo + Lead + Black (essential for R90/R95 DDTs).",
    ),
    WikiCounterSolution(
        tower_path="0-2-5 Spike Factory",
        upgrade_name="Perma-Spike (with Alch buff)",
        approx_cost=36000,
        threat_tags=("moab", "ddt", "bad", "ceramic"),
        why_it_works="End-of-track safety net catching leaking Ceramics, DDTs & BAD insides.",
    ),
)


HERO_NAMES: frozenset[str] = frozenset(
    {
        "Sauda",
        "Quincy",
        "Obyn",
        "Benjamin",
        "Etienne",
        "Geraldo",
        "Psi",
        "Adora",
        "Pat Fusty",
        "Gwendolin",
        "Striker Jones",
        "Captain Churchill",
        "Ezili",
        "Admiral Brickell",
        "Corvus",
        "Rosalia",
    }
)

KNOWN_BTD6_MAPS: tuple[str, ...] = (
    "Downstream",
    "Monkey Meadow",
    "Ravine",
    "Logs",
    "Cubism",
    "Resort",
    "Dark Castle",
    "Muddy Puddles",
    "Bloody Puddles",
    "#Ouch",
    "Quad",
    "Workshop",
    "Cargo",
    "Off The Coast",
    "Cornfield",
    "Peninsula",
    "High Finance",
    "Another Brick",
    "Underground",
    "Rake",
    "Chutes",
    "Bazaar",
    "Balance",
    "Encrypted",
    "Streambed",
    "Firing Range",
    "Moon Landing",
    "Haunted",
    "Spring Spring",
    "KartsNDarts",
    "Candy Falls",
    "Winter Park",
    "Carved",
    "Park Path",
    "Alpine Run",
    "Frozen Over",
    "In The Loop",
    "Lotus Island",
    "Four Circles",
    "Hedge",
    "End Of The Road",
    "Tree Stump",
    "Town Center",
    "Skates",
    "Spillway",
    "Geared",
    "Sanctuary",
    "Flooded Valley",
    "Infernal",
    "Tinkerton",
    "Lost Crevasse",
)


def detect_map_name(title_or_text: str) -> str | None:
    """Match a canonical BTD6 map name from video title or OCR text."""
    low = title_or_text.lower()
    for map_name in KNOWN_BTD6_MAPS:
        if map_name.lower() in low:
            return map_name
    return None


def mode_rules_summary(game_mode: str) -> str:
    """Return a concise BTD6 rule note explaining how coaching adapts to `game_mode`."""
    low = game_mode.lower()
    if "deflation" in low:
        return (
            "Deflation (R31–R60): Fixed $20,000 budget, $0 income, Farms disabled. "
            "Cash hoarding checks are disabled — focus on upfront Camo, Lead & R40/R59 coverage."
        )
    if "chimps" in low:
        return (
            "CHIMPS (R6–R100): 1 Life, no Banana Farms/income, no selling, no Monkey Knowledge. "
            "Eco advice is disabled — every dollar must maximize direct popping power & support."
        )
    if "alternate" in low or "abr" in low:
        return (
            "Alternate Bloons Rounds: Early Camo on R5/R7, Lead on R10, Fortified MOAB on R40. "
            "Prioritize early Camo-Lead solutions far sooner than Standard mode."
        )
    if "odyssey" in low:
        return (
            "Odyssey / Custom Mode: Limited crew & fixed round cap — plan Tier 4/5 save-ups "
            "carefully so you don't die to R40 MOAB or R63 Ceramics while saving."
        )
    return (
        f"{game_mode}: Balance Eco (Banana Farms) with killer-round defense "
        "(R24 Camo, R28 Lead, R40 MOAB, R63 Ceramics, R80 ZOMG)."
    )


def recommend_affordable_counters(
    threat_tags: tuple[str, ...],
    available_cash: int | None,
    *,
    limit: int = 2,
) -> list[WikiCounterSolution]:
    """Return proven Bloons Wiki counters matching threat tags, preferring affordable picks."""
    tags = set(threat_tags) or {"early", "ceramic"}
    matches = [c for c in WIKI_COUNTERS if tags & set(c.threat_tags)]
    if not matches:
        matches = list(WIKI_COUNTERS[:3])
    if available_cash is not None and available_cash > 0:
        affordable = [c for c in matches if c.approx_cost <= available_cash]
        if affordable:
            # Prefer strongest affordable counters (closer to mid-tier impact without overspending)
            affordable.sort(key=lambda c: (-len(tags & set(c.threat_tags)), c.approx_cost))
            return affordable[:limit]
    matches.sort(key=lambda c: (-len(tags & set(c.threat_tags)), c.approx_cost))
    return matches[:limit]
