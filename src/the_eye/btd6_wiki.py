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
