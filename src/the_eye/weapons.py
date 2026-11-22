"""CS2 weapon recoil & spray-pattern profiles for weapon-aware spray control grading."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WeaponProfile:
    """Real CS2 recoil characteristics for a specific weapon."""

    key: str
    display_name: str
    sprayable: bool
    expected_pulldown: float  # Expected vertical pull-down magnitude during sustained spray
    expected_h_tolerance: float  # Expected horizontal compensation tolerance
    spray_tip: str


WEAPON_PROFILES: dict[str, WeaponProfile] = {
    "auto": WeaponProfile(
        key="auto",
        display_name="Rifle (Auto)",
        sprayable=True,
        expected_pulldown=1.2,
        expected_h_tolerance=0.9,
        spray_tip="Pull smoothly straight down on bullets 1-7 before compensating horizontally.",
    ),
    "ak47": WeaponProfile(
        key="ak47",
        display_name="AK-47",
        sprayable=True,
        expected_pulldown=1.6,
        expected_h_tolerance=1.1,
        spray_tip=(
            "AK-47: Pull sharply down on bullets 1-7, then sweep right-to-left (bullets 8-14) "
            "and back right on the '7'-shaped tail."
        ),
    ),
    "m4a4": WeaponProfile(
        key="m4a4",
        display_name="M4A4",
        sprayable=True,
        expected_pulldown=1.4,
        expected_h_tolerance=1.0,
        spray_tip=(
            "M4A4 (666 RPM): Pull down firmly on bullets 1-6, then track right-to-left smoothly."
        ),
    ),
    "m4a1s": WeaponProfile(
        key="m4a1s",
        display_name="M4A1-S",
        sprayable=True,
        expected_pulldown=0.9,
        expected_h_tolerance=0.6,
        spray_tip=(
            "M4A1-S: Tight, forgiving recoil — use a gentle vertical pull-down with minimal "
            "horizontal correction."
        ),
    ),
    "galil": WeaponProfile(
        key="galil",
        display_name="Galil AR",
        sprayable=True,
        expected_pulldown=1.3,
        expected_h_tolerance=1.0,
        spray_tip=(
            "Galil AR: Pull down-and-slightly-left on bullets 1-6, then sweep right on mid-spray."
        ),
    ),
    "mp9": WeaponProfile(
        key="mp9",
        display_name="MP9",
        sprayable=True,
        expected_pulldown=1.0,
        expected_h_tolerance=0.8,
        spray_tip="MP9: Fast fire rate — burst 5-8 bullets with a quick pull-down while mobile.",
    ),
    "awp": WeaponProfile(
        key="awp",
        display_name="AWP (Bolt-Action)",
        sprayable=False,
        expected_pulldown=0.0,
        expected_h_tolerance=0.0,
        spray_tip=(
            "AWP is single-shot: focus on counter-strafe stop before firing and quick unscope."
        ),
    ),
    "deagle": WeaponProfile(
        key="deagle",
        display_name="Desert Eagle",
        sprayable=False,
        expected_pulldown=0.0,
        expected_h_tolerance=0.0,
        spray_tip=(
            "Deagle is tap-only: wait ~0.4s between shots for recoil reset instead of spamming."
        ),
    ),
}


def get_weapon_profile(weapon: str | None) -> WeaponProfile:
    key = (weapon or "auto").strip().lower().replace("-", "").replace("_", "")
    return WEAPON_PROFILES.get(key, WEAPON_PROFILES["auto"])
