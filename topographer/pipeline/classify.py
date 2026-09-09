"""Terrain classification.

Iteration 1 (stage 1) classifies on relief, slope and climate alone -- no land
cover yet.  The thresholds live in ``rules.yaml`` beside this module because
they are a starting hypothesis you will retune constantly, not settled truth.

The cascade is ordered: hexcrawl convention wants exactly one terrain per hex,
so precedence has to be explicit.  A secondary class and the underlying numbers
are always kept, so the viewer can say "Forest (hilly)" and the hex key can
write a real sentence.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_RULES = {
    "water_fraction_min": 0.5,
    "mountain_relief_m": 600.0,
    "mountain_slope_deg": 20.0,
    "hill_relief_m": 250.0,
    "rugged_relief_m": 120.0,
    "wetland_fraction_min": 0.25,
    "tree_fraction_min": 0.5,
    "bare_fraction_min": 0.5,
}

# Köppen first letters -> broad qualifier used to subtype a class.
KOPPEN_GROUPS = {
    "A": "tropical",
    "B": "arid",
    "C": "temperate",
    "D": "continental",
    "E": "polar",
}

FOREST_SUBTYPE = {
    "Af": "jungle", "Am": "jungle", "Aw": "monsoon woodland",
    "Cfb": "temperate forest", "Cfa": "temperate forest", "Cfc": "temperate forest",
    "Csa": "dry woodland", "Csb": "dry woodland",
    "Dfb": "boreal forest", "Dfc": "taiga", "Dfd": "taiga",
}


@dataclass
class Classification:
    terrain_class: str
    secondary_class: str | None
    qualifier: str | None
    reason: str


def load_rules(path: str | None = None) -> dict:
    path = path or os.path.join(os.path.dirname(__file__), "rules.yaml")
    if not os.path.exists(path):
        return dict(DEFAULT_RULES)
    try:
        import yaml
    except ImportError:
        return dict(DEFAULT_RULES)
    with open(path) as fh:
        loaded = yaml.safe_load(fh) or {}
    return {**DEFAULT_RULES, **loaded}


def classify_stage1(stats, koppen: str | None = None, is_coastal: bool = False,
                    rules: dict | None = None) -> Classification:
    """Relief + slope + climate.  No land cover -- that arrives at stage 2."""
    R = rules or load_rules()
    relief = stats.relief
    slope = stats.slope_mean
    group = KOPPEN_GROUPS.get((koppen or "")[:1], None)

    if relief >= R["mountain_relief_m"] and slope >= R["mountain_slope_deg"]:
        primary, why = "mountains", f"relief {relief:.0f} m, mean slope {slope:.1f} deg"
    elif relief >= R["hill_relief_m"]:
        primary, why = "hills", f"relief {relief:.0f} m"
    elif relief >= R["rugged_relief_m"]:
        primary, why = "broken", f"relief {relief:.0f} m"
    else:
        primary, why = "plains", f"relief {relief:.0f} m"

    qualifier = None
    if group == "arid":
        qualifier = "desert"
    elif group == "polar":
        qualifier = "tundra"
    elif group == "tropical":
        qualifier = "tropical"

    secondary = "coast" if is_coastal else None
    return Classification(primary, secondary, qualifier, why)


def describe(c: Classification) -> str:
    """Short human label, e.g. 'desert hills (coast)'."""
    parts = [p for p in (c.qualifier, c.terrain_class) if p]
    label = " ".join(parts)
    if c.secondary_class:
        label += f" ({c.secondary_class})"
    return label
