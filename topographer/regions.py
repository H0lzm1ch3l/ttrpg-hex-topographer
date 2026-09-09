"""Region definitions: bbox, titles and place labels.

Adding a test area should be data, not a code edit. Everything here is
declarative; ``scripts/make_hexmap.py --region <key>`` picks one up.

**Place labels are hand-entered from published coordinates and are the only
thing on a finished map that does not come out of the raster.** They live here,
separate from every derived layer, so that stays obvious.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Region:
    key: str
    title: str
    subtitle: str
    bbox: tuple[float, float, float, float]      # W, S, E, N
    places: list[tuple[str, float, float]] = field(default_factory=list)
    notes: str = ""

    @property
    def span_km(self) -> tuple[float, float]:
        import math

        w, s, e, n = self.bbox
        lat = (s + n) / 2.0
        return ((e - w) * 111.32 * math.cos(math.radians(lat)),
                (n - s) * 111.32)


SKYE = Region(
    key="skye",
    title="THE ISLE OF SKYE",
    subtitle="Loch Harport & the Cuillin · 6-mile hexes",
    # East edge pulled in from -5.60: the wider tile put a third of the page
    # under mainland Applecross, which is true but compositionally wrong.
    bbox=(-6.85, 57.02, -5.78, 57.72),
    places=[
        ("Portree",        -6.1956, 57.4125),
        ("Dunvegan",       -6.5833, 57.4417),
        ("Uig",            -6.3597, 57.5856),
        ("Carbost",        -6.3500, 57.3167),
        ("Sligachan",      -6.1700, 57.2900),
        ("Broadford",      -5.9083, 57.2417),
        ("Kyleakin",       -5.7325, 57.2758),
        ("Elgol",          -6.1050, 57.1450),
        ("Armadale",       -5.8833, 57.0667),
        ("Loch Harport",   -6.4100, 57.3300),
        ("Loch Bracadale", -6.5600, 57.3450),
        ("Sgurr Alasdair", -6.2478, 57.2114),
        ("The Storr",      -6.1789, 57.5069),
        ("Neist Point",    -6.7869, 57.4239),
    ],
    notes="Atlantic, heavily peated, effectively treeless. Max elevation ~959 m "
          "in this DEM (Sgurr Alasdair, 992 m surveyed).",
)

KERRY = Region(
    key="kerry",
    title="IVERAGH & THE REEKS",
    subtitle="Killarney, Carrauntoohil & the Kenmare River · 6-mile hexes",
    bbox=(-10.40, 51.70, -9.35, 52.20),
    places=[
        ("Killarney",      -9.5044, 52.0599),
        ("Kenmare",        -9.5836, 51.8806),
        ("Killorglin",     -9.7808, 52.1058),
        ("Glenbeigh",      -9.9375, 52.0567),
        ("Cahersiveen",   -10.2200, 51.9500),
        ("Waterville",    -10.1667, 51.8333),
        ("Sneem",          -9.8969, 51.8386),
        ("Portmagee",     -10.3628, 51.8869),
        ("Valentia I.",   -10.3167, 51.9167),
        ("Carrauntoohil",  -9.7427, 51.9994),   # 1038.6 m, highest in Ireland
        ("Lough Leane",    -9.5500, 52.0417),
        ("Gap of Dunloe",  -9.6500, 52.0000),
        ("Molls Gap",      -9.6333, 51.9167),
        ("Kenmare River",  -9.9000, 51.7900),
    ],
    notes="Same Atlantic climate as Skye, so the wetland and treeline models "
          "should transfer; higher and steeper (Carrauntoohil 1038.6 m), and "
          "unlike Skye it retains real native oakwood at Killarney. This is a "
          "test of portability rather than robustness.",
)

REGIONS: dict[str, Region] = {r.key: r for r in (SKYE, KERRY)}
