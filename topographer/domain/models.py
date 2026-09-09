"""Domain records.  Pure data -- no I/O, no framework, no ORM."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Literal, Protocol

Stage = Literal[1, 2]
Origin = Literal["random", "geographic", "edgematch"]


# --------------------------------------------------------------------------
# Provenance: where a tile came from and what it is like
# --------------------------------------------------------------------------

@dataclass
class ElevStats:
    elev_min: float
    elev_max: float
    elev_mean: float
    elev_median: float
    relief: float
    slope_mean: float
    slope_p85: float
    ruggedness: float
    aspect_deg: float


@dataclass
class EdgeSignature:
    """One hex edge, in the tile's own frame, traversed counterclockwise.

    Computed at stage 2 and stored forever.  Nothing in iteration 1 reads it:
    it exists so that edge-fit matching later is a scoring function over data
    already on disk, rather than a re-fetch of every raster for every tile.
    """

    profile: list[int]                      # elevation at N points along the edge
    mean_elev: float
    relief_band: float
    gradient_out: float                     # + means ground rises outward
    cover: dict[str, float] = field(default_factory=dict)
    water: dict[str, Any] = field(default_factory=dict)

    def reversed(self) -> "EdgeSignature":
        """Same edge traversed the other way.

        Two hexes sharing an edge traverse it in opposite directions, so
        exactly one side must be reversed before comparison.  Forgetting this
        is the classic edge-matching bug.
        """
        return EdgeSignature(
            profile=list(reversed(self.profile)),
            mean_elev=self.mean_elev,
            relief_band=self.relief_band,
            gradient_out=self.gradient_out,
            cover=dict(self.cover),
            water=dict(self.water),
        )


@dataclass
class Landmark:
    id: str
    kind: str
    name: str | None
    lat: float
    lon: float
    source: str
    source_id: str | None = None
    wikidata_qid: str | None = None
    summary: str | None = None
    prominence_m: float | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class HexTile:
    """An immutable, self-contained 6-mile hex cut from a real place.

    Shared across maps and content-addressed, so rolling the same spot twice
    costs a lookup rather than a fetch.
    """

    lat: float
    lon: float
    proj4: str
    hex_size_m: float
    stage: Stage
    stats: ElevStats
    koppen: str | None = None
    terrain_class: str = "unknown"
    secondary_class: str | None = None
    cover_fractions: dict[str, float] | None = None   # stage 2
    edge_signature: list[EdgeSignature] | None = None  # stage 2
    is_coastal: bool = False
    landmarks: list[Landmark] = field(default_factory=list)
    source_versions: dict[str, str] = field(default_factory=dict)
    thumb_png: bytes | None = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = self.content_id(
                self.lat, self.lon, self.hex_size_m, self.source_versions
            )

    @staticmethod
    def content_id(lat: float, lon: float, size: float, versions: dict[str, str]) -> str:
        """Stable hash of provenance.

        Coordinates round to 4 decimals (~11 m), well inside a hex, so the same
        spot resolves to the same tile.  Dataset versions are part of the key,
        so bumping a source invalidates cleanly instead of silently mixing
        vintages within one map.
        """
        payload = json.dumps(
            {
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "size": round(size, 2),
                "v": dict(sorted(versions.items())),
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:20]

    def edges_as_placed(self, rotation: int = 0) -> list[EdgeSignature] | None:
        from .hexgrid import rotate_edges
        if self.edge_signature is None:
            return None
        return rotate_edges(self.edge_signature, rotation)


# --------------------------------------------------------------------------
# Placement: where a tile sits in someone's map.  Deliberately separate.
# --------------------------------------------------------------------------

@dataclass
class Placement:
    map_id: str
    q: int
    r: int
    tile_id: str
    rotation: int = 0            # always 0 in iteration 1; the column is the point
    origin: Origin = "random"
    origin_edge: tuple[int, int, int] | None = None   # (q, r, edge) the user clicked
    # Vertical shift applied only in this placement, chosen so the tile's
    # shared edge lines up with its already-placed neighbour's -- an offset,
    # never a scale, so a tile's real relief is never squashed to fit. Lives
    # here rather than on HexTile for the same reason rotation does. 0.0 for
    # a tile with nothing yet to align to.
    elev_offset_m: float = 0.0


@dataclass
class SamplerFilters:
    """Constraints the sampler can honour from the prescreen grid alone,
    before any network call."""

    terrain: Literal["any", "mountainous", "hilly", "flat"] = "any"
    coastal: bool | None = None
    climate: list[str] | None = None
    min_relief_m: float | None = None
    max_relief_m: float | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class PlacementContext:
    """Everything a candidate source may consider.

    ``neighbours`` carries *all* adjacent placed tiles, not just the one the
    user clicked from.  Iteration 1's sources ignore it; the later
    edge-matching source needs it, because a slot in a nook can be constrained
    by up to six neighbours at once.  Passing only ``origin_edge`` would have
    forced an interface change later.
    """

    map_id: str
    slot: tuple[int, int]
    neighbours: dict[int, Placement] = field(default_factory=dict)
    origin_edge: tuple[int, int, int] | None = None
    filters: SamplerFilters = field(default_factory=SamplerFilters)


class CandidateSource(Protocol):
    """How a hex gets proposed for an empty slot."""

    name: str

    def propose(self, ctx: PlacementContext, n: int = 1) -> list[HexTile]:
        ...
