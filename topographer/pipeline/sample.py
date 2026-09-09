"""Sampling a hex from anywhere on Earth, and the candidate sources that use it."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from ..domain import hexgrid as hg
from ..domain.models import (
    CandidateSource, HexTile, PlacementContext, SamplerFilters,
)
from ..sources.landmask import TERRAIN_BANDS, LandMask, PrescreenGrid
from ..sources.tiles import TileTransport, fetch_mosaic, zoom_for_resolution
from . import classify as cls
from .zonal import HexRaster, build_hex_raster, edge_signatures, elev_stats

SOURCE_VERSIONS = {"dem": "terrarium-2023", "hex": "v1"}


def random_land_point(mask: LandMask, rng: random.Random,
                      max_tries: int = 400) -> tuple[float, float]:
    """Uniform on the sphere, rejected to land.

    ``asin`` on a uniform draw, not ``uniform(-90, 90)`` -- the naive form
    oversamples the poles badly and would fill the reroll loop with ice.
    """
    for _ in range(max_tries):
        lat = math.degrees(math.asin(rng.uniform(-1.0, 1.0)))
        lon = rng.uniform(-180.0, 180.0)
        if mask.is_land(lat, lon):
            return lat, lon
    raise RuntimeError("no land point found; is the land mask loaded?")


@dataclass
class HexSampler:
    """Builds a stage-1 tile for a point, and rolls random ones."""

    transport: TileTransport
    mask: LandMask
    prescreen: PrescreenGrid | None = None
    size: float = hg.SIX_MILE_HEX_SIZE_M
    grid: int = 320
    target_res_m: float = 55.0     # ~z11 in mid latitudes
    rng: random.Random = None      # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.rng is None:
            self.rng = random.Random()

    # -- filters ---------------------------------------------------------

    def passes_prescreen(self, lat: float, lon: float, f: SamplerFilters) -> bool:
        """Cheap local rejection, before any network call."""
        if self.prescreen is None:
            return True
        info = self.prescreen.lookup(lat, lon)
        lo, hi = TERRAIN_BANDS[f.terrain]
        if not (lo <= info["relief"] < hi):
            return False
        if f.coastal is not None and info["coastal"] != f.coastal:
            return False
        if f.min_relief_m is not None and info["relief"] < f.min_relief_m:
            return False
        if f.max_relief_m is not None and info["relief"] > f.max_relief_m:
            return False
        return True

    # -- building --------------------------------------------------------

    def sample_at(self, lat: float, lon: float, stage: int = 1) -> HexTile:
        tile, _hr = self.sample_at_with_raster(lat, lon, stage)
        return tile

    def sample_at_with_raster(self, lat: float, lon: float,
                              stage: int = 1) -> tuple[HexTile, HexRaster]:
        """Same as ``sample_at``, but also returns the raster used to build it.

        Callers that need to render a thumbnail (the web layer, in particular)
        would otherwise have to re-fetch and re-mosaic the same tiles just to
        get the grid back.
        """
        frame = hg.LocalFrame(lat=lat, lon=lon, size=self.size)
        z = zoom_for_resolution(lat, self.target_res_m)
        mosaic = fetch_mosaic(self.transport, frame.bbox_lonlat(), z)
        hr = build_hex_raster(frame, mosaic, grid=self.grid)
        stats = elev_stats(hr)

        coastal = False
        if self.prescreen is not None:
            coastal = self.prescreen.lookup(lat, lon)["coastal"]

        c = cls.classify_stage1(stats, koppen=None, is_coastal=coastal)

        tile = HexTile(
            lat=lat, lon=lon, proj4=frame.proj4, hex_size_m=self.size,
            stage=1, stats=stats, terrain_class=c.terrain_class,
            secondary_class=c.secondary_class, is_coastal=coastal,
            source_versions=dict(SOURCE_VERSIONS),
        )
        if stage >= 2:
            # Stage 2 proper also pulls WorldCover, Overpass and Wikidata; the
            # edge signatures are computed here because recomputing them later
            # would mean re-fetching every raster for every tile in the library.
            tile.edge_signature = edge_signatures(hr)
            tile.stage = 2
        return tile, hr

    def roll(self, filters: SamplerFilters | None = None,
             max_tries: int = 200) -> HexTile:
        tile, _hr = self.roll_with_raster(filters, max_tries)
        return tile

    def roll_with_raster(self, filters: SamplerFilters | None = None,
                         max_tries: int = 200, stage: int = 1) -> tuple[HexTile, HexRaster]:
        f = filters or SamplerFilters()
        for _ in range(max_tries):
            lat, lon = random_land_point(self.mask, self.rng)
            if self.passes_prescreen(lat, lon, f):
                return self.sample_at_with_raster(lat, lon, stage=stage)
        raise RuntimeError(f"no candidate matched filters after {max_tries} tries")

    def sample_neighbor_with_raster(self, tile: HexTile, edge: int,
                                    stage: int = 2) -> tuple[HexTile, HexRaster]:
        """The real-world neighbour of ``tile`` across ``edge``.

        Pure function of provenance, independent of where either tile sits in
        a map -- the same operation ``GeographicSource`` wraps, but returning
        the raster too so a caller can render a thumbnail without re-fetching.
        Defaults to stage 2 because a tile reached by growing the map is a
        candidate for a further edge click, which needs its edge signature to
        align against.
        """
        frame = hg.LocalFrame(lat=tile.lat, lon=tile.lon, size=tile.hex_size_m)
        lat, lon = frame.neighbor_center(edge)
        return self.sample_at_with_raster(lat, lon, stage=stage)


# --------------------------------------------------------------------------
# Candidate sources -- the two iteration-1 policies, plus a placeholder for
# the one that is deliberately not built yet.
# --------------------------------------------------------------------------

class RandomSource:
    """Roll anywhere on Earth.  Ignores neighbours entirely."""

    name = "random"

    def __init__(self, sampler: HexSampler):
        self.sampler = sampler

    def propose(self, ctx: PlacementContext, n: int = 1) -> list[HexTile]:
        return [self.sampler.roll(ctx.filters) for _ in range(n)]


class GeographicSource:
    """The real-world neighbour of the tile whose edge the user clicked.

    Requires ``origin_edge``: because the user clicks an *edge* rather than a
    cell, the source tile is never ambiguous even when a slot touches several
    placed tiles.  That is also the gesture the edge-matching source needs.
    """

    name = "geographic"

    def __init__(self, sampler: HexSampler, resolve_tile):
        self.sampler = sampler
        self.resolve_tile = resolve_tile     # (map_id, q, r) -> HexTile

    def propose(self, ctx: PlacementContext, n: int = 1) -> list[HexTile]:
        if ctx.origin_edge is None:
            raise ValueError("GeographicSource needs an origin_edge (q, r, edge)")
        q, r, edge = ctx.origin_edge
        src = self.resolve_tile(ctx.map_id, q, r)
        frame = hg.LocalFrame(lat=src.lat, lon=src.lon, size=src.hex_size_m)
        lat, lon = frame.neighbor_center(edge)
        return [self.sampler.sample_at(lat, lon)]


class EdgeMatchSource:
    """NOT BUILT -- iteration 2.

    Deliberately left as a stub so the shape is on the record.  It is
    ``RandomSource`` plus a scoring filter: sample K candidates, score each
    against *every* already-placed neighbour in ``ctx.neighbours``, return the
    best.  Everything it needs is already stored on the tile at stage 2, so
    turning it on requires no re-fetch and no schema change.
    """

    name = "edgematch"

    def __init__(self, sampler: HexSampler, k: int = 24):
        self.sampler = sampler
        self.k = k

    def propose(self, ctx: PlacementContext, n: int = 1) -> list[HexTile]:
        raise NotImplementedError(
            "edge matching is iteration 2; the stored edge signatures are its inputs"
        )

    @staticmethod
    def score(a, i: int, b, j: int,
              w=(1.0, 0.0, 0.0, 250.0)) -> float:
        """Reference scoring function -- unused, kept honest by a unit test.

        Lower is better.  Note the last term: two edges fit when the ground
        falls out of one and rises into the other, so their outward gradients
        should *sum* to zero rather than match.
        """
        ea = a.edge_signature[i]
        eb = b.edge_signature[j].reversed()
        pa = np.asarray(ea.profile, dtype=float)
        pb = np.asarray(eb.profile, dtype=float)
        # compare shape, not absolute height -- a plateau should still match a
        # plateau 300 m higher up
        pa = pa - pa.mean()
        pb = pb - pb.mean()
        profile_d = float(np.sqrt(((pa - pb) ** 2).mean()))
        cover_d = 0.0
        water_d = 0.0
        grad_d = abs(ea.gradient_out + eb.gradient_out)
        w1, w2, w3, w4 = w
        return w1 * profile_d + w2 * cover_d + w3 * water_d + w4 * grad_d


def default_sampler(cache_dir: str = ".tilecache", grid: int = 320,
                    seed: int | None = None) -> HexSampler:
    """Live HTTP tiles + Natural Earth land test -- the sampler iteration 1 uses.

    Assembled here, not in ``web/``, so that the only layers that import
    concrete source adapters (``HttpTileTransport``, ``NaturalEarthLandMask``)
    are ``cli.py`` and this function. The design doc's one rule for staying
    healthy is that ``web/`` may not import ``sources`` directly.
    """
    import sys

    from ..sources.landmask import AlwaysLand, NaturalEarthLandMask
    from ..sources.tiles import HttpTileTransport

    transport = HttpTileTransport(cache_dir=cache_dir)
    try:
        mask: LandMask = NaturalEarthLandMask()
    except FileNotFoundError as exc:
        print(f"warning: {exc}\n         falling back to AlwaysLand", file=sys.stderr)
        mask = AlwaysLand()
    return HexSampler(transport=transport, mask=mask, grid=grid,
                      rng=random.Random(seed))


def make_source(kind: str, sampler: HexSampler, resolve_tile=None) -> CandidateSource:
    if kind == "random":
        return RandomSource(sampler)
    if kind == "geographic":
        if resolve_tile is None:
            raise ValueError("geographic source needs a resolve_tile callable")
        return GeographicSource(sampler, resolve_tile)
    if kind == "edgematch":
        return EdgeMatchSource(sampler)
    raise ValueError(f"unknown candidate source: {kind}")
