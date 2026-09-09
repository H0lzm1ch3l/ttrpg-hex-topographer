import math
import random

import numpy as np
import pytest

from topographer.domain import hexgrid as hg
from topographer.domain.models import (
    HexTile, PlacementContext, SamplerFilters,
)
from topographer.pipeline import classify as cls
from topographer.pipeline.render import render_hex_png
from topographer.pipeline.sample import (
    EdgeMatchSource, HexSampler, RandomSource, random_land_point,
)
from topographer.pipeline.zonal import (
    build_hex_raster, edge_signatures, elev_stats, hillshade, slope_aspect,
)
from topographer.sources import tiles as T
from topographer.sources.landmask import AlwaysLand


# --------------------------------------------------------------------------
# Tile maths
# --------------------------------------------------------------------------

def test_slippy_round_trip():
    for lon, lat in [(0, 0), (10.31, 46.52), (-122.4, 37.8), (151.2, -33.9)]:
        for z in (8, 11, 12):
            fx, fy = T.lonlat_to_tile_frac(lon, lat, z)
            blon, blat = T.tile_to_lonlat(fx, fy, z)
            assert blon == pytest.approx(lon, abs=1e-6)
            assert blat == pytest.approx(lat, abs=1e-6)


def test_terrarium_decode_round_trip():
    heights = np.array([[-412.0, 0.0, 1.5, 3000.25, 8000.0]])
    v = np.clip(heights + 32768.0, 0, 65535.999)
    r = np.floor(v / 256.0).astype(np.uint8)
    g = np.floor(v - r.astype(np.float64) * 256.0).astype(np.uint8)
    b = np.floor((v - np.floor(v)) * 256.0).astype(np.uint8)
    out = T.decode_terrarium(np.dstack([r, g, b]))
    assert np.allclose(out, heights, atol=1 / 256.0)


def test_ground_resolution_and_zoom_choice():
    # ~30 m native source resolution puts the useful ceiling at z12
    assert T.ground_resolution(45.0, 11) == pytest.approx(54.05, abs=0.1)
    assert T.ground_resolution(45.0, 12) == pytest.approx(27.02, abs=0.1)
    assert T.zoom_for_resolution(45.0, 55.0) == 11
    assert T.zoom_for_resolution(45.0, 1.0) == 12          # capped


def test_hex_needs_few_tiles():
    """The reroll loop only works if a candidate is a handful of fetches."""
    for lat in (0, 30, 45, 60):
        frame = hg.LocalFrame(lat=lat, lon=0.0)
        z = T.zoom_for_resolution(lat, 55.0)
        assert T.tile_count(frame.bbox_lonlat(), z) <= 6


# --------------------------------------------------------------------------
# Zonal statistics, verified against terrain with a known answer
# --------------------------------------------------------------------------

class PlaneTransport:
    """A perfectly tilted plane: elevation = grade * easting, in metres."""

    def __init__(self, grade: float = 0.10):
        self.grade = grade

    def fetch(self, z, x, y):
        n = 2 ** z
        u = (np.arange(T.TILE_PX) + 0.5) / T.TILE_PX
        lon = ((x + u[None, :]) / n) * 360.0 - 180.0
        east_m = np.radians(lon) * 6378137.0
        h = np.repeat(self.grade * east_m, T.TILE_PX, axis=0)
        v = np.clip(h + 32768.0, 0, 65535.999)
        r = np.floor(v / 256.0).astype(np.uint8)
        g = np.floor(v - r.astype(np.float64) * 256.0).astype(np.uint8)
        b = np.floor((v - np.floor(v)) * 256.0).astype(np.uint8)
        return np.dstack([r, g, b])


def _raster(transport, lat=0.0, lon=0.0, grid=200):
    frame = hg.LocalFrame(lat=lat, lon=lon)
    z = T.zoom_for_resolution(lat, 55.0)
    mosaic = T.fetch_mosaic(transport, frame.bbox_lonlat(), z)
    return build_hex_raster(frame, mosaic, grid=grid)


def test_mask_area_matches_the_hexagon():
    hr = _raster(T.SyntheticTileTransport())
    cell = hr.px_m * hr.py_m
    assert hr.mask.sum() * cell == pytest.approx(hg.area(hr.frame.size), rel=5e-3)


def test_grid_cells_are_square():
    """Anisotropic cells would skew slope and hillshade and make a flat-top hex
    render as pointy-top."""
    hr = _raster(T.SyntheticTileTransport(), grid=240)
    assert hr.px_m == pytest.approx(hr.py_m, rel=0.01)
    assert hr.aspect_ratio == pytest.approx(2 / math.sqrt(3), rel=0.01)


def test_slope_on_a_known_plane():
    grade = 0.10
    hr = _raster(PlaneTransport(grade))
    slope, _ = slope_aspect(hr)
    assert math.degrees(float(slope[hr.mask].mean())) == pytest.approx(
        math.degrees(math.atan(grade)), abs=0.3)


def test_relief_on_a_known_plane():
    """A 10% plane across 11.15 km of hex: relief must equal grade x width."""
    grade = 0.10
    hr = _raster(PlaneTransport(grade))
    stats = elev_stats(hr)
    expected = grade * hg.across_corners(hr.frame.size)
    assert stats.relief == pytest.approx(expected, rel=0.02)


def test_aspect_on_an_east_facing_plane():
    hr = _raster(PlaneTransport(0.10))
    stats = elev_stats(hr)
    # rising to the east means the surface faces west: bearing ~270
    assert stats.aspect_deg == pytest.approx(270.0, abs=5.0)


def test_flat_ground_has_zero_relief_and_slope():
    class Flat:
        def fetch(self, z, x, y):
            v = 500.0 + 32768.0
            r = int(v // 256); g = int(v - r * 256)
            return np.dstack([
                np.full((T.TILE_PX, T.TILE_PX), r, np.uint8),
                np.full((T.TILE_PX, T.TILE_PX), g, np.uint8),
                np.zeros((T.TILE_PX, T.TILE_PX), np.uint8)])

    stats = elev_stats(_raster(Flat()))
    assert stats.relief == pytest.approx(0.0, abs=0.01)
    assert stats.slope_mean == pytest.approx(0.0, abs=0.01)
    assert stats.elev_mean == pytest.approx(500.0, abs=0.01)


def test_hillshade_is_bounded():
    hs = hillshade(_raster(T.SyntheticTileTransport()))
    assert hs.min() >= 0.0 and hs.max() <= 1.0


def test_hillshade_flat_ground_equals_sine_of_altitude():
    class Flat:
        def fetch(self, z, x, y):
            v = 500.0 + 32768.0
            r = int(v // 256); g = int(v - r * 256)
            return np.dstack([np.full((T.TILE_PX, T.TILE_PX), r, np.uint8),
                              np.full((T.TILE_PX, T.TILE_PX), g, np.uint8),
                              np.zeros((T.TILE_PX, T.TILE_PX), np.uint8)])

    hs = hillshade(_raster(Flat()), altitude_deg=45.0)
    assert float(hs.mean()) == pytest.approx(math.sin(math.radians(45.0)), abs=1e-6)


def test_hillshade_lights_the_slope_that_faces_the_sun():
    """A west-facing slope must be brighter under a NW sun than an east-facing
    one.  Pins the sign convention that the trigonometric form gets wrong."""
    west_facing = _raster(PlaneTransport(+0.20))   # rises east -> faces west
    east_facing = _raster(PlaneTransport(-0.20))   # rises west -> faces east
    lit = hillshade(west_facing, azimuth_deg=315.0)[west_facing.mask].mean()
    dark = hillshade(east_facing, azimuth_deg=315.0)[east_facing.mask].mean()
    assert lit > dark


def test_stats_are_finite_on_real_looking_terrain():
    stats = elev_stats(_raster(T.SyntheticTileTransport(seed=3)))
    for v in vars(stats).values():
        assert math.isfinite(v)
    assert stats.elev_min <= stats.elev_median <= stats.elev_max
    assert stats.relief == pytest.approx(stats.elev_max - stats.elev_min)


# --------------------------------------------------------------------------
# Edge signatures
# --------------------------------------------------------------------------

def test_edge_signatures_shape_and_order():
    hr = _raster(T.SyntheticTileTransport(seed=1))
    sigs = edge_signatures(hr, samples=16)
    assert len(sigs) == 6
    assert all(len(s.profile) == 16 for s in sigs)


def test_adjacent_edges_share_their_corner_heights():
    """Edge i ends where edge i+1 begins -- the counterclockwise convention."""
    hr = _raster(T.SyntheticTileTransport(seed=2))
    sigs = edge_signatures(hr, samples=32)
    for e in range(6):
        assert sigs[e].profile[-1] == pytest.approx(sigs[(e + 1) % 6].profile[0], abs=2)


def test_gradient_out_is_positive_where_ground_rises_outward():
    """A cone rising away from the centre must show positive outward gradient
    on every edge."""

    class Bowl:
        def fetch(self, z, x, y):
            n = 2 ** z
            u = (np.arange(T.TILE_PX) + 0.5) / T.TILE_PX
            lon = ((x + u[None, :]) / n) * 360.0 - 180.0
            lat_r = np.arctan(np.sinh(np.pi * (1 - 2 * (y + u[:, None]) / n)))
            ex = np.radians(lon) * 6378137.0
            ny = lat_r * 6378137.0
            h = 0.05 * np.hypot(np.broadcast_to(ex, (T.TILE_PX, T.TILE_PX)),
                                np.broadcast_to(ny, (T.TILE_PX, T.TILE_PX)))
            v = np.clip(h + 32768.0, 0, 65535.999)
            r = np.floor(v / 256.0).astype(np.uint8)
            g = np.floor(v - r.astype(np.float64) * 256.0).astype(np.uint8)
            b = np.floor((v - np.floor(v)) * 256.0).astype(np.uint8)
            return np.dstack([r, g, b])

    sigs = edge_signatures(_raster(Bowl()))
    assert all(s.gradient_out > 0 for s in sigs)


def test_edge_match_score_prefers_the_true_seam():
    """A tile scored against its own real-world neighbour should beat a
    randomly rolled tile.  The scorer is not wired into the app; this pins the
    convention so turning it on later is a one-line change."""
    tr = T.SyntheticTileTransport(seed=7)
    sampler = HexSampler(transport=tr, mask=AlwaysLand(), grid=160,
                         rng=random.Random(1))

    a = sampler.sample_at(20.0, 5.0, stage=2)
    frame = hg.LocalFrame(20.0, 5.0)
    nlat, nlon = frame.neighbor_center(0)
    true_neighbour = sampler.sample_at(nlat, nlon, stage=2)
    stranger = sampler.sample_at(-40.0, 130.0, stage=2)

    good = EdgeMatchSource.score(a, 0, true_neighbour, hg.opposite_edge(0))
    bad = EdgeMatchSource.score(a, 0, stranger, hg.opposite_edge(0))
    assert good < bad


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def test_sphere_sampling_is_equal_area():
    """Uniform latitude would oversample the poles; asin must not."""
    rng = random.Random(0)
    lats = [math.degrees(math.asin(rng.uniform(-1, 1))) for _ in range(40000)]
    # equal-area bands: |lat| < 30 covers exactly half the sphere
    frac = sum(1 for v in lats if abs(v) < 30.0) / len(lats)
    assert frac == pytest.approx(0.5, abs=0.02)


def test_random_land_point_respects_the_mask():
    class OnlyNorth:
        def is_land(self, lat, lon):
            return lat > 40.0

    lat, lon = random_land_point(OnlyNorth(), random.Random(0))
    assert lat > 40.0
    assert -180 <= lon <= 180


def test_sample_neighbor_with_raster_matches_manual_neighbor_center():
    sampler = HexSampler(transport=T.SyntheticTileTransport(seed=3), mask=AlwaysLand(),
                         grid=120, rng=random.Random(2))
    tile = sampler.sample_at(20.0, 5.0, stage=2)

    got, hr = sampler.sample_neighbor_with_raster(tile, edge=0)

    frame = hg.LocalFrame(20.0, 5.0)
    want_lat, want_lon = frame.neighbor_center(0)
    expected = sampler.sample_at(want_lat, want_lon, stage=2)

    assert got.id == expected.id
    assert got.stage == 2 and got.edge_signature is not None
    assert hr.mask.any()


def test_roll_with_raster_stage_defaults_to_one_but_is_overridable():
    sampler = HexSampler(transport=T.SyntheticTileTransport(), mask=AlwaysLand(),
                         grid=120, rng=random.Random(9))
    plain, _ = sampler.roll_with_raster(SamplerFilters())
    assert plain.stage == 1 and plain.edge_signature is None

    staged, _ = sampler.roll_with_raster(SamplerFilters(), stage=2)
    assert staged.stage == 2 and staged.edge_signature is not None


def test_roll_produces_a_usable_tile():
    sampler = HexSampler(transport=T.SyntheticTileTransport(), mask=AlwaysLand(),
                         grid=160, rng=random.Random(5))
    tile = sampler.roll(SamplerFilters())
    assert tile.stage == 1
    assert tile.terrain_class in {"plains", "broken", "hills", "mountains"}
    assert tile.id and len(tile.id) == 20
    assert tile.edge_signature is None      # stage 2 only


def test_random_source_honours_the_protocol():
    sampler = HexSampler(transport=T.SyntheticTileTransport(), mask=AlwaysLand(),
                         grid=120, rng=random.Random(2))
    src = RandomSource(sampler)
    out = src.propose(PlacementContext(map_id="m", slot=(0, 0)), n=2)
    assert len(out) == 2 and all(isinstance(t, HexTile) for t in out)


def test_content_id_is_stable_and_position_independent():
    a = HexTile.content_id(46.52, 10.31, hg.SIX_MILE_HEX_SIZE_M, {"dem": "v1"})
    b = HexTile.content_id(46.520004, 10.310001, hg.SIX_MILE_HEX_SIZE_M, {"dem": "v1"})
    c = HexTile.content_id(46.52, 10.31, hg.SIX_MILE_HEX_SIZE_M, {"dem": "v2"})
    assert a == b        # ~11 m rounding: the same spot is the same tile
    assert a != c        # a source version bump invalidates cleanly


# --------------------------------------------------------------------------
# Classification and rendering
# --------------------------------------------------------------------------

@pytest.mark.parametrize("relief,slope,expected", [
    (900, 25, "mountains"),
    (900, 5, "hills"),        # high relief but gentle: not mountains
    (300, 8, "hills"),
    (150, 3, "broken"),
    (40, 1, "plains"),
])
def test_classification_cascade(relief, slope, expected):
    from topographer.domain.models import ElevStats
    s = ElevStats(0, relief, relief / 2, relief / 2, relief, slope, slope * 1.5, 10, 180)
    assert cls.classify_stage1(s).terrain_class == expected


def test_climate_qualifies_the_class():
    from topographer.domain.models import ElevStats
    s = ElevStats(0, 40, 20, 20, 40, 1, 2, 5, 180)
    assert cls.classify_stage1(s, koppen="BWh").qualifier == "desert"
    assert cls.classify_stage1(s, koppen="ET").qualifier == "tundra"
    assert cls.classify_stage1(s, koppen="Cfb").qualifier is None
    assert "desert" in cls.describe(cls.classify_stage1(s, koppen="BWh"))


def test_render_produces_a_transparent_masked_png():
    from PIL import Image
    import io

    png = render_hex_png(_raster(T.SyntheticTileTransport(seed=4), grid=160), size_px=256)
    im = Image.open(io.BytesIO(png))
    assert im.mode == "RGBA"
    # flat-top proportions: wider than tall, 2 : sqrt(3)
    assert im.size == (256, round(256 * math.sqrt(3) / 2))
    a = np.asarray(im)[..., 3]
    h, w = a.shape
    assert a[0, 0] == 0                    # image corners fall outside the hexagon
    assert a[h // 2, w // 2] == 255        # the centre is inside it

    # Flat-top signature: the top row runs along a flat edge and is opaque for
    # about half the width, while the left column touches only the west vertex.
    top_run = (a[0, :] > 128).sum() / w
    left_run = (a[:, 0] > 128).sum() / h
    assert 0.35 < top_run < 0.65
    assert left_run < 0.10
