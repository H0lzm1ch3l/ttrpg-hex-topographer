import math

import numpy as np
import pytest
from pyproj import Geod

from topographer.domain import hexgrid as hg
from topographer.domain.models import EdgeSignature

GEOD = Geod(ellps="WGS84")


def test_six_mile_hex_dimensions():
    s = hg.SIX_MILE_HEX_SIZE_M
    assert hg.across_flats(s) == pytest.approx(6 * 1609.344, abs=0.01)
    assert s == pytest.approx(5574.93, abs=0.1)
    assert hg.across_corners(s) == pytest.approx(11149.86, abs=0.1)
    # the classic tabletop figure for a 6-mile hex
    assert hg.area(s) / 2.589988e6 == pytest.approx(31.18, abs=0.01)
    assert hg.area(s) / 1e6 == pytest.approx(80.75, abs=0.01)


def test_h3_does_not_fit_a_six_mile_hex():
    """Documents why there is a custom lattice rather than H3."""
    km2 = hg.area(hg.SIX_MILE_HEX_SIZE_M) / 1e6
    assert 36.13 < km2 < 252.90       # strictly between H3 res 6 and res 5
    assert km2 / 36.129 > 2.2          # more than twice res 6
    assert 252.904 / km2 > 3.1         # less than a third of res 5


def test_contains_matches_analytic_area():
    s = hg.SIX_MILE_HEX_SIZE_M
    n = 900
    xs = np.linspace(-s, s, n)
    ys = np.linspace(-math.sqrt(3) / 2 * s, math.sqrt(3) / 2 * s, n)
    gx, gy = np.meshgrid(xs, ys)
    inside = hg.contains(gx, gy, s)
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    assert inside.sum() * cell == pytest.approx(hg.area(s), rel=2e-3)


def test_corners_lie_on_the_circumcircle_and_edges_on_the_inradius():
    s = hg.SIX_MILE_HEX_SIZE_M
    c = hg.hex_corners(s)
    assert np.allclose(np.hypot(c[:, 0], c[:, 1]), s)
    for e in range(6):
        a, b = hg.edge_endpoints(s, e)
        mid = (a + b) / 2
        assert np.hypot(*mid) == pytest.approx(math.sqrt(3) / 2 * s, rel=1e-9)


def test_edge_faces_its_neighbour_direction():
    """Edge i must point at neighbour i -- everything else depends on it."""
    s = hg.SIX_MILE_HEX_SIZE_M
    for e in range(6):
        a, b = hg.edge_endpoints(s, e)
        mid = (a + b) / 2
        dq, dr = hg.DIRECTIONS[e]
        nx, ny = hg.axial_to_xy(dq, dr, s)
        assert math.degrees(math.atan2(mid[1], mid[0])) == pytest.approx(
            math.degrees(math.atan2(ny, nx)), abs=1e-6)


def test_direction_names_match_bearings():
    s = hg.SIX_MILE_HEX_SIZE_M
    expected = {"NE": 45, "N": 0, "NW": -45, "SW": -135, "S": 180, "SE": 135}
    for e, name in enumerate(hg.DIRECTION_NAMES):
        x, y = hg.axial_to_xy(*hg.DIRECTIONS[e], s)
        bearing = math.degrees(math.atan2(x, y))
        # flat-top diagonals sit at +/-30 from the axis, not 45; check the quadrant
        assert math.copysign(1, bearing) == math.copysign(1, expected[name]) or expected[name] in (0, 180)


def test_neighbour_distance_is_exactly_across_flats():
    """A neighbour centre must be one across-flats away, at any latitude."""
    for lat, lon in [(0, 0), (46.52, 10.31), (-33.9, 151.2), (68.0, -50.0)]:
        frame = hg.LocalFrame(lat=lat, lon=lon)
        for e in range(6):
            nlat, nlon = frame.neighbor_center(e)
            _, _, dist = GEOD.inv(lon, lat, nlon, nlat)
            assert dist == pytest.approx(hg.across_flats(frame.size), rel=1e-4)


@pytest.mark.parametrize("lat,budget_m", [(0, 0.5), (30, 9), (45, 15), (70, 40)])
def test_neighbour_round_trip_drift_is_bounded(lat, budget_m):
    """Stepping out and back does NOT land exactly where you started.

    Each tile is cut in its own LAEA, so "north" differs between the two
    frames by the meridian convergence -- zero at the equator, growing with
    latitude.  Measured: 0 m at the equator, 12.6 m at 45 deg, 34.7 m at
    70 deg, i.e. at worst 0.36% of a hex width.

    That is invisible in a collage, where adjacency is a fiction anyway, so it
    is deliberately not corrected for.  It would be worth correcting in a
    contiguous real-region map, which is exactly the design this project
    isn't.  The test pins the magnitude so a real regression is still caught.
    """
    frame = hg.LocalFrame(lat=lat, lon=10.0)
    for e in range(6):
        nlat, nlon = frame.neighbor_center(e)
        back = hg.LocalFrame(lat=nlat, lon=nlon)
        blat, blon = back.neighbor_center(hg.opposite_edge(e))
        _, _, dist = GEOD.inv(10.0, lat, blon, blat)
        assert dist < budget_m
        assert dist < 0.005 * hg.across_flats(frame.size)


def test_chained_neighbour_walk_keeps_its_scale():
    """Drift is directional, not cumulative in distance: 20 steps east still
    covers 20 hex widths to within a few metres."""
    lat, lon = 46.5, 10.0
    for _ in range(20):
        lat, lon = hg.LocalFrame(lat=lat, lon=lon).neighbor_center(0)
    _, _, dist = GEOD.inv(10.0, 46.5, lon, lat)
    expected = 20 * hg.across_flats(hg.SIX_MILE_HEX_SIZE_M)
    assert dist == pytest.approx(expected, abs=50.0)


def test_opposite_edge_is_an_involution():
    for e in range(6):
        assert hg.opposite_edge(hg.opposite_edge(e)) == e
        assert hg.opposite_edge(e) != e


def test_rotate_edges_is_a_cyclic_shift():
    sig = list(range(6))
    assert hg.rotate_edges(sig, 0) == sig
    assert hg.rotate_edges(sig, 1) == [5, 0, 1, 2, 3, 4]
    assert hg.rotate_edges(sig, 6) == sig
    # rotating by k then by -k is identity, for every k
    for k in range(6):
        assert hg.rotate_edges(hg.rotate_edges(sig, k), -k) == sig


def test_edge_signature_reverse_is_an_involution():
    e = EdgeSignature(profile=[1, 2, 3, 4], mean_elev=2.5, relief_band=3,
                      gradient_out=0.1)
    assert e.reversed().profile == [4, 3, 2, 1]
    assert e.reversed().reversed().profile == e.profile


def test_hex_polygon_closes_and_has_six_corners():
    ring = hg.LocalFrame(lat=46.52, lon=10.31).hex_polygon_lonlat()
    assert len(ring) == 7
    assert ring[0] == ring[-1]


def test_axial_distance():
    assert hg.axial_distance((0, 0), (0, 0)) == 0
    for e in range(6):
        assert hg.axial_distance((0, 0), hg.neighbor_axial(0, 0, e)) == 1
    assert hg.axial_distance((0, 0), (3, -1)) == 3
