"""Exercises the real Natural Earth land mask, if the asset is present."""

import os
import random

import pytest

from topographer.pipeline.sample import random_land_point
from topographer.sources.landmask import ASSETS, NaturalEarthLandMask

pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(ASSETS, "ne_50m_land.geojson")),
    reason="run scripts/build_assets.py first",
)


@pytest.fixture(scope="module")
def mask():
    return NaturalEarthLandMask()


@pytest.mark.parametrize("name,lat,lon,land", [
    ("Ortler, Alps",       46.51,   10.54, True),
    ("Kansas",             38.50,  -98.00, True),
    ("Sahara",             23.00,   12.00, True),
    ("central Siberia",    64.00,  100.00, True),
    ("Amazon basin",       -3.50,  -62.00, True),
    ("mid-Pacific",         0.00, -150.00, False),
    ("mid-Atlantic",       30.00,  -40.00, False),
    ("Southern Ocean",    -55.00,   20.00, False),
    ("Bay of Bengal",      15.00,   88.00, False),
])
def test_known_points(mask, name, lat, lon, land):
    assert mask.is_land(lat, lon) is land, name


def test_sampled_points_are_all_on_land(mask):
    rng = random.Random(42)
    pts = [random_land_point(mask, rng) for _ in range(200)]
    assert all(mask.is_land(la, lo) for la, lo in pts)


def test_land_fraction_of_uniform_sphere_draws(mask):
    """Land is ~29% of the surface; equal-area sampling should find that.

    A latitude-uniform sampler would come out noticeably different, so this
    doubles as a regression test on the sampling scheme itself.
    """
    import math

    rng = random.Random(7)
    hits = 0
    n = 4000
    for _ in range(n):
        lat = math.degrees(math.asin(rng.uniform(-1, 1)))
        lon = rng.uniform(-180, 180)
        hits += mask.is_land(lat, lon)
    assert 0.24 < hits / n < 0.34
