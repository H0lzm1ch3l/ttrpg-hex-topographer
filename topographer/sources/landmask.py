"""Local land test and prescreen grid.

Both exist so the sampler can reject a candidate **before** any network call.
Rejection costs ~3.4 draws per hit (land is 29.2% of the surface), and paying
for tile fetches on rejected draws would dominate the reroll loop's latency.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Protocol

import numpy as np

ASSETS = os.environ.get(
    "TOPOGRAPHER_ASSETS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "assets"),
)


class LandMask(Protocol):
    def is_land(self, lat: float, lon: float) -> bool:
        ...


class NaturalEarthLandMask:
    """Point-in-polygon against Natural Earth land.

    1:50m is the default.  Dropping tiny islands is a feature here -- rolling
    onto a 2 km atoll produces a hex that is almost entirely ocean.
    """

    def __init__(self, path: str | None = None, scale: str = "50m"):
        from shapely.geometry import shape
        from shapely.prepared import prep
        from shapely.strtree import STRtree

        path = path or os.path.join(ASSETS, f"ne_{scale}_land.geojson")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"land polygons not found at {path}. Run scripts/build_assets.py first."
            )
        with open(path) as fh:
            gj = json.load(fh)

        self._geoms = [shape(f["geometry"]) for f in gj["features"]]
        self._tree = STRtree(self._geoms)
        self._prepared = [prep(g) for g in self._geoms]

    def is_land(self, lat: float, lon: float) -> bool:
        from shapely.geometry import Point

        p = Point(lon, lat)
        for idx in self._tree.query(p):
            if self._prepared[int(idx)].contains(p):
                return True
        return False


class AlwaysLand:
    """Test double."""

    def is_land(self, lat: float, lon: float) -> bool:
        return True


# --------------------------------------------------------------------------
# Prescreen grid
# --------------------------------------------------------------------------

@dataclass
class PrescreenGrid:
    """Coarse global summary, ~0.1 degree, built once by scripts/build_assets.py.

    Without this the reroll loop is dominated by flat Siberia, flat Canada,
    flat Sahara and flat Australia -- and every rejected candidate the user has
    to look at costs a round of tile fetches.  Screening on relief before
    fetching turns a nice-to-have filter into a latency optimisation.
    """

    relief: np.ndarray      # (rows, cols) float32 metres
    elev_mean: np.ndarray   # (rows, cols) float32
    coastal: np.ndarray     # (rows, cols) bool
    step_deg: float = 0.1

    @property
    def shape(self) -> tuple[int, int]:
        return self.relief.shape

    def _index(self, lat: float, lon: float) -> tuple[int, int]:
        rows, cols = self.shape
        row = int((90.0 - lat) / 180.0 * rows)
        col = int((lon + 180.0) / 360.0 * cols)
        return min(max(row, 0), rows - 1), min(max(col, 0), cols - 1)

    def lookup(self, lat: float, lon: float) -> dict:
        r, c = self._index(lat, lon)
        return {
            "relief": float(self.relief[r, c]),
            "elev_mean": float(self.elev_mean[r, c]),
            "coastal": bool(self.coastal[r, c]),
        }

    def save(self, path: str) -> None:
        np.savez_compressed(path, relief=self.relief, elev_mean=self.elev_mean,
                            coastal=self.coastal, step_deg=self.step_deg)

    @classmethod
    def load(cls, path: str) -> "PrescreenGrid":
        d = np.load(path)
        return cls(relief=d["relief"], elev_mean=d["elev_mean"],
                   coastal=d["coastal"], step_deg=float(d["step_deg"]))


TERRAIN_BANDS = {
    "flat":        (0.0, 120.0),
    "hilly":       (120.0, 500.0),
    "mountainous": (500.0, math.inf),
    "any":         (0.0, math.inf),
}
