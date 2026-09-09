"""Flat-top hexagon geometry in a per-tile local equal-area projection.

Conventions fixed here, and depended on everywhere else
-------------------------------------------------------
* Hexes are **flat-top**: vertices left/right, flat edges top/bottom.
* ``size`` is the circumradius == side length ``s``.  The "6 mile hex" of
  tabletop convention is 6 miles **across the flats**, so
  ``s = 6 * 1609.344 / sqrt(3) = 5574.9 m``.
* Corner ``k`` sits at angle ``60k`` degrees from the centre.
* Edge ``i`` runs from corner ``i`` to corner ``i+1`` and faces the
  direction ``30 + 60i`` degrees.  Traversal is therefore
  **counterclockwise** around the hex.
* Neighbour direction ``i`` is the hex across edge ``i``.  Directions run
  counterclockwise starting at NE:
  ``NE(0) N(1) NW(2) SW(3) S(4) SE(5)``.
* The hex across edge ``i`` meets it on *its* edge ``(i + 3) % 6``.  The two
  traversals run in opposite directions, so any edge-to-edge comparison must
  reverse one profile.  See ``opposite_edge``.
* Axes are mathematical: ``+y`` is north, ``+x`` is east, units are metres in
  the tile's own Lambert Azimuthal Equal Area projection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from pyproj import CRS, Transformer

MILE_M = 1609.344
#: Circumradius of a hex that is 6 miles across the flats.
SIX_MILE_HEX_SIZE_M = 6.0 * MILE_M / math.sqrt(3.0)

#: Axial (q, r) offsets, counterclockwise from NE.  Index == edge index.
DIRECTIONS: tuple[tuple[int, int], ...] = (
    (+1, 0),   # 0 NE
    (0, +1),   # 1 N
    (-1, +1),  # 2 NW
    (-1, 0),   # 3 SW
    (0, -1),   # 4 S
    (+1, -1),  # 5 SE
)

DIRECTION_NAMES = ("NE", "N", "NW", "SW", "S", "SE")


def opposite_edge(edge: int) -> int:
    """Edge index on the neighbour that abuts ``edge`` on this hex."""
    return (edge + 3) % 6


def rotate_edges(signature: list, rotation: int) -> list:
    """Re-index a tile-local 6-element edge array for a placed rotation.

    Signatures are stored in the tile's own frame.  A tile placed with
    ``rotation = k`` presents its edge ``(i - k) % 6`` at the map's edge ``i``.
    Rotation is therefore a cyclic shift and never a re-derivation -- which is
    the whole reason signatures are stored tile-local.
    """
    if len(signature) != 6:
        raise ValueError("edge signature must have exactly 6 entries")
    k = rotation % 6
    return [signature[(i - k) % 6] for i in range(6)]


# --------------------------------------------------------------------------
# Axial coordinates
# --------------------------------------------------------------------------

def axial_to_xy(q: int, r: int, size: float) -> tuple[float, float]:
    """Axial (q, r) -> local projected metres, flat-top layout."""
    x = size * 1.5 * q
    y = size * math.sqrt(3.0) * (0.5 * q + r)
    return x, y


def neighbor_axial(q: int, r: int, edge: int) -> tuple[int, int]:
    dq, dr = DIRECTIONS[edge % 6]
    return q + dq, r + dr


def axial_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Hex distance via cube coordinates."""
    aq, ar = a
    bq, br = b
    return (abs(aq - bq) + abs(aq + ar - bq - br) + abs(ar - br)) // 2


def hex_corners(size: float, cx: float = 0.0, cy: float = 0.0) -> np.ndarray:
    """The 6 corners, counterclockwise, corner k at angle 60k degrees."""
    ang = np.radians(np.arange(6) * 60.0)
    return np.column_stack([cx + size * np.cos(ang), cy + size * np.sin(ang)])


def edge_endpoints(size: float, edge: int) -> tuple[np.ndarray, np.ndarray]:
    """Corners bounding ``edge``, in counterclockwise traversal order."""
    c = hex_corners(size)
    return c[edge % 6], c[(edge + 1) % 6]


def contains(x: np.ndarray, y: np.ndarray, size: float) -> np.ndarray:
    """Vectorised point-in-flat-top-hexagon, centred at the origin.

    Analytic and branch-free -- far faster than shapely for the dense grids
    used in zonal statistics.
    """
    ax = np.abs(x)
    ay = np.abs(y)
    root3 = math.sqrt(3.0)
    return (ay <= root3 / 2.0 * size + 1e-9) & (root3 * ax + ay <= root3 * size + 1e-9)


def across_flats(size: float) -> float:
    return math.sqrt(3.0) * size


def across_corners(size: float) -> float:
    return 2.0 * size


def area(size: float) -> float:
    return 1.5 * math.sqrt(3.0) * size * size


# --------------------------------------------------------------------------
# Per-tile local projection
# --------------------------------------------------------------------------

def local_crs(lat: float, lon: float) -> CRS:
    """Lambert Azimuthal Equal Area centred on the tile.

    Cutting every tile in a projection centred on itself is what makes each
    hex exactly 6 miles across regardless of latitude.  A single shared
    projection across a map would *not* have this property.
    """
    return CRS.from_proj4(
        f"+proj=laea +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0 "
        f"+datum=WGS84 +units=m +no_defs"
    )


@dataclass(frozen=True)
class LocalFrame:
    """A tile's own projected frame, plus transformers to and from WGS84."""

    lat: float
    lon: float
    size: float = SIX_MILE_HEX_SIZE_M

    @property
    def proj4(self) -> str:
        # pyproj warns that to_proj4 can lose information. For a plain LAEA on
        # WGS84 the proj4 string is complete, and it is what gets stored on the
        # tile row, so the warning is noise here.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return local_crs(self.lat, self.lon).to_proj4()

    def _to_wgs(self) -> Transformer:
        return Transformer.from_crs(local_crs(self.lat, self.lon), "EPSG:4326", always_xy=True)

    def _from_wgs(self) -> Transformer:
        return Transformer.from_crs("EPSG:4326", local_crs(self.lat, self.lon), always_xy=True)

    def to_lonlat(self, x, y):
        return self._to_wgs().transform(x, y)

    def from_lonlat(self, lon, lat):
        return self._from_wgs().transform(lon, lat)

    def hex_polygon_lonlat(self) -> list[tuple[float, float]]:
        """Hex corners as (lon, lat), counterclockwise, ring closed."""
        c = hex_corners(self.size)
        lons, lats = self.to_lonlat(c[:, 0], c[:, 1])
        ring = list(zip(lons, lats))
        return ring + [ring[0]]

    def bbox_lonlat(self, pad: float = 1.05) -> tuple[float, float, float, float]:
        """(west, south, east, north) covering the hex, with a little padding."""
        hw = across_corners(self.size) / 2.0 * pad
        hh = across_flats(self.size) / 2.0 * pad
        xs = np.array([-hw, hw, hw, -hw])
        ys = np.array([-hh, -hh, hh, hh])
        lons, lats = self.to_lonlat(xs, ys)
        return float(min(lons)), float(min(lats)), float(max(lons)), float(max(lats))

    def neighbor_center(self, edge: int) -> tuple[float, float]:
        """Real-world (lat, lon) of the hex across ``edge``.

        Pure function of provenance -- nothing to do with where either tile
        sits in a map.

        Stepping out in this tile's frame and back in the neighbour's does not
        return exactly to the start: "north" differs between the two LAEAs by
        the meridian convergence.  Measured drift is 0 m at the equator, 12.6 m
        at 45 deg and 34.7 m at 70 deg -- at worst 0.36% of a hex width, and
        it does not accumulate along a straight walk.  In a collage, where
        adjacency is a fiction, that is invisible and deliberately uncorrected.
        """
        dq, dr = DIRECTIONS[edge % 6]
        x, y = axial_to_xy(dq, dr, self.size)
        lon, lat = self.to_lonlat(x, y)
        return float(lat), float(lon)
