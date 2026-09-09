"""Woodland: what is there now, and what would be there without us.

Two different layers, and for Skye the second is the interesting one.

**Current woodland** needs a land-cover raster; topography cannot tell you where
someone planted Sitka spruce. `from_worldcover` handles that.

**Potential woodland** is modelled from terrain. This is not a fudge for missing
data -- for Skye it is the more meaningful layer. The island carries about 4%
woodland today and was already effectively treeless by Roy's survey in 1750, so
an "ancient woodland" polygon layer comes back nearly empty. The early-Holocene
birch-hazel wildwood is a pollen-core fact, not a mappable boundary. What *can*
be mapped is where trees would grow if grazing and drainage stopped, and for a
hexcrawl that is what you actually want on the page.

The model
---------
In the north-west Highlands the limit on tree growth is **wind**, not
temperature. So the treeline here is not a contour: it is a function of shelter.

    treeline(cell) = TREELINE_EXPOSED + shelter · (TREELINE_SHELTERED − TREELINE_EXPOSED)

Shelter combines two terms:

* **Topographic position** — elevation minus the local mean. Negative in glens,
  corries and hollows; positive on ridges and summits.
* **Upwind protection** — how much higher the ground is in the prevailing wind
  direction (south-west on Skye) at a few hundred metres' distance. This is what
  distinguishes a lee-side glen from an exposed western headland at the same
  elevation.

Woodland is then everything below its local treeline that is not open water,
not too steep to hold soil, and not mire.

Excluding mire matters and is easy to get wrong. Blanket bog in this climate is
a **climax community**, not deforested ground waiting to revert -- it is
self-sustaining once formed, and trees do not colonise active peat. Left in, it
put 81% of the mire mask under potential woodland and pushed the total to 53% of
land, which would have drawn a wildwood over the middle of every bog on the
island.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

TREELINE_EXPOSED = 110.0      # m: a bare western headland
TREELINE_SHELTERED = 400.0    # m: a deep lee-side glen
PREVAILING_BEARING = 225.0    # degrees; south-westerly, the Atlantic airflow
SHELTER_RADIUS_M = 900.0
UPWIND_DISTANCES_M = (300.0, 700.0, 1300.0)
SLOPE_MAX = 42.0              # beyond this it is crag, not woodland


@dataclass
class Woodland:
    shelter: np.ndarray      # 0 (fully exposed) .. 1 (fully sheltered)
    treeline: np.ndarray     # metres, per cell
    potential: np.ndarray    # bool
    current: np.ndarray | None = None


def _shift(a: np.ndarray, dr: float, dc: float) -> np.ndarray:
    """Shift with edge replication, so coastal cells are not given phantom
    upwind mountains from the opposite side of the array."""
    return ndimage.shift(a, (dr, dc), order=1, mode="nearest")


def shelter_index(elev: np.ndarray, cell_m: float,
                  bearing_deg: float = PREVAILING_BEARING,
                  radius_m: float = SHELTER_RADIUS_M) -> np.ndarray:
    """0 = fully exposed, 1 = fully sheltered."""
    e = np.nan_to_num(elev, nan=0.0).astype(np.float64)

    # Topographic position: below the local mean means in a hollow.
    r_cells = max(radius_m / cell_m, 2.0)
    tpi = e - ndimage.gaussian_filter(e, r_cells)
    # 40 m below local mean counts as fully sheltered on this term
    tpi_term = np.clip(-tpi / 40.0, 0.0, 1.0)

    # Upwind protection: is the ground higher towards the weather?
    # Screen coords: row increases south, col increases east.
    theta = np.radians(bearing_deg)
    dr_unit = np.cos(theta)      # bearing 225 -> upwind is south-west
    dc_unit = -np.sin(theta)
    rise = np.zeros_like(e)
    for dist in UPWIND_DISTANCES_M:
        n = dist / cell_m
        up = _shift(e, dr_unit * n, dc_unit * n)
        # normalise the rise by distance: a hill 300 m away shelters more than
        # the same hill 1.3 km away
        rise = np.maximum(rise, (up - e) / (dist / 300.0))
    upwind_term = np.clip(rise / 120.0, 0.0, 1.0)

    return np.clip(0.45 * tpi_term + 0.55 * upwind_term, 0.0, 1.0)


def potential(elev: np.ndarray, slope_deg: np.ndarray, cell_m: float,
              exclude: np.ndarray | None = None,
              sea_level: float = 1.0) -> Woodland:
    e = np.nan_to_num(elev, nan=-9999.0)
    land = e > sea_level
    slope = np.nan_to_num(np.asarray(slope_deg, dtype=float), nan=99.0)

    sh = shelter_index(e, cell_m)
    treeline = TREELINE_EXPOSED + sh * (TREELINE_SHELTERED - TREELINE_EXPOSED)

    wood = land & (e < treeline) & (slope < SLOPE_MAX)
    if exclude is not None:
        wood = wood & ~exclude

    # coherent stands, not confetti
    wood = ndimage.binary_opening(wood, np.ones((3, 3)))
    wood = ndimage.binary_closing(wood, np.ones((3, 3)))

    return Woodland(shelter=sh, treeline=treeline, potential=wood)


# --------------------------------------------------------------------------
# Current woodland from ESA WorldCover
# --------------------------------------------------------------------------

#: WorldCover v200 class codes.
WC_TREE = 10
WC_SHRUB = 20
WC_GRASS = 30
WC_CROP = 40
WC_BUILT = 50
WC_BARE = 60
WC_SNOW = 70
WC_WATER = 80
WC_WETLAND = 90
WC_MANGROVE = 95
WC_MOSS = 100

WC_NAMES = {
    WC_TREE: "tree cover", WC_SHRUB: "shrubland", WC_GRASS: "grassland",
    WC_CROP: "cropland", WC_BUILT: "built-up", WC_BARE: "bare/sparse",
    WC_SNOW: "snow & ice", WC_WATER: "permanent water",
    WC_WETLAND: "herbaceous wetland", WC_MANGROVE: "mangroves",
    WC_MOSS: "moss & lichen",
}


def from_worldcover(lc: np.ndarray) -> dict[str, np.ndarray]:
    """Split a WorldCover class raster into the masks this renderer draws.

    Nearest-neighbour sampling is essential upstream: these are class codes, and
    bilinear interpolation between 'tree cover' (10) and 'grassland' (30) would
    invent 'shrubland' (20) along every boundary.
    """
    lc = np.asarray(lc)
    return {
        "tree": lc == WC_TREE,
        "shrub": lc == WC_SHRUB,
        "wetland": lc == WC_WETLAND,
        "water": lc == WC_WATER,
        "moss": lc == WC_MOSS,
        "bare": lc == WC_BARE,
    }
