"""Drainage network extraction: where the water actually goes.

Why the slope raster is not the input
-------------------------------------
Slope cannot locate a river. Slope says how steep a cell is; a river is where
water *accumulates*, which depends on how much land drains through the cell. A
bare crag has high slope and no water on it; the strath below has gentle slope
and carries the burn. Channels come from **flow accumulation** over a
hydrologically corrected DEM.

The slope raster earns its place one step later, classifying each reach once the
network exists -- steep reach with a small catchment is a torrent or a fall,
gentle reach with a large one is a meander over soft ground. That is the
distinction that matters at the table: whether the party can ford it.

Why pysheds rather than a hand-rolled D8
----------------------------------------
The naive pipeline -- fill depressions by morphological reconstruction, then
route by steepest descent -- was tried first and produces a network that looks
almost empty. Measured on this DEM at 108 m cells: only 1.8% of land cells end
up with no downhill neighbour, but because every one of them *severs a flow
path*, mean path length collapses to ~55 cells, maximum catchment comes out at
16 km², and the channel raster fragments into specks that get filtered away.

Correct handling needs all three of pit filling, depression filling and **flat
resolution** (filled depressions are perfectly level, so steepest descent has
nothing to choose). With those, maximum catchment on Skye goes to 84.7 km² and
the network becomes continuous. pysheds does all three, numba-accelerated, in
about 1.4 s for a 1080x900 grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# pysheds 0.5 predates the numpy 2 removal of np.in1d.
if not hasattr(np, "in1d"):          # pragma: no cover
    np.in1d = np.isin

from affine import Affine                       # noqa: E402
from pysheds.grid import Grid                   # noqa: E402
from pysheds.view import Raster, ViewFinder     # noqa: E402

#: Catchment area in km² at which each band begins. Chosen so band 1 is a
#: hill burn you step over and band 4 is something you need a bridge for.
ORDER_BANDS_KM2 = (0.8, 3.0, 10.0, 30.0)

TORRENT_SLOPE_DEG = 11.0     # above this a reach is falls and rapids
SLUGGISH_SLOPE_DEG = 2.0     # below this it wanders and the ground is wet


@dataclass
class Drainage:
    acc_km2: np.ndarray      # catchment area draining through each cell
    order: np.ndarray        # 0 = not a channel, 1-4 = size band
    channel: np.ndarray      # bool
    slope_deg: np.ndarray
    cell_m: float


@dataclass
class Reach:
    pts: np.ndarray          # (N, 2) as (row, col) on the working grid
    band: int
    slope_deg: float

    @property
    def kind(self) -> str:
        if self.slope_deg >= TORRENT_SLOPE_DEG:
            return "torrent"
        if self.slope_deg <= SLUGGISH_SLOPE_DEG:
            return "sluggish"
        return "normal"


def extract(elev: np.ndarray, slope_deg: np.ndarray, cell_m: float,
            sea_level: float = 1.0,
            min_km2: float = ORDER_BANDS_KM2[0]) -> Drainage:
    e = np.nan_to_num(elev, nan=-9999.0).astype(np.float64)
    rows, cols = e.shape

    # pysheds works through a viewfinder; the affine is in the field's own
    # projected metres, so cell area is exact and catchments come out in km².
    aff = Affine(cell_m, 0.0, 0.0, 0.0, -cell_m, rows * cell_m)
    vf = ViewFinder(affine=aff, shape=e.shape, nodata=-9999.0)
    grid = Grid(viewfinder=vf)
    raster = Raster(e, viewfinder=vf)

    inflated = grid.resolve_flats(grid.fill_depressions(grid.fill_pits(raster)))
    fdir = grid.flowdir(inflated)
    acc_cells = np.asarray(grid.accumulation(fdir), dtype=np.float64)

    km2_per_cell = (cell_m / 1000.0) ** 2
    acc_km2 = acc_cells * km2_per_cell

    land = e > sea_level
    channel = (acc_km2 >= min_km2) & land

    order = np.zeros(e.shape, dtype=np.int8)
    for i, lo in enumerate(ORDER_BANDS_KM2):
        order[channel & (acc_km2 >= lo)] = i + 1

    return Drainage(acc_km2=acc_km2, order=order, channel=channel,
                    slope_deg=slope_deg, cell_m=cell_m)


def trace(dr: Drainage, min_cells: int = 6) -> list[Reach]:
    """Vectorise the channel raster into per-band polylines.

    Splitting by band before labelling means each drawn reach has one line
    weight; reaches still join visually because bands share cells at their
    boundaries.
    """
    out: list[Reach] = []
    for band in range(1, int(dr.order.max()) + 1):
        mask = dr.order == band
        if not mask.any():
            continue
        lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
        for comp in range(1, n + 1):
            rows, cols = np.nonzero(lab == comp)
            if rows.size < min_cells:
                continue
            for path in _walk(rows, cols):
                if len(path) >= min_cells:
                    idx = path.astype(int)
                    out.append(Reach(
                        pts=path, band=band,
                        slope_deg=float(np.nanmean(dr.slope_deg[idx[:, 0], idx[:, 1]])),
                    ))
    return out


def _walk(rows: np.ndarray, cols: np.ndarray, jump: float = 2.9) -> list[np.ndarray]:
    """Greedy nearest-neighbour walks over one connected component.

    A channel component is a tree, not a simple path, so a single walk would
    either double back or leave branches unvisited. Instead the walk restarts
    whenever the nearest unvisited cell is further than one diagonal step,
    emitting each branch as its own polyline.
    """
    pts = np.column_stack([rows, cols]).astype(float)
    remaining = np.ones(len(pts), dtype=bool)
    paths: list[np.ndarray] = []

    while remaining.any():
        cand = np.flatnonzero(remaining)
        # start each branch at the cell furthest north, which for a drainage
        # tree is usually a headwater
        start = cand[int(np.argmin(pts[cand, 0]))]
        path = [start]
        remaining[start] = False
        cur = pts[start]
        while True:
            cand = np.flatnonzero(remaining)
            if cand.size == 0:
                break
            d = np.hypot(*(pts[cand] - cur).T)
            j = int(np.argmin(d))
            if d[j] > jump:
                break
            nxt = cand[j]
            path.append(nxt)
            remaining[nxt] = False
            cur = pts[nxt]
        if len(path) > 1:
            paths.append(pts[path])
    return paths
