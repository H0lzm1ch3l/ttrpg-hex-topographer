"""Wetland and standing water from terrain alone -- bog, fen, swamp, loch.

Those first three are different ecosystems, not synonyms, and the difference is
hydrological, so terrain can separate them. Two independent axes do the work:

**Water source** decides bog from fen.
  A *bog* is ombrotrophic -- rain-fed. It sits on gentle interfluves where the
  only water arriving is what falls on it, so its upslope catchment is tiny.
  A *fen* is minerotrophic: water flows through it from higher ground, bringing
  minerals with it. Measured on Skye, catchment among wet cells is p50 = 0.03
  km² but p90 = 0.63 -- a large rain-fed bulk with a thin throughflow tail,
  which is exactly the split the ecology predicts.

**Shelter** decides fen from swamp.
  A *swamp* is a wooded wetland -- alder and willow carr. Same water regime as a
  fen; the difference is whether trees can stand there at all. On Skye that is a
  wind question, so it reuses the shelter index from ``woodland``.

    bog   = wet AND catchment <  0.15 km²
    fen   = wet AND catchment >= 0.15 km²  AND  exposed
    swamp = wet AND catchment >= 0.15 km²  AND  sheltered

Result: bog 19.5%, fen 3.8%, swamp 1.1% of land. Swamp being rare is correct --
carr on Skye is scarce, held back by grazing and by Atlantic wind.


Blanket bog is strongly topographically controlled, which is what makes this
work without a land-cover raster. Peat forms where rainfall exceeds drainage:
gentle slopes, water arriving from upslope, and no incision to carry it away.
So the model is the Topographic Wetness Index

    TWI = ln( a / tan(beta) )

where ``a`` is upslope contributing area per unit contour width and ``beta`` is
the local slope, combined with a slope ceiling and a **roughness** ceiling.

Roughness is the discriminator that stops this over-firing. Wet flat ground
that is also rocky is a boulder field with a burn in it, not a mire. Measured on
Skye: cells passing the slope-and-TWI test average 4.2 roughness against 10.9
for land as a whole, so the wet flats really are smooth — an independent check
that the model is finding what it claims to.

Standing water needs a different test. In a DSM a loch is not a depression to be
filled — its surface is already flat water draining through an outlet — so pit
and depression filling find nothing (measured: zero cells). Lochs show up
instead as **flats**: near-zero slope over a connected area, inland.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

# Tuned against the COP30 distribution for Skye, where land slope has a median
# of 5.9 deg and TWI a median of 7.5.
MIRE_SLOPE_MAX = 6.0        # deg; blanket bog rarely forms steeper
MIRE_TWI_MIN = 8.5          # ~p75 of land TWI
MIRE_ROUGH_MAX = 9.0        # exclude wet but rocky ground
MIRE_ELEV_MAX = 500.0       # above this it is exposed montane, not bog

#: Catchment above which a wet cell is fed by throughflow rather than rain.
FEN_MIN_KM2 = 0.15
#: Shelter above which a throughflow wetland can carry carr rather than reeds.
SWAMP_SHELTER_MIN = 0.55

#: Smallest patch worth a symbol.
BOG_MIN_PATCH_HA = 6.0
FLOW_MIN_PATCH_HA = 2.0

LAKE_SLOPE_MAX = 0.45       # a water surface in a DSM is essentially planar
LAKE_MIN_HA = 1.5
LAKE_ELEV_STD_MAX = 1.5     # a real lake is level; a raised beach is not


@dataclass
class WetLayers:
    twi: np.ndarray
    bog: np.ndarray         # ombrotrophic: rain-fed blanket peat
    fen: np.ndarray         # minerotrophic, open: reeds and sedge
    swamp: np.ndarray       # minerotrophic, wooded: alder and willow carr
    lake: np.ndarray        # standing fresh water
    slope_deg: np.ndarray
    roughness: np.ndarray

    @property
    def any_wet(self) -> np.ndarray:
        return self.bog | self.fen | self.swamp | self.lake


def topographic_wetness(acc_cells: np.ndarray, slope_deg: np.ndarray,
                        cell_m: float, smooth_cells: float = 2.0) -> np.ndarray:
    """Standard TWI, lightly smoothed.

    Feed this the **D-infinity** accumulation, not D8. Slope is floored so flats
    do not divide by zero, and the result is blurred a couple of cells to close
    the residual single-cell striping that any flow-routing scheme leaves.
    """
    a = np.maximum(acc_cells, 1.0) * cell_m
    tanb = np.maximum(np.tan(np.radians(np.nan_to_num(slope_deg, nan=0.0))), 1e-3)
    twi = np.log(a / tanb)
    return ndimage.gaussian_filter(twi, smooth_cells) if smooth_cells else twi


def _drop_small(mask: np.ndarray, min_ha: float, cell_m: float) -> np.ndarray:
    """Remove connected components below an area threshold.

    Written out rather than using ``skimage.remove_small_objects`` because that
    function's ``min_size``/``max_size`` semantics changed between versions, and
    a silent off-by-one here would quietly delete the smallest real patches.
    """
    if not mask.any():
        return mask
    min_cells = max(int(round(min_ha * 1e4 / (cell_m ** 2))), 3)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.arange(1, n + 1)] = sizes >= min_cells
    return keep[lab]


def find_lakes(elev: np.ndarray, slope_deg: np.ndarray, land: np.ndarray,
               cell_m: float) -> np.ndarray:
    """Inland flats that behave like a water surface."""
    flat = land & (np.nan_to_num(slope_deg, nan=99.0) < LAKE_SLOPE_MAX)
    lab, n = ndimage.label(flat)
    if n == 0:
        return np.zeros_like(flat)

    min_cells = LAKE_MIN_HA * 1e4 / (cell_m ** 2)
    sizes = ndimage.sum(flat, lab, range(1, n + 1))
    stds = ndimage.standard_deviation(elev, lab, range(1, n + 1))

    keep = np.zeros(n + 1, dtype=bool)
    ids = np.arange(1, n + 1)
    keep[ids] = (sizes >= min_cells) & (stds <= LAKE_ELEV_STD_MAX)
    return keep[lab]


def derive(elev: np.ndarray, slope_deg: np.ndarray, roughness: np.ndarray,
           acc_cells: np.ndarray, cell_m: float,
           sea_level: float = 1.0,
           shelter: np.ndarray | None = None) -> WetLayers:
    e = np.nan_to_num(elev, nan=-9999.0)
    land = e > sea_level
    slope = np.nan_to_num(np.asarray(slope_deg, dtype=float), nan=99.0)
    rough = np.nan_to_num(np.asarray(roughness, dtype=float), nan=0.0)

    twi = topographic_wetness(acc_cells, slope, cell_m)
    acc_km2 = acc_cells * (cell_m / 1000.0) ** 2
    lake = find_lakes(e, slope, land, cell_m)

    # One wet mask first, then split it. Deriving three masks independently
    # would let them overlap and double-draw the same ground.
    wet = (land & ~lake
           & (slope < MIRE_SLOPE_MAX)
           & (twi > MIRE_TWI_MIN)
           & (rough < MIRE_ROUGH_MAX)
           & (e < MIRE_ELEV_MAX))

    if shelter is None:
        from .woodland import shelter_index
        shelter = shelter_index(e, cell_m)

    throughflow = wet & (acc_km2 >= FEN_MIN_KM2)
    bog = wet & ~throughflow
    swamp = throughflow & (shelter >= SWAMP_SHELTER_MIN)
    fen = throughflow & ~swamp

    # Drop specks by AREA, not by shape. A 3x3 binary_opening was tried first
    # and is badly wrong here: fen and swamp follow valley bottoms, so they are
    # naturally linear, and opening erodes exactly those away -- it took fen
    # from 3.8% of land to 0.05%, and bog from 19.5% to 10.2%. Removing small
    # connected components keeps a one-cell-wide strath intact while still
    # discarding noise.
    # Fen and swamp get a lower floor than bog: they are valley-bottom strips
    # and are genuinely smaller features, not noise.
    bog = _drop_small(bog, BOG_MIN_PATCH_HA, cell_m)
    fen = _drop_small(fen, FLOW_MIN_PATCH_HA, cell_m)
    swamp = _drop_small(swamp, FLOW_MIN_PATCH_HA, cell_m)

    return WetLayers(twi=twi, bog=bog, fen=fen, swamp=swamp, lake=lake,
                     slope_deg=slope, roughness=rough)


def scatter(mask: np.ndarray, spacing_cells: float,
            rng: np.random.Generator) -> np.ndarray:
    """One jittered point per grid cell of ``spacing``, kept where mask is True.

    Cheap blue-noise: symbols land inside the actual wet patches rather than
    being spread evenly over the hex, so the map shows *where* the bog is.
    """
    rows, cols = mask.shape
    step = max(int(round(spacing_cells)), 2)
    gr = np.arange(0, rows, step)
    gc = np.arange(0, cols, step)
    if gr.size == 0 or gc.size == 0:
        return np.empty((0, 2))
    mr, mc = np.meshgrid(gr, gc, indexing="ij")
    jr = np.clip(mr + rng.integers(0, step, mr.shape), 0, rows - 1)
    jc = np.clip(mc + rng.integers(0, step, mc.shape), 0, cols - 1)
    ok = mask[jr, jc]
    return np.column_stack([jr[ok], jc[ok]]).astype(float)


def fractions(mask_stack: dict[str, np.ndarray], sel: np.ndarray) -> dict[str, float]:
    """Area fractions of each mask within a boolean selection (one hex)."""
    n = int(sel.sum())
    if n == 0:
        return {k: 0.0 for k in mask_stack}
    return {k: float((m & sel).sum()) / n for k, m in mask_stack.items()}
