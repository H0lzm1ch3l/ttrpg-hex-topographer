"""Shadowdark-style hexmap rendering: black ink, no greys, no raster effects.

Everything is drawn as SVG paths from the DEM. The look comes from four things,
in order of how much they matter:

1. **Symbolic, not photographic.** A hillshade is the wrong answer -- OSR maps
   read as *drawings*. Relief becomes inked peak and hill glyphs placed at the
   DEM's actual local maxima, so the symbols are true even though they are
   stylised.
2. **Coastal ripples.** Progressively offset copies of the coastline into the
   sea, from a distance transform. This single technique does more for the
   old-chart feel than anything else on the page.
3. **Wobble.** Every path is arc-length resampled and displaced by smooth
   noise, so no line is mechanically straight.
4. **Two tones only.** Paper and ink. Density, not grey, carries tone --
   hatching and stipple where a shaded map would use a ramp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dcfield

import numpy as np
from scipy import ndimage
from skimage import measure

from ..domain import hexgrid as hg
from .dem import Field, SEA_LEVEL_M
from .inkstyle import SVG, ValueNoise, hatch, path_d, resample, smooth_d, stipple, wobble

INK = "#111111"
PAPER = "#f4efe4"      # warm zine stock; use #ffffff for pure mono
PAPER_PURE = "#ffffff"

# Mutable one-slot holder so glyph helpers fill with the active paper colour
# without every call site having to pass it through.
PAPER_FILL = [PAPER]


# --------------------------------------------------------------------------
# Terrain reading
# --------------------------------------------------------------------------

@dataclass
class HexCell:
    q: int
    r: int
    cx: float          # centre in LAEA metres
    cy: float
    land_frac: float
    relief: float
    max_elev: float
    mean_elev: float
    slope_mean: float
    terrain: str
    label: str
    peaks: list = dcfield(default_factory=list)   # (x_m, y_m, elev, prominence)


# Absolute floors. Percentiles alone would call the top quartile of a flat
# region "mountains" -- the Biebrza marshes top out around 160 m and have no
# mountains at all. A threshold is therefore the *higher* of the region's own
# percentile and these.
MOUNTAIN_RELIEF_FLOOR = 350.0
MOUNTAIN_ELEV_FLOOR = 500.0
HILL_RELIEF_FLOOR = 120.0
HILL_SLOPE_FLOOR = 6.0


@dataclass
class TerrainScale:
    """Classification thresholds derived from one region's own distribution.

    Absolute metre thresholds do not travel. The first version was hand-tuned
    to Skye (mountain above 560 m); dropped on Kerry it would over-call, and on
    a lowland fen region it would find nothing. Taking percentiles of the
    region's own hexes and clamping them to floors makes the same code work in
    both places.
    """

    mountain_relief: float
    mountain_max_elev: float
    hill_relief: float
    hill_slope: float
    source: str = ""

    def describe(self) -> str:
        return (f"mountain: relief>{self.mountain_relief:.0f} m or peak>"
                f"{self.mountain_max_elev:.0f} m; hill: relief>{self.hill_relief:.0f} m "
                f"or slope>{self.hill_slope:.1f}°")


DEFAULT_SCALE = TerrainScale(500.0, 560.0, 230.0, 9.5, "fixed defaults")


def calibrate(cells: list["HexCell"]) -> TerrainScale:
    land = [c for c in cells if c.land_frac >= 0.10 and c.relief > 0.0]
    if len(land) < 8:
        return TerrainScale(*vars(DEFAULT_SCALE).values())

    relief = np.array([c.relief for c in land])
    maxe = np.array([c.max_elev for c in land])
    slope = np.array([c.slope_mean for c in land])
    return TerrainScale(
        mountain_relief=max(float(np.percentile(relief, 78)), MOUNTAIN_RELIEF_FLOOR),
        mountain_max_elev=max(float(np.percentile(maxe, 82)), MOUNTAIN_ELEV_FLOOR),
        hill_relief=max(float(np.percentile(relief, 38)), HILL_RELIEF_FLOOR),
        hill_slope=max(float(np.percentile(slope, 55)), HILL_SLOPE_FLOOR),
        source=f"percentiles over {len(land)} land hexes",
    )


def classify(land_frac: float, relief: float, max_elev: float, slope: float,
             scale: TerrainScale = DEFAULT_SCALE) -> str:
    """Elevation and slope only -- no land cover, which is honest for Skye and
    Kerry alike: both are sea, moor, hill and bare mountain."""
    if land_frac < 0.10:
        return "sea"
    if max_elev > scale.mountain_max_elev or relief > scale.mountain_relief:
        return "mountain"
    if relief > scale.hill_relief or slope > scale.hill_slope:
        return "hill"
    if land_frac < 0.80:
        return "coast"
    return "moor"


def find_peaks(fld: Field, min_elev: float = 140.0, sep_m: float = 1500.0,
               min_prom: float = 110.0) -> np.ndarray:
    """Local maxima with a crude prominence filter.

    Returns (N, 4): x_m, y_m, elevation, prominence. Prominence here is the
    drop to the lowest point of a neighbourhood -- enough to tell a summit from
    a bump on a ridge, which is all the glyph sizing needs.
    """
    e = np.nan_to_num(fld.elev, nan=-9999.0)
    k = max(int(round(sep_m / fld.m_per_px)), 3)
    mx = ndimage.maximum_filter(e, size=k, mode="nearest")
    mn = ndimage.minimum_filter(e, size=k * 2 + 1, mode="nearest")
    is_peak = (e == mx) & (e > min_elev) & ((e - mn) > min_prom)

    rows, cols = np.nonzero(is_peak)
    if rows.size == 0:
        return np.empty((0, 4))

    x0, y0, x1, y1 = fld.extent_m
    h, w = e.shape
    xs = x0 + cols / (w - 1) * (x1 - x0)
    ys = y1 - rows / (h - 1) * (y1 - y0)
    return np.column_stack([xs, ys, e[rows, cols], (e - mn)[rows, cols]])


def build_hexes(fld: Field, size_m: float, peaks: np.ndarray,
                scale: TerrainScale | None = None) -> tuple[list[HexCell], TerrainScale]:
    """Lay the flat-top lattice over the field and read each cell.

    Two passes: every hex's statistics are collected first, then thresholds are
    calibrated from that distribution, then terrain is assigned. Classifying
    inline would mean the thresholds could not depend on the region.
    """
    x0, y0, x1, y1 = fld.extent_m
    e = fld.elev
    slope = fld.slope_deg()
    h, w = e.shape

    # axial range that covers the rectangle, with a margin
    qs = range(int(math.floor(x0 / (1.5 * size_m))) - 1,
               int(math.ceil(x1 / (1.5 * size_m))) + 2)
    root3 = math.sqrt(3.0)

    cells: list[HexCell] = []
    for q in qs:
        rlo = int(math.floor((y0 / (root3 * size_m)) - 0.5 * q)) - 1
        rhi = int(math.ceil((y1 / (root3 * size_m)) - 0.5 * q)) + 2
        for r in range(rlo, rhi):
            cx, cy = hg.axial_to_xy(q, r, size_m)
            if not (x0 - size_m < cx < x1 + size_m and y0 - size_m < cy < y1 + size_m):
                continue

            # sample the hex on the field grid
            hw = hg.across_corners(size_m) / 2.0
            hh = hg.across_flats(size_m) / 2.0
            c_lo = int(max((cx - hw - x0) / (x1 - x0) * (w - 1), 0))
            c_hi = int(min((cx + hw - x0) / (x1 - x0) * (w - 1), w - 1))
            r_lo = int(max((y1 - (cy + hh)) / (y1 - y0) * (h - 1), 0))
            r_hi = int(min((y1 - (cy - hh)) / (y1 - y0) * (h - 1), h - 1))
            if c_hi <= c_lo or r_hi <= r_lo:
                continue

            sub = e[r_lo:r_hi + 1, c_lo:c_hi + 1]
            ss = slope[r_lo:r_hi + 1, c_lo:c_hi + 1]
            gx = x0 + np.arange(c_lo, c_hi + 1) / (w - 1) * (x1 - x0) - cx
            gy = y1 - np.arange(r_lo, r_hi + 1) / (h - 1) * (y1 - y0) - cy
            mgx, mgy = np.meshgrid(gx, gy)
            inside = hg.contains(mgx, mgy, size_m)
            if inside.sum() < 12:
                continue

            vals = np.nan_to_num(sub[inside], nan=-9999.0)
            land = vals[vals > SEA_LEVEL_M]
            land_frac = float(land.size) / float(vals.size)

            if land.size >= 8:
                relief = float(np.percentile(land, 98) - np.percentile(land, 2))
                max_elev = float(land.max())
                mean_elev = float(land.mean())
                sl = float(np.nanmean(ss[inside][vals > SEA_LEVEL_M]))
            else:
                relief = max_elev = mean_elev = sl = 0.0

            cell = HexCell(q=q, r=r, cx=cx, cy=cy, land_frac=land_frac,
                           relief=relief, max_elev=max_elev, mean_elev=mean_elev,
                           slope_mean=sl, terrain="", label="")
            if peaks.size:
                d = hg.contains(peaks[:, 0] - cx, peaks[:, 1] - cy, size_m)
                sel = peaks[d]
                if sel.size:
                    order = np.argsort(-sel[:, 3])
                    cell.peaks = [tuple(v) for v in sel[order]]
            cells.append(cell)

    scale = scale or calibrate(cells)
    for c in cells:
        c.terrain = classify(c.land_frac, c.relief, c.max_elev, c.slope_mean, scale)

    # Hexcrawl labels: columns numbered west to east, rows numbered north to
    # south *within* each column. Deriving the row from y directly gets the
    # half-hex column offset wrong, which is why this ranks per column instead.
    by_col: dict[int, list[HexCell]] = {}
    for c in cells:
        by_col.setdefault(c.q, []).append(c)
    if cells:
        qmin = min(by_col)
        for q, group in by_col.items():
            for i, c in enumerate(sorted(group, key=lambda z: -z.cy), start=1):
                c.label = f"{q - qmin + 1:02d}{i:02d}"
    return cells, scale


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

@dataclass
class Layout:
    map_w: float
    map_h: float
    left: float
    top: float
    page_w: float
    page_h: float
    fld: Field

    def m_to_px(self, x, y):
        x0, y0, x1, y1 = self.fld.extent_m
        px = self.left + (np.asarray(x) - x0) / (x1 - x0) * self.map_w
        py = self.top + (y1 - np.asarray(y)) / (y1 - y0) * self.map_h
        return px, py

    def rc_to_px(self, rows, cols, shape=None):
        """Grid indices to page pixels. ``shape`` lets a coarser working grid
        (the hydrology one, say) map onto the same page."""
        h, w = shape if shape is not None else self.fld.shape
        return (self.left + np.asarray(cols) / (w - 1) * self.map_w,
                self.top + np.asarray(rows) / (h - 1) * self.map_h)

    @property
    def px_per_m(self) -> float:
        x0, _, x1, _ = self.fld.extent_m
        return self.map_w / (x1 - x0)


# --------------------------------------------------------------------------
# Glyphs
# --------------------------------------------------------------------------

def mountain_glyph(svg: SVG, x: float, y: float, half_w: float, height: float,
                   noise: ValueNoise, seed: float, lw: float = 2.2,
                   shoulders: int = 2) -> None:
    """Outlined peak, paper-filled, with the right face solid black.

    Hatched flanks were tried first and lose at map scale: at 30-40 px a peak
    has room for maybe six strokes, which reads as a comb rather than a
    mountain. A solid shadow face carries the same information -- lit from the
    upper left, like every other relief convention -- and survives being
    printed small. Shoulders are drawn before the main peak so it overlaps them.
    """
    order: list[tuple[float, float]] = []
    if shoulders >= 1:
        order.append((-half_w * 0.88, 0.60))
    if shoulders >= 2:
        order.append((half_w * 0.92, 0.48))
    order.append((0.0, 1.0))

    for ox, sc in order:
        hh, ww = height * sc, half_w * sc
        apex = np.array([x + ox, y - hh])
        bl = np.array([x + ox - ww, y])
        br = np.array([x + ox + ww, y])
        tri = wobble(resample(np.vstack([bl, apex, br, bl]), max(hh / 8.0, 2.0)),
                     noise, amp=hh * 0.025, scale=28, phase=seed + ox)
        svg.path(smooth_d(tri, close=True), width=lw, stroke=INK, fill=PAPER_FILL[0])
        shade = wobble(
            resample(np.vstack([apex, br, np.array([x + ox + ww * 0.16, y]), apex]),
                     max(hh / 8.0, 2.0)),
            noise, amp=hh * 0.02, scale=28, phase=seed + ox + 5)
        svg.path(smooth_d(shade, close=True), width=1.0, stroke=INK, fill=INK)


def hill_glyph(svg: SVG, x: float, y: float, w: float, h: float,
               noise: ValueNoise, seed: float, lw: float = 2.0) -> None:
    """Rounded hump, paper-filled, right flank in solid shadow.

    Closed with a straight base rather than a smoothed one: running the
    Catmull-Rom through the base corners loops it and leaves ink blobs at both
    ends of the hill.
    """
    t = np.linspace(math.pi, 0.0, 20)
    dome = np.column_stack([x + np.cos(t) * w / 2.0, y - np.sin(t) * h])
    dome = wobble(dome, noise, amp=h * 0.05, scale=26, phase=seed)
    svg.path(smooth_d(dome) + "Z", width=lw, stroke=INK, fill=PAPER_FILL[0])

    t2 = np.linspace(math.pi * 0.40, 0.0, 12)
    shade = np.column_stack([x + np.cos(t2) * w / 2.0, y - np.sin(t2) * h])
    shade = wobble(shade, noise, amp=h * 0.04, scale=26, phase=seed + 4)
    svg.path(smooth_d(shade) + "Z", width=1.0, stroke=INK, fill=INK)


def moor_tuft(svg: SVG, x: float, y: float, s: float, lw: float = 1.3) -> None:
    """Two splayed ticks -- reads as rough grazing, not forest. Skye is
    effectively treeless, so a tree symbol here would be a lie."""
    d = (f"M{x - s*0.5:.1f},{y:.1f}q{s*0.25:.1f},{-s*0.9:.1f} {s*0.45:.1f},{-s*0.15:.1f}"
         f"M{x + s*0.15:.1f},{y:.1f}q{s*0.2:.1f},{-s*1.0:.1f} {s*0.5:.1f},{-s*0.25:.1f}")
    svg.path(d, width=lw, stroke=INK)


def bog_glyph(svg: SVG, x: float, y: float, s_: float = 7.0, lw: float = 1.5) -> None:
    """Stacked short horizontals -- the standard peat/bog symbol.

    No reeds: blanket bog is not reedbed, and keeping the two symbols distinct
    is what lets a reader tell wet moor from a fen at a glance.
    """
    d = (f"M{x - s_:.1f},{y:.1f}h{s_ * 1.5:.1f}"
         f"M{x - s_ * 0.55:.1f},{y + s_ * 0.42:.1f}h{s_ * 1.05:.1f}"
         f"M{x - s_ * 0.3:.1f},{y - s_ * 0.42:.1f}h{s_ * 0.85:.1f}")
    svg.path(d, width=lw, stroke=INK, cap="butt")


def fen_glyph(svg: SVG, x: float, y: float, s_: float = 7.0, lw: float = 1.5) -> None:
    """Horizontals plus upright reeds -- open minerotrophic fen.

    Reeds are the tell. A fen has water moving through it and grows sedge and
    reed; a bog is rain-fed and does not. Keeping the symbols distinct is what
    lets a reader tell wet moor from a fen at a glance.
    """
    d = (f"M{x - s_:.1f},{y:.1f}h{s_ * 1.6:.1f}"
         f"M{x - s_ * 0.5:.1f},{y + s_ * 0.45:.1f}h{s_ * 1.0:.1f}")
    svg.path(d, width=lw, stroke=INK, cap="butt")
    r = (f"M{x - s_ * 0.45:.1f},{y - s_ * 0.1:.1f}l{-s_ * 0.12:.1f},{-s_ * 0.72:.1f}"
         f"M{x:.1f},{y - s_ * 0.1:.1f}l0,{-s_ * 0.9:.1f}"
         f"M{x + s_ * 0.45:.1f},{y - s_ * 0.1:.1f}l{s_ * 0.12:.1f},{-s_ * 0.72:.1f}")
    svg.path(r, width=lw * 0.8, stroke=INK)


def swamp_glyph(svg: SVG, x: float, y: float, s_: float, noise: ValueNoise,
                seed: float, lw: float = 1.4) -> None:
    """Water lines with a tree standing in them -- alder and willow carr.

    Same water regime as a fen; the difference is that it is wooded, so the
    symbol is the fen's water lines carrying a miniature of the wood glyph.
    """
    d = (f"M{x - s_ * 1.15:.1f},{y:.1f}h{s_ * 1.0:.1f}"
         f"M{x + s_ * 0.30:.1f},{y:.1f}h{s_ * 0.85:.1f}"
         f"M{x - s_ * 0.75:.1f},{y + s_ * 0.45:.1f}h{s_ * 1.5:.1f}")
    svg.path(d, width=lw, stroke=INK, cap="butt")
    tree_glyph(svg, x, y - s_ * 0.05, s_ * 1.9, noise, seed, lw=lw * 0.9)


def tree_glyph(svg: SVG, x: float, y: float, h: float, noise: ValueNoise,
               seed: float, lw: float = 1.6) -> None:
    """Round-canopy broadleaf on a visible trunk, shaded lower-right.

    ``h`` is the total height from the ground line. Proportions matter more
    than they look: an earlier version centred the canopy so low that it
    swallowed the trunk, and the symbol read as a bubble rather than a tree.
    Ten shallow lobes read as foliage; six deep ones read as a star.

    Broadleaf rather than conifer by default -- Skye's native woodland is birch
    and hazel, and drawing spruce triangles would assert a plantation that
    neither the model nor WorldCover distinguishes.
    """
    svg.path(f"M{x:.1f},{y:.1f}L{x:.1f},{y - h * 0.42:.1f}",
             width=lw * 1.15, stroke=INK)

    cy = y - h * 0.66
    r0 = h * 0.36
    t = np.linspace(0.0, 2 * math.pi, 140)
    r = r0 * (1.0 + 0.09 * np.cos(10 * t + seed))
    pts = np.column_stack([x + np.cos(t) * r, cy - np.sin(t) * r * 0.92])
    svg.path(smooth_d(pts, close=True), width=lw, stroke=INK, fill=PAPER_FILL[0])

    # shadow crescent on the lower right, matching the lighting on the peaks
    t2 = np.linspace(-0.62 * math.pi, 0.18 * math.pi, 50)
    ro = r0 * (1.0 + 0.09 * np.cos(10 * t2 + seed))
    outer = np.column_stack([x + np.cos(t2) * ro, cy - np.sin(t2) * ro * 0.92])
    inner = np.column_stack([x + np.cos(t2) * ro * 0.52,
                             cy - np.sin(t2) * ro * 0.52 * 0.92])
    svg.path(smooth_d(np.vstack([outer, inner[::-1]]), close=True),
             width=0.7, stroke=INK, fill=INK)


def crag_ticks(svg: SVG, pts: np.ndarray, lw: float = 1.4) -> None:
    """Hachure ticks along a steep break of slope."""
    if len(pts) < 2:
        return
    segs = []
    for i in range(0, len(pts) - 1, 2):
        a, b = pts[i], pts[i + 1]
        v = b - a
        L = np.hypot(*v)
        if L < 1e-6:
            continue
        nrm = np.array([-v[1], v[0]]) / L
        segs.append(np.array([a, a + nrm * 4.0]))
    svg.segments(segs, width=lw, stroke=INK)


def draw_lakes(svg: SVG, lay: "Layout", lake: np.ndarray, shape, noise: ValueNoise) -> int:
    """Outline each loch and rule a couple of water lines inside it."""
    n = 0
    for c in measure.find_contours(lake.astype(float), 0.5):
        if len(c) < 16:
            continue
        px, py = lay.rc_to_px(c[:, 0], c[:, 1], shape=shape)
        pts = resample(np.column_stack([px, py]), 3.2)
        if len(pts) < 6:
            continue
        pts = wobble(pts, noise, amp=0.7, scale=30, phase=n * 2.1)
        svg.path(smooth_d(pts, close=True), width=1.9, stroke=INK, fill=PAPER_FILL[0])

        y0, y1 = pts[:, 1].min(), pts[:, 1].max()
        x0, x1 = pts[:, 0].min(), pts[:, 0].max()
        segs = []
        for yy in np.arange(y0 + 5.0, y1 - 2.0, 10.0):
            inside = pts[np.abs(pts[:, 1] - yy) < 3.8]
            if len(inside) >= 2:
                a, b = inside[:, 0].min(), inside[:, 0].max()
                if b - a > 7:
                    pad = (b - a) * 0.18
                    segs.append(np.array([[a + pad, yy], [b - pad, yy]]))
        svg.segments(segs, width=0.8, stroke=INK, opacity=0.38)
        n += 1
    return n


def draw_wetland(svg: SVG, lay: "Layout", wl, shape, rng, noise: ValueNoise,
                 spacing_px: float = 30.0,
                 px_per_cell: float = 1.0, land: np.ndarray | None = None
                 ) -> dict[str, int]:
    """Scatter bog, fen and swamp symbols inside their own patches.

    Spacing widens where the wetland covers more of the region. Skye is 17% bog
    and a fixed spacing suits it; Kerry is 26% and the same spacing carpeted the
    page, taking total ink from 9.6% to 12.8% and burying the relief. A symbol
    scatter has to indicate extent, not tile it.
    """
    from . import wetland as W

    cover = 1.0
    if land is not None and land.any():
        frac = float((wl.bog | wl.fen | wl.swamp).sum()) / float(land.sum())
        cover = max(frac / 0.15, 1.0) ** 0.5      # 15% coverage is the baseline
    sc = max(spacing_px * cover / max(px_per_cell, 1e-6), 2.0)
    counts: dict[str, int] = {}

    for name, mask, sp in (("bog", wl.bog, sc),
                           ("fen", wl.fen, sc * 1.05),
                           ("swamp", wl.swamp, sc * 1.35)):
        pts = W.scatter(mask, sp, rng)
        for i, (r, c) in enumerate(pts):
            gx, gy = lay.rc_to_px(r, c, shape=shape)
            size = float(rng.uniform(5.5, 8.0))
            if name == "bog":
                bog_glyph(svg, float(gx), float(gy), size)
            elif name == "fen":
                fen_glyph(svg, float(gx), float(gy), size)
            else:
                swamp_glyph(svg, float(gx), float(gy), size, noise, i)
        counts[name] = len(pts)
    return counts


def draw_woodland(svg: SVG, lay: "Layout", wood: np.ndarray, shape, rng,
                  noise: ValueNoise, spacing_px: float = 27.0,
                  px_per_cell: float = 1.0) -> int:
    from . import wetland as W

    pts = W.scatter(wood, max(spacing_px / max(px_per_cell, 1e-6), 2.0), rng)
    for i, (r, c) in enumerate(pts):
        gx, gy = lay.rc_to_px(r, c, shape=shape)
        tree_glyph(svg, float(gx), float(gy), float(rng.uniform(20.0, 27.0)), noise, i)
    return len(pts)


def draw_hachures(svg: SVG, lay: "Layout", slope_deg: np.ndarray, aspect_deg: np.ndarray,
                  land: np.ndarray, rng, shape,
                  min_slope: float = 10.0, spacing_px: float = 6.0,
                  light_bearing: float = 315.0, max_slope: float = 36.0,
                  strength: float = 1.0) -> int:
    """Lehmann hachures: short strokes running straight downslope.

    This is the pre-contour answer to showing three dimensions in pure line
    art, and it is what the map was missing. Discrete glyphs tell you a hex is
    mountainous; they do not tell you where the ridge runs or which way the
    ground falls. Hachures do both, because each stroke *is* the fall line.

    Following Lehmann: steeper ground gets shorter, heavier, denser strokes.
    Weight is additionally modulated by illumination, so slopes facing away from
    the north-west light carry more ink and the relief reads as shaded without
    a single grey pixel.
    """
    rows, cols = shape
    step_r = max(spacing_px / (lay.map_h / (rows - 1)), 1.0)
    step_c = max(spacing_px / (lay.map_w / (cols - 1)), 1.0)

    rr = np.arange(0, rows, step_r)
    cc = np.arange(0, cols, step_c)
    gr, gc = np.meshgrid(rr, cc, indexing="ij")
    gr = np.clip(gr + rng.uniform(-step_r * .4, step_r * .4, gr.shape), 0, rows - 1)
    gc = np.clip(gc + rng.uniform(-step_c * .4, step_c * .4, gc.shape), 0, cols - 1)
    ri = gr.astype(int).ravel()
    ci = gc.astype(int).ravel()

    sl = np.nan_to_num(slope_deg[ri, ci], nan=0.0)
    asp = np.nan_to_num(aspect_deg[ri, ci], nan=-1.0)
    ok = land[ri, ci] & (sl >= min_slope) & (asp >= 0.0)
    if not ok.any():
        return 0

    ri, ci, sl, asp = ri[ok], ci[ok], sl[ok], asp[ok]
    t = np.clip((sl - min_slope) / (max_slope - min_slope), 0.0, 1.0)

    # Density proportional to steepness. A uniform scatter was the first
    # version's mistake -- even weight everywhere reads as fur, because the eye
    # gets no gradient to infer form from. Lehmann's whole point is that ink
    # per unit area *is* the slope.
    keep = rng.random(t.shape) < (0.18 + 0.82 * t)
    ri, ci, sl, asp, t = ri[keep], ci[keep], sl[keep], asp[keep], t[keep]
    if not len(ri):
        return 0
    px, py = lay.rc_to_px(ri, ci, shape=shape)

    length = 8.0 - 3.6 * t                       # steeper -> shorter
    theta = np.radians(asp)
    dx, dy = np.sin(theta), -np.cos(theta)       # downslope, screen y is south

    # 0 where the face is lit from the NW, 1 where it is in shadow. The strong
    # lit/shadow split is what turns a texture into modelled relief.
    shade = (1.0 - np.cos(np.radians(asp - light_bearing))) / 2.0
    width = (0.40 + 1.70 * t * (0.20 + 0.80 * shade)) * strength
    alpha = np.clip((0.16 + 0.74 * shade) * (0.40 + 0.60 * t) * strength, 0.05, 0.92)

    # bucket by weight so near-identical strokes share one path element
    drawn = 0
    for lo, hi in ((0.0, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 0.85), (0.85, 1.01)):
        m = (shade >= lo) & (shade < hi)
        if not m.any():
            continue
        d = "".join(
            f"M{a:.1f},{b:.1f}l{c:.1f},{e:.1f}"
            for a, b, c, e in zip(px[m], py[m], dx[m] * length[m], dy[m] * length[m])
        )
        svg.path(d, width=float(np.median(width[m])), stroke=INK,
                 opacity=float(np.median(alpha[m])), cap="round")
        drawn += int(m.sum())
    return drawn


# Line weight by catchment band: a burn you step over up to something needing
# a bridge.
RIVER_WIDTH = {1: 0.9, 2: 1.5, 3: 2.2, 4: 3.0}


def draw_rivers(svg: SVG, lay: "Layout", reaches, hydro_shape, noise: ValueNoise,
                min_band: int = 1) -> int:
    """Draw traced reaches, tapering each by the catchment it actually carries.

    Coastlines and rivers were reading alike -- both mid-weight wobbled black
    lines. The distinction that fixes it is also the truthful one: a coast has
    uniform weight along its length, a river does not. Width here follows
    ``acc_km2`` point by point, so a burn visibly thickens as tributaries join
    it, and the largest reaches are drawn with two banks rather than one line.

    Slope still sets character: a torrent is confined by its gorge and runs
    nearly straight, a sluggish reach over soft ground wanders.
    """
    amp_for = {"torrent": 0.55, "normal": 1.1, "sluggish": 2.0}
    drawn = 0

    for i, rch in enumerate(reaches):
        if rch.band < min_band:
            continue
        px, py = lay.rc_to_px(rch.pts[:, 0], rch.pts[:, 1], shape=hydro_shape)
        raw = np.column_stack([px, py])
        if len(raw) < 4:
            continue

        # carry accumulation through the arc-length resampling
        acc = rch.acc_km2 if rch.acc_km2 is not None else np.full(len(raw), 1.0)
        seg = np.hypot(*(raw[1:] - raw[:-1]).T)
        sdist = np.concatenate([[0.0], np.cumsum(seg)])
        if sdist[-1] < 4.0:
            continue
        tt = np.arange(0.0, sdist[-1], 4.0)
        pts = np.column_stack([np.interp(tt, sdist, raw[:, 0]),
                               np.interp(tt, sdist, raw[:, 1])])
        acc_i = np.interp(tt, sdist, acc)
        pts = wobble(pts, noise, amp=amp_for[rch.kind], scale=34, phase=i * 1.7)

        w = _river_width(acc_i)
        if rch.band >= 4 and float(np.median(w)) >= 2.6:
            _draw_banks(svg, pts, w)
        else:
            _draw_tapered(svg, pts, w)
        drawn += 1

        if rch.kind == "torrent" and rch.band >= 2:
            segs = []
            for j in range(4, len(pts) - 2, 13):
                v = pts[j + 1] - pts[j - 1]
                L = float(np.hypot(*v))
                if L < 1e-6:
                    continue
                nrm = np.array([-v[1], v[0]]) / L
                segs.append(np.array([pts[j] - nrm * 2.4, pts[j] + nrm * 2.4]))
            svg.segments(segs, width=float(np.median(w)) * 0.7, stroke=INK)
    return drawn


def _river_width(acc_km2: np.ndarray) -> np.ndarray:
    """Catchment to stroke width. Cube root because channel width scales with
    something close to the cube root of discharge, and it keeps a 90 km²
    catchment from being thirty times the weight of a 1 km² one."""
    return np.clip(0.55 + 1.15 * np.cbrt(np.maximum(acc_km2, 0.05)), 0.6, 4.2)


def _draw_tapered(svg: SVG, pts: np.ndarray, w: np.ndarray, chunk: int = 6) -> None:
    """One path per width band, so the taper is visible without emitting a
    separate element per segment."""
    for k in range(0, len(pts) - 1, chunk):
        seg = pts[k:k + chunk + 1]
        if len(seg) < 2:
            continue
        svg.path(smooth_d(seg), width=float(w[k:k + chunk + 1].mean()), stroke=INK)


def _draw_banks(svg: SVG, pts: np.ndarray, w: np.ndarray) -> None:
    """Two banks instead of one line -- the classic symbol for a river wide
    enough to matter, and unmistakable against a coastline."""
    d = np.gradient(pts, axis=0)
    L = np.hypot(d[:, 0], d[:, 1])
    L[L < 1e-6] = 1.0
    nrm = np.column_stack([-d[:, 1] / L, d[:, 0] / L])
    half = (w * 0.42)[:, None]
    for side in (+1.0, -1.0):
        svg.path(smooth_d(pts + nrm * half * side), width=1.25, stroke=INK)


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------

@dataclass
class MapStyle:
    paper: str = PAPER
    px_per_km: float = 26.0
    hex_lw: float = 1.15
    coast_lw: float = 3.4
    ripples: int = 4
    ripple_step_km: float = 1.25
    show_labels: bool = True
    show_hex_ids: bool = True
    hachure_strength: float = 1.7
    seed: int = 11


def render(fld: Field, size_m: float, title: str, subtitle: str,
           style: MapStyle | None = None,
           places: list[tuple[str, float, float]] | None = None,
           reaches=None, hydro_shape=None, wet=None, wood=None,
           slope_deg=None, aspect_deg=None
           ) -> tuple[SVG, list[HexCell], TerrainScale]:
    st = style or MapStyle()
    PAPER_FILL[0] = st.paper
    noise = ValueNoise(seed=st.seed)
    rng = np.random.default_rng(st.seed)

    x0, y0, x1, y1 = fld.extent_m
    map_w = (x1 - x0) / 1000.0 * st.px_per_km
    map_h = (y1 - y0) / 1000.0 * st.px_per_km

    margin = 46.0
    title_h = 128.0
    legend_h = 152.0
    page_w = map_w + margin * 2
    page_h = map_h + margin * 2 + title_h + legend_h
    lay = Layout(map_w, map_h, margin, margin + title_h, page_w, page_h, fld)

    svg = SVG(page_w, page_h, bg=st.paper)
    # Everything from here to close_group() is clipped to the map rectangle.
    # Without this the lattice runs under the title slab and past the frame,
    # because hexes are generated with a margin beyond the region.
    svg.clip_rect("mapclip", margin, margin + title_h, map_w, map_h)
    svg.open_group(clip_path="url(#mapclip)")

    peaks = find_peaks(fld)
    cells, scale = build_hexes(fld, size_m, peaks)

    # ---- sea: ripples then coastline ------------------------------------
    sea = fld.sea
    land = ~sea
    dist_px = ndimage.distance_transform_edt(sea) * fld.m_per_px / 1000.0  # km into sea
    smooth_e = ndimage.gaussian_filter(np.nan_to_num(fld.elev, nan=-50.0), 1.2)

    for i in range(st.ripples):
        lvl = (i + 1) * st.ripple_step_km
        try:
            contours = measure.find_contours(dist_px, lvl)
        except Exception:
            continue
        fade = (1.0 - i / st.ripples) ** 1.6
        for c in contours:
            if len(c) < 26:
                continue
            px, py = lay.rc_to_px(c[:, 0], c[:, 1])
            pts = resample(np.column_stack([px, py]), 5.0)
            if len(pts) < 6:
                continue
            pts = wobble(pts, noise, amp=1.5 + i * 0.35, scale=55, phase=i * 3.1)
            svg.path(smooth_d(pts), width=1.0, stroke=INK, opacity=0.16 + 0.46 * fade)

    for c in measure.find_contours(smooth_e, SEA_LEVEL_M):
        if len(c) < 14:
            continue
        px, py = lay.rc_to_px(c[:, 0], c[:, 1])
        pts = resample(np.column_stack([px, py]), 3.6)
        if len(pts) < 5:
            continue
        pts = wobble(pts, noise, amp=0.85, scale=40, phase=1.7)
        svg.path(smooth_d(pts), width=st.coast_lw, stroke=INK)

    # ---- hachures --------------------------------------------------------
    # Under everything else: they are the ground the rest sits on.
    if slope_deg is not None and aspect_deg is not None:
        n_h = draw_hachures(svg, lay, slope_deg, aspect_deg, ~fld.sea, rng,
                            fld.shape, strength=st.hachure_strength)
        svg.add(f"<!-- {n_h} hachures -->")

    # ---- lochs -----------------------------------------------------------
    if wet is not None:
        px_per_cell = lay.map_w / (hydro_shape[1] - 1)
        n_lake = draw_lakes(svg, lay, wet.lake, hydro_shape, noise)
        svg.add(f"<!-- {n_lake} lochs -->")

    # ---- watercourses ----------------------------------------------------
    # Drawn after the coast but before the landforms, so peaks overlap the
    # burns that run past them rather than the other way round.
    if reaches:
        n_drawn = draw_rivers(svg, lay, reaches, hydro_shape, noise)
        svg.add(f"<!-- {n_drawn} reaches -->")

    # ---- wetland and woodland -------------------------------------------
    # Below the landforms: a mountain should overlap the bog at its foot.
    if wet is not None:
        hland = np.nan_to_num(fld.elev, nan=-9999.0) > SEA_LEVEL_M
        wc = draw_wetland(svg, lay, wet, hydro_shape, rng, noise,
                          px_per_cell=px_per_cell,
                          land=~wet.lake & (wet.twi > -1e9))
        svg.add(f"<!-- wetland symbols: {wc} -->")
    if wood is not None:
        nw = draw_woodland(svg, lay, wood, hydro_shape, rng, noise,
                           px_per_cell=lay.map_w / (hydro_shape[1] - 1))
        svg.add(f"<!-- {nw} trees -->")

    # ---- terrain glyphs --------------------------------------------------
    ppm = lay.px_per_m
    for cell in cells:
        if cell.terrain == "sea":
            continue
        cxp, cyp = lay.m_to_px(cell.cx, cell.cy)
        seedv = (cell.q * 31 + cell.r * 17) % 997

        if cell.terrain in ("mountain", "hill"):
            k = 4 if cell.terrain == "mountain" else 5
            drawn = 0
            used: list[tuple[float, float]] = []
            for (pxm, pym, pe, prom) in cell.peaks:
                if drawn >= k:
                    break
                gx, gy = lay.m_to_px(pxm, pym)
                sep = 40.0 if cell.terrain == "mountain" else 26.0
                if any(math.hypot(gx - ux, gy - uy) < sep for ux, uy in used):
                    continue
                used.append((gx, gy))
                if cell.terrain == "mountain":
                    hgt = 19.0 + min(prom, 520.0) / 520.0 * 22.0
                    sh = 2 if prom > 260 else (1 if prom > 130 else 0)
                    mountain_glyph(svg, gx, gy + hgt * 0.40, hgt * 0.80, hgt,
                                   noise, seedv + drawn, lw=2.2, shoulders=sh)
                else:
                    hh = 9.5 + min(prom, 260.0) / 260.0 * 8.0
                    hill_glyph(svg, gx, gy + hh * 0.4, hh * 2.9, hh,
                               noise, seedv + drawn, lw=2.0)
                drawn += 1
            if drawn == 0 and cell.land_frac > 0.25:
                hill_glyph(svg, cxp, cyp, 32, 11, noise, seedv, lw=2.0)

        if cell.terrain in ("moor", "coast", "hill", "mountain"):
            # Rough grazing between the landforms. Without this the land reads
            # as blank paper wherever no peak happened to be detected.
            base = {"moor": 26, "coast": 20, "hill": 14, "mountain": 7}[cell.terrain]
            n = int(base * (0.4 + 0.6 * cell.land_frac))
            hw = hg.across_corners(size_m) / 2.0 * ppm * 0.72
            hh = hg.across_flats(size_m) / 2.0 * ppm * 0.72
            for _ in range(n):
                ox = rng.uniform(-hw, hw)
                oy = rng.uniform(-hh, hh)
                if abs(oy) > hh - abs(ox) * 0.5:
                    continue
                gx, gy = cxp + ox, cyp + oy
                # keep tufts on land
                fc = int(np.clip((gx - lay.left) / lay.map_w * (fld.shape[1] - 1), 0, fld.shape[1] - 1))
                fr = int(np.clip((gy - lay.top) / lay.map_h * (fld.shape[0] - 1), 0, fld.shape[0] - 1))
                if sea[fr, fc]:
                    continue
                # don't stipple grazing over ground already marked bog, fen or wood
                if wet is not None or wood is not None:
                    hc = int(np.clip((gx - lay.left) / lay.map_w * (hydro_shape[1] - 1),
                                     0, hydro_shape[1] - 1))
                    hr = int(np.clip((gy - lay.top) / lay.map_h * (hydro_shape[0] - 1),
                                     0, hydro_shape[0] - 1))
                    if wet is not None and wet.any_wet[hr, hc]:
                        continue
                    if wood is not None and wood[hr, hc]:
                        continue
                moor_tuft(svg, gx, gy, rng.uniform(7.0, 10.5))

    # ---- hex lattice -----------------------------------------------------
    for cell in cells:
        corners = hg.hex_corners(size_m, cell.cx, cell.cy)
        px, py = lay.m_to_px(corners[:, 0], corners[:, 1])
        pts = resample(np.vstack([np.column_stack([px, py]),
                                  np.array([[px[0], py[0]]])]), 7.0)
        pts = wobble(pts, noise, amp=0.9, scale=70, phase=cell.q * 2.3 + cell.r)
        svg.path(smooth_d(pts), width=st.hex_lw, stroke=INK, opacity=0.62)

        if st.show_hex_ids:
            lx, ly = lay.m_to_px(cell.cx, cell.cy + hg.across_flats(size_m) * 0.34)
            svg.text(lx, ly, cell.label, size=9.5, family="DejaVu Serif Condensed",
                     fill=INK, opacity=0.55, spacing=0.6, halo=st.paper, halo_w=2.6)

    # ---- place labels ----------------------------------------------------
    if places and st.show_labels:
        for name, lon, lat in places:
            mx, my = fld.from_lonlat(lon, lat)
            gx, gy = lay.m_to_px(mx, my)
            if not (lay.left <= gx <= lay.left + map_w and lay.top <= gy <= lay.top + map_h):
                continue
            svg.text(gx, gy + 2.5, name.upper(), size=12.0,
                     family="DejaVu Serif Condensed", fill=INK, spacing=2.0,
                     halo=st.paper, halo_w=4.0)

    svg.close_group()
    _frame(svg, lay, st, title, subtitle, size_m, cells)
    return svg, cells, scale


def _frame(svg: SVG, lay: Layout, st: MapStyle, title: str, subtitle: str,
           size_m: float, cells: list[HexCell]) -> None:
    """Heavy border, reversed-out title slab, legend, scale bar, compass."""
    W, H = lay.page_w, lay.page_h
    m = 16.0
    svg.rect(m, m, W - 2 * m, H - 2 * m, fill="none", stroke=INK, width=5.0)
    svg.rect(m + 9, m + 9, W - 2 * m - 18, H - 2 * m - 18, fill="none", stroke=INK, width=1.4)

    # title slab
    sh = 74.0
    sx, sy = m + 26, m + 20
    sw = W - 2 * (m + 26)
    svg.rect(sx, sy, sw, sh, fill=INK)
    svg.text(sx + sw / 2, sy + 34, title.upper(), size=32,
             family="DejaVu Serif Condensed", fill=st.paper, weight="bold", spacing=7.5)
    svg.text(sx + sw / 2, sy + 60, subtitle, size=13.5,
             family="LM Roman Caps 10", fill=st.paper, spacing=3.2, opacity=0.9)

    # Legend. Two rows: nine entries no longer fit on one at this page width.
    ly = H - m - 92
    lx = m + 34
    noise = ValueNoise(seed=st.seed + 5)
    svg.text(lx, ly - 12, "TERRAIN", size=11, family="DejaVu Serif Condensed",
             fill=INK, anchor="start", weight="bold", spacing=2.4)

    def label(px, py, text):
        svg.text(px, py, text, size=10.5, family="LM Roman Caps 10",
                 fill=INK, anchor="start", spacing=1.2)

    col = 96.0
    r1, r2 = ly + 14, ly + 46          # baselines of the two rows

    gx = lx + 8
    mountain_glyph(svg, gx, r1 + 6, 11, 20, noise, 3, lw=1.9, shoulders=1)
    label(gx + 20, r1, "MOUNTAIN"); gx += col + 20
    hill_glyph(svg, gx, r1 + 6, 26, 10, noise, 9, lw=1.9)
    label(gx + 18, r1, "HILL"); gx += col - 22
    moor_tuft(svg, gx - 4, r1 + 5, 8); moor_tuft(svg, gx + 5, r1 + 3, 7)
    label(gx + 18, r1, "MOOR"); gx += col - 16
    tree_glyph(svg, gx, r1 + 9, 20, noise, 2, lw=1.5)
    label(gx + 14, r1, "WOOD")

    gx = lx + 8
    bog_glyph(svg, gx, r2, 7.0)
    label(gx + 16, r2, "BOG"); gx += col - 8
    fen_glyph(svg, gx, r2, 7.0)
    label(gx + 16, r2, "FEN"); gx += col - 12
    swamp_glyph(svg, gx, r2 - 1, 6.0, noise, 4)
    label(gx + 16, r2, "SWAMP"); gx += col + 6
    rv = np.array([[gx - 8, r2 + 6], [gx + 1, r2 - 3], [gx + 9, r2 + 4], [gx + 18, r2 - 6]])
    svg.path(smooth_d(wobble(resample(rv, 3.0), noise, amp=0.8, scale=20)),
             width=2.0, stroke=INK)
    label(gx + 24, r2, "RIVER"); gx += col
    svg.segments([np.array([[gx - 6, r2 - 5 + i * 4.5], [gx + 14, r2 - 5 + i * 4.5]])
                  for i in range(3)], width=1.0, opacity=0.6)
    label(gx + 22, r2, "SEA")

    # Scale bar: exactly one hex across the flats. Right-aligned from its own
    # width -- a fixed offset ran it off the page.
    hexpx = hg.across_flats(size_m) * lay.px_per_m
    bx = W - m - 40 - hexpx
    by = ly + 20
    svg.path(path_d(np.array([[bx, by], [bx + hexpx, by]])), width=3.0, stroke=INK)
    for t in (0, 1):
        svg.path(path_d(np.array([[bx + t * hexpx, by - 6], [bx + t * hexpx, by + 6]])),
                 width=2.0, stroke=INK)
    miles = hg.across_flats(size_m) / hg.MILE_M
    svg.text(bx + hexpx / 2, by + 21, f"{miles:.0f} MILES — ONE HEX", size=10.5,
             family="LM Roman Caps 10", fill=INK, spacing=1.4)

    # compass, tucked left of the scale bar
    cxp, cyp = bx - 52, ly + 16
    svg.path(path_d(np.array([[cxp, cyp + 16], [cxp, cyp - 16]])), width=2.0, stroke=INK)
    svg.path(path_d(np.array([[cxp - 5, cyp - 6], [cxp, cyp - 17], [cxp + 5, cyp - 6]]),
                    close=True), width=1.6, stroke=INK, fill=INK)
    svg.text(cxp, cyp - 22, "N", size=12, family="DejaVu Serif Condensed",
             fill=INK, weight="bold")

    counts: dict[str, int] = {}
    for c in cells:
        counts[c.terrain] = counts.get(c.terrain, 0) + 1
    land = sum(v for k, v in counts.items() if k != "sea")
    svg.text(W / 2, H - m - 16,
             f"{land} land hexes  ·  {counts.get('mountain', 0)} mountain  ·  "
             f"{counts.get('hill', 0)} hill  ·  {counts.get('moor', 0) + counts.get('coast', 0)} moor & coast",
             size=10, family="DejaVu Serif Condensed", fill=INK, opacity=0.65, spacing=1.0)
