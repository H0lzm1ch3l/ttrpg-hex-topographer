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


def classify(land_frac: float, relief: float, max_elev: float, slope: float) -> str:
    """Elevation and slope only -- no land cover, which is honest for Skye:
    the island really is sea, moor, hill and bare mountain."""
    if land_frac < 0.10:
        return "sea"
    # Tuned against the COP30 distribution for Skye: median land hex has 296 m
    # of relief and a 366 m high point, so the pre-tuning thresholds called
    # almost the whole island mountainous. Mountain is now roughly the top
    # quartile -- the Cuillin, the Red Hills and the Trotternish ridge.
    if max_elev > 560.0 or relief > 500.0:
        return "mountain"
    if relief > 230.0 or slope > 9.5:
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


def build_hexes(fld: Field, size_m: float, peaks: np.ndarray) -> list[HexCell]:
    """Lay the flat-top lattice over the field and read each cell."""
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

            terrain = classify(land_frac, relief, max_elev, sl)
            cell = HexCell(q=q, r=r, cx=cx, cy=cy, land_frac=land_frac,
                           relief=relief, max_elev=max_elev, mean_elev=mean_elev,
                           slope_mean=sl, terrain=terrain, label="")
            if peaks.size:
                d = hg.contains(peaks[:, 0] - cx, peaks[:, 1] - cy, size_m)
                sel = peaks[d]
                if sel.size:
                    order = np.argsort(-sel[:, 3])
                    cell.peaks = [tuple(v) for v in sel[order]]
            cells.append(cell)

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
    return cells


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


# Line weight by catchment band: a burn you step over up to something needing
# a bridge.
RIVER_WIDTH = {1: 0.9, 2: 1.5, 3: 2.2, 4: 3.0}


def draw_rivers(svg: SVG, lay: "Layout", reaches, hydro_shape, noise: ValueNoise,
                min_band: int = 1) -> int:
    """Draw traced reaches, letting the slope raster set their character.

    Wobble amplitude is the tell: a torrent is confined by its gorge and runs
    nearly straight, a sluggish reach over soft ground wanders. Torrents in the
    larger bands also get rapids ticks.
    """
    drawn = 0
    amp_for = {"torrent": 0.55, "normal": 1.1, "sluggish": 2.0}
    for i, rch in enumerate(reaches):
        if rch.band < min_band:
            continue
        px, py = lay.rc_to_px(rch.pts[:, 0], rch.pts[:, 1], shape=hydro_shape)
        pts = resample(np.column_stack([px, py]), 4.0)
        if len(pts) < 4:
            continue
        pts = wobble(pts, noise, amp=amp_for[rch.kind], scale=34, phase=i * 1.7)
        w = RIVER_WIDTH.get(rch.band, 1.0)
        svg.path(smooth_d(pts), width=w, stroke=INK)
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
            svg.segments(segs, width=w * 0.7, stroke=INK)
    return drawn


# --------------------------------------------------------------------------
# The map
# --------------------------------------------------------------------------

@dataclass
class MapStyle:
    paper: str = PAPER
    px_per_km: float = 26.0
    hex_lw: float = 1.15
    coast_lw: float = 2.6
    ripples: int = 4
    ripple_step_km: float = 1.25
    show_labels: bool = True
    show_hex_ids: bool = True
    seed: int = 11


def render(fld: Field, size_m: float, title: str, subtitle: str,
           style: MapStyle | None = None,
           places: list[tuple[str, float, float]] | None = None,
           reaches=None, hydro_shape=None) -> tuple[SVG, list[HexCell]]:
    st = style or MapStyle()
    PAPER_FILL[0] = st.paper
    noise = ValueNoise(seed=st.seed)
    rng = np.random.default_rng(st.seed)

    x0, y0, x1, y1 = fld.extent_m
    map_w = (x1 - x0) / 1000.0 * st.px_per_km
    map_h = (y1 - y0) / 1000.0 * st.px_per_km

    margin = 46.0
    title_h = 128.0
    legend_h = 128.0
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
    cells = build_hexes(fld, size_m, peaks)

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

    # ---- watercourses ----------------------------------------------------
    # Drawn after the coast but before the landforms, so peaks overlap the
    # burns that run past them rather than the other way round.
    if reaches:
        n_drawn = draw_rivers(svg, lay, reaches, hydro_shape, noise)
        svg.add(f"<!-- {n_drawn} reaches -->")

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
    return svg, cells


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

    # legend
    ly = H - m - 78
    lx = m + 34
    noise = ValueNoise(seed=st.seed + 5)
    svg.text(lx, ly - 14, "TERRAIN", size=11, family="DejaVu Serif Condensed",
             fill=INK, anchor="start", weight="bold", spacing=2.4)

    gx = lx + 8
    mountain_glyph(svg, gx, ly + 20, 13, 24, noise, 3, lw=2.0, shoulders=1)
    svg.text(gx + 24, ly + 14, "MOUNTAIN", size=10.5, family="LM Roman Caps 10",
             fill=INK, anchor="start", spacing=1.2)
    gx += 118
    hill_glyph(svg, gx, ly + 20, 30, 12, noise, 9, lw=2.0)
    svg.text(gx + 22, ly + 14, "HILL", size=10.5, family="LM Roman Caps 10",
             fill=INK, anchor="start", spacing=1.2)
    gx += 92
    moor_tuft(svg, gx - 4, ly + 16, 8)
    moor_tuft(svg, gx + 5, ly + 14, 7)
    svg.text(gx + 22, ly + 14, "MOOR", size=10.5, family="LM Roman Caps 10",
             fill=INK, anchor="start", spacing=1.2)
    gx += 96
    rv = np.array([[gx - 8, ly + 22], [gx + 2, ly + 12], [gx + 10, ly + 20], [gx + 20, ly + 10]])
    svg.path(smooth_d(wobble(resample(rv, 3.0), noise, amp=0.8, scale=20)),
             width=2.0, stroke=INK)
    svg.text(gx + 28, ly + 14, "RIVER", size=10.5, family="LM Roman Caps 10",
             fill=INK, anchor="start", spacing=1.2)
    gx += 100
    segs = [np.array([[gx - 6, ly + 8 + i * 4.5], [gx + 16, ly + 8 + i * 4.5]]) for i in range(3)]
    svg.segments(segs, width=1.0, opacity=0.6)
    svg.text(gx + 26, ly + 14, "SEA", size=10.5, family="LM Roman Caps 10",
             fill=INK, anchor="start", spacing=1.2)

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
