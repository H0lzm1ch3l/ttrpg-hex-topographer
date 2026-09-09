"""Pen-and-ink drawing primitives for SVG cartography.

Everything here is deterministic: the same seed gives the same wobble, so a map
is reproducible. No raster filters, no image models -- just paths.

The whole trick to a hand-inked look is that no line is straight and no two
lines are the same weight. `wobble` handles the first; varying `stroke-width`
per element handles the second.
"""

from __future__ import annotations

import math

import numpy as np


# --------------------------------------------------------------------------
# Deterministic smooth noise
# --------------------------------------------------------------------------

class ValueNoise:
    """Tileable smooth value noise on a lattice, sampled bilinearly with a
    smoothstep easing. Cheap, vectorised, and stable across runs."""

    def __init__(self, seed: int = 0, grid: int = 256):
        rng = np.random.default_rng(seed)
        self.g = rng.random((grid, grid)) * 2.0 - 1.0
        self.n = grid

    def sample(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n = self.n
        xi = np.floor(x).astype(np.int64)
        yi = np.floor(y).astype(np.int64)
        xf, yf = x - xi, y - yi
        u = xf * xf * (3.0 - 2.0 * xf)
        v = yf * yf * (3.0 - 2.0 * yf)
        g = self.g
        a = g[yi % n, xi % n]
        b = g[yi % n, (xi + 1) % n]
        c = g[(yi + 1) % n, xi % n]
        d = g[(yi + 1) % n, (xi + 1) % n]
        return (a * (1 - u) + b * u) * (1 - v) + (c * (1 - u) + d * u) * v

    def fbm(self, x, y, octaves: int = 3):
        total = np.zeros(np.shape(x), dtype=float)
        amp, freq = 1.0, 1.0
        norm = 0.0
        for _ in range(octaves):
            total = total + amp * self.sample(np.asarray(x) * freq, np.asarray(y) * freq)
            norm += amp
            amp *= 0.5
            freq *= 2.0
        return total / norm


# --------------------------------------------------------------------------
# Polyline helpers
# --------------------------------------------------------------------------

def resample(pts: np.ndarray, step: float) -> np.ndarray:
    """Even arc-length resampling, so wobble amplitude reads uniformly."""
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return p
    seg = np.hypot(*(p[1:] - p[:-1]).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < step or not np.isfinite(total):
        return p
    t = np.arange(0.0, total, step)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def wobble(pts, noise: ValueNoise, amp: float = 1.2, scale: float = 45.0,
           phase: float = 0.0) -> np.ndarray:
    """Displace points by smooth noise. This is what stops lines looking CAD-drawn."""
    p = np.asarray(pts, dtype=float)
    if len(p) == 0:
        return p
    dx = noise.sample(p[:, 0] / scale + phase, p[:, 1] / scale + phase * 0.7)
    dy = noise.sample(p[:, 0] / scale + 41.3 + phase, p[:, 1] / scale + 17.7 + phase * 0.7)
    return p + np.column_stack([dx, dy]) * amp


def path_d(pts, close: bool = False, prec: int = 1) -> str:
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return ""
    out = [f"M{p[0, 0]:.{prec}f},{p[0, 1]:.{prec}f}"]
    out += [f"L{x:.{prec}f},{y:.{prec}f}" for x, y in p[1:]]
    if close:
        out.append("Z")
    return "".join(out)


def smooth_d(pts, close: bool = False, tension: float = 0.5, prec: int = 1) -> str:
    """Catmull-Rom through the points, emitted as cubic beziers.

    Ink lines curve; polylines of short segments read as faceted at print size.
    """
    p = np.asarray(pts, dtype=float)
    if len(p) < 3:
        return path_d(p, close, prec)
    if close:
        p = np.vstack([p[-1], p, p[0], p[1]])
    else:
        p = np.vstack([p[0], p, p[-1]])

    d = [f"M{p[1, 0]:.{prec}f},{p[1, 1]:.{prec}f}"]
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        c1 = p1 + (p2 - p0) / 6.0 * tension * 2
        c2 = p2 - (p3 - p1) / 6.0 * tension * 2
        d.append(f"C{c1[0]:.{prec}f},{c1[1]:.{prec}f} {c2[0]:.{prec}f},{c2[1]:.{prec}f} "
                 f"{p2[0]:.{prec}f},{p2[1]:.{prec}f}")
    if close:
        d.append("Z")
    return "".join(d)


# --------------------------------------------------------------------------
# Hatching
# --------------------------------------------------------------------------

def hatch(mask: np.ndarray, extent: tuple[float, float, float, float],
          angle_deg: float, spacing: float, step: float = 2.0,
          min_len: float = 3.0, jitter: float = 0.0,
          rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Parallel line segments clipped to a boolean mask.

    ``mask`` covers ``extent`` = (x0, y0, x1, y1) in output units, row 0 at y0.
    Returns a list of (2, 2) endpoint arrays. This is the workhorse for shading
    -- cross-hatching is two calls at different angles.
    """
    x0, y0, x1, y1 = extent
    rows, cols = mask.shape
    ang = math.radians(angle_deg)
    dirv = np.array([math.cos(ang), math.sin(ang)])
    nrm = np.array([-dirv[1], dirv[0]])
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = math.hypot(x1 - x0, y1 - y0) / 2.0 + spacing

    segs: list[np.ndarray] = []
    ks = np.arange(-half, half, spacing)
    ts = np.arange(-half, half, step)
    for k in ks:
        off = k + (rng.uniform(-jitter, jitter) if (rng is not None and jitter) else 0.0)
        base = np.array([cx, cy]) + nrm * off
        pts = base[None, :] + ts[:, None] * dirv[None, :]
        col = ((pts[:, 0] - x0) / (x1 - x0) * (cols - 1)).round().astype(int)
        row = ((pts[:, 1] - y0) / (y1 - y0) * (rows - 1)).round().astype(int)
        ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
        inside = np.zeros(len(ts), dtype=bool)
        inside[ok] = mask[row[ok], col[ok]]

        idx = np.flatnonzero(inside)
        if idx.size == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks, [idx.size - 1]])
        for s, e in zip(starts, ends):
            a, b = pts[idx[s]], pts[idx[e]]
            if math.hypot(*(b - a)) >= min_len:
                segs.append(np.array([a, b]))
    return segs


def stipple(mask: np.ndarray, extent: tuple[float, float, float, float],
            density: float, rng: np.random.Generator) -> np.ndarray:
    """Random dots inside a mask. Density is dots per square output unit."""
    x0, y0, x1, y1 = extent
    rows, cols = mask.shape
    n = int(abs((x1 - x0) * (y1 - y0)) * density)
    if n <= 0:
        return np.empty((0, 2))
    xs = rng.uniform(x0, x1, n)
    ys = rng.uniform(y0, y1, n)
    col = ((xs - x0) / (x1 - x0) * (cols - 1)).round().astype(int)
    row = ((ys - y0) / (y1 - y0) * (rows - 1)).round().astype(int)
    ok = (col >= 0) & (col < cols) & (row >= 0) & (row < rows)
    keep = np.zeros(n, dtype=bool)
    keep[ok] = mask[row[ok], col[ok]]
    return np.column_stack([xs[keep], ys[keep]])


# --------------------------------------------------------------------------
# SVG assembly
# --------------------------------------------------------------------------

class SVG:
    """Minimal SVG document builder. Deliberately not a library -- the output
    has to stay readable so the toolchain is inspectable."""

    def __init__(self, width: float, height: float, bg: str = "#ffffff"):
        self.w, self.h = width, height
        self.parts: list[str] = []
        self.defs: list[str] = []
        self.bg = bg

    def add(self, s: str) -> None:
        self.parts.append(s)

    def define(self, s: str) -> None:
        self.defs.append(s)

    def path(self, d: str, stroke: str = "#000", width: float = 1.0,
             fill: str = "none", opacity: float | None = None,
             cap: str = "round", join: str = "round", extra: str = "") -> None:
        if not d:
            return
        o = f' opacity="{opacity:.3f}"' if opacity is not None else ""
        self.add(f'<path d="{d}" fill="{fill}" stroke="{stroke}" '
                 f'stroke-width="{width:.2f}" stroke-linecap="{cap}" '
                 f'stroke-linejoin="{join}"{o}{extra}/>')

    def segments(self, segs, stroke: str = "#000", width: float = 1.0,
                 opacity: float | None = None) -> None:
        if len(segs) == 0:
            return
        d = "".join(f"M{a[0]:.1f},{a[1]:.1f}L{b[0]:.1f},{b[1]:.1f}" for a, b in segs)
        self.path(d, stroke=stroke, width=width, opacity=opacity, cap="round")

    def dots(self, pts, r: float = 0.7, fill: str = "#000") -> None:
        if len(pts) == 0:
            return
        d = "".join(f"M{x:.1f},{y:.1f}m{-r},0a{r},{r} 0 1,0 {2*r},0a{r},{r} 0 1,0 {-2*r},0"
                    for x, y in pts)
        self.add(f'<path d="{d}" fill="{fill}" stroke="none"/>')

    def rect(self, x, y, w, h, fill="none", stroke="none", width=1.0) -> None:
        self.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{width:.2f}"/>')

    def text(self, x, y, s: str, size: float = 14, family: str = "DejaVu Serif",
             fill: str = "#000", anchor: str = "middle", weight: str = "normal",
             spacing: float = 0.0, style: str = "normal", opacity=None,
             halo: str | None = None, halo_w: float = 3.5) -> None:
        """``halo`` keeps a label legible over busy line work without an opaque
        rectangle punching a hole in the coastline underneath.

        It is emitted as a separate stroked copy *below* the filled text rather
        than as ``paint-order="stroke"`` on one element: cairosvg ignores
        paint-order, so the single-element form paints the paper stroke over the
        glyphs and the label disappears.
        """
        esc = (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
        o = f' opacity="{opacity}"' if opacity is not None else ""
        common = (f'x="{x:.1f}" y="{y:.1f}" font-family="{family}" '
                  f'font-size="{size:.1f}" text-anchor="{anchor}" '
                  f'font-weight="{weight}" font-style="{style}" '
                  f'letter-spacing="{spacing:.2f}"')
        if halo:
            self.add(f'<text {common} fill="none" stroke="{halo}" '
                     f'stroke-width="{halo_w:.1f}" stroke-linejoin="round"'
                     f'{o}>{esc}</text>')
        self.add(f'<text {common} fill="{fill}"{o}>{esc}</text>')

    def group(self, content: str, **attrs) -> None:
        a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.add(f"<g {a}>{content}</g>")

    def open_group(self, **attrs) -> None:
        a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        self.add(f"<g {a}>")

    def close_group(self) -> None:
        self.add("</g>")

    def clip_rect(self, name: str, x, y, w, h) -> None:
        self.define(f'<clipPath id="{name}"><rect x="{x:.1f}" y="{y:.1f}" '
                    f'width="{w:.1f}" height="{h:.1f}"/></clipPath>')

    def tostring(self) -> str:
        defs = f"<defs>{''.join(self.defs)}</defs>" if self.defs else ""
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w:.0f}" '
            f'height="{self.h:.0f}" viewBox="0 0 {self.w:.0f} {self.h:.0f}">'
            f'{defs}<rect width="100%" height="100%" fill="{self.bg}"/>'
            f'{"".join(self.parts)}</svg>'
        )

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            fh.write(self.tostring())
