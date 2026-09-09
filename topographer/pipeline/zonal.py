"""Per-hex zonal statistics, sampled on a regular grid in the tile's own frame.

The grid is built in local LAEA metres, masked to the hexagon analytically,
then transformed out to WGS84 to sample the Web Mercator mosaic.  Doing it in
this direction -- rather than reprojecting the raster -- keeps every hex exactly
6 miles across and avoids a resampling step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..domain import hexgrid as hg
from ..domain.models import EdgeSignature, ElevStats
from ..sources.tiles import Mosaic

DEFAULT_GRID = 320   # ~35 m per cell across a 6-mile hex


@dataclass
class HexRaster:
    """Elevation sampled over a hex, plus the mask and grid metadata.

    Cells are square: a flat-top hex is wider than it is tall (2 : sqrt(3)), so
    the grid has proportionally fewer rows than columns.  Anisotropic cells
    would skew slope and hillshade and make the rendered hex look pointy-top.
    """

    elev: np.ndarray       # (rows, cols) float32; cells outside the hex are still filled
    mask: np.ndarray       # (rows, cols) bool, True inside the hexagon
    px_m: float            # metres per cell in x
    py_m: float            # metres per cell in y
    frame: hg.LocalFrame

    @property
    def inside(self) -> np.ndarray:
        return self.elev[self.mask]

    @property
    def aspect_ratio(self) -> float:
        return self.elev.shape[1] / self.elev.shape[0]


def build_hex_raster(frame: hg.LocalFrame, mosaic: Mosaic, grid: int = DEFAULT_GRID) -> HexRaster:
    """``grid`` is the number of cells across the hex's *width*."""
    half_w = hg.across_corners(frame.size) / 2.0
    half_h = hg.across_flats(frame.size) / 2.0

    cols = int(grid)
    rows = max(int(round(grid * half_h / half_w)), 2)

    xs = np.linspace(-half_w, half_w, cols)
    ys = np.linspace(half_h, -half_h, rows)          # row 0 is north
    gx, gy = np.meshgrid(xs, ys)

    mask = hg.contains(gx, gy, frame.size)
    lon, lat = frame.to_lonlat(gx.ravel(), gy.ravel())
    elev = mosaic.sample(np.asarray(lon), np.asarray(lat)).reshape(rows, cols)

    return HexRaster(
        elev=elev.astype(np.float32),
        mask=mask,
        px_m=(2 * half_w) / (cols - 1),
        py_m=(2 * half_h) / (rows - 1),
        frame=frame,
    )


# --------------------------------------------------------------------------
# Derived surfaces
# --------------------------------------------------------------------------

def _gradients(hr: HexRaster, z_factor: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """dz/d(east), dz/d(north) in metres per metre.

    ``np.gradient`` differentiates along rows first, and row 0 is north, so the
    row derivative is the negative of the northward one.
    """
    d_row, d_col = np.gradient(hr.elev.astype(np.float64) * z_factor, hr.py_m, hr.px_m)
    return d_col, -d_row


def slope_aspect(hr: HexRaster) -> tuple[np.ndarray, np.ndarray]:
    """Slope in radians, and aspect as the compass bearing the slope faces.

    Aspect is the *downhill* bearing, measured clockwise from north, which is
    the usual GIS convention: ground rising to the east faces west, 270 deg.
    """
    dz_de, dz_dn = _gradients(hr)
    slope = np.arctan(np.hypot(dz_de, dz_dn))
    aspect = (np.degrees(np.arctan2(-dz_de, -dz_dn)) + 360.0) % 360.0
    return slope, aspect


def hillshade(hr: HexRaster, azimuth_deg: float = 315.0, altitude_deg: float = 45.0,
              z_factor: float = 1.0) -> np.ndarray:
    """Lambertian hillshade, 0..1.

    Expressed as the dot product of the surface normal with the light vector
    rather than the ESRI slope/aspect identity -- same result, but the sign
    conventions are checkable by hand, which the trigonometric form is not.
    """
    dz_de, dz_dn = _gradients(hr, z_factor)
    nrm = np.sqrt(dz_de ** 2 + dz_dn ** 2 + 1.0)
    n_e, n_n, n_u = -dz_de / nrm, -dz_dn / nrm, 1.0 / nrm

    az = math.radians(azimuth_deg)
    alt = math.radians(altitude_deg)
    l_e = math.sin(az) * math.cos(alt)
    l_n = math.cos(az) * math.cos(alt)
    l_u = math.sin(alt)

    return np.clip(n_e * l_e + n_n * l_n + n_u * l_u, 0.0, 1.0)


def elev_stats(hr: HexRaster) -> ElevStats:
    vals = hr.inside
    if vals.size == 0:
        raise ValueError("hex mask selected no cells")

    slope, aspect = slope_aspect(hr)
    s_in = slope[hr.mask]
    # bearings are clockwise from north; convert to standard angles for the
    # circular mean, then convert back
    a_in = np.radians(90.0 - aspect[hr.mask])

    # Circular mean of aspect, weighted by slope: flat ground has no meaningful
    # facing, so it should not drag the average around.
    w = np.sin(s_in)
    if w.sum() > 1e-9:
        ang = math.atan2(float((w * np.sin(a_in)).sum()),
                         float((w * np.cos(a_in)).sum()))
        mean_aspect = (90.0 - math.degrees(ang)) % 360.0
    else:
        mean_aspect = 0.0

    return ElevStats(
        elev_min=float(vals.min()),
        elev_max=float(vals.max()),
        elev_mean=float(vals.mean()),
        elev_median=float(np.median(vals)),
        relief=float(vals.max() - vals.min()),
        slope_mean=float(np.degrees(s_in.mean())),
        slope_p85=float(np.degrees(np.percentile(s_in, 85))),
        ruggedness=float(vals.std()),
        aspect_deg=float(mean_aspect),
    )


# --------------------------------------------------------------------------
# Edge signatures -- stage 2 data, computed here so it is never re-fetched
# --------------------------------------------------------------------------

def edge_signatures(hr: HexRaster, samples: int = 16,
                    inset_m: float = 400.0) -> list[EdgeSignature]:
    """Sample all six edges in tile-local, counterclockwise order.

    ``gradient_out`` is measured by comparing the edge itself against a line
    inset toward the centre: positive means the ground rises as you leave the
    hex.  Two edges fit when their outward gradients *sum* to zero, not when
    they match -- ground falling out of one should rise into the other.
    """
    sigs: list[EdgeSignature] = []
    for e in range(6):
        a, b = hg.edge_endpoints(hr.frame.size, e)
        t = np.linspace(0.0, 1.0, samples)
        ex = a[0] + (b[0] - a[0]) * t
        ey = a[1] + (b[1] - a[1]) * t

        # inset copy, pulled toward the hex centre
        norm = np.hypot(ex, ey)
        norm[norm == 0] = 1.0
        ix = ex * (1.0 - inset_m / norm)
        iy = ey * (1.0 - inset_m / norm)

        edge_h = _sample_local(hr, ex, ey)
        in_h = _sample_local(hr, ix, iy)

        sigs.append(EdgeSignature(
            profile=[int(round(v)) for v in edge_h],
            mean_elev=float(edge_h.mean()),
            relief_band=float(edge_h.max() - edge_h.min()),
            gradient_out=float((edge_h.mean() - in_h.mean()) / inset_m),
            cover={},     # filled at stage 2 once WorldCover is wired in
            water={},
        ))
    return sigs


def _sample_local(hr: HexRaster, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Bilinear sample of the hex raster at local projected coordinates."""
    half_w = hg.across_corners(hr.frame.size) / 2.0
    half_h = hg.across_flats(hr.frame.size) / 2.0
    rows, cols = hr.elev.shape

    col = (x + half_w) / (2 * half_w) * (cols - 1)
    row = (half_h - y) / (2 * half_h) * (rows - 1)

    c0 = np.clip(np.floor(col).astype(int), 0, cols - 1)
    r0 = np.clip(np.floor(row).astype(int), 0, rows - 1)
    c1 = np.clip(c0 + 1, 0, cols - 1)
    r1 = np.clip(r0 + 1, 0, rows - 1)
    fc = np.clip(col - c0, 0, 1)
    fr = np.clip(row - r0, 0, 1)

    return (hr.elev[r0, c0] * (1 - fc) * (1 - fr)
            + hr.elev[r0, c1] * fc * (1 - fr)
            + hr.elev[r1, c0] * (1 - fc) * fr
            + hr.elev[r1, c1] * fc * fr)
