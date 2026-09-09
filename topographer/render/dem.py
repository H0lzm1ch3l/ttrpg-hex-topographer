"""DEM sources and resampling into a hexmap's local projected frame.

The map is drawn in a Lambert Azimuthal Equal Area projection centred on the
region, for the same reason tiles are (see domain/hexgrid.py): it keeps every
hex the same true size, which a Web Mercator grid would not at 57 deg north.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pyproj import CRS, Transformer

SEA_LEVEL_M = 1.0   # COP30 is a DSM; open water sits at ~0 with a little noise


class DemSource(Protocol):
    name: str

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Elevation in metres at WGS84 coordinates. NaN where unknown."""
        ...


class GeoTiffDem:
    """Any north-up GeoTIFF in EPSG:4326 -- what OpenTopography's global DEM
    API returns for COP30/SRTM/NASADEM."""

    def __init__(self, path: str, nearest: bool = False, bbox=None, pad: float = 0.02):
        """``nearest`` is mandatory for class rasters such as WorldCover:
        interpolating between class codes 10 (tree) and 30 (grass) would invent
        code 20 (shrub) along every boundary.

        ``bbox`` reads only the window covering (W, S, E, N) instead of the whole
        band. This is not an optimisation but a requirement for WorldCover: its
        tiles are 3 x 3 degrees at 10 m, so a full band is 36000 x 36000 cells
        and 5 GB as float32 -- reading one is an immediate out-of-memory kill.
        The files are COGs precisely so that a window can be read cheaply.
        """
        import rasterio

        self.nearest = nearest
        self.path = path
        self.src = rasterio.open(path)
        self.name = f"GeoTIFF {path}"

        transform = self.src.transform
        window = None
        if bbox is not None:
            from rasterio.windows import from_bounds

            w, s_, e, n = bbox
            b = self.src.bounds
            w, s_ = max(w - pad, b.left), max(s_ - pad, b.bottom)
            e, n = min(e + pad, b.right), min(n + pad, b.top)
            if e > w and n > s_:
                window = from_bounds(w, s_, e, n, self.src.transform)
                transform = self.src.window_transform(window)

        # Cast before filling: WorldCover is uint8 and a NaN fill value cannot
        # be represented in an integer masked array.
        band = self.src.read(1, masked=True, window=window)
        self.band = band.astype(np.float32).filled(np.nan)
        self.inv = ~transform
        if self.src.crs is not None and self.src.crs.to_epsg() not in (4326, None):
            raise ValueError(
                f"expected EPSG:4326, got {self.src.crs}. Reproject first: "
                f"gdalwarp -t_srs EPSG:4326 in.tif out.tif"
            )

    @property
    def bounds(self):
        return tuple(self.src.bounds)

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        cols, rows = self.inv * (np.asarray(lon), np.asarray(lat))
        if self.nearest:
            return _nearest(self.band, np.asarray(rows), np.asarray(cols))
        return _bilinear(self.band, np.asarray(rows), np.asarray(cols))


class SyntheticIslandDem:
    """A plausible island, for iterating on drawing style with no network.

    Not a stand-in for the real thing in any output the user sees -- it exists
    so the renderer can be developed and tested before the DEM arrives.
    """

    name = "synthetic island (NOT REAL TERRAIN)"

    def __init__(self, bbox, seed: int = 7, peak_m: float = 950.0):
        self.w, self.s, self.e, self.n = bbox
        self.peak = peak_m
        rng = np.random.default_rng(seed)
        self.lobes = np.column_stack([
            rng.uniform(0.18, 0.82, 9), rng.uniform(0.18, 0.82, 9),
            rng.uniform(0.10, 0.26, 9),
        ])
        self.ridge = np.column_stack([
            rng.uniform(0.35, 0.62, 4), rng.uniform(0.30, 0.55, 4),
        ])
        self.g = rng.random((256, 256)) * 2 - 1

    def _noise(self, u, v, f):
        x, y = u * f, v * f
        xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
        xf, yf = x - xi, y - yi
        sx = xf * xf * (3 - 2 * xf)
        sy = yf * yf * (3 - 2 * yf)
        g, n = self.g, 256
        a = g[yi % n, xi % n]; b = g[yi % n, (xi + 1) % n]
        c = g[(yi + 1) % n, xi % n]; d = g[(yi + 1) % n, (xi + 1) % n]
        return (a * (1 - sx) + b * sx) * (1 - sy) + (c * (1 - sx) + d * sx) * sy

    def sample(self, lon, lat):
        u = (np.asarray(lon) - self.w) / (self.e - self.w)
        v = (np.asarray(lat) - self.s) / (self.n - self.s)
        land = np.full(np.shape(u), -1.0)
        for cx, cy, r in self.lobes:
            land = np.maximum(land, 1.0 - np.hypot(u - cx, v - cy) / r)
        land += 0.30 * self._noise(u, v, 11) + 0.15 * self._noise(u, v, 27)

        ridge = np.full(np.shape(u), 0.0)
        for cx, cy in self.ridge:
            ridge = np.maximum(ridge, np.clip(1.0 - np.hypot(u - cx, v - cy) / 0.13, 0, 1))
        rough = 0.5 + 0.5 * self._noise(u, v, 60)

        h = np.where(land > 0, land * 260 + ridge ** 1.6 * self.peak * rough, land * 60)
        return h.astype(np.float32)


def _nearest(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    r = np.clip(np.round(rows).astype(np.int64), 0, h - 1)
    c = np.clip(np.round(cols).astype(np.int64), 0, w - 1)
    out = arr[r, c].astype(np.float32)
    outside = (rows < -1) | (rows > h) | (cols < -1) | (cols > w)
    return np.where(outside, np.nan, out)


def _bilinear(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    r0 = np.clip(np.floor(rows).astype(np.int64), 0, h - 1)
    c0 = np.clip(np.floor(cols).astype(np.int64), 0, w - 1)
    r1 = np.clip(r0 + 1, 0, h - 1)
    c1 = np.clip(c0 + 1, 0, w - 1)
    fr = np.clip(rows - r0, 0, 1)
    fc = np.clip(cols - c0, 0, 1)
    out = (arr[r0, c0] * (1 - fc) * (1 - fr) + arr[r0, c1] * fc * (1 - fr)
           + arr[r1, c0] * (1 - fc) * fr + arr[r1, c1] * fc * fr)
    outside = (rows < -1) | (rows > h) | (cols < -1) | (cols > w)
    out = np.where(outside, np.nan, out)
    return out


# --------------------------------------------------------------------------
# The projected working grid
# --------------------------------------------------------------------------

@dataclass
class Field:
    """Elevation on a regular grid in a local LAEA, plus everything derived
    from it that the renderer needs."""

    elev: np.ndarray                 # (rows, cols) metres, row 0 = north
    extent_m: tuple[float, float, float, float]   # x0, y0, x1, y1 in LAEA metres
    m_per_px: float
    crs: CRS
    lat0: float
    lon0: float

    @property
    def shape(self):
        return self.elev.shape

    @property
    def sea(self) -> np.ndarray:
        return np.nan_to_num(self.elev, nan=-9999.0) <= SEA_LEVEL_M

    def to_lonlat(self, x, y):
        t = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        return t.transform(x, y)

    def from_lonlat(self, lon, lat):
        t = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        return t.transform(lon, lat)

    def slope_deg(self) -> np.ndarray:
        e = np.nan_to_num(self.elev, nan=0.0).astype(np.float64)
        dr, dc = np.gradient(e, self.m_per_px, self.m_per_px)
        return np.degrees(np.arctan(np.hypot(dc, -dr)))


def build_field(dem: DemSource, bbox, px_width: int = 1800) -> Field:
    """Resample a DEM onto a LAEA grid covering ``bbox``.

    Sampling the source in the target's frame (rather than warping the raster)
    keeps this dependency-light and matches how the tile pipeline already works.
    """
    w, s, e, n = bbox
    lat0 = (s + n) / 2.0
    lon0 = (w + e) / 2.0
    crs = CRS.from_proj4(
        f"+proj=laea +lat_0={lat0} +lon_0={lon0} +x_0=0 +y_0=0 "
        f"+datum=WGS84 +units=m +no_defs"
    )
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    rev = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    cor_lon = [w, e, e, w]
    cor_lat = [s, s, n, n]
    xs, ys = fwd.transform(cor_lon, cor_lat)
    x0, x1 = float(min(xs)), float(max(xs))
    y0, y1 = float(min(ys)), float(max(ys))

    m_per_px = (x1 - x0) / px_width
    cols = px_width
    rows = max(int(round((y1 - y0) / m_per_px)), 2)

    gx = np.linspace(x0, x1, cols)
    gy = np.linspace(y1, y0, rows)          # row 0 is north
    mx, my = np.meshgrid(gx, gy)
    lon, lat = rev.transform(mx.ravel(), my.ravel())
    elev = dem.sample(np.asarray(lon), np.asarray(lat)).reshape(rows, cols)

    return Field(elev=elev.astype(np.float32), extent_m=(x0, y0, x1, y1),
                 m_per_px=m_per_px, crs=crs, lat0=lat0, lon0=lon0)


def sample_onto(fld: Field, src: DemSource) -> np.ndarray:
    """Resample another raster (slope, roughness, ...) onto an existing grid.

    Keeping auxiliary layers on the *same* grid as the elevation means every
    per-cell combination -- slope at a channel cell, roughness in a hex -- is a
    plain array index rather than another reprojection.
    """
    rows, cols = fld.shape
    x0, y0, x1, y1 = fld.extent_m
    gx = np.linspace(x0, x1, cols)
    gy = np.linspace(y1, y0, rows)
    mx, my = np.meshgrid(gx, gy)
    lon, lat = fld.to_lonlat(mx.ravel(), my.ravel())
    return src.sample(np.asarray(lon), np.asarray(lat)).reshape(rows, cols).astype(np.float32)
