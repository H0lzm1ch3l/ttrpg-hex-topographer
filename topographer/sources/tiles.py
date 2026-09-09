"""AWS Terrain Tiles (terrarium) reader.

Bucket ``elevation-tiles-prod`` -- no API key, no requester-pays, no rate
limit.  This is why elevation comes from here and not from OpenTopography,
whose free non-academic tier allows 50 calls per 24 hours.

Terrarium encodes metres in RGB::

    elevation = (R * 256 + G + B / 256) - 32768

Zoom guidance (see the design doc): source resolution is ~30 m, so z12 is the
practical ceiling and z13 is interpolation.  z11 gives ~179 px across a 6-mile
hex, which is ample for statistics at a quarter of z12's tile count.
"""

from __future__ import annotations

import io
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256
EARTH_CIRCUMFERENCE = 40075016.685578488


# --------------------------------------------------------------------------
# Slippy tile maths (Web Mercator, EPSG:3857)
# --------------------------------------------------------------------------

def lonlat_to_tile_frac(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(min(lat, 85.05112878), -85.05112878))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[int, int]:
    x, y = lonlat_to_tile_frac(lon, lat, z)
    return int(math.floor(x)), int(math.floor(y))


def tile_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lon, lat


def ground_resolution(lat: float, z: int) -> float:
    """Metres per pixel at a latitude and zoom."""
    return EARTH_CIRCUMFERENCE * math.cos(math.radians(lat)) / (TILE_PX * 2 ** z)


def zoom_for_resolution(lat: float, target_m_per_px: float, cap: int = 12) -> int:
    """Smallest zoom meeting a resolution target, capped at the useful limit."""
    z = math.log2(EARTH_CIRCUMFERENCE * math.cos(math.radians(lat)) / (TILE_PX * target_m_per_px))
    return max(0, min(cap, int(math.ceil(z))))


def decode_terrarium(rgb: np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 -> (H, W) float32 metres."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    return (r * 256.0 + g + b / 256.0) - 32768.0


# --------------------------------------------------------------------------
# Transport -- pluggable so the pipeline is testable with no network
# --------------------------------------------------------------------------

class TileTransport(Protocol):
    def fetch(self, z: int, x: int, y: int) -> np.ndarray:
        """Return an (H, W, 3) uint8 terrarium tile."""
        ...


class HttpTileTransport:
    """The real thing.  Tiles are immutable, so the disk cache never expires."""

    def __init__(self, cache_dir: str | None = None, timeout: float = 20.0,
                 url_template: str = TERRARIUM_URL):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.url_template = url_template
        self._lock = threading.Lock()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, z: int, x: int, y: int) -> str | None:
        if not self.cache_dir:
            return None
        d = os.path.join(self.cache_dir, str(z), str(x))
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{y}.png")

    def fetch(self, z: int, x: int, y: int) -> np.ndarray:
        import requests

        path = self._cache_path(z, x, y)
        if path and os.path.exists(path):
            return np.asarray(Image.open(path).convert("RGB"))

        url = self.url_template.format(z=z, x=x, y=y)
        resp = requests.get(url, timeout=self.timeout,
                            headers={"User-Agent": "topographer/0.1 (hexcrawl map tool)"})
        resp.raise_for_status()
        data = resp.content
        if path:
            with self._lock:
                tmp = path + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(data)
                os.replace(tmp, path)
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


class SyntheticTileTransport:
    """Deterministic procedural terrain, for tests and offline development.

    Not a toy stand-in for correctness testing of the *pipeline*: it produces a
    real height field with ridges, basins and a sea level, so masks, statistics,
    slope and hillshade all exercise their real code paths without network.
    """

    #: Base spatial frequency in cycles per world width.  ~1500 puts one cycle
    #: at roughly 27 km, so a 6-mile hex spans a third of one and sees real
    #: relief rather than a near-flat slice of a planet-scale wave.
    BASE_FREQ = 1500.0

    def __init__(self, seed: int = 0, relief: float = 900.0):
        self.seed = seed
        self.relief = relief

    def fetch(self, z: int, x: int, y: int) -> np.ndarray:
        n = 2 ** z
        u = (np.arange(TILE_PX) + 0.5) / TILE_PX
        gx = (x + u[None, :]) / n
        gy = (y + u[:, None]) / n
        h = np.zeros((TILE_PX, TILE_PX), dtype=np.float64)
        amp, freq = 1.0, self.BASE_FREQ
        for octave in range(6):
            ph = self.seed * 0.7 + octave * 1.31
            h += amp * (
                np.sin(2 * np.pi * freq * gx + ph)
                * np.cos(2 * np.pi * freq * gy * 1.3 + ph * 0.5)
            )
            # ridged component -- gives believable crests rather than blobs
            h += 0.5 * amp * (1.0 - np.abs(np.sin(2 * np.pi * freq * (gx + gy) + ph)))
            amp *= 0.5
            freq *= 2.02
        h = h / 2.4 * self.relief
        h = np.clip(h, -400.0, 8000.0)

        v = np.clip(h + 32768.0, 0, 65535.999)
        r = np.floor(v / 256.0).astype(np.uint8)
        g = np.floor(v - r.astype(np.float64) * 256.0).astype(np.uint8)
        b = np.floor((v - np.floor(v)) * 256.0).astype(np.uint8)
        return np.dstack([r, g, b])


# --------------------------------------------------------------------------
# Mosaic
# --------------------------------------------------------------------------

@dataclass
class Mosaic:
    """A decoded elevation mosaic addressed in fractional Web Mercator tile
    coordinates, so any point on Earth can be sampled without a resampling
    step in between."""

    elev: np.ndarray      # (H, W) float32 metres
    z: int
    x0: int               # tile index of the left-most column
    y0: int               # tile index of the top-most row

    def sample(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
        """Bilinear sample at WGS84 coordinates."""
        n = 2.0 ** self.z
        tx = (lon + 180.0) / 360.0 * n
        lat_c = np.clip(lat, -85.05112878, 85.05112878)
        ty = (1.0 - np.arcsinh(np.tan(np.radians(lat_c))) / np.pi) / 2.0 * n

        px = (tx - self.x0) * TILE_PX - 0.5
        py = (ty - self.y0) * TILE_PX - 0.5

        h, w = self.elev.shape
        x0 = np.clip(np.floor(px).astype(np.int64), 0, w - 1)
        y0 = np.clip(np.floor(py).astype(np.int64), 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        y1 = np.clip(y0 + 1, 0, h - 1)
        fx = np.clip(px - x0, 0.0, 1.0)
        fy = np.clip(py - y0, 0.0, 1.0)

        return (
            self.elev[y0, x0] * (1 - fx) * (1 - fy)
            + self.elev[y0, x1] * fx * (1 - fy)
            + self.elev[y1, x0] * (1 - fx) * fy
            + self.elev[y1, x1] * fx * fy
        ).astype(np.float32)


def fetch_mosaic(transport: TileTransport, bbox: tuple[float, float, float, float],
                 z: int, max_workers: int = 8) -> Mosaic:
    """Fetch and stitch the tiles covering ``bbox`` = (west, south, east, north)."""
    west, south, east, north = bbox
    x0, y0 = lonlat_to_tile(west, north, z)
    x1, y1 = lonlat_to_tile(east, south, z)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)

    n = 2 ** z
    coords = [(xx % n, yy) for yy in range(y0, y1 + 1) for xx in range(x0, x1 + 1)
              if 0 <= yy < n]
    if not coords:
        raise ValueError(f"bbox {bbox} covers no tiles at z{z}")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        rgbs = list(pool.map(lambda c: transport.fetch(z, c[0], c[1]), coords))

    cols = x1 - x0 + 1
    rows = y1 - y0 + 1
    out = np.zeros((rows * TILE_PX, cols * TILE_PX), dtype=np.float32)
    for (cx, cy), rgb in zip(coords, rgbs):
        # cx came back modulo n; recover the un-wrapped column for placement
        rel_x = ((cx - x0) % n) if x1 >= x0 else 0
        rel_y = cy - y0
        out[rel_y * TILE_PX:(rel_y + 1) * TILE_PX,
            rel_x * TILE_PX:(rel_x + 1) * TILE_PX] = decode_terrarium(rgb)

    return Mosaic(elev=out, z=z, x0=x0, y0=y0)


def tile_count(bbox: tuple[float, float, float, float], z: int) -> int:
    west, south, east, north = bbox
    x0, y0 = lonlat_to_tile(west, north, z)
    x1, y1 = lonlat_to_tile(east, south, z)
    return (abs(x1 - x0) + 1) * (abs(y1 - y0) + 1)
