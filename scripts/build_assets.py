#!/usr/bin/env python3
"""Build the local assets the sampler needs.

These are one-off downloads, not runtime dependencies.  Write and run this
early: it is the only part of the project with an awkward manual step, and you
do not want it blocking you halfway through building the reroll loop.

    python scripts/build_assets.py land          # Natural Earth land polygons
    python scripts/build_assets.py prescreen     # coarse global relief grid
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import requests

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")

NE_BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"


def build_land(scale: str = "50m") -> None:
    os.makedirs(ASSETS, exist_ok=True)
    out = os.path.join(ASSETS, f"ne_{scale}_land.geojson")
    url = f"{NE_BASE}/ne_{scale}_land.geojson"
    print(f"fetching {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(out, "wb") as fh:
        fh.write(r.content)
    print(f"wrote {out} ({len(r.content) / 1e6:.2f} MB)")


def build_prescreen(step_deg: float = 0.1, z: int = 6, workers: int = 8) -> None:
    """Coarse global relief / mean-elevation / coastal grid from terrain tiles.

    Runs once, takes a few minutes, and is what stops the reroll loop from
    being dominated by flat Siberia.  It also lets the sampler honour terrain
    filters before making any network call.
    """
    sys.path.insert(0, os.path.dirname(ASSETS))
    from topographer.sources.landmask import PrescreenGrid
    from topographer.sources.tiles import (
        HttpTileTransport, TILE_PX, decode_terrarium, tile_to_lonlat,
    )

    rows = int(round(180.0 / step_deg))
    cols = int(round(360.0 / step_deg))
    relief = np.zeros((rows, cols), np.float32)
    elev_mean = np.zeros((rows, cols), np.float32)
    coastal = np.zeros((rows, cols), bool)

    transport = HttpTileTransport(cache_dir=os.path.join(ASSETS, ".prescreen-cache"))
    n = 2 ** z
    print(f"building {rows}x{cols} prescreen grid from {n * n} z{z} tiles")

    for ty in range(n):
        for tx in range(n):
            try:
                elev = decode_terrarium(transport.fetch(z, tx, ty))
            except Exception as exc:               # a missing tile is ocean
                print(f"  skip {z}/{tx}/{ty}: {exc}", file=sys.stderr)
                continue

            lon0, lat0 = tile_to_lonlat(tx, ty, z)
            lon1, lat1 = tile_to_lonlat(tx + 1, ty + 1, z)

            r0 = int((90.0 - lat0) / 180.0 * rows)
            r1 = max(int((90.0 - lat1) / 180.0 * rows), r0 + 1)
            c0 = int((lon0 + 180.0) / 360.0 * cols)
            c1 = max(int((lon1 + 180.0) / 360.0 * cols), c0 + 1)
            r0, r1 = max(r0, 0), min(r1, rows)
            c0, c1 = max(c0, 0), min(c1, cols)

            for rr in range(r0, r1):
                for cc in range(c0, c1):
                    py0 = int((rr - r0) / max(r1 - r0, 1) * TILE_PX)
                    py1 = max(int((rr - r0 + 1) / max(r1 - r0, 1) * TILE_PX), py0 + 1)
                    px0 = int((cc - c0) / max(c1 - c0, 1) * TILE_PX)
                    px1 = max(int((cc - c0 + 1) / max(c1 - c0, 1) * TILE_PX), px0 + 1)
                    block = elev[py0:py1, px0:px1]
                    if block.size == 0:
                        continue
                    relief[rr, cc] = float(block.max() - block.min())
                    elev_mean[rr, cc] = float(block.mean())
                    coastal[rr, cc] = bool((block > 0).any() and (block <= 0).any())
        print(f"  row {ty + 1}/{n}", end="\r", flush=True)

    out = os.path.join(ASSETS, "prescreen.npz")
    PrescreenGrid(relief, elev_mean, coastal, step_deg).save(out)
    print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["land", "prescreen", "all"])
    p.add_argument("--scale", default="50m", choices=["10m", "50m", "110m"])
    p.add_argument("--step", type=float, default=0.1)
    a = p.parse_args()

    if a.what in ("land", "all"):
        build_land(a.scale)
    if a.what in ("prescreen", "all"):
        build_prescreen(a.step)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
