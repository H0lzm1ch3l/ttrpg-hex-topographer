#!/usr/bin/env python3
"""Render a Shadowdark-style hexmap from a DEM.

    python scripts/make_hexmap.py --dem data/skye_cop30.tif --out out/skye
    python scripts/make_hexmap.py --synthetic --out out/dryrun

Writes .svg (vector, the real deliverable), .png and .pdf. No image models are
involved at any point -- every mark on the page is an SVG path computed from
the elevation raster.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topographer.domain.hexgrid import MILE_M          # noqa: E402
from topographer.render import shadowdark as sd        # noqa: E402
from topographer.render.dem import (                   # noqa: E402
    GeoTiffDem, SyntheticIslandDem, build_field,
)

# Skye, west of Kyle of Lochalsh: Rubha Hunish down to Point of Sleat.
SKYE_BBOX = (-6.85, 57.02, -5.78, 57.72)   # trimmed east: -5.60 filled a third of the page with mainland

# Hand-placed from published coordinates, NOT derived from the DEM. Kept
# explicitly separate from the terrain so the map's provenance stays honest.
SKYE_PLACES = [
    ("Portree",        -6.1956, 57.4125),
    ("Dunvegan",       -6.5833, 57.4417),
    ("Uig",            -6.3597, 57.5856),
    ("Carbost",        -6.3500, 57.3167),
    ("Sligachan",      -6.1700, 57.2900),
    ("Broadford",      -5.9083, 57.2417),
    ("Kyleakin",       -5.7325, 57.2758),
    ("Elgol",          -6.1050, 57.1450),
    ("Armadale",       -5.8833, 57.0667),
    ("Loch Harport",   -6.4100, 57.3300),
    ("Loch Bracadale", -6.5600, 57.3450),
    ("Sgurr Alasdair", -6.2478, 57.2114),
    ("The Storr",      -6.1789, 57.5069),
    ("Neist Point",    -6.7869, 57.4239),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dem", help="GeoTIFF in EPSG:4326 (OpenTopography output)")
    p.add_argument("--synthetic", action="store_true",
                   help="procedural island, for style work with no DEM")
    p.add_argument("--slope", help="slope GeoTIFF (gdaldem slope output); if "
                                   "omitted it is computed from the DEM")
    p.add_argument("--rivers", action="store_true", help="extract and draw watercourses")
    p.add_argument("--river-grid", type=int, default=900,
                   help="working grid width for hydrology (coarser than the map)")
    p.add_argument("--min-km2", type=float, default=2.0,
                   help="smallest catchment drawn as a watercourse")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    p.add_argument("--hex-miles", type=float, default=6.0)
    p.add_argument("--px-per-km", type=float, default=26.0)
    p.add_argument("--grid-width", type=int, default=1700,
                   help="working raster width in cells")
    p.add_argument("--title", default="THE ISLE OF SKYE")
    p.add_argument("--subtitle", default="Loch Harport & the Cuillin · 6-mile hexes")
    p.add_argument("--paper", default="warm", choices=["warm", "pure"])
    p.add_argument("--no-places", action="store_true")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="out/hexmap")
    a = p.parse_args()

    bbox = tuple(a.bbox) if a.bbox else SKYE_BBOX

    if a.synthetic:
        dem = SyntheticIslandDem(bbox, seed=a.seed)
        title, subtitle = "DRY RUN", "synthetic terrain · not a real place"
        places = None
    elif a.dem:
        dem = GeoTiffDem(a.dem)
        title, subtitle = a.title, a.subtitle
        places = None if a.no_places else SKYE_PLACES
    else:
        p.error("pass --dem PATH or --synthetic")

    print(f"source : {dem.name}")
    print(f"bbox   : {bbox}")

    fld = build_field(dem, bbox, px_width=a.grid_width)
    x0, y0, x1, y1 = fld.extent_m
    print(f"region : {(x1-x0)/1000:.1f} x {(y1-y0)/1000:.1f} km "
          f"@ {fld.m_per_px:.0f} m/cell  grid {fld.shape}")

    size_m = a.hex_miles * MILE_M / (3 ** 0.5)
    style = sd.MapStyle(px_per_km=a.px_per_km, seed=a.seed,
                        paper=sd.PAPER if a.paper == "warm" else sd.PAPER_PURE)

    reaches, hydro_shape = None, None
    if a.rivers:
        from topographer.render import hydro
        from topographer.render.dem import sample_onto

        hfld = build_field(dem, bbox, px_width=a.river_grid)
        if a.slope:
            slope = sample_onto(hfld, GeoTiffDem(a.slope))
            slope_src = a.slope
        else:
            slope = hfld.slope_deg()
            slope_src = "computed from DEM"
        drn = hydro.extract(hfld.elev, slope, hfld.m_per_px, min_km2=a.min_km2)
        reaches = hydro.trace(drn)
        hydro_shape = hfld.shape
        kinds: dict[str, int] = {}
        for rr in reaches:
            kinds[rr.kind] = kinds.get(rr.kind, 0) + 1
        print(f"slope  : {slope_src}")
        print(f"hydro  : grid {hfld.shape} @ {hfld.m_per_px:.0f} m  "
              f"max catchment {drn.acc_km2[drn.channel].max():.1f} km²"
              if drn.channel.any() else "hydro  : no channels")
        print(f"rivers : {len(reaches)} reaches  {kinds}")

    svg, cells = sd.render(fld, size_m, title, subtitle, style=style, places=places,
                           reaches=reaches, hydro_shape=hydro_shape)

    counts: dict[str, int] = {}
    for c in cells:
        counts[c.terrain] = counts.get(c.terrain, 0) + 1
    print(f"hexes  : {len(cells)}  {counts}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    svg.save(a.out + ".svg")
    print(f"wrote  : {a.out}.svg  ({os.path.getsize(a.out + '.svg')/1e6:.2f} MB, "
          f"{svg.w:.0f}x{svg.h:.0f})")

    import io

    import cairosvg
    from PIL import Image

    # cairosvg bakes SUBPIXEL antialiasing into its PNG output -- measured up to
    # 148 levels of RGB channel spread on pure #111 text, i.e. visible colour
    # fringing. On a black-and-white brief that is a defect, so the PNG is
    # supersampled at 2x and flattened to 8-bit greyscale, which removes the
    # fringing and sharpens the line work at the same time. The SVG and PDF are
    # vector and unaffected.
    raw = cairosvg.svg2png(url=a.out + ".svg", scale=2.0)
    im = Image.open(io.BytesIO(raw)).convert("L")
    im = im.resize((int(svg.w), int(svg.h)), Image.LANCZOS)
    im.save(a.out + ".png", optimize=True)

    im.resize((int(svg.w * 2), int(svg.h * 2)), Image.LANCZOS).save(a.out + "@2x.png")
    cairosvg.svg2pdf(url=a.out + ".svg", write_to=a.out + ".pdf")

    ex = np.asarray(im)
    print(f"wrote  : {a.out}.png ({im.size[0]}x{im.size[1]}, greyscale), "
          f"{a.out}@2x.png, {a.out}.pdf")
    print(f"tone   : ink {float((ex < 128).mean())*100:.1f}% of page")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
