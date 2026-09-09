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
from topographer.regions import REGIONS                # noqa: E402
from topographer.render import shadowdark as sd        # noqa: E402
from topographer.render.dem import (                   # noqa: E402
    GeoTiffDem, SyntheticIslandDem, build_field,
)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dem", help="GeoTIFF in EPSG:4326 (OpenTopography output)")
    p.add_argument("--synthetic", action="store_true",
                   help="procedural island, for style work with no DEM")
    p.add_argument("--slope", help="slope GeoTIFF (gdaldem slope output); if "
                                   "omitted it is computed from the DEM")
    p.add_argument("--rivers", action="store_true", help="extract and draw watercourses")
    p.add_argument("--wetland", action="store_true",
                   help="derive bog / marsh / loch from terrain and draw them")
    p.add_argument("--roughness", help="roughness GeoTIFF (gdaldem roughness)")
    p.add_argument("--aspect", help="aspect GeoTIFF (gdaldem aspect)")
    p.add_argument("--hachure-strength", type=float, default=1.0,
                   help="ink weight of the relief hachures")
    p.add_argument("--hachures", action="store_true",
                   help="draw downslope hachures for three-dimensional relief")
    p.add_argument("--potential-wood", action="store_true",
                   help="model where woodland would grow without grazing/drainage")
    p.add_argument("--landcover", nargs="+",
                   help="ESA WorldCover class GeoTIFF(s) for CURRENT woodland")
    p.add_argument("--river-grid", type=int, default=900,
                   help="working grid width for hydrology (coarser than the map)")
    p.add_argument("--min-km2", type=float, default=2.0,
                   help="smallest catchment drawn as a watercourse")
    p.add_argument("--region", choices=sorted(REGIONS), default="skye",
                   help="named region: bbox, titles and place labels")
    p.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                   help="override the region's bbox")
    p.add_argument("--hex-miles", type=float, default=6.0)
    p.add_argument("--px-per-km", type=float, default=26.0)
    p.add_argument("--grid-width", type=int, default=1700,
                   help="working raster width in cells")
    p.add_argument("--title")
    p.add_argument("--subtitle")
    p.add_argument("--paper", default="warm", choices=["warm", "pure"])
    p.add_argument("--no-places", action="store_true")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="out/hexmap")
    a = p.parse_args()

    region = REGIONS[a.region]
    bbox = tuple(a.bbox) if a.bbox else region.bbox

    if a.synthetic:
        dem = SyntheticIslandDem(bbox, seed=a.seed)
        title, subtitle = "DRY RUN", "synthetic terrain · not a real place"
        places = None
    elif a.dem:
        dem = GeoTiffDem(a.dem)
        title = a.title or region.title
        subtitle = a.subtitle or region.subtitle
        places = None if a.no_places else region.places
    else:
        p.error("pass --dem PATH or --synthetic")

    print(f"region : {region.key}  {region.span_km[0]:.0f} x {region.span_km[1]:.0f} km")

    print(f"source : {dem.name}")
    print(f"bbox   : {bbox}")

    fld = build_field(dem, bbox, px_width=a.grid_width)
    x0, y0, x1, y1 = fld.extent_m
    print(f"grid   : {(x1-x0)/1000:.1f} x {(y1-y0)/1000:.1f} km "
          f"@ {fld.m_per_px:.0f} m/cell  grid {fld.shape}")

    size_m = a.hex_miles * MILE_M / (3 ** 0.5)
    style = sd.MapStyle(px_per_km=a.px_per_km, seed=a.seed,
                        hachure_strength=a.hachure_strength,
                        paper=sd.PAPER if a.paper == "warm" else sd.PAPER_PURE)

    reaches, hydro_shape, wet, wood = None, None, None, None
    if a.rivers or a.wetland or a.potential_wood or a.landcover:
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
        hydro_shape = hfld.shape
        reaches = hydro.trace(drn) if a.rivers else None

        if a.wetland:
            from topographer.render import wetland as W

            if a.roughness:
                rough = sample_onto(hfld, GeoTiffDem(a.roughness))
            else:
                rough = np.zeros_like(hfld.elev)
            wet = W.derive(hfld.elev, slope, rough, drn.acc_wet, hfld.m_per_px)
            land_n = max(int((np.nan_to_num(hfld.elev, nan=-9999) > 1.0).sum()), 1)
            print(f"wetland: bog {wet.bog.sum()/land_n*100:.1f}%  "
                  f"fen {wet.fen.sum()/land_n*100:.2f}%  "
                  f"swamp {wet.swamp.sum()/land_n*100:.2f}%  "
                  f"loch {wet.lake.sum()/land_n*100:.2f}%  of land")

        hland = np.nan_to_num(hfld.elev, nan=-9999.0) > 1.0
        if a.landcover:
            from topographer.render import woodland as WD

            lc = None
            for path in a.landcover:
                part = sample_onto(hfld, GeoTiffDem(path, nearest=True, bbox=bbox))
                lc = part if lc is None else np.where(np.isnan(lc) | (lc == 0), part, lc)
            masks = WD.from_worldcover(np.nan_to_num(lc, nan=0).astype(int))
            wood = masks["tree"] & hland
            print(f"landcov: tree {wood.sum()/hland.sum()*100:.1f}%  "
                  f"wetland {(masks['wetland']&hland).sum()/hland.sum()*100:.1f}%  of land")
            if wet is not None:
                # trust the observed wetland class where it fires
                # trust the observed wetland class, splitting it the same way
                obs = masks["wetland"] & hland & ~wet.any_wet
                wet.fen = wet.fen | obs
        elif a.potential_wood:
            from topographer.render import woodland as WD

            excl = None
            if wet is not None:
                # bog and open fen stay treeless; swamp is already wooded
                excl = wet.bog | wet.fen | wet.lake | wet.swamp
            wd = WD.potential(hfld.elev, slope, hfld.m_per_px, exclude=excl)
            wood = wd.potential
            print(f"wood   : potential {wood.sum()/hland.sum()*100:.1f}% of land "
                  f"(modelled, treeline {np.percentile(wd.treeline[hland],50):.0f} m median)")
        kinds: dict[str, int] = {}
        for rr in reaches:
            kinds[rr.kind] = kinds.get(rr.kind, 0) + 1
        print(f"slope  : {slope_src}")
        print(f"hydro  : grid {hfld.shape} @ {hfld.m_per_px:.0f} m  "
              f"max catchment {drn.acc_km2[drn.channel].max():.1f} km²"
              if drn.channel.any() else "hydro  : no channels")
        print(f"rivers : {len(reaches)} reaches  {kinds}")

    map_slope = map_aspect = None
    if a.hachures:
        from topographer.render.dem import sample_onto

        map_slope = (sample_onto(fld, GeoTiffDem(a.slope)) if a.slope
                     else fld.slope_deg())
        if a.aspect:
            map_aspect = sample_onto(fld, GeoTiffDem(a.aspect))
        else:
            from topographer.render.zonalcompat import aspect_from_dem
            map_aspect = aspect_from_dem(fld)
        print(f"hachure: slope {'raster' if a.slope else 'from DEM'}, "
              f"aspect {'raster' if a.aspect else 'from DEM'}")

    svg, cells, scale = sd.render(fld, size_m, title, subtitle, style=style,
                                  places=places, reaches=reaches,
                                  hydro_shape=hydro_shape, wet=wet, wood=wood,
                                  slope_deg=map_slope, aspect_deg=map_aspect)
    print(f"scale  : {scale.describe()}   [{scale.source}]")

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
