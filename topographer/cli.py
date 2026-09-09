"""Command line entry point.

Deliberately built before any web code.  Every pipeline stage has to be
runnable from a terminal -- debugging a projection bug through an HTTP request
and a browser is miserable, and this takes two seconds.

    python -m topographer.cli roll --filter mountainous --png out.png
    python -m topographer.cli at --lat 46.52 --lon 10.31 --png alps.png
    python -m topographer.cli neighbour --lat 46.52 --lon 10.31 --edge 0
    python -m topographer.cli geometry

Set ``--synthetic`` to run the whole pipeline with procedural terrain and no
network at all.
"""

from __future__ import annotations

import argparse
import json
import random
import sys

from .domain import hexgrid as hg
from .domain.models import SamplerFilters
from .pipeline import classify as cls
from .pipeline.render import render_hex_png
from .pipeline.sample import HexSampler
from .pipeline.zonal import build_hex_raster, elev_stats
from .sources.landmask import AlwaysLand, NaturalEarthLandMask
from .sources.tiles import (
    HttpTileTransport, SyntheticTileTransport, fetch_mosaic, tile_count,
    zoom_for_resolution,
)


def build_sampler(args) -> HexSampler:
    transport = (SyntheticTileTransport(seed=args.seed)
                 if args.synthetic
                 else HttpTileTransport(cache_dir=args.cache))
    try:
        mask = AlwaysLand() if args.synthetic else NaturalEarthLandMask()
    except FileNotFoundError as exc:
        print(f"warning: {exc}\n         falling back to AlwaysLand", file=sys.stderr)
        mask = AlwaysLand()
    return HexSampler(transport=transport, mask=mask, grid=args.grid,
                      rng=random.Random(args.seed))


def report(tile, hr=None) -> None:
    s = tile.stats
    c = cls.classify_stage1(s, koppen=tile.koppen, is_coastal=tile.is_coastal)
    print(f"tile {tile.id}")
    print(f"  where     {tile.lat:.4f}, {tile.lon:.4f}")
    print(f"  terrain   {cls.describe(c)}   ({c.reason})")
    print(f"  elevation {s.elev_min:.0f} .. {s.elev_max:.0f} m  "
          f"(mean {s.elev_mean:.0f}, median {s.elev_median:.0f})")
    print(f"  relief    {s.relief:.0f} m")
    print(f"  slope     mean {s.slope_mean:.1f} deg, p85 {s.slope_p85:.1f} deg")
    print(f"  rugged    {s.ruggedness:.1f}   aspect {s.aspect_deg:.0f} deg")
    if tile.edge_signature:
        print("  edges     " + "  ".join(
            f"{hg.DIRECTION_NAMES[i]}:{e.mean_elev:.0f}m/{e.gradient_out:+.3f}"
            for i, e in enumerate(tile.edge_signature)))


def cmd_geometry(args) -> None:
    s = hg.SIX_MILE_HEX_SIZE_M
    print(f"6-mile hex (flat-top)")
    print(f"  side / circumradius   {s:,.1f} m")
    print(f"  across flats          {hg.across_flats(s):,.1f} m")
    print(f"  across corners        {hg.across_corners(s):,.1f} m")
    print(f"  area                  {hg.area(s) / 1e6:,.2f} km²  "
          f"({hg.area(s) / 2.589988e6:,.2f} sq mi)")
    print("  H3 comparison         res 5 = 252.90 km², res 6 = 36.13 km² — neither fits")
    print()
    for lat in (0, 30, 45, 60):
        z = zoom_for_resolution(lat, 55.0)
        frame = hg.LocalFrame(lat=lat, lon=0.0)
        print(f"  lat {lat:>2}°: z{z}, {tile_count(frame.bbox_lonlat(), z)} tiles per hex")


def cmd_at(args) -> None:
    sampler = build_sampler(args)
    tile = sampler.sample_at(args.lat, args.lon, stage=2 if args.edges else 1)
    report(tile)
    if args.png:
        frame = hg.LocalFrame(args.lat, args.lon)
        z = zoom_for_resolution(args.lat, 55.0)
        mosaic = fetch_mosaic(sampler.transport, frame.bbox_lonlat(), z)
        hr = build_hex_raster(frame, mosaic, grid=args.grid)
        with open(args.png, "wb") as fh:
            fh.write(render_hex_png(hr))
        print(f"  wrote {args.png}")
    if args.json:
        print(json.dumps({"id": tile.id, "lat": tile.lat, "lon": tile.lon,
                          "terrain": tile.terrain_class,
                          "relief": tile.stats.relief}, indent=2))


def cmd_roll(args) -> None:
    sampler = build_sampler(args)
    filters = SamplerFilters(terrain=args.filter)
    tile = sampler.roll(filters)
    report(tile)
    if args.png:
        frame = hg.LocalFrame(tile.lat, tile.lon)
        z = zoom_for_resolution(tile.lat, 55.0)
        mosaic = fetch_mosaic(sampler.transport, frame.bbox_lonlat(), z)
        hr = build_hex_raster(frame, mosaic, grid=args.grid)
        with open(args.png, "wb") as fh:
            fh.write(render_hex_png(hr))
        print(f"  wrote {args.png}")


def cmd_neighbour(args) -> None:
    frame = hg.LocalFrame(args.lat, args.lon)
    lat, lon = frame.neighbor_center(args.edge)
    print(f"edge {args.edge} ({hg.DIRECTION_NAMES[args.edge]}) of "
          f"{args.lat:.4f},{args.lon:.4f}")
    print(f"  -> {lat:.5f}, {lon:.5f}")
    print(f"  meets that tile on its edge {hg.opposite_edge(args.edge)} "
          f"({hg.DIRECTION_NAMES[hg.opposite_edge(args.edge)]})")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="topographer")
    p.add_argument("--synthetic", action="store_true",
                   help="procedural terrain, no network")
    p.add_argument("--cache", default=".tilecache")
    p.add_argument("--grid", type=int, default=320)
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("geometry").set_defaults(func=cmd_geometry)

    a = sub.add_parser("at", help="sample a specific place")
    a.add_argument("--lat", type=float, required=True)
    a.add_argument("--lon", type=float, required=True)
    a.add_argument("--png"); a.add_argument("--json", action="store_true")
    a.add_argument("--edges", action="store_true", help="also compute edge signatures")
    a.set_defaults(func=cmd_at)

    r = sub.add_parser("roll", help="roll a random hex")
    r.add_argument("--filter", default="any",
                   choices=["any", "flat", "hilly", "mountainous"])
    r.add_argument("--png")
    r.set_defaults(func=cmd_roll)

    n = sub.add_parser("neighbour", help="real-world neighbour across an edge")
    n.add_argument("--lat", type=float, required=True)
    n.add_argument("--lon", type=float, required=True)
    n.add_argument("--edge", type=int, required=True, choices=range(6))
    n.set_defaults(func=cmd_neighbour)

    args = p.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
