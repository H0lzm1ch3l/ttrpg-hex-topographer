# Topographer — P0 scaffold

Real-world topography → 6-mile hexcrawl tiles. This is the **P0 skeleton** from
`claude/tile-collage-model.md`: hex geometry, elevation fetch, zonal statistics,
the random land sampler, edge signatures, and the store. CLI only — no web layer
yet, deliberately.

```bash
pip install numpy pillow requests pyproj shapely flask pytest
python scripts/build_assets.py land        # Natural Earth land polygons (~1.6 MB)
python -m pytest tests -q                  # 65 tests, no network needed
```

## Try it

```bash
python -m topographer.cli geometry
python -m topographer.cli --synthetic roll --filter mountainous --png hex.png
python -m topographer.cli at --lat 46.52 --lon 10.31 --edges --png alps.png
python -m topographer.cli neighbour --lat 46.52 --lon 10.31 --edge 0
```

Or the web viewer -- roll a random 6-mile hex and reroll with back/forward
history:

```bash
python -m topographer.web.app     # http://127.0.0.1:5000
```

`--synthetic` swaps in procedural terrain so the whole pipeline runs with no
network at all. Without it, elevation comes from AWS terrain tiles over HTTP.

> The synthetic generator's diagonal streaks are an artefact of its ridged-noise
> term, not of the pipeline. Real terrain does not look like that.

## What's here

| Module | Does |
|---|---|
| `domain/hexgrid.py` | Flat-top axial coords, per-tile LAEA, edge geometry, rotation |
| `domain/models.py` | `HexTile`, `Placement`, `EdgeSignature`, `CandidateSource` |
| `sources/tiles.py` | Slippy maths, terrarium decode, mosaic, pluggable transport |
| `sources/landmask.py` | Natural Earth land test, prescreen grid |
| `pipeline/zonal.py` | Hex mask, elevation stats, slope/aspect, hillshade, edge signatures |
| `pipeline/classify.py` | Stage-1 terrain cascade (relief + slope + climate) |
| `pipeline/sample.py` | Equal-area sampler, `RandomSource`, `GeographicSource` |
| `pipeline/render.py` | Hex thumbnail PNG |
| `store/` | SQLite schema and repository |
| `web/` | Flask viewer: roll, reroll, back/forward, grow the map by clicking a hex edge |
| `web/layout.py` | Pure pixel geometry for the map view -- no Flask, independently tested |

## Three things worth knowing before you edit

**Provenance and placement never share a table.** `hex_tiles` is a real place;
`placements` is where a tile sits in a map. Edge matching is precisely "try this
tile at a different slot and rotation", so collapsing them would make that
feature expensive later.

**Edge signatures are computed at stage 2 and stored forever.** Nothing reads
them yet. They exist so that edge-fit matching is a scoring function over data
already on disk rather than a re-fetch of every raster for every tile.
`EdgeMatchSource.score()` is the reference implementation, unused but pinned by
a test.

**Conventions in `hexgrid.py` are load-bearing.** Edge order is tile-local and
counterclockwise; rotation is a cyclic shift; two hexes traverse their shared
edge in opposite directions, so one profile must be reversed before comparison.
The module docstring is the spec.

## Verified, and not

Tested against terrain with a known answer — a tilted plane, a cone, flat
ground — rather than only against itself:

- 6-mile hex is 80.75 km² / 31.18 sq mi; H3 res 5 (252.9) and res 6 (36.1)
  both miss it, which is why there's a custom lattice
- slope, relief and aspect on a 10% plane match the analytic values
- hillshade on flat ground equals sin(altitude); a west-facing slope is brighter
  under a NW sun than an east-facing one
- neighbour centres are one across-flats away by geodesic distance at every
  latitude tested
- the land mask puts the Alps, Kansas, the Sahara, Siberia and the Amazon on
  land and the mid-Pacific, mid-Atlantic, Southern Ocean and Bay of Bengal in
  water; equal-area sampling recovers the ~29% land fraction

**Not verified: the live HTTP fetch.** `s3.amazonaws.com` was blocked by egress
policy in the environment this was written in, so `HttpTileTransport` has never
made a real request. Every other layer was exercised through the transport
interface. Run `python -m topographer.cli at --lat 46.52 --lon 10.31 --png x.png`
as the first thing on your machine — the Ortler massif should come back with
roughly 1,500–2,500 m of relief. If that works, the whole pipeline works.

**Known and deliberate:** stepping to a real-world neighbour and back does not
land exactly where you started — the two LAEA frames disagree about north by
the meridian convergence. Measured at 0 m on the equator, 12.6 m at 45°, 34.7 m
at 70°, i.e. at worst 0.36% of a hex width, and it does not accumulate along a
straight walk. Invisible in a collage; it would matter in a contiguous map,
which this isn't.

## Not built yet

Stage 2 land cover (WorldCover COGs), Köppen lookup, Overpass landmarks,
Wikidata enrichment, the prescreen grid builder's tuning, exports.
`EdgeMatchSource` raises `NotImplementedError` on purpose — it's iteration 2,
and its inputs are already being stored.

The web viewer renders a real spatial map: click a direction on the current
hex to lock it into `placements` and step to its real-world geographic
neighbour there, which becomes the new reroll target. Every locked hex draws
at its actual `(q, r)` position (`web/layout.py` — pure pixel geometry reusing
the same axial layout the pipeline uses in metres; no MapLibre, because map
space in the collage model is abstract, not real geography, so a
geographic-projection widget is the wrong tool here). There's still no way
back into a locked hex. Each placement also carries `elev_offset_m`, a
vertical shift computed from edge signatures so a locked hex's shared edge
lines up with whatever it was grown from; nothing renders it visually yet
(no elevation-aware shading across hexes), but it's there for when that's
built.
