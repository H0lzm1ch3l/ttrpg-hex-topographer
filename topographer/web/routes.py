"""Roll / view / history / edge-growth routes.

One map, one *current* slot at a time -- the one the roll/reroll/back/forward
UI targets. Clicking an edge locks the current hex into ``placements``
(computing its ``elev_offset_m`` against whatever it was grown from, if
anything) and moves the current slot to that hex's real-world neighbour
across the clicked edge. Every locked hex renders at its real ``(q, r)``
position via ``layout.py``, so the map is an actual spatial grid, not just a
chain -- but there is still no way back into a locked hex yet.

``roll_history`` never loses a row: back/forward only move
``slot_state.current_seq``, so rerolling after going back simply appends past
the old tip rather than truncating it.
"""

from __future__ import annotations

from flask import Blueprint, Response, current_app, g, redirect, render_template, url_for

from ..domain import hexgrid as hg
from ..domain.models import Placement, SamplerFilters
from ..pipeline.render import render_hex_png
from ..store.repo import Repo
from .layout import layout_cells

bp = Blueprint("roll", __name__)

MAP_NAME = "default"
HEX_W_PX = 130.0


def _repo() -> Repo:
    if "repo" not in g:
        g.repo = Repo(current_app.config["DB_PATH"])
    return g.repo


@bp.teardown_app_request
def _close_repo(exc: BaseException | None) -> None:
    repo = g.pop("repo", None)
    if repo is not None:
        repo.close()


def _sampler():
    return current_app.extensions["sampler"]


def _map_id() -> str:
    return _repo().get_or_create_map(MAP_NAME, hg.SIX_MILE_HEX_SIZE_M)


def _store_and_record(map_id: str, q: int, r: int, tile, hr, origin: str) -> None:
    tile.thumb_png = render_hex_png(hr)
    repo = _repo()
    repo.put_tile(tile)
    repo.record_roll(map_id, q, r, tile.id, origin=origin)


def _roll_new_tile(map_id: str, q: int, r: int) -> None:
    """Sample a fresh random hex for the given slot and make it the tip of
    that slot's history. Stage 2 so an edge signature is ready the moment the
    user clicks an edge on it."""
    tile, hr = _sampler().roll_with_raster(SamplerFilters(), stage=2)
    _store_and_record(map_id, q, r, tile, hr, origin="random")


def _elev_offset_for_lock(repo: Repo, map_id: str, tile, entry) -> float:
    """The vertical shift that lines ``tile``'s shared edge up with whatever
    it was grown from. 0.0 for a slot with no entry -- the root of a chain
    defines the map's own baseline."""
    if entry is None:
        return 0.0
    from_q, from_r, from_edge = entry
    from_placement = repo.placement_at(map_id, from_q, from_r)
    from_tile = repo.get_tile(from_placement.tile_id)
    to_edge = hg.opposite_edge(from_edge)
    return (from_tile.edge_signature[from_edge].mean_elev + from_placement.elev_offset_m
            - tile.edge_signature[to_edge].mean_elev)


@bp.route("/")
def index():
    repo = _repo()
    map_id = _map_id()
    q, r = repo.get_current_slot(map_id)

    seq = repo.get_current_seq(map_id, q, r)
    if seq is None:
        repo.create_slot(map_id, q, r, entry_edge=None)
        _roll_new_tile(map_id, q, r)
        seq = repo.get_current_seq(map_id, q, r)

    history = dict(repo.roll_history(map_id, q, r))
    tile = repo.get_tile(history[seq])

    chain = repo.placements(map_id)
    cells = [(p.q, p.r) for p in chain if (p.q, p.r) != (q, r)] + [(q, r)]
    layout = layout_cells(cells, hex_w=HEX_W_PX)

    map_hexes = [
        {"tile_id": p.tile_id, "left": layout.cells[(p.q, p.r)].left,
         "top": layout.cells[(p.q, p.r)].top, "locked": True}
        for p in chain if (p.q, p.r) != (q, r)
    ]
    map_hexes.append({
        "tile_id": tile.id, "left": layout.cells[(q, r)].left,
        "top": layout.cells[(q, r)].top, "locked": False,
    })

    seqs = sorted(history)
    return render_template(
        "index.html",
        tile=tile,
        seq_index=seqs.index(seq) + 1,
        seq_count=len(seqs),
        can_back=seq > seqs[0],
        can_forward=seq < seqs[-1],
        map_hexes=map_hexes,
        map_width=layout.width,
        map_height=layout.height,
        hex_w=layout.hex_w,
        hex_h=layout.hex_h,
        edge_anchors=layout.edge_anchors(q, r),
        directions=dict(enumerate(hg.DIRECTION_NAMES)),
    )


@bp.route("/roll", methods=["POST"])
def roll():
    map_id = _map_id()
    q, r = _repo().get_current_slot(map_id)
    _roll_new_tile(map_id, q, r)
    return redirect(url_for("roll.index"))


@bp.route("/back", methods=["POST"])
def back():
    _step(-1)
    return redirect(url_for("roll.index"))


@bp.route("/forward", methods=["POST"])
def forward():
    _step(+1)
    return redirect(url_for("roll.index"))


def _step(direction: int) -> None:
    repo = _repo()
    map_id = _map_id()
    q, r = repo.get_current_slot(map_id)
    seqs = sorted(s for s, _ in repo.roll_history(map_id, q, r))
    current = repo.get_current_seq(map_id, q, r)
    if current is None or not seqs:
        return
    candidates = [s for s in seqs if (s < current if direction < 0 else s > current)]
    if candidates:
        repo.set_current_seq(map_id, q, r, max(candidates) if direction < 0 else min(candidates))


@bp.route("/edge/<int:edge>", methods=["POST"])
def click_edge(edge: int):
    """Lock the current hex in and step to its real-world neighbour."""
    if not 0 <= edge <= 5:
        return Response(status=404)

    repo = _repo()
    map_id = _map_id()
    q, r = repo.get_current_slot(map_id)

    seq = repo.get_current_seq(map_id, q, r)
    tile = repo.get_tile(dict(repo.roll_history(map_id, q, r))[seq])
    entry = repo.get_slot_entry(map_id, q, r)
    origin = repo.get_slot_origin(map_id, q, r)

    offset = _elev_offset_for_lock(repo, map_id, tile, entry)
    repo.place(Placement(map_id=map_id, q=q, r=r, tile_id=tile.id,
                         origin=origin, origin_edge=entry, elev_offset_m=offset))

    nq, nr = hg.neighbor_axial(q, r, edge)
    new_tile, hr = _sampler().sample_neighbor_with_raster(tile, edge)
    repo.create_slot(map_id, nq, nr, entry_edge=(q, r, edge))
    _store_and_record(map_id, nq, nr, new_tile, hr, origin="geographic")
    repo.set_current_slot(map_id, nq, nr)

    return redirect(url_for("roll.index"))


@bp.route("/tile/<tile_id>.png")
def tile_png(tile_id: str):
    tile = _repo().get_tile(tile_id)
    if tile is None or not tile.thumb_png:
        return Response(status=404)
    return Response(tile.thumb_png, mimetype="image/png")
