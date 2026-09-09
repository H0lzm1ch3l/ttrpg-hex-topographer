"""Screen-pixel layout for the hex map view. Pure geometry, no Flask.

Reuses ``domain.hexgrid``'s flat-top axial layout -- the same formula real
tiles use in metres -- at a pixel scale, so the grid the browser sees tiles
exactly like the grid the pipeline measures distances on. There is no
MapLibre here on purpose: map space in the collage model is abstract
``(q, r)``, not real geography (see ``tile_collage_model.md`` S1), so a
geographic-projection map widget is the wrong tool regardless of what the
older, superseded design doc recommended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..domain import hexgrid as hg

PAD = 24.0


@dataclass
class CellBox:
    q: int
    r: int
    left: float
    top: float


@dataclass
class EdgeAnchor:
    edge: int
    left: float
    top: float


@dataclass
class Layout:
    cells: dict[tuple[int, int], CellBox]
    width: float
    height: float
    hex_w: float
    hex_h: float

    def center(self, q: int, r: int) -> tuple[float, float]:
        box = self.cells[(q, r)]
        return box.left + self.hex_w / 2.0, box.top + self.hex_h / 2.0

    def edge_anchors(self, q: int, r: int, push: float = 1.3) -> list[EdgeAnchor]:
        """Screen positions for six edge hotspots around cell ``(q, r)``,
        pushed ``push``x past the hex boundary so they sit outside the image.
        """
        cx, cy = self.center(q, r)
        out = []
        for i in range(6):
            a, b = hg.edge_endpoints(self.hex_w / 2.0, i)
            mx = (a[0] + b[0]) / 2.0 * push
            my = (a[1] + b[1]) / 2.0 * push
            out.append(EdgeAnchor(edge=i, left=cx + mx, top=cy - my))  # -my: math y is north-positive, screen y grows downward
        return out


def layout_cells(cells: list[tuple[int, int]], hex_w: float = 130.0) -> Layout:
    """Position every ``(q, r)`` in ``cells`` on a screen-pixel canvas.

    ``hex_w`` is the rendered image width -- across corners, matching
    ``render_hex_png``'s own convention -- so a thumbnail dropped in at its
    box's (left, top) tiles the grid exactly, no gaps or overlap.
    """
    if not cells:
        raise ValueError("layout_cells needs at least one cell")

    size_px = hex_w / 2.0
    hex_h = hex_w * math.sqrt(3.0) / 2.0

    raw = {(q, r): (x, -y) for (q, r), (x, y) in
           ((qr, hg.axial_to_xy(qr[0], qr[1], size_px)) for qr in cells)}

    xs = [x for x, _ in raw.values()]
    ys = [y for _, y in raw.values()]
    minx, miny = min(xs), min(ys)
    maxx, maxy = max(xs), max(ys)

    boxes = {
        qr: CellBox(q=qr[0], r=qr[1], left=x - minx + PAD, top=y - miny + PAD)
        for qr, (x, y) in raw.items()
    }
    return Layout(
        cells=boxes,
        width=(maxx - minx) + hex_w + 2 * PAD,
        height=(maxy - miny) + hex_h + 2 * PAD,
        hex_w=hex_w, hex_h=hex_h,
    )
