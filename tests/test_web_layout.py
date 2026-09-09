import math

import pytest

from topographer.domain import hexgrid as hg
from topographer.web.layout import layout_cells


def test_single_cell_sits_at_the_padding_origin():
    layout = layout_cells([(0, 0)], hex_w=100.0)
    box = layout.cells[(0, 0)]
    assert box.left == pytest.approx(layout.width / 2 - 50.0)
    assert box.top == pytest.approx(layout.height / 2 - layout.hex_h / 2)
    assert layout.hex_h == pytest.approx(100.0 * math.sqrt(3) / 2)


def test_adjacent_cells_are_spaced_one_hex_width_across_flats():
    """Regardless of direction, neighbour centres in a flat-top grid are
    exactly ``across_flats`` apart -- this is the same invariant real tiles
    satisfy in metres (see hexgrid tests), just at pixel scale."""
    hex_w = 120.0
    size_px = hex_w / 2.0
    expected = hg.across_flats(size_px)

    for edge in range(6):
        nq, nr = hg.neighbor_axial(0, 0, edge)
        layout = layout_cells([(0, 0), (nq, nr)], hex_w=hex_w)
        cx0, cy0 = layout.center(0, 0)
        cx1, cy1 = layout.center(nq, nr)
        dist = math.hypot(cx1 - cx0, cy1 - cy0)
        assert dist == pytest.approx(expected, abs=1e-6)


def test_no_two_cells_overlap_the_bounding_box():
    cells = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0)]
    layout = layout_cells(cells, hex_w=100.0)
    for q, r in cells:
        box = layout.cells[(q, r)]
        assert 0 <= box.left <= layout.width - layout.hex_w
        assert 0 <= box.top <= layout.height - layout.hex_h


def test_edge_anchor_directions_match_compass_names():
    """Edge 1 is N: its anchor should sit strictly above the hex centre.
    Edge 4 is S: strictly below. Screen y grows downward."""
    layout = layout_cells([(0, 0)], hex_w=100.0)
    cx, cy = layout.center(0, 0)
    anchors = {a.edge: a for a in layout.edge_anchors(0, 0)}
    assert anchors[1].top < cy
    assert anchors[4].top > cy
    assert anchors[1].left == pytest.approx(cx, abs=1e-6)


def test_layout_rejects_empty_input():
    with pytest.raises(ValueError):
        layout_cells([])
