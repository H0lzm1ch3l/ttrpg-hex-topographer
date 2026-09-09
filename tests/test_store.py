import random

import pytest

from topographer.domain import hexgrid as hg
from topographer.domain.models import Placement
from topographer.pipeline.sample import HexSampler
from topographer.sources.landmask import AlwaysLand
from topographer.sources.tiles import SyntheticTileTransport
from topographer.store.repo import Repo


@pytest.fixture
def repo(tmp_path):
    r = Repo(str(tmp_path / "t.db"))
    r.init_schema()
    yield r
    r.close()


@pytest.fixture
def sampler():
    return HexSampler(transport=SyntheticTileTransport(), mask=AlwaysLand(),
                      grid=120, rng=random.Random(11))


def test_tile_round_trip(repo, sampler):
    tile = sampler.sample_at(46.52, 10.31, stage=2)
    repo.put_tile(tile)
    back = repo.get_tile(tile.id)
    assert back is not None
    assert back.lat == pytest.approx(tile.lat)
    assert back.stats.relief == pytest.approx(tile.stats.relief)
    assert back.edge_signature is not None and len(back.edge_signature) == 6
    assert back.edge_signature[0].profile == tile.edge_signature[0].profile


def test_reroll_of_the_same_spot_reuses_the_tile(repo, sampler):
    a = sampler.sample_at(10.0, 20.0)
    b = sampler.sample_at(10.0, 20.0)
    assert a.id == b.id
    repo.put_tile(a)
    repo.put_tile(b)
    n = repo.conn.execute("SELECT COUNT(*) c FROM hex_tiles").fetchone()["c"]
    assert n == 1


def test_stage_2_upgrade_does_not_lose_signatures(repo, sampler):
    stage1 = sampler.sample_at(10.0, 20.0, stage=1)
    repo.put_tile(stage1)
    stage2 = sampler.sample_at(10.0, 20.0, stage=2)
    repo.put_tile(stage2)
    back = repo.get_tile(stage1.id)
    assert back.stage == 2 and back.edge_signature is not None


def test_placement_is_separate_from_provenance(repo, sampler):
    """The same real-world tile can sit at two slots, in two maps, at once."""
    tile = sampler.sample_at(46.52, 10.31)
    repo.put_tile(tile)
    m1 = repo.create_map("one", hg.SIX_MILE_HEX_SIZE_M)
    m2 = repo.create_map("two", hg.SIX_MILE_HEX_SIZE_M)
    repo.place(Placement(m1, 0, 0, tile.id, origin="random"))
    repo.place(Placement(m1, 3, -1, tile.id, origin="random"))
    repo.place(Placement(m2, 0, 0, tile.id, origin="random"))
    assert len(repo.placements(m1)) == 2
    assert len(repo.placements(m2)) == 1


def test_neighbours_are_keyed_by_edge(repo, sampler):
    tile = sampler.sample_at(46.52, 10.31)
    repo.put_tile(tile)
    m = repo.create_map("m", hg.SIX_MILE_HEX_SIZE_M)
    repo.place(Placement(m, 0, 0, tile.id))
    for e in (0, 2, 4):
        q, r = hg.neighbor_axial(0, 0, e)
        repo.place(Placement(m, q, r, tile.id))

    found = repo.neighbours(m, 0, 0)
    assert set(found.keys()) == {0, 2, 4}
    # and the empty slot NE of the origin sees the origin on its own edge 3
    ne_q, ne_r = hg.neighbor_axial(0, 0, 0)
    assert hg.opposite_edge(0) in repo.neighbours(m, ne_q, ne_r)


def test_origin_edge_survives_the_round_trip(repo, sampler):
    tile = sampler.sample_at(1.0, 1.0)
    repo.put_tile(tile)
    m = repo.create_map("m", hg.SIX_MILE_HEX_SIZE_M)
    repo.place(Placement(m, 1, 0, tile.id, origin="geographic", origin_edge=(0, 0, 3)))
    p = repo.placement_at(m, 1, 0)
    assert p.origin == "geographic" and p.origin_edge == (0, 0, 3)


def test_roll_history_is_ordered(repo, sampler):
    m = repo.create_map("m", hg.SIX_MILE_HEX_SIZE_M)
    ids = []
    for lat in (10.0, 20.0, 30.0):
        t = sampler.sample_at(lat, 5.0)
        repo.put_tile(t)
        repo.push_roll(m, 0, 0, t.id)
        ids.append(t.id)
    hist = repo.roll_history(m, 0, 0)
    assert [s for s, _ in hist] == [0, 1, 2]
    assert [t for _, t in hist] == ids


def test_elev_offset_round_trips_through_placement(repo, sampler):
    tile = sampler.sample_at(46.52, 10.31)
    repo.put_tile(tile)
    m = repo.create_map("m", hg.SIX_MILE_HEX_SIZE_M)
    repo.place(Placement(m, 0, 0, tile.id, elev_offset_m=-42.5))
    p = repo.placement_at(m, 0, 0)
    assert p.elev_offset_m == pytest.approx(-42.5)
    # default is 0.0, not None -- the root of a chain has nothing to align to
    repo.place(Placement(m, 1, 0, tile.id))
    assert repo.placement_at(m, 1, 0).elev_offset_m == 0.0


def test_current_slot_defaults_to_origin_and_moves(repo):
    m = repo.create_map("m", hg.SIX_MILE_HEX_SIZE_M)
    assert repo.get_current_slot(m) == (0, 0)
    repo.set_current_slot(m, 3, -1)
    assert repo.get_current_slot(m) == (3, -1)


def test_create_slot_records_entry_and_record_roll_tracks_origin(repo, sampler):
    m = repo.create_map("m", hg.SIX_MILE_HEX_SIZE_M)
    repo.create_slot(m, 0, 0, entry_edge=None)
    assert repo.get_slot_entry(m, 0, 0) is None
    assert repo.get_slot_origin(m, 0, 0) == "random"  # default before any roll

    repo.create_slot(m, 1, 0, entry_edge=(0, 0, 0))
    assert repo.get_slot_entry(m, 1, 0) == (0, 0, 0)

    tile = sampler.sample_at(46.52, 10.31, stage=2)
    repo.put_tile(tile)
    seq = repo.record_roll(m, 1, 0, tile.id, origin="geographic")
    assert seq == 0
    assert repo.get_current_seq(m, 1, 0) == 0
    assert repo.get_slot_origin(m, 1, 0) == "geographic"

    # a reroll on the same slot updates origin without disturbing entry_edge
    other = sampler.sample_at(-10.0, 40.0, stage=2)
    repo.put_tile(other)
    repo.record_roll(m, 1, 0, other.id, origin="random")
    assert repo.get_slot_origin(m, 1, 0) == "random"
    assert repo.get_slot_entry(m, 1, 0) == (0, 0, 0)
