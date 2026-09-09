"""SQLite repository.  The only module that knows about persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import asdict

from ..domain.models import EdgeSignature, ElevStats, HexTile, Placement

SCHEMA = os.path.join(os.path.dirname(__file__), "schema.sql")


class Repo:
    def __init__(self, path: str = "topographer.db"):
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")

    def init_schema(self) -> None:
        with open(SCHEMA) as fh:
            self.conn.executescript(fh.read())
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- tiles -----------------------------------------------------------

    def put_tile(self, t: HexTile) -> str:
        s = t.stats
        self.conn.execute(
            """INSERT INTO hex_tiles
               (id, lat, lon, proj4, hex_size_m, stage,
                elev_min, elev_max, elev_mean, elev_median, relief,
                slope_mean, slope_p85, ruggedness, aspect_deg,
                cover_fractions, koppen, terrain_class, secondary_class,
                edge_signature, is_coastal, thumb_png, source_versions)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 stage = MAX(hex_tiles.stage, excluded.stage),
                 cover_fractions = COALESCE(excluded.cover_fractions, hex_tiles.cover_fractions),
                 edge_signature  = COALESCE(excluded.edge_signature,  hex_tiles.edge_signature),
                 thumb_png       = COALESCE(excluded.thumb_png,       hex_tiles.thumb_png),
                 terrain_class   = excluded.terrain_class""",
            (t.id, t.lat, t.lon, t.proj4, t.hex_size_m, t.stage,
             s.elev_min, s.elev_max, s.elev_mean, s.elev_median, s.relief,
             s.slope_mean, s.slope_p85, s.ruggedness, s.aspect_deg,
             json.dumps(t.cover_fractions) if t.cover_fractions else None,
             t.koppen, t.terrain_class, t.secondary_class,
             json.dumps([asdict(e) for e in t.edge_signature]) if t.edge_signature else None,
             int(t.is_coastal), t.thumb_png, json.dumps(t.source_versions)),
        )
        self.conn.commit()
        return t.id

    def get_tile(self, tile_id: str) -> HexTile | None:
        row = self.conn.execute("SELECT * FROM hex_tiles WHERE id = ?", (tile_id,)).fetchone()
        return self._row_to_tile(row) if row else None

    @staticmethod
    def _row_to_tile(row: sqlite3.Row) -> HexTile:
        stats = ElevStats(
            elev_min=row["elev_min"], elev_max=row["elev_max"],
            elev_mean=row["elev_mean"], elev_median=row["elev_median"],
            relief=row["relief"], slope_mean=row["slope_mean"],
            slope_p85=row["slope_p85"], ruggedness=row["ruggedness"],
            aspect_deg=row["aspect_deg"],
        )
        sig = None
        if row["edge_signature"]:
            sig = [EdgeSignature(**e) for e in json.loads(row["edge_signature"])]
        return HexTile(
            id=row["id"], lat=row["lat"], lon=row["lon"], proj4=row["proj4"],
            hex_size_m=row["hex_size_m"], stage=row["stage"], stats=stats,
            koppen=row["koppen"], terrain_class=row["terrain_class"],
            secondary_class=row["secondary_class"],
            cover_fractions=json.loads(row["cover_fractions"]) if row["cover_fractions"] else None,
            edge_signature=sig, is_coastal=bool(row["is_coastal"]),
            thumb_png=row["thumb_png"],
            source_versions=json.loads(row["source_versions"] or "{}"),
        )

    # -- maps and placements ---------------------------------------------

    def create_map(self, name: str, hex_size_m: float, orientation: str = "flat") -> str:
        map_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO maps (id, name, hex_size_m, orientation) VALUES (?,?,?,?)",
            (map_id, name, hex_size_m, orientation),
        )
        self.conn.commit()
        return map_id

    def get_or_create_map(self, name: str, hex_size_m: float,
                          orientation: str = "flat") -> str:
        row = self.conn.execute("SELECT id FROM maps WHERE name = ?", (name,)).fetchone()
        if row:
            return row["id"]
        return self.create_map(name, hex_size_m, orientation)

    def get_current_slot(self, map_id: str) -> tuple[int, int]:
        row = self.conn.execute(
            "SELECT current_q, current_r FROM maps WHERE id = ?", (map_id,)).fetchone()
        return (row["current_q"], row["current_r"]) if row else (0, 0)

    def set_current_slot(self, map_id: str, q: int, r: int) -> None:
        self.conn.execute(
            "UPDATE maps SET current_q=?, current_r=? WHERE id=?", (q, r, map_id))
        self.conn.commit()

    def place(self, p: Placement) -> None:
        self.conn.execute(
            """INSERT INTO placements
               (map_id, q, r, tile_id, rotation, origin, origin_edge, elev_offset_m)
               VALUES (?,?,?,?,?,?,?,?)
               ON CONFLICT(map_id, q, r) DO UPDATE SET
                 tile_id = excluded.tile_id, rotation = excluded.rotation,
                 origin = excluded.origin, origin_edge = excluded.origin_edge,
                 elev_offset_m = excluded.elev_offset_m""",
            (p.map_id, p.q, p.r, p.tile_id, p.rotation, p.origin,
             json.dumps(p.origin_edge) if p.origin_edge else None, p.elev_offset_m),
        )
        self.conn.commit()

    def placements(self, map_id: str) -> list[Placement]:
        rows = self.conn.execute(
            "SELECT * FROM placements WHERE map_id = ? ORDER BY placed_at",
            (map_id,)).fetchall()
        return [self._row_to_placement(r) for r in rows]

    def placement_at(self, map_id: str, q: int, r: int) -> Placement | None:
        row = self.conn.execute(
            "SELECT * FROM placements WHERE map_id=? AND q=? AND r=?", (map_id, q, r)).fetchone()
        return self._row_to_placement(row) if row else None

    @staticmethod
    def _row_to_placement(row: sqlite3.Row) -> Placement:
        return Placement(
            map_id=row["map_id"], q=row["q"], r=row["r"], tile_id=row["tile_id"],
            rotation=row["rotation"], origin=row["origin"],
            origin_edge=tuple(json.loads(row["origin_edge"])) if row["origin_edge"] else None,
            elev_offset_m=row["elev_offset_m"],
        )

    def neighbours(self, map_id: str, q: int, r: int) -> dict[int, Placement]:
        """All placed tiles adjacent to a slot, keyed by edge index.

        Iteration 1's sources ignore this; edge matching needs every one of
        them, which is why the lookup exists now.
        """
        from ..domain.hexgrid import DIRECTIONS

        found: dict[int, Placement] = {}
        for edge, (dq, dr) in enumerate(DIRECTIONS):
            p = self.placement_at(map_id, q + dq, r + dr)
            if p:
                found[edge] = p
        return found

    # -- roll history ----------------------------------------------------

    def push_roll(self, map_id: str, q: int, r: int, tile_id: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(seq), -1) AS m FROM roll_history "
            "WHERE map_id=? AND slot_q=? AND slot_r=?", (map_id, q, r)).fetchone()
        seq = int(row["m"]) + 1
        self.conn.execute(
            "INSERT INTO roll_history (map_id, slot_q, slot_r, seq, tile_id) VALUES (?,?,?,?,?)",
            (map_id, q, r, seq, tile_id))
        self.conn.commit()
        return seq

    def roll_history(self, map_id: str, q: int, r: int) -> list[tuple[int, str]]:
        rows = self.conn.execute(
            "SELECT seq, tile_id FROM roll_history WHERE map_id=? AND slot_q=? AND slot_r=? "
            "ORDER BY seq", (map_id, q, r)).fetchall()
        return [(r["seq"], r["tile_id"]) for r in rows]

    # -- slot cursor -------------------------------------------------------

    def get_current_seq(self, map_id: str, q: int, r: int) -> int | None:
        row = self.conn.execute(
            "SELECT current_seq FROM slot_state WHERE map_id=? AND slot_q=? AND slot_r=?",
            (map_id, q, r)).fetchone()
        return int(row["current_seq"]) if row else None

    def set_current_seq(self, map_id: str, q: int, r: int, seq: int) -> None:
        self.conn.execute(
            """INSERT INTO slot_state (map_id, slot_q, slot_r, current_seq)
               VALUES (?,?,?,?)
               ON CONFLICT(map_id, slot_q, slot_r) DO UPDATE SET
                 current_seq = excluded.current_seq""",
            (map_id, q, r, seq),
        )
        self.conn.commit()

    def create_slot(self, map_id: str, q: int, r: int,
                    entry_edge: tuple[int, int, int] | None) -> None:
        """Register a slot before it has any rolls, recording how it was
        entered. A no-op if the slot already exists (rerolling never
        re-creates it)."""
        self.conn.execute(
            """INSERT INTO slot_state (map_id, slot_q, slot_r, current_seq, entry_edge)
               VALUES (?,?,?,-1,?)
               ON CONFLICT(map_id, slot_q, slot_r) DO NOTHING""",
            (map_id, q, r, json.dumps(entry_edge) if entry_edge else None),
        )
        self.conn.commit()

    def get_slot_entry(self, map_id: str, q: int, r: int) -> tuple[int, int, int] | None:
        row = self.conn.execute(
            "SELECT entry_edge FROM slot_state WHERE map_id=? AND slot_q=? AND slot_r=?",
            (map_id, q, r)).fetchone()
        return tuple(json.loads(row["entry_edge"])) if row and row["entry_edge"] else None

    def get_slot_origin(self, map_id: str, q: int, r: int) -> str:
        row = self.conn.execute(
            "SELECT origin FROM slot_state WHERE map_id=? AND slot_q=? AND slot_r=?",
            (map_id, q, r)).fetchone()
        return row["origin"] if row else "random"

    def record_roll(self, map_id: str, q: int, r: int, tile_id: str,
                    origin: str = "random") -> int:
        """Push a roll, move the cursor to it, and record how it was chosen --
        the one call the web layer needs for both a plain reroll and a fresh
        slot's first candidate."""
        seq = self.push_roll(map_id, q, r, tile_id)
        self.set_current_seq(map_id, q, r, seq)
        self.conn.execute(
            "UPDATE slot_state SET origin=? WHERE map_id=? AND slot_q=? AND slot_r=?",
            (origin, map_id, q, r),
        )
        self.conn.commit()
        return seq
