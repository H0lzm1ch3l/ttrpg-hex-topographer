-- Topographer store.  SQLite is ample at personal scale; the PostGIS path stays
-- open because only repo.py would change.
--
-- The load-bearing split: hex_tiles is PROVENANCE (a real place, immutable,
-- shared across maps, content-addressed) and placements is PLACEMENT (where a
-- tile sits in someone's map). Collapsing them would make edge matching --
-- which is precisely "try this tile at a different slot and rotation" --
-- expensive to add later.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS hex_tiles (
  id              TEXT PRIMARY KEY,   -- hash(lat, lon, hex_size_m, source_versions)
  lat             REAL NOT NULL,
  lon             REAL NOT NULL,
  proj4           TEXT NOT NULL,      -- this tile's own LAEA
  hex_size_m      REAL NOT NULL,
  stage           INTEGER NOT NULL,   -- 1 = preview, 2 = full

  elev_min        REAL, elev_max    REAL,
  elev_mean       REAL, elev_median REAL,
  relief          REAL, slope_mean  REAL,
  slope_p85       REAL, ruggedness  REAL,
  aspect_deg      REAL,

  cover_fractions TEXT,               -- JSON; NULL at stage 1
  koppen          TEXT,
  terrain_class   TEXT,
  secondary_class TEXT,
  edge_signature  TEXT,               -- JSON[6]; NULL at stage 1
  is_coastal      INTEGER DEFAULT 0,
  thumb_png       BLOB,
  source_versions TEXT,
  fetched_at      TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_tiles_terrain ON hex_tiles(terrain_class);
CREATE INDEX IF NOT EXISTS idx_tiles_relief  ON hex_tiles(relief);

CREATE TABLE IF NOT EXISTS landmarks (
  id           TEXT PRIMARY KEY,
  tile_id      TEXT NOT NULL REFERENCES hex_tiles(id) ON DELETE CASCADE,
  source       TEXT NOT NULL,        -- 'osm' | 'geonames' | 'wikidata'
  source_id    TEXT,
  kind         TEXT,
  name         TEXT,
  lat          REAL, lon REAL,
  wikidata_qid TEXT,
  summary      TEXT,
  prominence_m REAL,
  attrs        TEXT
);

CREATE INDEX IF NOT EXISTS idx_landmarks_tile ON landmarks(tile_id);

CREATE TABLE IF NOT EXISTS maps (
  id          TEXT PRIMARY KEY,
  name        TEXT,
  hex_size_m  REAL NOT NULL,
  orientation TEXT NOT NULL DEFAULT 'flat',   -- 'flat' | 'pointy'
  -- The slot the roll/reroll/back/forward UI currently targets. Moves when
  -- an edge click grows the map; everything behind it is locked in.
  current_q   INTEGER NOT NULL DEFAULT 0,
  current_r   INTEGER NOT NULL DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS placements (
  map_id      TEXT NOT NULL REFERENCES maps(id) ON DELETE CASCADE,
  q           INTEGER NOT NULL,
  r           INTEGER NOT NULL,
  tile_id     TEXT NOT NULL REFERENCES hex_tiles(id),
  -- Always 0 in iteration 1. Present from day one because adding it later
  -- would touch the renderer, the exporter, the signature lookup and every
  -- stored map.
  rotation    INTEGER NOT NULL DEFAULT 0,
  origin      TEXT NOT NULL DEFAULT 'random', -- random | geographic | edgematch
  origin_edge TEXT,                           -- JSON [q, r, edge]
  elev_offset_m REAL NOT NULL DEFAULT 0.0,    -- see Placement.elev_offset_m
  placed_at   TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (map_id, q, r)
);

-- Back/forward through rerolls for one slot.
CREATE TABLE IF NOT EXISTS roll_history (
  map_id     TEXT NOT NULL,
  slot_q     INTEGER NOT NULL,
  slot_r     INTEGER NOT NULL,
  seq        INTEGER NOT NULL,
  tile_id    TEXT NOT NULL REFERENCES hex_tiles(id),
  created_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY (map_id, slot_q, slot_r, seq)
);

-- Where the back/forward cursor currently sits for one slot. Separate from
-- roll_history itself so going back and then rerolling never has to delete
-- history -- it just appends past whatever the pointer was on.
CREATE TABLE IF NOT EXISTS slot_state (
  map_id      TEXT NOT NULL REFERENCES maps(id) ON DELETE CASCADE,
  slot_q      INTEGER NOT NULL,
  slot_r      INTEGER NOT NULL,
  current_seq INTEGER NOT NULL,
  -- How this slot came to exist: JSON [from_q, from_r, edge] of the click
  -- that grew it, or NULL for a map's root slot. Constant for the slot's
  -- life; only the tile occupying it changes as it's (re)rolled.
  entry_edge  TEXT,
  -- How the *current* tip tile was chosen. A plain reroll always produces
  -- 'random'; a freshly grown slot starts 'geographic' until rerolled.
  origin      TEXT NOT NULL DEFAULT 'random',
  PRIMARY KEY (map_id, slot_q, slot_r)
);
