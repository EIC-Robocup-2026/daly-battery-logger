import json
import sqlite3
import time

import pandas as pd

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS readings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        REAL NOT NULL,
    device_id TEXT NOT NULL DEFAULT 'default',
    voltage   REAL,
    current   REAL,
    soc       REAL,
    cell_min  REAL,
    cell_max  REAL,
    temp_min  REAL,
    temp_max  REAL,
    mode        TEXT,
    errors      TEXT,
    power       REAL,
    capacity_ah REAL
);
CREATE INDEX IF NOT EXISTS idx_ts ON readings(ts);
"""

INSERT_SQL = """
INSERT INTO readings
    (ts, device_id, voltage, current, soc, cell_min, cell_max,
     temp_min, temp_max, mode, errors, power, capacity_ah)
VALUES
    (:ts, :device_id, :voltage, :current, :soc, :cell_min, :cell_max,
     :temp_min, :temp_max, :mode, :errors, :power, :capacity_ah)
"""

_MIGRATIONS = [
    "ALTER TABLE readings ADD COLUMN device_id TEXT NOT NULL DEFAULT 'default'",
    "ALTER TABLE readings ADD COLUMN power REAL",
    "ALTER TABLE readings ADD COLUMN capacity_ah REAL",
]


class DataLogger:
    def __init__(self, db_path: str, device_id: str = "default"):
        self._db_path = db_path
        self._device_id = device_id
        self._conn = None

    def open(self):
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.isolation_level = None  # autocommit
        self._conn.executescript(CREATE_SQL)
        self._run_migrations()

    def _run_migrations(self):
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(readings)")}
        if "device_id" not in existing:
            self._conn.execute(_MIGRATIONS[0])
        # Always ensure index exists (handles both new and migrated databases)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_device_id ON readings(device_id)"
        )
        if "power" not in existing:
            self._conn.execute(_MIGRATIONS[1])
        if "capacity_ah" not in existing:
            self._conn.execute(_MIGRATIONS[2])

    def insert(self, record: dict):
        if self._conn is None:
            return
        row = {
            "ts": record.get("ts", time.time()),
            "device_id": self._device_id,
            "voltage": record.get("voltage"),
            "current": record.get("current"),
            "soc": record.get("soc"),
            "cell_min": record.get("cell_min"),
            "cell_max": record.get("cell_max"),
            "temp_min": record.get("temp_min"),
            "temp_max": record.get("temp_max"),
            "mode": record.get("mode"),
            "errors": json.dumps(record.get("errors", [])),
            "power": record.get("power"),
            "capacity_ah": record.get("capacity_ah"),
        }
        self._conn.execute(INSERT_SQL, row)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


def query_range(
    db_path: str,
    start_ts: float,
    end_ts: float,
    device_id: str | None = None,
) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    try:
        if device_id:
            df = pd.read_sql_query(
                "SELECT * FROM readings WHERE ts >= ? AND ts <= ? AND device_id = ? ORDER BY ts",
                conn,
                params=(start_ts, end_ts, device_id),
            )
        else:
            df = pd.read_sql_query(
                "SELECT * FROM readings WHERE ts >= ? AND ts <= ? ORDER BY ts",
                conn,
                params=(start_ts, end_ts),
            )
    finally:
        conn.close()
    return df


def list_devices(db_path: str) -> list[str]:
    """Return all distinct device_ids that have readings."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT device_id FROM readings ORDER BY device_id"
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def query_new_since(db_path: str, since_ts: float) -> list[dict]:
    """Return all readings with ts > since_ts ordered by ts as plain dicts (JSON-ready)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM readings WHERE ts > ? ORDER BY ts", (since_ts,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def query_latest(db_path: str) -> list[dict]:
    """Return the most recent reading for each device_id."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM readings WHERE id IN "
            "(SELECT MAX(id) FROM readings GROUP BY device_id)"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
