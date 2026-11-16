"""SQLite store for emitted signals and data-quality issues. Re-running a session is idempotent."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
  signal_id     TEXT PRIMARY KEY,
  detector      TEXT NOT NULL,
  kind          TEXT NOT NULL,
  ticker        TEXT NOT NULL,
  market        TEXT NOT NULL,
  session_date  TEXT NOT NULL,
  severity_hint INTEGER NOT NULL,
  payload       TEXT NOT NULL,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS signals_by_session ON signals (market, session_date);
CREATE TABLE IF NOT EXISTS quality_issues (
  market        TEXT NOT NULL,
  session_date  TEXT NOT NULL,
  ticker        TEXT NOT NULL,
  issue         TEXT NOT NULL,
  detail        TEXT,
  created_at    TEXT NOT NULL,
  PRIMARY KEY (market, session_date, ticker, issue)
);
"""


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.executescript(SCHEMA)

    def save_signals(self, signals: list[dict]) -> list[dict]:
        """Insert signals; returns only those not stored before."""
        now = datetime.now(timezone.utc).isoformat()
        new = []
        with self.db:
            for s in signals:
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO signals VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        s["signal_id"], s["detector"], s["kind"], s["topics"][0].split(":", 1)[1],
                        s["market"], s["session_date"], s["severity_hint"], json.dumps(s), now,
                    ),
                )
                if cur.rowcount:
                    new.append(s)
        return new

    def save_issues(self, market: str, session_date: str, issues: list[dict]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.executemany(
                "INSERT OR REPLACE INTO quality_issues VALUES (?,?,?,?,?,?)",
                [(market, session_date, i["ticker"], i["issue"], i.get("detail"), now) for i in issues],
            )
