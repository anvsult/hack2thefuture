"""SQLite storage for tickets and the alert-sent log. Survives Streamlit reruns and restarts."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent / "claims_alerts.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    claim_id      TEXT NOT NULL,
    policy_id     TEXT NOT NULL,
    rule          TEXT NOT NULL,
    tier          TEXT NOT NULL,
    sim_date      TEXT NOT NULL,      -- simulated "today" when the rule fired
    created_at    TEXT NOT NULL,      -- wall-clock time
    team          TEXT,
    city          TEXT,
    owner         TEXT,
    reason        TEXT,
    next_step     TEXT,
    premium_at_risk REAL,
    summary       TEXT,
    summary_source TEXT,              -- 'gemini' or 'template'
    status        TEXT NOT NULL DEFAULT 'open',
    slack_status  TEXT,               -- 'sent', 'dry-run', 'failed: ...'
    UNIQUE (claim_id, rule, tier)     -- dedupe: one ticket per claim, rule and tier
);
"""


@contextmanager
def connect(path: Path | None = None):
    con = sqlite3.connect(path or DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def already_sent(con: sqlite3.Connection, claim_id: str, rule: str, tier: str) -> bool:
    row = con.execute("SELECT 1 FROM tickets WHERE claim_id=? AND rule=? AND tier=?",
                      (claim_id, rule, tier)).fetchone()
    return row is not None


def reserve(con: sqlite3.Connection, alert: dict, sim_date: str) -> int | None:
    """Insert the ticket before any side effect. Returns the new id, or None if it already exists."""
    cur = con.execute(
        """INSERT OR IGNORE INTO tickets
           (claim_id, policy_id, rule, tier, sim_date, created_at, team, city, owner,
            reason, next_step, premium_at_risk)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (alert["claim_id"], alert["policy_id"], alert["rule"], alert["tier"], sim_date,
         datetime.now().isoformat(timespec="seconds"), alert["team"], alert["city"],
         alert["owner"], alert["reason"], alert["next_step"], float(alert["premium_at_risk"])))
    con.commit()
    return cur.lastrowid if cur.rowcount else None


def finish(con: sqlite3.Connection, ticket_id: int, summary: str, source: str, slack: str) -> None:
    con.execute("UPDATE tickets SET summary=?, summary_source=?, slack_status=? WHERE id=?",
                (summary, source, slack, ticket_id))


def set_status(con: sqlite3.Connection, ticket_id: int, status: str) -> None:
    con.execute("UPDATE tickets SET status=? WHERE id=?", (status, ticket_id))


def tickets(con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM tickets ORDER BY id DESC", con)


def reset(con: sqlite3.Connection) -> None:
    con.execute("DELETE FROM tickets")
