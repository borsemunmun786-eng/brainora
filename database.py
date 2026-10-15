"""
database.py  –  SQLite persistence layer
All tables are created on first import. No ORM dependency needed.
"""
import sqlite3
import os
import json
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

DATA_DIR = os.environ.get("APP_DATA_DIR") or os.path.dirname(__file__)
DB_PATH = os.environ.get("DATABASE_PATH") or os.path.join(DATA_DIR, "studyquiz.db")
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)


# ── Connection helper ──────────────────────────────────────────────────────────
@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    con = sqlite3.connect(DB_PATH)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA foreign_keys=ON")
        yield con
    except Exception:
        con.rollback()
        raise
    else:
        con.commit()
    finally:
        con.close()


# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    username    TEXT    NOT NULL UNIQUE,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS quiz_attempts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    pdf_name     TEXT    NOT NULL DEFAULT '',
    score_pct    REAL    NOT NULL,
    correct      INTEGER NOT NULL,
    total        INTEGER NOT NULL,
    attempted_at TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS topic_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    attempt_id  INTEGER NOT NULL REFERENCES quiz_attempts(id) ON DELETE CASCADE,
    topic       TEXT    NOT NULL,
    correct     INTEGER NOT NULL,
    total       INTEGER NOT NULL,
    pct         REAL    NOT NULL
);

CREATE TABLE IF NOT EXISTS revision_tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    topic        TEXT    NOT NULL,
    due_date     TEXT    NOT NULL,
    interval_days INTEGER NOT NULL DEFAULT 1,
    ease_factor  REAL    NOT NULL DEFAULT 2.5,
    repetitions  INTEGER NOT NULL DEFAULT 0,
    status       TEXT    NOT NULL DEFAULT 'pending',
    created_at   TEXT    NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS study_goals (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL UNIQUE,
    exam_date    TEXT,
    daily_hours  REAL    NOT NULL DEFAULT 1.0,
    goal_text    TEXT    NOT NULL DEFAULT '',
    updated_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_attempts_session  ON quiz_attempts(session_id);
CREATE INDEX IF NOT EXISTS ix_topics_attempt    ON topic_scores(attempt_id);
CREATE INDEX IF NOT EXISTS ix_revision_session  ON revision_tasks(session_id);
"""


def init_db() -> None:
    """Create all tables if they do not exist."""
    with _conn() as con:
        con.executescript(SCHEMA)


# ── Quiz attempts ──────────────────────────────────────────────────────────────
def save_attempt(session_id: str, pdf_name: str, score_pct: float,
                 correct: int, total: int,
                 topic_chart: List[Dict]) -> int:
    """Persist one quiz attempt and its per-topic scores. Returns attempt id."""
    with _conn() as con:
        cur = con.execute(
            "INSERT INTO quiz_attempts (session_id, pdf_name, score_pct, correct, total) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, pdf_name, score_pct, correct, total)
        )
        attempt_id = cur.lastrowid
        for t in topic_chart:
            con.execute(
                "INSERT INTO topic_scores (attempt_id, topic, correct, total, pct) "
                "VALUES (?, ?, ?, ?, ?)",
                (attempt_id, t["topic"], t["correct"], t["total"], t["pct"])
            )
    return attempt_id


def get_attempts(session_id: str, limit: int = 20) -> List[Dict]:
    """Return recent attempts for a session, newest first."""
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM quiz_attempts WHERE session_id=? "
            "ORDER BY attempted_at DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
    return [dict(r) for r in rows]


def get_topic_history(session_id: str) -> List[Dict]:
    """
    Return per-topic aggregated stats across all attempts for a session.
    Columns: topic, attempts, avg_pct, last_pct, trend
    """
    with _conn() as con:
        rows = con.execute("""
            SELECT ts.topic,
                   COUNT(*)          AS attempts,
                   ROUND(AVG(ts.pct),1) AS avg_pct,
                   ts.pct            AS last_pct
            FROM topic_scores ts
            JOIN quiz_attempts qa ON qa.id = ts.attempt_id
            WHERE qa.session_id = ?
            GROUP BY ts.topic
            ORDER BY avg_pct ASC
        """, (session_id,)).fetchall()
    results = []
    for r in rows:
        d = dict(r)
        # Trend: compare last attempt to average
        trend = "stable"
        if d["last_pct"] > d["avg_pct"] + 5:
            trend = "improving"
        elif d["last_pct"] < d["avg_pct"] - 5:
            trend = "declining"
        d["trend"] = trend
        results.append(d)
    return results


# ── Revision tasks ─────────────────────────────────────────────────────────────
def get_revision_tasks(session_id: str) -> List[Dict]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM revision_tasks WHERE session_id=? ORDER BY due_date ASC",
            (session_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_revision_task(session_id: str, topic: str, due_date: str,
                         interval_days: int, ease_factor: float,
                         repetitions: int) -> None:
    with _conn() as con:
        existing = con.execute(
            "SELECT id FROM revision_tasks WHERE session_id=? AND topic=? AND status='pending'",
            (session_id, topic)
        ).fetchone()
        if existing:
            con.execute(
                "UPDATE revision_tasks SET due_date=?, interval_days=?, "
                "ease_factor=?, repetitions=? WHERE id=?",
                (due_date, interval_days, ease_factor, repetitions, existing["id"])
            )
        else:
            con.execute(
                "INSERT INTO revision_tasks "
                "(session_id, topic, due_date, interval_days, ease_factor, repetitions) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, topic, due_date, interval_days, ease_factor, repetitions)
            )


def complete_revision_task(task_id: int) -> None:
    with _conn() as con:
        con.execute(
            "UPDATE revision_tasks SET status='done', completed_at=datetime('now') WHERE id=?",
            (task_id,)
        )


# ── Study goals ────────────────────────────────────────────────────────────────
def save_goal(session_id: str, exam_date: Optional[str],
              daily_hours: float, goal_text: str) -> None:
    with _conn() as con:
        con.execute("""
            INSERT INTO study_goals (session_id, exam_date, daily_hours, goal_text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                exam_date=excluded.exam_date,
                daily_hours=excluded.daily_hours,
                goal_text=excluded.goal_text,
                updated_at=datetime('now')
        """, (session_id, exam_date, daily_hours, goal_text))


def get_goal(session_id: str) -> Optional[Dict]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM study_goals WHERE session_id=?", (session_id,)
        ).fetchone()
    return dict(row) if row else None


# Initialise on import
init_db()
