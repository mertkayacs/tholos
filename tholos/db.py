import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE settings(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE models(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, base_url TEXT NOT NULL,
 model TEXT NOT NULL, api_key TEXT, json_mode TEXT NOT NULL DEFAULT 'schema',
 temperature REAL NOT NULL DEFAULT 0.2, max_tokens INTEGER NOT NULL DEFAULT 512,
 timeout REAL NOT NULL DEFAULT 120);
CREATE TABLE agents(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, role TEXT NOT NULL,
 model_id INTEGER REFERENCES models ON DELETE SET NULL, tools TEXT NOT NULL,
 max_steps INTEGER NOT NULL DEFAULT 12, paused INTEGER NOT NULL DEFAULT 0,
 created_at TEXT NOT NULL);
CREATE TABLE schedules(id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL REFERENCES agents
 ON DELETE CASCADE, every TEXT NOT NULL, prompt TEXT NOT NULL, next_at TEXT NOT NULL,
 enabled INTEGER NOT NULL DEFAULT 1);
CREATE TABLE tasks(id INTEGER PRIMARY KEY, title TEXT NOT NULL, details TEXT NOT NULL,
 agent_id INTEGER REFERENCES agents ON DELETE SET NULL, created_by TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'todo', result TEXT, parent_id INTEGER REFERENCES tasks
 ON DELETE SET NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE runs(id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL REFERENCES agents
 ON DELETE CASCADE, task_id INTEGER REFERENCES tasks ON DELETE SET NULL, trigger TEXT NOT NULL,
 trigger_kind TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued', due_at TEXT NOT NULL,
 lease_until TEXT, fence INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
 messages TEXT NOT NULL DEFAULT '[]', steps INTEGER NOT NULL DEFAULT 0,
 tokens_in INTEGER NOT NULL DEFAULT 0, tokens_out INTEGER NOT NULL DEFAULT 0, error TEXT,
 dedupe_key TEXT UNIQUE, created_at TEXT NOT NULL, started_at TEXT, ended_at TEXT);
CREATE TABLE steps(id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs
 ON DELETE CASCADE, n INTEGER NOT NULL, thought TEXT NOT NULL, tool TEXT NOT NULL,
 args TEXT NOT NULL, result TEXT NOT NULL, status TEXT NOT NULL, ms INTEGER NOT NULL,
 created_at TEXT NOT NULL, UNIQUE(run_id, n));
CREATE TABLE approvals(id INTEGER PRIMARY KEY, run_id INTEGER NOT NULL REFERENCES runs
 ON DELETE CASCADE, step_id INTEGER NOT NULL REFERENCES steps ON DELETE CASCADE,
 agent_id INTEGER NOT NULL REFERENCES agents ON DELETE CASCADE, kind TEXT NOT NULL,
 tool TEXT NOT NULL, args TEXT NOT NULL, args_hash TEXT NOT NULL, preview TEXT NOT NULL,
 status TEXT NOT NULL DEFAULT 'pending', answer TEXT, created_at TEXT NOT NULL, decided_at TEXT);
CREATE TABLE rules(id INTEGER PRIMARY KEY, agent TEXT NOT NULL DEFAULT '*', tool TEXT NOT NULL,
 match TEXT NOT NULL DEFAULT '*', decision TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE sheets(id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, columns TEXT NOT NULL,
 version INTEGER NOT NULL DEFAULT 1, created_by TEXT NOT NULL, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL);
CREATE TABLE rows(id INTEGER PRIMARY KEY, sheet_id INTEGER NOT NULL REFERENCES sheets
 ON DELETE CASCADE, data TEXT NOT NULL, version INTEGER NOT NULL DEFAULT 1,
 updated_by TEXT NOT NULL, updated_at TEXT NOT NULL, deleted INTEGER NOT NULL DEFAULT 0);
CREATE TABLE notes(id INTEGER PRIMARY KEY, title TEXT UNIQUE NOT NULL, body TEXT NOT NULL,
 version INTEGER NOT NULL DEFAULT 1, updated_by TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE history(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, ref_id INTEGER NOT NULL,
 version INTEGER NOT NULL, before TEXT, after TEXT, actor TEXT NOT NULL,
 run_id INTEGER REFERENCES runs ON DELETE SET NULL, at TEXT NOT NULL);
CREATE TABLE memories(id INTEGER PRIMARY KEY, agent_id INTEGER NOT NULL REFERENCES agents
 ON DELETE CASCADE, text TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE events(seq INTEGER PRIMARY KEY, at TEXT NOT NULL, actor TEXT NOT NULL,
 kind TEXT NOT NULL, ref TEXT NOT NULL, text TEXT NOT NULL);
CREATE VIRTUAL TABLE search USING fts5(kind UNINDEXED, ref UNINDEXED, title, body);
CREATE INDEX run_queue ON runs(status, due_at);
CREATE INDEX agent_runs ON runs(agent_id, status);
CREATE INDEX row_sheet ON rows(sheet_id, deleted);
CREATE INDEX history_ref ON history(kind, ref_id);
"""


def connect(path: str | None = None) -> sqlite3.Connection:
    if path is None:
        home = Path(os.environ.get("THOLOS_HOME", "~/.tholos")).expanduser()
        home.mkdir(mode=0o700, parents=True, exist_ok=True)
        (home / "exports").mkdir(mode=0o700, exist_ok=True)
        path = str(home / "tholos.db")
    if path != ":memory:":
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(fd, 0o600)
        os.close(fd)
    db = sqlite3.connect(path, isolation_level=None)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=5000")
    return db


def init(db: sqlite3.Connection) -> None:
    with tx(db):
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version > 1:
            raise ValueError("Database schema is newer than this application")
        if version == 0:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    db.execute(statement)
            db.execute("PRAGMA user_version=1")


@contextmanager
def tx(db: sqlite3.Connection) -> Iterator[None]:
    nested = db.in_transaction
    db.execute("SAVEPOINT nested" if nested else "BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        db.execute("ROLLBACK TO nested" if nested else "ROLLBACK")
        if nested:
            db.execute("RELEASE nested")
        raise
    else:
        db.execute("RELEASE nested" if nested else "COMMIT")


class Clock:
    """The runtime's only source of time.

    The benchmark pins it so runs repeat exactly. A pinned clock starts at the given
    aware datetime and moves one second per read, so timestamps stay strictly ordered.
    It is process-wide: pin it from one thread at a time.
    """

    def __init__(self) -> None:
        self._pinned: datetime | None = None

    def pin(self, start: datetime) -> None:
        self._pinned = start.astimezone(UTC)

    def unpin(self) -> None:
        self._pinned = None

    def now(self) -> datetime:
        if self._pinned is None:
            return datetime.now(UTC)
        at = self._pinned
        self._pinned = at + timedelta(seconds=1)
        return at


clock = Clock()


def now() -> str:
    return clock.now().strftime("%Y-%m-%dT%H:%M:%SZ")
