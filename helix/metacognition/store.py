"""Where metacognition keeps what it learns: one SQLite database per agent deployment.

- procedures: every version of every procedure, with its trust state and provenance
- episodes: every run in order, with its verdict, model, reasoning setting, cost, and
  the procedures it matched (its class), was offered, and called
- state: each procedure's online state (detector window, quarantine point, next attempt)
- learning: every learning attempt, with the gate's result and what it cost
- floors: the measured reasoning floors, by model, procedure ("*" for any) and arm
- leases: who holds a role, such as the learning worker, and until when
- meta: the schema version

A store opened without a path lives in memory, as the study's recorder did. A
store on disk uses SQLite's write-ahead log and waits up to 30 seconds for a
lock, so a serving process, a learning worker and the CLI can share it. A
store written by an earlier version is migrated when it is opened.
Trust states: candidate -> trusted -> quarantined; proposed (awaiting a
person), rejected (failed the gate), retired (replaced).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from helix.memory.learning_mechanisms import Episode

SCHEMA_VERSION = 3
ANY_PROCEDURE = "*"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS procedures (
    name TEXT NOT NULL, version INTEGER NOT NULL, source TEXT NOT NULL, state TEXT NOT NULL,
    created_by TEXT NOT NULL, parent INTEGER, created_at TEXT NOT NULL, info TEXT NOT NULL,
    PRIMARY KEY (name, version));
CREATE TABLE IF NOT EXISTS episodes (
    ord INTEGER PRIMARY KEY, run_id TEXT UNIQUE NOT NULL, case_id TEXT NOT NULL, case_data TEXT,
    request TEXT NOT NULL, content TEXT NOT NULL, verdict TEXT, verdict_source TEXT, reason TEXT,
    matched TEXT NOT NULL, offered TEXT NOT NULL, effort TEXT, dollars REAL, reasoning_tokens INTEGER,
    seconds REAL, created_at TEXT NOT NULL, model TEXT, called TEXT, scores TEXT);
CREATE TABLE IF NOT EXISTS state (name TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS learning (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, mechanism TEXT NOT NULL, version INTEGER,
    episodes INTEGER NOT NULL, writer_calls INTEGER NOT NULL, dollars REAL NOT NULL, gate TEXT, outcome TEXT NOT NULL,
    at_ord INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS floors (
    model TEXT NOT NULL, procedure TEXT NOT NULL, arm TEXT NOT NULL, setting TEXT NOT NULL, bar REAL,
    cases INTEGER, measured_at TEXT NOT NULL, PRIMARY KEY (model, procedure, arm));
CREATE TABLE IF NOT EXISTS leases (name TEXT PRIMARY KEY, holder TEXT NOT NULL, expires REAL NOT NULL);
"""

# The step from each version to the next.
_MIGRATIONS = {
    1: """
ALTER TABLE episodes ADD COLUMN model TEXT;
ALTER TABLE episodes ADD COLUMN called TEXT;
ALTER TABLE episodes ADD COLUMN scores TEXT;
""",
    2: """
CREATE TABLE IF NOT EXISTS floors (
    model TEXT NOT NULL, procedure TEXT NOT NULL, arm TEXT NOT NULL, setting TEXT NOT NULL, bar REAL,
    cases INTEGER, measured_at TEXT NOT NULL, PRIMARY KEY (model, procedure, arm));
CREATE TABLE IF NOT EXISTS leases (name TEXT PRIMARY KEY, holder TEXT NOT NULL, expires REAL NOT NULL);
""",
}

_EPISODE_COLUMNS = ("ord", "run_id", "case_id", "case_data", "request", "content", "verdict", "verdict_source",
                    "reason", "matched", "offered", "effort", "dollars", "reasoning_tokens", "seconds", "created_at",
                    "model", "called", "scores")
_JSON_COLUMNS = ("case_data", "content", "matched", "offered", "called", "scores")


_DROPPED = "{}"  # what stands in for an episode that was dropped to bound the store
_COMPACT_EVERY = 256  # runs between the passes that drop old episodes
_REPACK_EVERY = 16 * _COMPACT_EVERY  # runs between the passes that give the dropped episodes' space back


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MetaStore:
    """Runs, procedure versions, state and learning attempts, in SQLite.

    `keep` bounds what a long deployment stores: only the latest `keep` runs keep their episode and the data their
    world is rebuilt from. Older runs keep their number, verdict, reasoning setting, cost and time, so counts and
    summaries still cover every run. Without it every run is kept whole. A bounded store also repacks itself every
    few thousand runs (see `repack`), so an old run costs the few hundred bytes it still holds.
    """

    def __init__(self, path: str | Path | None = None, *, keep: int | None = None) -> None:
        self.keep = keep
        if path is None:
            self.path = None
            database = ":memory:"
        else:
            self.path = Path(path)
            self.path.mkdir(parents=True, exist_ok=True)
            database = str(self.path / "meta.sqlite")
        self._lock = threading.RLock()
        self._watch: sqlite3.Connection | None = None  # opened by stamp()
        self._conn = sqlite3.connect(database, check_same_thread=False, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        if self.path is not None:
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._migrate()

    def _migrate(self) -> None:
        conn = self._conn
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
        if row is not None:
            version = int(row[0])
        else:  # version 1 had no meta table
            exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'episodes'").fetchone()
            version = 1 if exists else 0
        if version > SCHEMA_VERSION:
            raise RuntimeError(f"this store was written by a newer metacog (schema {version}); "
                               f"this one reads up to schema {SCHEMA_VERSION}")
        if version == 0:
            conn.executescript(_SCHEMA)
        else:
            for step in range(version, SCHEMA_VERSION):
                conn.executescript(_MIGRATIONS[step])
        conn.execute("INSERT INTO meta VALUES ('schema_version', ?) ON CONFLICT(key) DO UPDATE SET value = "
                     "excluded.value", (str(SCHEMA_VERSION),))
        # Looking a case up by its id must not scan every run a long deployment has recorded.
        conn.execute("CREATE INDEX IF NOT EXISTS episodes_case ON episodes (case_id, ord)")
        conn.commit()

    @property
    def schema_version(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()[0])

    def stamp(self) -> str:
        """A value that changes whenever anyone writes to the store: this process (its own write count) or
        another one (SQLite's data_version, read on a second connection that sees every other commit)."""
        with self._lock:
            mine = self._conn.total_changes
            if self.path is None:
                return f"0:{mine}"
            if self._watch is None:
                self._watch = sqlite3.connect(str(self.path / "meta.sqlite"), check_same_thread=False, timeout=30.0)
            theirs = self._watch.execute("PRAGMA data_version").fetchone()[0]
            return f"{theirs}:{mine}"

    # ------------------------------------------------------------ settings kept with the deployment

    def put_setting(self, key: str, value: Any) -> None:
        """Keep a deployment setting, such as the calibrated router, as JSON."""
        with self._lock:
            self._conn.execute("INSERT INTO meta VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                               (f"setting:{key}", json.dumps(value)))
            self._conn.commit()

    def setting(self, key: str) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (f"setting:{key}",)).fetchone()
        return None if row is None else json.loads(row[0])

    @contextmanager
    def _immediate(self) -> Iterator[sqlite3.Connection]:
        """A write transaction that takes SQLite's write lock before it reads, so a read that numbers a new
        row (a MAX or a COUNT) and the write that uses it are one step, even across processes."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            self._conn.commit()

    # ------------------------------------------------------------ procedures

    def add_version(self, name: str, source: str, *, state: str, created_by: str, parent: int | None = None,
                    info: dict[str, Any] | None = None) -> int:
        with self._immediate() as conn:
            row = conn.execute("SELECT MAX(version) FROM procedures WHERE name = ?", (name,)).fetchone()
            version = (row[0] or 0) + 1
            conn.execute("INSERT INTO procedures VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (name, version, source, state, created_by, parent, _now(), json.dumps(info or {})))
        return version

    def set_state(self, name: str, version: int, state: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE procedures SET state = ? WHERE name = ? AND version = ?", (state, name, version))
            self._conn.commit()

    def versions(self, name: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM procedures" + (" WHERE name = ?" if name else "") + " ORDER BY name, version",
                (name,) if name else ()).fetchall()
        return [{**dict(r), "info": json.loads(r["info"])} for r in rows]

    def version(self, name: str, version: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM procedures WHERE name = ? AND version = ?",
                                     (name, version)).fetchone()
        return None if row is None else {**dict(row), "info": json.loads(row["info"])}

    def trusted(self, name: str) -> dict[str, Any] | None:
        # Asked several times for every request, so it reads one row and not every version a long stream has left.
        with self._lock:
            row = self._conn.execute("SELECT * FROM procedures WHERE name = ? AND state = 'trusted' "
                                     "ORDER BY version DESC LIMIT 1", (name,)).fetchone()
        return None if row is None else {**dict(row), "info": json.loads(row["info"])}

    # ------------------------------------------------------------ episodes

    def record(self, *, run_id: str, case_id: str, case_data: Any, request: str, episode: Episode,
               verdict: Any, matched: list[str], offered: list[str], effort: str | None, dollars: float,
               reasoning_tokens: int, seconds: float, answer: Any = None, model: str | None = None,
               called: list[str] | None = None, scores: dict[str, float] | None = None) -> int:
        # The run's own answer rides beside the book's episode content, for replayed practice.
        content = {**episode.to_content(), **({"answer": answer} if answer is not None else {})}
        with self._immediate() as conn:
            ord_ = int(conn.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM episodes").fetchone()[0])
            values = (ord_, run_id, case_id, json.dumps(case_data, default=str), request, json.dumps(content,
                      default=str), None if verdict is None else ("accepted" if verdict.accepted else "rejected"),
                      None if verdict is None else verdict.source, None if verdict is None else verdict.reason,
                      json.dumps(matched), json.dumps(offered), effort, dollars, reasoning_tokens, seconds, _now(),
                      model, json.dumps(called if called is not None else []),
                      json.dumps({k: round(v, 6) for k, v in (scores or {}).items()}))
            conn.execute(f"INSERT INTO episodes ({', '.join(_EPISODE_COLUMNS)}) VALUES "
                         f"({', '.join('?' for _ in _EPISODE_COLUMNS)})", values)
            if self.keep is not None and ord_ % _COMPACT_EVERY == 0:
                conn.execute(f"UPDATE episodes SET content = '{_DROPPED}', case_data = NULL WHERE ord < ? AND "
                             f"content != '{_DROPPED}'", (ord_ - self.keep,))
        if self.keep is not None and ord_ and ord_ % _REPACK_EVERY == 0:
            self.repack()
        return ord_

    def repack(self) -> bool:
        """Rewrite the database so the space of dropped episodes is given back. True if it ran.

        SQLite frees the pages a dropped episode overflowed into and leaves the row's own page as it was, so an
        old run goes on holding a few kilobytes it no longer uses. A repack needs the database to itself for a
        moment. When another process is using it, this pass is skipped and the next one tries again.
        """
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=200")  # serving must not wait long for a tidy-up
            try:
                self._conn.execute("VACUUM")
            except sqlite3.OperationalError:
                return False
            finally:
                self._conn.execute("PRAGMA busy_timeout=30000")
        return True

    def set_verdict(self, run_id: str, verdict: Any) -> dict[str, Any] | None:
        with self._lock:
            self._conn.execute("UPDATE episodes SET verdict = ?, verdict_source = ?, reason = ? WHERE run_id = ?",
                               ("accepted" if verdict.accepted else "rejected", verdict.source, verdict.reason,
                                run_id))
            self._conn.commit()
            row = self._conn.execute("SELECT * FROM episodes WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else self._row(row)

    def size(self) -> int:
        """Bytes of the database in use now: what a long deployment has to keep."""
        with self._lock:
            pages, free, page = (self._conn.execute(f"PRAGMA {name}").fetchone()[0]
                                 for name in ("page_count", "freelist_count", "page_size"))
        return int((pages - free) * page)

    def count(self) -> int:
        """How many runs have been recorded: the number the next run gets."""
        with self._lock:
            return int(self._conn.execute("SELECT COALESCE(MAX(ord), -1) + 1 FROM episodes").fetchone()[0])

    def judged(self, since: int = 0) -> list[tuple[str | None, float, float]]:
        """Each judged run at or after `since`, oldest first, as (reasoning setting, dollars, seconds): what a
        signal sums, without reading any episode."""
        with self._lock:
            rows = self._conn.execute("SELECT effort, dollars, seconds FROM episodes WHERE ord >= ? AND verdict "
                                      "IS NOT NULL ORDER BY ord", (since,)).fetchall()
        return [(row[0], row[1] or 0.0, row[2] or 0.0) for row in rows]

    def rejected_cases(self, since: int, effort: str) -> list[str]:
        """The cases of runs at `effort` that were rejected at or after `since`, oldest first, while the data to
        rebuild them is still kept."""
        with self._lock:
            rows = self._conn.execute("SELECT case_id FROM episodes WHERE ord >= ? AND effort = ? AND verdict = "
                                      "'rejected' AND case_data IS NOT NULL ORDER BY ord", (since, effort)).fetchall()
        return [row[0] for row in rows]

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in _JSON_COLUMNS:
            data[key] = json.loads(data[key]) if data.get(key) is not None else None
        return data

    def runs(self, since: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM episodes WHERE ord >= ? ORDER BY ord", (since,)).fetchall()
        return [self._row(r) for r in rows]

    def recent_runs(self, limit: int = 50, *, before: int | None = None, unjudged: bool = False
                    ) -> list[dict[str, Any]]:
        """The latest runs, newest first: before run number `before` if given, and only those still
        awaiting a verdict with `unjudged`."""
        query, params = "SELECT * FROM episodes WHERE 1 = 1", []
        if before is not None:
            query, params = query + " AND ord < ?", [*params, before]
        if unjudged:
            query += " AND verdict IS NULL"
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY ord DESC LIMIT ?", [*params, limit]).fetchall()
        return [self._row(r) for r in rows]

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM episodes WHERE run_id = ?", (run_id,)).fetchone()
        return None if row is None else self._row(row)

    def episodes(self, *, since: int = 0, procedure: str | None = None, last: int | None = None) -> list[Episode]:
        """Judged runs at or after `since`, oldest first, as a learner reads them; filtered to the runs that
        matched `procedure` when given, and only the latest `last` of them when given. A run whose episode was
        dropped to bound the store is left out."""
        if last is not None:
            # Read from the newest back and stop at `last`, so the cost does not grow with what came before.
            matched = () if procedure is None else (f'%"{procedure}"%',)
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT * FROM episodes WHERE ord >= ? AND verdict IS NOT NULL AND content != '{_DROPPED}'"
                    f"{' AND matched LIKE ?' if matched else ''} ORDER BY ord DESC LIMIT ?",
                    (since, *matched, last)).fetchall()
            runs = [self._row(r) for r in reversed(rows)]
        else:
            runs = self.runs(since)
        out = []
        for run in runs:
            if run["verdict"] is None or (procedure is not None and procedure not in run["matched"]):
                continue
            if not run["content"]:
                continue
            content = {**run["content"], "success": run["verdict"] == "accepted", "reason": run["reason"] or ""}
            out.append(Episode.from_content(content))
        return out

    def case(self, case_id: str) -> dict[str, Any] | None:
        """The latest run of a case: its request and the data a world factory rebuilds it from."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM episodes WHERE case_id = ? ORDER BY ord DESC LIMIT 1",
                                     (case_id,)).fetchone()
        return None if row is None else self._row(row)

    def accepted_run(self, case_id: str) -> dict[str, Any] | None:
        """The latest accepted run of a case: the world and outcome a replay checks a candidate against."""
        with self._lock:
            row = self._conn.execute("SELECT * FROM episodes WHERE case_id = ? AND verdict = 'accepted'"
                                     " ORDER BY ord DESC LIMIT 1", (case_id,)).fetchone()
        return None if row is None else self._row(row)

    # ------------------------------------------------------------ state and learning

    def state(self, name: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("SELECT value FROM state WHERE name = ?", (name,)).fetchone()
        return json.loads(row[0]) if row else {}

    def put_state(self, name: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO state VALUES (?, ?) ON CONFLICT(name) DO UPDATE SET value = excluded.value",
                               (name, json.dumps(value)))
            self._conn.commit()

    def log_learning(self, *, name: str, mechanism: str, version: int | None, episodes: int, writer_calls: int,
                     dollars: float, gate: dict[str, Any] | None, outcome: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO learning (name, mechanism, version, episodes, writer_calls, dollars, gate, outcome,"
                " at_ord, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, mechanism, version, episodes, writer_calls, dollars, json.dumps(gate) if gate else None,
                 outcome, self.count(), _now()))
            self._conn.commit()

    def learning(self, name: str | None = None, *, within: timedelta | None = None,
                 since_run: int | None = None) -> list[dict[str, Any]]:
        """Learning attempts, oldest first: all, or one procedure's, or those started within a recent span of
        time, or at or after run number `since_run`."""
        query, params = "SELECT * FROM learning WHERE 1 = 1", []
        if name is not None:
            query, params = query + " AND name = ?", [*params, name]
        if since_run is not None:
            query, params = query + " AND at_ord >= ?", [*params, since_run]
        if within is not None:
            query, params = query + " AND created_at >= ?", [*params, (datetime.now(UTC) - within).isoformat()]
        with self._lock:
            rows = self._conn.execute(query + " ORDER BY id", params).fetchall()
        return [{**dict(r), "gate": json.loads(r["gate"]) if r["gate"] else None} for r in rows]

    # ------------------------------------------------------------ floors

    def put_floors(self, model: str, table: dict[str, str], *, procedure: str = ANY_PROCEDURE,
                   bar: float | None = None, cases: int | None = None) -> None:
        """Record measured floors ({"memory": ..., "no_memory": ...}) for a model and procedure."""
        from helix.metacognition.ladders import bare

        with self._lock:
            for arm, setting in table.items():
                self._conn.execute(
                    "INSERT INTO floors VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(model, procedure, arm) DO UPDATE"
                    " SET setting = excluded.setting, bar = excluded.bar, cases = excluded.cases,"
                    " measured_at = excluded.measured_at", (bare(model), procedure, arm, setting, bar, cases, _now()))
            self._conn.commit()

    def floors(self) -> dict[str, dict[str, Any]]:
        """Measured floors by model, as `Controller.per_model` reads them."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM floors ORDER BY model, procedure, arm").fetchall()
        out: dict[str, dict[str, Any]] = {}
        for row in rows:
            entry = out.setdefault(row["model"], {})
            if row["procedure"] == ANY_PROCEDURE:
                entry.setdefault("table", {})[row["arm"]] = row["setting"]
            else:
                entry.setdefault("per_procedure", {}).setdefault(row["procedure"], {})[row["arm"]] = row["setting"]
        return out

    # ------------------------------------------------------------ leases

    def acquire_lease(self, name: str, holder: str, seconds: float, *, now: float | None = None) -> bool:
        """Take or renew the lease `name` for `holder` for `seconds`, unless another holder's lease is live.

        The check and the write happen in one immediate transaction, so two processes can't both win.
        """
        now = time.time() if now is None else now
        with self._immediate() as conn:
            row = conn.execute("SELECT holder, expires FROM leases WHERE name = ?", (name,)).fetchone()
            if row is not None and row["holder"] != holder and row["expires"] > now:
                return False
            conn.execute("INSERT INTO leases VALUES (?, ?, ?) ON CONFLICT(name) DO UPDATE SET"
                         " holder = excluded.holder, expires = excluded.expires", (name, holder, now + seconds))
            return True

    def release_lease(self, name: str, holder: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE name = ? AND holder = ?", (name, holder))
            self._conn.commit()

    def lease(self, name: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM leases WHERE name = ?", (name,)).fetchone()
        return None if row is None else dict(row)

    def close(self) -> None:
        if self._watch is not None:
            self._watch.close()
        self._conn.close()
