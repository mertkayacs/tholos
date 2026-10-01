import asyncio
import json
import logging
from contextlib import suppress
from datetime import UTC, datetime, timedelta

import httpx

from tholos import prompt, runner
from tholos import workspace as w
from tholos.db import connect, init, now, tx

log = logging.getLogger("tholos.worker")


def recover(db: w.DB, at: str | None = None) -> None:
    at = at or now()
    with tx(db):
        expired = w._many(
            db,
            "SELECT * FROM runs WHERE status='running' AND (lease_until IS NULL OR lease_until<=?)",
            (at,),
        )
        for run in expired:
            failed = run["attempts"] + 1 > 3
            db.execute(
                "UPDATE runs SET status=?,attempts=attempts+1,fence=fence+1,lease_until=NULL,"
                "error=?,ended_at=? WHERE id=?",
                (
                    "failed" if failed else "queued",
                    "recovery limit" if failed else None,
                    at if failed else None,
                    run["id"],
                ),
            )
            w.add_event(
                db,
                "worker",
                "run",
                str(run["id"]),
                "Recovery failed" if failed else "Run recovered",
            )


def enqueue(db: w.DB, at: str | None = None) -> None:
    at = at or now()
    with tx(db):
        for schedule in w._many(
            db, "SELECT * FROM schedules WHERE enabled=1 AND next_at<=? ORDER BY id", (at,)
        ):
            due = schedule["next_at"]
            w.queue_run(
                db,
                schedule["agent_id"],
                prompt.schedule_trigger(schedule["prompt"]),
                "schedule",
                due_at=due,
                dedupe_key=f"sched:{schedule['id']}:{due[:16]}",
            )
            local = datetime.fromisoformat(at).astimezone(w.timezone(db))
            next_at = (
                w.parse_every(schedule["every"])(local)
                .astimezone(UTC)
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            )
            db.execute("UPDATE schedules SET next_at=? WHERE id=?", (next_at, schedule["id"]))


def claim(db: w.DB, at: str | None = None, run_id: int | None = None) -> dict | None:
    at = at or now()
    with tx(db):
        candidate = db.execute(
            "SELECT r.id,coalesce(m.timeout,120) AS timeout FROM runs r "
            "JOIN agents a ON a.id=r.agent_id LEFT JOIN models m ON m.id=a.model_id "
            "WHERE r.status='queued' AND r.due_at<=? AND a.paused=0 "
            "AND (? IS NULL OR r.id=?) ORDER BY r.due_at,r.id LIMIT 1",
            (at, run_id, run_id),
        ).fetchone()
        if candidate is None:
            return None
        lease = (datetime.fromisoformat(at) + timedelta(
            seconds=runner.lease_seconds(candidate["timeout"])
        )).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = db.execute(
            "UPDATE runs SET status='running',lease_until=?,fence=fence+1,"
            "started_at=coalesce(started_at,?) WHERE id=? RETURNING *",
            (lease, at, candidate["id"]),
        ).fetchone()
        if row:
            run = dict(row)
            run["messages"] = json.loads(run["messages"])
            w.add_event(db, "worker", "run", str(run["id"]), "Run started")
            return run
    return None


class Worker:
    def __init__(
        self, db: w.DB | None = None, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.db = db
        self.transport = transport
        self._owned = db is None
        self._task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._jobs: set[asyncio.Task] = set()

    def start(self) -> None:
        if self._task and not self._task.done():
            return
        if self.db is None:
            self.db = connect()
            init(self.db)
        recover(self.db)
        self._task = asyncio.create_task(self._loop())

    def wake(self) -> None:
        self._wake.set()

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None
        if self._owned and self.db is not None:
            self.db.close()
            self.db = None

    async def _job(self, run: dict) -> None:
        try:
            await runner.run_one(self.db, run, self.transport)
        finally:
            self.wake()

    async def _loop(self) -> None:
        try:
            while True:
                try:
                    self._wake.clear()
                    recover(self.db)
                    enqueue(self.db)
                    self._jobs = {job for job in self._jobs if not job.done()}
                    concurrency = max(1, int(w.get_setting(self.db, "workers", 1)))
                    while len(self._jobs) < concurrency:
                        run = claim(self.db)
                        if run is None:
                            break
                        self._jobs.add(asyncio.create_task(self._job(run)))
                    with suppress(TimeoutError):
                        await asyncio.wait_for(self._wake.wait(), timeout=2)
                except Exception:
                    # A supervisor loop must survive transient errors such as a locked database.
                    log.exception("worker loop failed")
                    await asyncio.sleep(5)
        finally:
            for job in self._jobs:
                job.cancel()
            await asyncio.gather(*self._jobs, return_exceptions=True)
            self._jobs.clear()
