import asyncio
import json
import sys
import time
from datetime import UTC, datetime, timedelta

import httpx

from tholos import worker
from tholos import workspace as w
from tholos.db import connect, init


async def main():
    db = connect(sys.argv[1])
    init(db)
    if sys.argv[2] == "restart":
        # Advance recovery past the lease instead of waiting for it to expire.
        row = db.execute(
            "SELECT MAX(lease_until) FROM runs WHERE status='running'"
        ).fetchone()
        until = row[0] or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        past = (datetime.fromisoformat(until) + timedelta(seconds=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        worker.recover(db, past)

    def handler(request):
        body = json.loads(request.content)
        count = sum(message["role"] == "assistant" for message in body["messages"])
        if count == 0:
            value = {
                "thought": "Record it.",
                "tool": "table_add",
                "args": {"table": "items", "rows": [{"title": "Recorded once"}]},
            }
        else:
            if sys.argv[2] == "first":
                time.sleep(60)
            value = {"thought": "Done.", "tool": "finish", "args": {"summary": "Recorded"}}
        return httpx.Response(200, json={"choices": [{"message": {"content": w.dumps(value)}}]})

    service = worker.Worker(db, httpx.MockTransport(handler))
    service.start()
    try:
        for _ in range(1000):
            if w.list_runs(db, status="done"):
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("Worker did not finish")
    finally:
        await service.stop()
        db.close()


if __name__ == "__main__":
    asyncio.run(main())
