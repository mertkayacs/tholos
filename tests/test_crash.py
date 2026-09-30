import subprocess
import sys
import time
from pathlib import Path

from tholos import workspace as w


def test_kill_and_restart_exactly_once(db):
    path = db.execute("PRAGMA database_list").fetchone()[2]
    mid = w.save_model(db, None, "Local", "http://local.test/v1", "small", None, "schema", 0, 512)
    aid = w.save_agent(db, None, "Recorder", "You record items.", mid, ["table_add", "finish"])
    w.create_table(db, "items", ["title"], "you")
    rid = w.queue_run(db, aid, "Record an item", "message")
    script = str(Path(__file__).with_name("crash_worker.py"))
    process = subprocess.Popen([sys.executable, script, path, "first"])
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if w.get_run(db, rid)["step_count"] == 1:
                break
            assert process.poll() is None
            time.sleep(0.02)
        assert w.get_run(db, rid)["step_count"] == 1
        assert w.get_run(db, rid)["status"] == "running"
        process.kill()
        assert process.wait(timeout=5) == -9
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
    subprocess.run([sys.executable, script, path, "restart"], check=True, timeout=15)
    run = w.get_run(db, rid)
    assert run["status"] == "done" and run["attempts"] == 1 and run["step_count"] == 2
    assert [step["tool"] for step in run["steps"]] == ["table_add", "finish"]
    assert len(w.get_table(db, "items")["rows"]) == 1
    assert len(w.recent_changes(db, "row")) == 1
    assert len(run["messages"]) == 6
