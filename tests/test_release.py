import re
import tomllib
from pathlib import Path

from starlette.testclient import TestClient

from tholos import web

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DEPS = {"starlette", "uvicorn", "httpx", "jinja2", "openpyxl", "python-multipart"}


def test_health_is_public_and_contains_no_workspace_data(tmp_path, monkeypatch):
    monkeypatch.setenv("THOLOS_HOME", str(tmp_path))
    monkeypatch.setenv("THOLOS_HOST", "0.0.0.0")
    with TestClient(web.app) as client:
        response = client.get("/healthz", follow_redirects=False)
        assert response.status_code == 200 and response.text == "ok"
        assert response.headers["content-type"].startswith("text/plain")
        assert "set-cookie" not in response.headers
        assert client.head("/healthz").status_code == 200
        assert client.get("/", follow_redirects=False).status_code == 303


def test_ci_uses_least_privilege_permissions():
    text = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "permissions:" in text and "contents: read" in text


def test_workflows_pin_actions_to_commit_shas():
    workflows = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert workflows
    for path in workflows:
        for line in path.read_text().splitlines():
            if "uses:" not in line:
                continue
            match = re.search(r"uses:\s*(\S+)@(\S+)\s+#\s*(\S+)", line)
            assert match, f"{path.name}: {line.strip()!r} is not SHA-pinned with a tag comment"
            _, sha, tag = match.groups()
            assert re.fullmatch(r"[0-9a-f]{40}", sha), f"{path.name}: {line.strip()!r}"
            assert re.fullmatch(r"v\d+(\.\d+)*", tag), f"{path.name}: {line.strip()!r}"


def test_runtime_dependencies_carry_lockfile_floors():
    with open(ROOT / "pyproject.toml", "rb") as file:
        deps = tomllib.load(file)["project"]["dependencies"]
    locked = {}
    for chunk in (ROOT / "uv.lock").read_text().split("[[package]]")[1:]:
        name = re.search(r'name = "([^"]+)"', chunk).group(1)
        locked[name] = re.search(r'version = "([^"]+)"', chunk).group(1)
    assert {re.split(r"[><=]", dep, maxsplit=1)[0] for dep in deps} == RUNTIME_DEPS
    for dep in deps:
        match = re.fullmatch(r"([a-z0-9-]+)>=(\d+\.\d+)", dep)
        assert match, f"{dep} needs a major.minor floor"
        name, floor = match.groups()
        assert floor == ".".join(locked[name].split(".")[:2]), (dep, locked[name])
