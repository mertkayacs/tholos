from starlette.testclient import TestClient

from tholos import web


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
