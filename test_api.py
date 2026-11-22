"""API tests. They need `fastapi` and `httpx` (both in requirements.txt)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402

client = TestClient(app)


def test_health_and_meta():
    assert client.get("/api/health").json()["status"] == "ok"
    meta = client.get("/api/meta").json()
    assert len(meta["policies"]) == 4 and len(meta["actions"]) == 4


def test_dashboard_compare_transactions():
    d = client.get("/api/dashboard").json()
    assert d["total_transactions"] == 800
    assert len(client.get("/api/compare?split=dev").json()["policies"]) == 4
    assert client.get("/api/transactions?risk=High&limit=5").json()["returned"] == 5
    assert client.get("/api/dashboard?split=bogus").status_code == 400
    assert "log" in client.get("/api/training-log").json()


def test_episode_flow_report_and_export():
    assert client.post("/api/episode/start", json={"split": "dev", "episode_index": 0, "policy": "rl_agent"}).status_code == 200
    assert client.post("/api/episode/step", json={}).status_code == 200
    run = client.post("/api/episode/run").json()
    assert run["state"]["done"]
    assert client.post("/api/episode/step", json={}).status_code == 409           # episode finished
    assert client.get("/api/episode/current").json()["status"] == "completed"
    assert client.get("/api/report?source=session").json()["summary"]["transactions"] == 40
    csv = client.get("/api/report/export?format=csv&source=session")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    assert client.get("/api/report/export?format=json&source=session").status_code == 200


def test_action_validation_and_manual_play():
    client.post("/api/episode/start", json={"split": "dev", "episode_index": 1, "policy": "manual"})
    assert client.post("/api/episode/step", json={"action": "dance"}).status_code == 400
    assert client.post("/api/episode/step", json={"action": 9}).status_code == 400
    assert client.post("/api/episode/step", json={"action": 1.5}).status_code == 400
    assert client.post("/api/episode/step", json={"action": "investigate"}).status_code == 200
    assert client.post("/api/episode/start", json={"policy": "nope"}).status_code == 400
