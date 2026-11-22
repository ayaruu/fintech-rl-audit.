import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_dashboard_matches_comparison(service):
    d, c = service.dashboard("dev", "rl_agent"), service.compare("dev")
    rl = next(p for p in c["policies"] if p["policy"] == "rl_agent")
    assert d["total_reward"] == rl["total_reward"] and d["accuracy"] == rl["accuracy"]
    assert d["total_transactions"] == 800


def test_metrics_match_trajectories_jsonl(service):
    """Numbers on the website must equal the numbers recorded in trajectories.jsonl."""
    totals = {}
    for line in (ROOT / "trajectories.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["type"] == "episode_result" and row["split"] == "dev":
            totals[row["policy"]] = totals.get(row["policy"], 0) + row["result"]["total_reward"]
    for p in service.compare("dev")["policies"]:
        assert totals[p["policy"]] == pytest.approx(p["total_reward"], abs=0.01)


def test_trajectory_lines_have_required_fields():
    first = json.loads((ROOT / "trajectories.jsonl").read_text().splitlines()[0])
    for k in ("policy", "episode_id", "observation", "action_name", "reward", "total_reward"):
        assert k in first


def test_transaction_filters(service):
    all_ = service.transactions()
    assert all_["total"] == 800
    high = service.transactions(risk="High")
    assert 0 < high["total"] < 800 and all(t["risk_level"] == "High" for t in high["transactions"])
    assert all(t["split"] == "dev" for t in service.transactions(split="dev")["transactions"])
    assert service.transactions(search="TX-D00-001")["total"] == 1
    with pytest.raises(ValueError):
        service.transactions(split="bogus")


def test_session_matches_batch_evaluation(service):
    service.start_episode("dev", 4, "risk_threshold")
    state = service.run_session()["state"]
    batch = service.compare("dev")["policies"][2]["per_episode_rewards"][4]
    assert state["done"] and state["total_reward"] == batch


def test_session_errors(service):
    service.start_episode("dev", 0, "manual")
    with pytest.raises(ValueError):
        service.step_session("dance")
    with pytest.raises(ValueError):
        service.step_session()                 # manual needs an action
    with pytest.raises(ValueError):
        service.run_session()
    service.start_episode("dev", 0, "rule_based")
    with pytest.raises(ValueError):
        service.step_session("flag")           # automatic policy: no manual override
    with pytest.raises(ValueError):
        service.start_episode("dev", 50, "rl_agent")


def test_report_and_exports(service):
    service.start_episode("dev", 0, "rl_agent"); service.run_session()
    rep = service.report(source="session")
    for k in ("anomalies", "incorrect_transactions", "missing_documents", "risk_assessment", "summary"):
        assert k in rep
    m = rep["summary"]["metrics"]
    assert rep["summary"]["flagged"] == m["true_positive"] + m["false_positive"]
    content, mt, fn = service.export_report("csv", source="session")
    assert mt == "text/csv" and fn.endswith(".csv") and content.count("\n") == 41
    assert json.loads(service.export_report("json", source="session")[0])["meta"]["episode_id"] == "dev-00"
    with pytest.raises(ValueError):
        service.export_report("xml")
