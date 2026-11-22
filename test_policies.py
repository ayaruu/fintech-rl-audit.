import numpy as np
import pytest

from environment.config import APPROVE, FLAG, INVESTIGATE, OBS_INDEX, N_OBS, REQUEST_DOCS
from environment.policies import POLICY_KEYS, RandomPolicy, RiskThresholdPolicy, RuleBasedPolicy, make_policy
from environment.rl_agent import QLearningAgent
from environment.runner import evaluate_policy, run_episode


@pytest.fixture(scope="module")
def agent():
    from pathlib import Path
    return QLearningAgent.load(Path(__file__).resolve().parent.parent / "data" / "q_table.json")


def obs_with(**kw):
    o = np.zeros(N_OBS, dtype=np.float32)
    for k, v in kw.items():
        o[OBS_INDEX[k]] = v
    return o


def test_random_policy_is_reproducible():
    a, b = RandomPolicy(5), RandomPolicy(5)
    o = obs_with()
    assert [a.act(o) for _ in range(30)] == [b.act(o) for _ in range(30)]
    assert all(0 <= a.act(o) <= 3 for _ in range(30))


def test_rule_based_rules():
    p = RuleBasedPolicy()
    assert p.act(obs_with()) == APPROVE
    assert p.act(obs_with(doc_status=1 / 3)) == REQUEST_DOCS
    assert p.act(obs_with(doc_status=1.0)) == FLAG
    assert p.act(obs_with(amount_ratio=0.5)) == FLAG
    assert p.act(obs_with(near_threshold=1.0)) == FLAG


def test_risk_threshold_policy():
    p = RiskThresholdPolicy()
    assert p.act(obs_with(risk_score=0.1)) == APPROVE
    assert p.act(obs_with(risk_score=0.7)) == FLAG
    assert p.act(obs_with(risk_score=0.3)) == INVESTIGATE
    assert p.act(obs_with(risk_score=0.3, investigation_result=1.0)) == FLAG
    assert p.act(obs_with(risk_score=0.3, investigation_result=0.5)) == APPROVE


def test_make_policy_errors():
    with pytest.raises(ValueError):
        make_policy("nope")
    with pytest.raises(ValueError):
        make_policy("rl_agent")      # needs a trained agent


@pytest.mark.parametrize("key", POLICY_KEYS)
def test_every_policy_completes_an_episode_deterministically(key, episodes, agent):
    r1 = run_episode(episodes["dev"], 0, make_policy(key, agent=agent))
    r2 = run_episode(episodes["dev"], 0, make_policy(key, agent=agent))
    assert r1["summary"] == r2["summary"]
    assert r1["steps"][-1]["done"] and r1["summary"]["undecided"] == 0


def test_comparison_has_all_metrics(episodes, agent):
    ev = evaluate_policy(episodes["dev"], "rule_based")
    for k in ("total_reward", "accuracy", "precision", "recall", "investigations"):
        assert k in ev["aggregate"]
    assert ev["aggregate"]["episodes"] == 10


def test_rl_agent_beats_random_on_unseen_episodes(episodes, agent):
    rl = evaluate_policy(episodes["dev"], "rl_agent", agent)["aggregate"]
    rnd = evaluate_policy(episodes["dev"], "random")["aggregate"]
    assert rl["total_reward"] > rnd["total_reward"] and rl["accuracy"] > rnd["accuracy"]
