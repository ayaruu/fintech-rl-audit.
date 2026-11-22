import numpy as np
import pytest

from environment.audit_env import AuditEnv
from environment.config import APPROVE, FLAG, INVESTIGATE, MAX_STEPS_PER_TX, N_ACTIONS, N_OBS, REQUEST_DOCS, REWARDS


def test_initialization(episodes):
    env = AuditEnv(episodes["train"])
    assert env.action_space.n == N_ACTIONS == 4
    assert env.observation_space.shape == (N_OBS,)
    with pytest.raises(ValueError):
        AuditEnv([])


def test_step_before_reset_fails(episodes):
    with pytest.raises(RuntimeError):
        AuditEnv(episodes["train"]).step(0)


def test_reset_returns_valid_observation(episodes):
    env = AuditEnv(episodes["dev"])
    obs, info = env.reset(seed=1, options={"episode_index": 2})
    assert env.observation_space.contains(obs) and obs.dtype == np.float32
    assert info["episode_id"] == "dev-02"
    with pytest.raises(IndexError):
        env.reset(options={"episode_index": 99})


def test_step_returns_gymnasium_tuple(episodes):
    env = AuditEnv(episodes["dev"]); env.reset(seed=0)
    out = env.step(APPROVE)
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float) and terminated is False and truncated is False
    assert info["decision_made"] is True


@pytest.mark.parametrize("bad", [-1, 4, 99, 1.5, "flag", None, True])
def test_invalid_actions_rejected(tiny_env, bad):
    with pytest.raises(ValueError):
        tiny_env.step(bad)


def test_numpy_int_action_is_valid(tiny_env):
    tiny_env.step(np.int64(APPROVE))


def test_rewards_decisions(tiny_episodes):
    def play(actions):
        env = AuditEnv(tiny_episodes); env.reset(seed=0)
        return [env.step(a)[1] for a in actions]
    assert play([APPROVE, FLAG]) == [REWARDS["true_negative"], REWARDS["true_positive"]]
    assert play([FLAG, APPROVE]) == [REWARDS["false_positive"], REWARDS["false_negative"]]


def test_rewards_information_actions(tiny_env):
    assert tiny_env.step(INVESTIGATE)[1] == REWARDS["investigate"]
    assert tiny_env.step(INVESTIGATE)[1] == REWARDS["repeat_investigate"]       # unnecessary repeat
    assert tiny_env.step(REQUEST_DOCS)[1] == REWARDS["unnecessary_request"]     # docs are not missing


def test_request_docs_reveals_information(episodes):
    ep = next(e for e in episodes["dev"] if any(t["doc_status_initial"] == "missing" for t in e["transactions"]))
    idx = next(i for i, t in enumerate(ep["transactions"]) if t["doc_status_initial"] == "missing")
    import copy
    ep = copy.deepcopy(ep); ep["transactions"] = ep["transactions"][idx:idx + 1]
    env = AuditEnv([ep]); obs, _ = env.reset()
    assert round(obs[3] * 3) == 1                                                # missing
    obs, r, *_ = env.step(REQUEST_DOCS)
    assert r == REWARDS["request_docs"] and round(obs[3] * 3) in (2, 3)          # received / refused


def test_episode_terminates_after_last_decision(tiny_env):
    tiny_env.step(APPROVE)
    _, _, terminated, truncated, _ = tiny_env.step(FLAG)
    assert terminated and not truncated and tiny_env.done
    with pytest.raises(RuntimeError):
        tiny_env.step(APPROVE)


def test_episode_truncates_when_agent_wastes_steps(tiny_env):
    for _ in range(2 * MAX_STEPS_PER_TX):
        _, _, terminated, truncated, _ = tiny_env.step(INVESTIGATE)
    assert truncated and not terminated and tiny_env.get_summary()["undecided"] == 2


def test_summary_metrics(tiny_env):
    tiny_env.step(APPROVE); tiny_env.step(FLAG)
    s = tiny_env.get_summary()
    assert (s["true_negative"], s["true_positive"]) == (1, 1)
    assert s["accuracy"] == s["precision"] == s["recall"] == 1.0
    assert s["total_reward"] == REWARDS["true_negative"] + REWARDS["true_positive"]


def test_observation_hides_ground_truth(episodes):
    env = AuditEnv(episodes["dev"]); env.reset()
    assert "label" not in env.current_transaction()
