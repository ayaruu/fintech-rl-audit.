"""
Runs policies inside the environment and records everything that happens.
Used by the website backend, the data pipeline and the tests, so numbers always match.
"""
from __future__ import annotations

from .audit_env import AuditEnv
from .config import ACTION_NAMES, POLICY_SEED, obs_to_dict
from .metrics import aggregate
from .policies import POLICY_INFO, POLICY_KEYS, make_policy


def step_with_record(env: AuditEnv, obs, action):
    """Perform one env step and return (record, next_obs, done). The record is JSON-friendly."""
    view = env.current_transaction()
    obs_named = obs_to_dict(obs)
    a = int(action)
    next_obs, reward, terminated, truncated, info = env.step(a)
    record = {
        "step": info["step"],
        "episode_id": info["episode_id"],
        "transaction_id": view["transaction_id"] if view else None,
        "transaction": view,
        "observation": [round(float(x), 4) for x in obs],
        "observation_named": obs_named,
        "action": a,
        "action_name": ACTION_NAMES[a],
        "reward": round(float(reward), 2),
        "total_reward": info["total_reward"],
        "decision_made": info["decision_made"],
        "outcome": info["outcome"],
        "is_suspicious": info["is_suspicious"],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(terminated or truncated),
    }
    return record, next_obs, bool(terminated or truncated)


def run_episode(episodes, episode_index: int, policy, seed: int = POLICY_SEED, rewards=None) -> dict:
    """Run one full episode with a policy. Deterministic for a fixed seed."""
    env = AuditEnv(episodes, rewards)
    obs, _ = env.reset(seed=seed, options={"episode_index": episode_index})
    policy.reset(seed=seed + episode_index)
    steps, done = [], False
    while not done:
        record, obs, done = step_with_record(env, obs, policy.act(obs))
        steps.append(record)
    return {"episode_id": env.episode["episode_id"], "episode_index": episode_index,
            "policy": policy.key, "steps": steps, "summary": env.get_summary()}


def evaluate_policy(episodes, policy_key: str, agent=None) -> dict:
    """Run one policy on every episode of a split."""
    runs = [run_episode(episodes, i, make_policy(policy_key, agent=agent)) for i in range(len(episodes))]
    per_episode = [r["summary"] for r in runs]
    return {"policy": policy_key, "name": POLICY_INFO[policy_key][0], "runs": runs,
            "per_episode": per_episode, "aggregate": aggregate(per_episode)}


def evaluate_all(episodes_by_split: dict, agent) -> dict:
    """{split: {policy_key: evaluation}} for all four policies."""
    return {split: {key: evaluate_policy(eps, key, agent) for key in POLICY_KEYS}
            for split, eps in episodes_by_split.items()}
