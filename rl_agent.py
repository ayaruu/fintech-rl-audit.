"""
A small, honest reinforcement-learning agent: TABULAR Q-LEARNING.

Why not Stable-Baselines3? Our problem is tiny (15 numbers in, 4 actions out) and
the important information can be reduced to a few buckets, so a Q-table is easier
to understand, trains in seconds on a laptop and needs no GPU/PyTorch.

How it learns
    1. The observation is converted to a small "state" (risk bucket, document
       status, investigation result, big-amount flag, earlier-flags flag).
    2. For each state we keep 4 numbers (Q-values): how good each action is.
    3. During training the agent tries actions (epsilon-greedy), receives rewards
       from the environment and updates:  Q(s,a) += alpha * (r + gamma*max Q(s',.) - Q(s,a))
    4. It is trained ONLY on the 10 training episodes and evaluated on the 10
       development episodes it has never seen.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import N_ACTIONS, OBS_INDEX

RISK_EDGES = [0.10, 0.25, 0.40, 0.55]          # 5 risk buckets
Q_SHAPE = (5, 4, 3, 2, 2, N_ACTIONS)            # risk, doc, investigation, big-amount, prior-flags, action


class QLearningAgent:
    def __init__(self):
        self.q = np.zeros(Q_SHAPE)
        self.meta: dict = {"trained": False}

    # ------------------------------------------------------------- state
    @staticmethod
    def state_of(obs) -> tuple:
        """Reduce the 15-number observation to a small discrete state."""
        risk_bucket = int(np.searchsorted(RISK_EDGES, float(obs[OBS_INDEX["risk_score"]]), side="right"))
        doc = int(round(float(obs[OBS_INDEX["doc_status"]]) * 3))
        inv = int(round(float(obs[OBS_INDEX["investigation_result"]]) * 2))
        big = int(float(obs[OBS_INDEX["amount_ratio"]]) > 0.3)       # amount > 3x category norm
        prior = int(float(obs[OBS_INDEX["account_prior_flags"]]) > 0)
        return (risk_bucket, doc, inv, big, prior)

    # --------------------------------------------------------------- act
    def act(self, obs, greedy: bool = True, epsilon: float = 0.0, rng=None) -> int:
        if not greedy and rng is not None and rng.random() < epsilon:
            return int(rng.integers(N_ACTIONS))
        return int(np.argmax(self.q[self.state_of(obs)]))

    # ------------------------------------------------------------- train
    def train(self, env, n_epochs: int = 300, seed: int = 42, alpha: float = 0.1,
              gamma: float = 0.95, eps_start: float = 1.0, eps_end: float = 0.05) -> list[dict]:
        """Train on every episode of `env` for n_epochs passes. Returns a learning log."""
        rng = np.random.default_rng(seed)
        n_eps = len(env.episodes)
        log = []
        for epoch in range(n_epochs):
            # exploration rate falls linearly from eps_start to eps_end over the first 70% of epochs
            eps = max(eps_end, eps_start - (eps_start - eps_end) * epoch / (0.7 * n_epochs))
            rewards = []
            for ep_idx in rng.permutation(n_eps):
                obs, _ = env.reset(options={"episode_index": int(ep_idx)})
                done = False
                while not done:
                    s = self.state_of(obs)
                    a = self.act(obs, greedy=False, epsilon=eps, rng=rng)
                    obs, r, terminated, truncated, _ = env.step(a)
                    s2 = self.state_of(obs)
                    target = r if terminated else r + gamma * float(np.max(self.q[s2]))
                    self.q[s + (a,)] += alpha * (target - self.q[s + (a,)])
                    done = terminated or truncated
                rewards.append(env.total_reward)
            log.append({"epoch": epoch + 1, "epsilon": round(eps, 3),
                        "mean_reward": round(float(np.mean(rewards)), 2)})
        self.meta = {
            "trained": True, "algorithm": "tabular Q-learning", "epochs": n_epochs, "seed": seed,
            "alpha": alpha, "gamma": gamma, "train_episodes": [e["episode_id"] for e in env.episodes],
        }
        return log

    # --------------------------------------------------------- save/load
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"meta": self.meta, "shape": list(Q_SHAPE),
                                          "q": self.q.flatten().round(6).tolist()}))

    @classmethod
    def load(cls, path) -> "QLearningAgent":
        data = json.loads(Path(path).read_text())
        agent = cls()
        agent.q = np.array(data["q"], dtype=float).reshape(tuple(data["shape"]))
        agent.meta = data.get("meta", {})
        return agent
