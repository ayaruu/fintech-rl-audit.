"""
The four policies that can audit an episode.

A policy only receives the OBSERVATION (never the hidden ground truth) and
returns an action number (0 approve, 1 flag, 2 request_docs, 3 investigate).

    1. RandomPolicy         - picks any action at random (a "floor" for comparison)
    2. RuleBasedPolicy      - fixed if-then audit rules, never investigates
    3. RiskThresholdPolicy  - uses the risk score; investigates the grey zone
    4. RLPolicy             - the trained Q-learning agent
"""
from __future__ import annotations

import numpy as np

from .config import APPROVE, FLAG, INVESTIGATE, N_ACTIONS, OBS_INDEX, POLICY_SEED, REQUEST_DOCS

I = OBS_INDEX


class BasePolicy:
    key = "base"
    name = "Base policy"
    description = ""

    def reset(self, seed: int | None = None) -> None:  # called at the start of every episode
        pass

    def act(self, obs) -> int:
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    key = "random"
    name = "Random Policy"
    description = "Chooses one of the 4 actions uniformly at random (seeded, so it is reproducible)."

    def __init__(self, seed: int = POLICY_SEED):
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None):
        self.rng = np.random.default_rng(POLICY_SEED if seed is None else seed)

    def act(self, obs) -> int:
        return int(self.rng.integers(N_ACTIONS))


class RuleBasedPolicy(BasePolicy):
    key = "rule_based"
    name = "Rule-Based Policy"
    description = ("Fixed audit rules: request missing documents, flag refused documents, "
                   "flag amounts > 3x the category norm, flag foreign off-hours payments and "
                   "amounts just under the review limit. Otherwise approve. Never investigates.")

    def act(self, obs) -> int:
        doc = int(round(float(obs[I["doc_status"]]) * 3))     # 0 ok, 1 missing, 2 received, 3 refused
        if doc == 1:
            return REQUEST_DOCS                                # ask for the missing document first
        if doc == 3:
            return FLAG                                        # document request was refused
        if float(obs[I["amount_ratio"]]) > 0.3:                # amount > 3x category norm
            return FLAG
        if obs[I["country_mismatch"]] > 0.5 and obs[I["off_hours"]] > 0.5:
            return FLAG
        if obs[I["near_threshold"]] > 0.5:
            return FLAG
        return APPROVE


class RiskThresholdPolicy(BasePolicy):
    key = "risk_threshold"
    name = "Risk-Threshold Policy"
    description = ("risk >= 0.50: flag. risk 0.25-0.50: investigate once, then flag only if the "
                   "investigation found a problem. risk < 0.25: approve.")

    def __init__(self, flag_threshold: float = 0.50, review_threshold: float = 0.25):
        self.flag_threshold = flag_threshold
        self.review_threshold = review_threshold

    def act(self, obs) -> int:
        risk = float(obs[I["risk_score"]])
        inv = float(obs[I["investigation_result"]])            # 0 unknown, 0.5 clean, 1 problem
        if inv == 1.0 or risk >= self.flag_threshold:
            return FLAG
        if risk >= self.review_threshold:
            return INVESTIGATE if inv == 0.0 else APPROVE
        return APPROVE


class RLPolicy(BasePolicy):
    key = "rl_agent"
    name = "RL Agent (Q-learning)"
    description = ("Tabular Q-learning agent trained on the 10 training episodes; evaluated on the "
                   "10 unseen development episodes. It learns when investigating is worth its cost.")

    def __init__(self, agent):
        self.agent = agent

    def act(self, obs) -> int:
        return self.agent.act(obs, greedy=True)


POLICY_KEYS = ["random", "rule_based", "risk_threshold", "rl_agent"]
POLICY_INFO = {
    "random": (RandomPolicy.name, RandomPolicy.description),
    "rule_based": (RuleBasedPolicy.name, RuleBasedPolicy.description),
    "risk_threshold": (RiskThresholdPolicy.name, RiskThresholdPolicy.description),
    "rl_agent": (RLPolicy.name, RLPolicy.description),
}


def make_policy(key: str, seed: int = POLICY_SEED, agent=None) -> BasePolicy:
    """Create a policy by its key. The RL policy needs a trained agent."""
    if key == "random":
        return RandomPolicy(seed)
    if key == "rule_based":
        return RuleBasedPolicy()
    if key == "risk_threshold":
        return RiskThresholdPolicy()
    if key == "rl_agent":
        if agent is None:
            raise ValueError("The RL policy needs a trained agent.")
        return RLPolicy(agent)
    raise ValueError(f"Unknown policy {key!r}. Choose one of {POLICY_KEYS}.")
