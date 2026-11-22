"""
AuditEnv - a custom Gymnasium environment for financial auditing.

HOW AN EPISODE WORKS
--------------------
An episode is a batch of 40 transactions. The agent looks at ONE transaction at a time:

    observation (what the agent sees)  ->  action  ->  reward + next observation

Actions
    0 approve        final decision: transaction looks fine
    1 flag           final decision: transaction is suspicious
    2 request_docs   gather information: ask for the missing document
    3 investigate    gather information: deeper look (noisy but informative)

approve / flag finish the current transaction and move to the next one.
request_docs / investigate keep the same transaction open and update the observation.
The episode ends (terminated) after the last transaction is decided. It is cut off
(truncated) if the agent wastes too many steps (n_transactions * 6).

The reward function is documented in config.REWARDS and in README.md.
"""
from __future__ import annotations

import math

import numpy as np

from ._compat import Box, Discrete, Env
from .config import (
    ACTION_NAMES, APPROVE, CATEGORIES, DOC_STATUS_NAMES, FLAG, INVESTIGATE, MAX_STEPS_PER_TX,
    N_ACTIONS, N_OBS, REQUEST_DOCS, REWARDS,
)
from .data_generator import visible_transaction
from .metrics import compute_metrics


class AuditEnv(Env):
    """Gymnasium environment. Use reset() to start an episode and step(action) to act."""

    metadata = {"render_modes": []}

    def __init__(self, episodes: list[dict], rewards: dict | None = None):
        super().__init__()
        if not episodes:
            raise ValueError("AuditEnv needs at least one episode.")
        self.episodes = list(episodes)
        self.rewards = {**REWARDS, **(rewards or {})}

        # Gymnasium requires these two attributes
        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Box(low=0.0, high=1.0, shape=(N_OBS,), dtype=np.float32)

        self._next_episode = 0
        self._started = False
        self._done = True
        self._truncated = False

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed=None, options=None):
        """Start an episode. options={'episode_index': 3} picks a specific one."""
        super().reset(seed=seed)
        options = options or {}
        if options.get("episode_index") is not None:
            idx = int(options["episode_index"])
            if not 0 <= idx < len(self.episodes):
                raise IndexError(f"episode_index must be between 0 and {len(self.episodes) - 1}")
        else:
            idx = self._next_episode
        self._next_episode = (idx + 1) % len(self.episodes)

        self.episode_index = idx
        self.episode = self.episodes[idx]
        self._txs = self.episode["transactions"]
        self._pos = 0
        self._decisions: list[dict] = []
        self._flags_by_account: dict[str, int] = {}
        self._n_flagged = 0
        self._step_count = 0
        self._total_reward = 0.0
        self._n_investigate = 0
        self._n_request_docs = 0
        self._max_steps = len(self._txs) * MAX_STEPS_PER_TX
        self._start_transaction()
        self._started = True
        self._done = False
        self._truncated = False
        return self._observation(), self._info(None, None, None, False, 0.0)

    def _start_transaction(self):
        self._investigated = False
        self._docs_requested = False

    # ------------------------------------------------------------------- step
    def step(self, action):
        """Apply one action. Returns (observation, reward, terminated, truncated, info)."""
        if not self._started:
            raise RuntimeError("Call reset() before step().")
        if self._done:
            raise RuntimeError("The episode is finished. Call reset() to start a new one.")
        action = self._validate_action(action)

        tx = self._txs[self._pos]
        label = tx["label"]
        suspicious = bool(label["is_suspicious"])
        R = self.rewards
        outcome = None
        decision_made = False

        if action == APPROVE:
            outcome = "false_negative" if suspicious else "true_negative"
            reward, decision_made = R[outcome], True
        elif action == FLAG:
            outcome = "true_positive" if suspicious else "false_positive"
            reward, decision_made = R[outcome], True
            self._n_flagged += 1
            self._flags_by_account[tx["account_id"]] = self._flags_by_account.get(tx["account_id"], 0) + 1
        elif action == REQUEST_DOCS:
            self._n_request_docs += 1
            if tx["doc_status_initial"] == "missing" and not self._docs_requested:
                reward = R["request_docs"]
            else:
                reward = R["unnecessary_request"]
            self._docs_requested = True
        else:  # INVESTIGATE
            self._n_investigate += 1
            reward = R["repeat_investigate"] if self._investigated else R["investigate"]
            self._investigated = True

        self._step_count += 1
        self._total_reward += reward

        if decision_made:
            self._decisions.append({
                "transaction_id": tx["transaction_id"], "action": action,
                "action_name": ACTION_NAMES[action], "is_suspicious": suspicious,
                "outcome": outcome, "reward": reward,
            })
            self._pos += 1
            self._start_transaction()

        terminated = self._pos >= len(self._txs)
        truncated = (not terminated) and self._step_count >= self._max_steps
        self._done = terminated or truncated
        self._truncated = truncated
        info = self._info(tx, action, outcome, decision_made, reward)
        return self._observation(), float(reward), terminated, truncated, info

    def _validate_action(self, action) -> int:
        """Accept 0-3 (ints or numpy ints). Anything else raises ValueError."""
        try:
            a = int(action)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid action {action!r}: expected an integer 0-{N_ACTIONS - 1}.") from None
        if a != action or isinstance(action, bool) or not 0 <= a < N_ACTIONS:
            raise ValueError(f"Invalid action {action!r}: expected an integer 0-{N_ACTIONS - 1}.")
        return a

    # ------------------------------------------------------------ observation
    def _doc_status_code(self, tx) -> int:
        if tx["doc_status_initial"] == "available":
            return 0
        if not self._docs_requested:
            return 1
        return 2 if tx["label"]["doc_request_outcome"] == "received" else 3

    def _observation(self) -> np.ndarray:
        if self._pos >= len(self._txs):
            return np.zeros(N_OBS, dtype=np.float32)
        tx = self._txs[self._pos]
        if not self._investigated:
            inv = 0.0
        else:
            inv = 1.0 if tx["label"]["investigation_signal"] == "problem" else 0.5
        decided = len(self._decisions)
        vec = [
            min(math.log10(1 + tx["amount"]) / 5.0, 1.0),
            CATEGORIES.index(tx["category"]) / (len(CATEGORIES) - 1),
            min(tx["account_age_days"], 365) / 365.0,
            self._doc_status_code(tx) / 3.0,
            min(tx["amount_ratio"], 10.0) / 10.0,
            min(tx["velocity"], 6) / 6.0,
            float(tx["country_mismatch"]),
            float(tx["off_hours"]),
            float(tx["round_amount"]),
            float(tx["near_threshold"]),
            float(tx["risk_score"]),
            inv,
            min(self._flags_by_account.get(tx["account_id"], 0), 3) / 3.0,
            (self._n_flagged / decided) if decided else 0.0,
            self._pos / len(self._txs),
        ]
        return np.clip(np.array(vec, dtype=np.float32), 0.0, 1.0)

    def _info(self, tx, action, outcome, decision_made, reward) -> dict:
        return {
            "episode_id": self.episode["episode_id"],
            "step": self._step_count,
            "transaction_id": tx["transaction_id"] if tx else None,
            "action": action,
            "action_name": ACTION_NAMES[action] if action is not None else None,
            "decision_made": decision_made,
            "outcome": outcome,
            # ground truth is only revealed once a final decision was made (as feedback)
            "is_suspicious": bool(tx["label"]["is_suspicious"]) if (tx and decision_made) else None,
            "reward": float(reward),
            "total_reward": round(self._total_reward, 2),
        }

    # -------------------------------------------------------- helpers for the UI
    def current_transaction(self) -> dict | None:
        """The transaction the agent is looking at (visible fields only, no ground truth)."""
        if not self._started or self._pos >= len(self._txs):
            return None
        tx = self._txs[self._pos]
        view = visible_transaction(tx)
        view["doc_status"] = DOC_STATUS_NAMES[self._doc_status_code(tx)]
        if not self._investigated:
            view["investigation_result"] = "not investigated"
        else:
            view["investigation_result"] = "problem found" if tx["label"]["investigation_signal"] == "problem" else "looks clean"
        return view

    @property
    def done(self) -> bool:
        return self._done

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def total_reward(self) -> float:
        return round(self._total_reward, 2)

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def decisions(self) -> list[dict]:
        return list(self._decisions)

    def get_summary(self) -> dict:
        """Episode metrics (reward, accuracy, precision, recall, #investigations ...)."""
        summary = compute_metrics(self._decisions, self._n_investigate, self._n_request_docs, self._total_reward)
        summary.update({
            "episode_id": self.episode["episode_id"],
            "n_transactions": len(self._txs),
            "steps": self._step_count,
            "undecided": len(self._txs) - len(self._decisions),
            "truncated": self._truncated,
        })
        return summary
