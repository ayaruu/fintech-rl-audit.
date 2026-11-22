"""
All the "business logic" of the website lives here (no web code), so it is easy to
read and to test. main.py only turns these methods into HTTP endpoints.
"""
from __future__ import annotations

import json
from pathlib import Path

from environment.audit_env import AuditEnv
from environment.config import (
    ANOMALY_TYPES, ACTION_IDS, ACTION_LABELS, ACTION_NAMES, N_ACTIONS, OBS_DESCRIPTIONS, OBS_FIELDS,
    POLICY_SEED, REWARD_DESCRIPTIONS, REWARDS, obs_to_dict,
)
from environment.pipeline import build_all, data_is_ready, flatten_transactions
from environment.policies import POLICY_INFO, POLICY_KEYS, make_policy
from environment.rl_agent import QLearningAgent
from environment.runner import evaluate_policy, run_episode, step_with_record

from .reports import build_report, report_to_csv, report_to_json

ROOT = Path(__file__).resolve().parent.parent
MANUAL = "manual"
TABLE_COLUMNS = ["transaction_id", "episode_id", "split", "date", "time", "amount", "category", "account_id",
                 "status", "risk_level", "risk_score", "doc_status_initial", "is_suspicious", "anomaly_type"]


class AuditService:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)
        if not data_is_ready(self.root):          # first start: create datasets + train the agent (~5 s)
            build_all(self.root, verbose=False)
        data = self.root / "data"
        self.episodes = {
            "train": json.loads((data / "train_episodes.json").read_text()),
            "dev": json.loads((data / "dev_episodes.json").read_text()),
        }
        self.agent = QLearningAgent.load(data / "q_table.json")
        self.training_log = json.loads((data / "training_log.json").read_text())
        self.tx_df = flatten_transactions(self.episodes)
        self._eval_cache: dict[str, dict] = {}
        self.session: dict | None = None          # the audit episode currently open in the UI
        self.last_completed: dict | None = None   # the last finished run (used by the report page)

    # ------------------------------------------------------------------ meta
    def meta(self) -> dict:
        return {
            "policies": [{"key": k, "name": POLICY_INFO[k][0], "description": POLICY_INFO[k][1]} for k in POLICY_KEYS],
            "actions": [{"id": i, "name": n, "label": ACTION_LABELS[n]} for i, n in enumerate(ACTION_NAMES)],
            "rewards": [{"key": k, "value": v, "description": REWARD_DESCRIPTIONS[k]} for k, v in REWARDS.items()],
            "observation_fields": [{"index": i, "name": n, "description": OBS_DESCRIPTIONS[n]} for i, n in enumerate(OBS_FIELDS)],
            "episodes": {
                split: [{"episode_index": e["index"], "episode_id": e["episode_id"],
                         "n_transactions": e["n_transactions"]} for e in eps]
                for split, eps in self.episodes.items()},
            "agent": self.agent.meta,
            "categories": sorted(self.tx_df["category"].unique().tolist()),
            "anomaly_types": ANOMALY_TYPES,
        }

    # ------------------------------------------------------ policy evaluation
    def evaluation(self, split: str) -> dict:
        """Run all 4 policies on every episode of a split (cached; deterministic)."""
        self._check_split(split)
        if split not in self._eval_cache:
            result = {}
            for key in POLICY_KEYS:
                ev = evaluate_policy(self.episodes[split], key, self.agent)
                result[key] = {"name": ev["name"], "aggregate": ev["aggregate"], "per_episode": ev["per_episode"]}
            self._eval_cache[split] = result
        return self._eval_cache[split]

    def compare(self, split: str = "dev") -> dict:
        ev = self.evaluation(split)
        rows = []
        for key in POLICY_KEYS:
            rows.append({"policy": key, "name": ev[key]["name"], "description": POLICY_INFO[key][1],
                         **ev[key]["aggregate"],
                         "per_episode_rewards": [s["total_reward"] for s in ev[key]["per_episode"]]})
        best = max(rows, key=lambda r: r["total_reward"])["policy"]
        return {"split": split, "policies": rows, "best_total_reward": best,
                "episode_ids": [e["episode_id"] for e in self.episodes[split]],
                "note": ("The RL agent was trained on the 'train' split only. Results on 'dev' are on unseen episodes; "
                         "results on 'train' are on the episodes it learned from.")}

    def dashboard(self, split: str = "dev", policy: str = "rl_agent") -> dict:
        self._check_split(split)
        if policy not in POLICY_KEYS:
            raise ValueError(f"Unknown policy {policy!r}.")
        ev = self.evaluation(split)
        agg = ev[policy]["aggregate"]
        return {
            "split": split, "policy": policy, "policy_name": ev[policy]["name"],
            "total_transactions": int(len(self.tx_df)),
            "split_transactions": int((self.tx_df["split"] == split).sum()),
            "audited_transactions": agg["decisions"],
            "suspicious_flagged": agg["true_positive"] + agg["false_positive"],
            "actual_suspicious": agg["true_positive"] + agg["false_negative"],
            "accuracy": agg["accuracy"], "precision": agg["precision"], "recall": agg["recall"],
            "total_reward": agg["total_reward"], "mean_reward": agg["mean_reward"],
            "episodes": agg["episodes"],
            "episode_ids": [e["episode_id"] for e in self.episodes[split]],
            "per_episode": {k: [s["total_reward"] for s in ev[k]["per_episode"]] for k in POLICY_KEYS},
            "policy_names": {k: ev[k]["name"] for k in POLICY_KEYS},
        }

    # ----------------------------------------------------------- transactions
    def transactions(self, split="all", episode="", search="", category="", risk="", status="",
                     suspicious="all", limit=1000) -> dict:
        if split not in ("all", "train", "dev"):
            raise ValueError("split must be 'all', 'train' or 'dev'.")
        df = self.tx_df
        if split != "all":
            df = df[df["split"] == split]
        if episode:
            df = df[df["episode_id"] == episode]
        if category:
            df = df[df["category"] == category]
        if risk:
            df = df[df["risk_level"].str.lower() == risk.lower()]
        if status:
            df = df[df["status"].str.lower() == status.lower()]
        if suspicious == "yes":
            df = df[df["is_suspicious"]]
        elif suspicious == "no":
            df = df[~df["is_suspicious"]]
        if search:
            s = search.strip().lower()
            mask = (df["transaction_id"].str.lower().str.contains(s, regex=False)
                    | df["account_id"].str.lower().str.contains(s, regex=False)
                    | df["category"].str.lower().str.contains(s, regex=False)
                    | df["status"].str.lower().str.contains(s, regex=False)
                    | df["amount"].astype(str).str.contains(s, regex=False))
            df = df[mask]
        df = df.sort_values(["date", "time", "transaction_id"], kind="stable")
        limit = max(1, min(int(limit), 5000))
        rows = json.loads(df[TABLE_COLUMNS].head(limit).to_json(orient="records"))
        return {"total": int(len(df)), "returned": len(rows), "transactions": rows}

    # ---------------------------------------------------------- audit session
    def start_episode(self, split: str = "dev", episode_index: int = 0, policy: str = "rl_agent") -> dict:
        self._check_split(split)
        if policy != MANUAL and policy not in POLICY_KEYS:
            raise ValueError(f"Unknown policy {policy!r}. Choose one of {POLICY_KEYS + [MANUAL]}.")
        episodes = self.episodes[split]
        if not 0 <= int(episode_index) < len(episodes):
            raise ValueError(f"episode_index must be between 0 and {len(episodes) - 1}.")
        episode_index = int(episode_index)
        env = AuditEnv(episodes)
        obs, _ = env.reset(seed=POLICY_SEED, options={"episode_index": episode_index})
        pol = None
        if policy != MANUAL:
            pol = make_policy(policy, agent=self.agent)
            pol.reset(seed=POLICY_SEED + episode_index)   # same seeding as run_episode -> identical results
        self.session = {"env": env, "policy": pol, "policy_key": policy, "split": split,
                        "episode_index": episode_index, "obs": obs, "steps": []}
        return self.session_state()

    def _require_session(self) -> dict:
        if self.session is None:
            raise RuntimeError("No active episode. Start one first.")
        return self.session

    @staticmethod
    def parse_action(action) -> int:
        """Accept 0-3 or 'approve'/'flag'/'request_docs'/'investigate'."""
        if isinstance(action, bool):
            raise ValueError(f"Invalid action {action!r}.")
        if isinstance(action, str):
            a = action.strip().lower()
            if a in ACTION_IDS:
                return ACTION_IDS[a]
            if a.isdigit():
                action = int(a)
            else:
                raise ValueError(f"Invalid action {action!r}. Use one of {ACTION_NAMES} or 0-{N_ACTIONS - 1}.")
        if isinstance(action, float) and action.is_integer():
            action = int(action)
        if not isinstance(action, int) or not 0 <= action < N_ACTIONS:
            raise ValueError(f"Invalid action {action!r}. Use one of {ACTION_NAMES} or 0-{N_ACTIONS - 1}.")
        return action

    def step_session(self, action=None) -> dict:
        """One step. action=None -> the chosen policy decides; otherwise manual play."""
        s = self._require_session()
        env = s["env"]
        if env.done:
            raise RuntimeError("This episode is finished. Start a new episode.")
        if action is None:
            if s["policy"] is None:
                raise ValueError("Manual episode: please choose an action.")
            a = s["policy"].act(s["obs"])
        else:
            if s["policy"] is not None:
                raise ValueError("This episode is run by an automatic policy. Start a 'manual' episode to choose actions yourself.")
            a = self.parse_action(action)
        record, s["obs"], done = step_with_record(env, s["obs"], a)
        s["steps"].append(record)
        if done:
            self.last_completed = self._completed_run(s)
        return record

    def run_session(self) -> dict:
        """Run the active episode to the end with its policy."""
        s = self._require_session()
        if s["policy"] is None:
            raise ValueError("Manual episode: choose the actions yourself, or start an episode with a policy.")
        if s["env"].done:
            raise RuntimeError("This episode is finished. Start a new episode.")
        new_steps = []
        while not s["env"].done:
            new_steps.append(self.step_session())
        return {"new_steps": new_steps, "state": self.session_state()}

    def session_state(self) -> dict:
        s = self.session
        if s is None:
            return {"active": False}
        env = s["env"]
        key = s["policy_key"]
        return {
            "active": True,
            "episode_id": env.episode["episode_id"], "split": s["split"], "episode_index": s["episode_index"],
            "policy": key, "policy_name": "Manual (you)" if key == MANUAL else POLICY_INFO[key][0],
            "step": env.step_count, "total_reward": env.total_reward,
            "done": env.done, "truncated": env.truncated,
            "status": "truncated" if env.truncated else ("completed" if env.done else "running"),
            "decided": len(env.decisions), "n_transactions": len(env.episode["transactions"]),
            "current_observation": None if env.done else obs_to_dict(s["obs"]),
            "current_transaction": env.current_transaction(),
            "last_step": s["steps"][-1] if s["steps"] else None,
            "steps": s["steps"],
            "summary": env.get_summary() if env.done else None,
        }

    def _completed_run(self, s: dict) -> dict:
        key = s["policy_key"]
        return {"episode_id": s["env"].episode["episode_id"], "split": s["split"], "episode_index": s["episode_index"],
                "policy": key, "policy_name": "Manual (you)" if key == MANUAL else POLICY_INFO[key][0],
                "steps": list(s["steps"]), "summary": s["env"].get_summary()}

    # ----------------------------------------------------------------- reports
    def report(self, source: str = "episode", split: str = "dev", episode_index: int = 0,
               policy: str = "rl_agent") -> dict:
        if source == "session":
            if self.last_completed is None:
                raise LookupError("No finished audit run yet. Run an episode on the 'AI Audit Agent' page first.")
            run = self.last_completed
        else:
            self._check_split(split)
            if policy not in POLICY_KEYS:
                raise ValueError(f"Unknown policy {policy!r}.")
            if not 0 <= int(episode_index) < len(self.episodes[split]):
                raise ValueError(f"episode_index must be between 0 and {len(self.episodes[split]) - 1}.")
            res = run_episode(self.episodes[split], int(episode_index), make_policy(policy, agent=self.agent))
            run = {**res, "split": split, "policy_name": POLICY_INFO[policy][0]}
        return build_report(run, self.episodes[run["split"]][run["episode_index"]])

    def export_report(self, fmt: str = "json", **kwargs) -> tuple[str, str, str]:
        """Returns (content, media_type, filename)."""
        if fmt not in ("json", "csv"):
            raise ValueError("format must be 'json' or 'csv'.")
        rep = self.report(**kwargs)
        name = f"audit_report_{rep['meta']['episode_id']}_{rep['meta']['policy']}"
        if fmt == "json":
            return report_to_json(rep), "application/json", name + ".json"
        return report_to_csv(rep), "text/csv", name + ".csv"

    # ----------------------------------------------------------------- helpers
    def _check_split(self, split: str) -> None:
        if split not in self.episodes:
            raise ValueError("split must be 'train' or 'dev'.")
