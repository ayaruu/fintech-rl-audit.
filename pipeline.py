"""
Builds every data file of the project:

    data/train_episodes.json     10 training episodes
    data/dev_episodes.json       10 development episodes
    data/transactions.csv        all transactions in one flat table
    data/q_table.json            the trained Q-learning agent
    data/training_log.json       learning curve of the agent
    data/evaluation_summary.json metrics of the 4 policies (train + dev)
    data/dataset_summary.json    counts describing the datasets
    trajectories.jsonl           every step of every policy on every episode

Run it with:  python build_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .audit_env import AuditEnv
from .config import N_DEV_EPISODES, N_TRAIN_EPISODES, POLICY_SEED
from .data_generator import make_episode
from .rl_agent import QLearningAgent
from .runner import evaluate_all

DATA_FILES = ["train_episodes.json", "dev_episodes.json", "q_table.json", "training_log.json"]


def flatten_transactions(episodes_by_split: dict) -> pd.DataFrame:
    """One row per transaction (visible fields + ground-truth label columns)."""
    rows = []
    for split, episodes in episodes_by_split.items():
        for ep in episodes:
            for tx in ep["transactions"]:
                row = {k: v for k, v in tx.items() if k != "label"}
                row["split"] = split
                row["is_suspicious"] = tx["label"]["is_suspicious"]
                row["anomaly_type"] = tx["label"]["anomaly_type"]
                rows.append(row)
    return pd.DataFrame(rows)


def dataset_summary(episodes_by_split: dict) -> dict:
    df = flatten_transactions(episodes_by_split)
    out = {}
    for split, g in df.groupby("split"):
        out[split] = {
            "episodes": int(g["episode_id"].nunique()),
            "transactions": int(len(g)),
            "suspicious": int(g["is_suspicious"].sum()),
            "legitimate": int((~g["is_suspicious"]).sum()),
            "missing_documents": int((g["doc_status_initial"] == "missing").sum()),
            "record_mismatches": int((g["invoice_amount"] != g["amount"]).sum()),
            "anomaly_types": {k: int(v) for k, v in g["anomaly_type"].value_counts().items()},
        }
    return out


def write_trajectories(evaluation: dict, path: Path) -> int:
    """Write trajectories.jsonl (one JSON object per line). Returns number of lines."""
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for split, by_policy in evaluation.items():
            for policy_key, result in by_policy.items():
                for run in result["runs"]:
                    for s in run["steps"]:
                        line = {
                            "type": "step", "split": split, "policy": policy_key,
                            "episode_id": run["episode_id"], "step": s["step"],
                            "transaction_id": s["transaction_id"],
                            "observation": s["observation"],
                            "action": s["action"], "action_name": s["action_name"],
                            "reward": s["reward"], "total_reward": s["total_reward"],
                            "outcome": s["outcome"], "done": s["done"],
                        }
                        f.write(json.dumps(line) + "\n")
                        n += 1
                    f.write(json.dumps({"type": "episode_result", "split": split, "policy": policy_key,
                                        "episode_id": run["episode_id"], "result": run["summary"]}) + "\n")
                    n += 1
    return n


def build_all(root: Path, n_epochs: int = 300, verbose: bool = True) -> dict:
    root = Path(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train = [make_episode("train", i) for i in range(N_TRAIN_EPISODES)]
    dev = [make_episode("dev", i) for i in range(N_DEV_EPISODES)]
    episodes = {"train": train, "dev": dev}
    (data_dir / "train_episodes.json").write_text(json.dumps(train, indent=1))
    (data_dir / "dev_episodes.json").write_text(json.dumps(dev, indent=1))
    flatten_transactions(episodes).to_csv(data_dir / "transactions.csv", index=False)
    summary = dataset_summary(episodes)
    (data_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    if verbose:
        print("Datasets written:", json.dumps({k: v["transactions"] for k, v in summary.items()}))

    # Train the Q-learning agent on the TRAINING episodes only.
    agent = QLearningAgent()
    log = agent.train(AuditEnv(train), n_epochs=n_epochs, seed=POLICY_SEED)
    agent.save(data_dir / "q_table.json")
    (data_dir / "training_log.json").write_text(json.dumps(log))
    if verbose:
        print(f"Agent trained for {n_epochs} epochs. Last-epoch mean training reward: {log[-1]['mean_reward']}")

    # Evaluate all four policies and write the trajectories.
    evaluation = evaluate_all(episodes, agent)
    lines = write_trajectories(evaluation, root / "trajectories.jsonl")
    slim = {split: {k: {"name": v["name"], "aggregate": v["aggregate"], "per_episode": v["per_episode"]}
                    for k, v in by_policy.items()} for split, by_policy in evaluation.items()}
    (data_dir / "evaluation_summary.json").write_text(json.dumps(slim, indent=2))
    if verbose:
        print(f"trajectories.jsonl written ({lines} lines)")
        for split in ("dev", "train"):
            print(f"\n{split.upper()} results")
            for k, v in slim[split].items():
                a = v["aggregate"]
                print(f"  {v['name']:<24} reward={a['total_reward']:>8}  acc={a['accuracy']:.3f}  "
                      f"prec={a['precision']:.3f}  rec={a['recall']:.3f}  inv={a['investigations']}")
    return slim


def data_is_ready(root: Path) -> bool:
    root = Path(root)
    return all((root / "data" / f).exists() for f in DATA_FILES) and (root / "trajectories.jsonl").exists()
