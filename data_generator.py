"""
Synthetic data generator.

Creates "episodes". One episode = one audit batch of 40 transactions covering a
14-day window. All data is FAKE and generated with fixed random seeds, so the
same seed always produces exactly the same episode.

Each transaction has
  * visible fields   -> what the audit agent can see
  * a hidden "label" -> the ground truth, used ONLY by the environment to compute rewards
"""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np

from .config import (
    ANOMALY_PROBS, ANOMALY_TYPES, CATEGORIES, CATEGORY_PROBS, CATEGORY_STATS,
    DEV_SEED_BASE, EPISODE_DAYS, TRAIN_SEED_BASE, TX_PER_EPISODE, risk_level,
)

ACCOUNT_TYPES = ["Operating", "Payroll", "Vendor", "Treasury"]
OTHER_COUNTRIES = ["US", "DE", "AE", "GB", "TR"]
STRUCTURING_ACCOUNTS = [3, 8, 12]   # account indexes used by "structuring" cases
NEW_ACCOUNTS = [17, 18, 19]         # young accounts (< 60 days old)


def build_accounts() -> list[dict]:
    """20 fixed synthetic accounts (always the same, independent of the episode)."""
    rng = np.random.default_rng(7)
    accounts = []
    for i in range(20):
        age = [15, 30, 45][i - 17] if i >= 17 else int(rng.integers(120, 2500))
        home = "KZ" if i < 15 else OTHER_COUNTRIES[i - 15]
        accounts.append({
            "account_id": f"ACC-{1001 + i}",
            "account_type": ACCOUNT_TYPES[i % len(ACCOUNT_TYPES)],
            "account_age_days": age,
            "account_country": home,
        })
    return accounts


def compute_risk_score(tx: dict) -> float:
    """
    Heuristic risk score between 0 and 1 built ONLY from visible indicators.
    (It is imperfect on purpose: some normal transactions look risky and some
    suspicious ones look harmless. That is what makes auditing hard.)
    """
    ratio = tx["amount_ratio"]
    score = 0.0
    score += 0.30 if ratio > 3 else (0.10 if ratio > 2 else 0.0)
    score += 0.15 if tx["velocity"] >= 3 else (0.05 if tx["velocity"] == 2 else 0.0)
    score += 0.15 * tx["country_mismatch"]
    score += 0.10 * tx["off_hours"]
    score += 0.10 * tx["round_amount"]
    score += 0.15 * tx["near_threshold"]
    score += 0.10 * tx["new_account"]
    score += 0.25 * (tx["doc_status_initial"] == "missing")
    return round(min(score, 1.0), 3)


def visible_transaction(tx: dict) -> dict:
    """The transaction without its hidden ground-truth label."""
    return {k: v for k, v in tx.items() if k != "label"}


def _raw_tx(rng, accounts, anomaly, acct_idx=None, day=None):
    """Draw one transaction. `anomaly` is None (legitimate) or an anomaly type."""
    cat = CATEGORIES[int(rng.choice(len(CATEGORIES), p=CATEGORY_PROBS))]
    median, sigma = CATEGORY_STATS[cat]
    amount = median * float(np.exp(rng.normal(0, sigma)))
    if acct_idx is None:
        acct_idx = int(rng.integers(len(accounts)))
    if day is None:
        day = int(rng.integers(0, EPISODE_DAYS))
    off_p, flip_p, miss_p = 0.08, 0.06, 0.08   # normal "noise" in legitimate data
    invoice_factor = 1.0

    if anomaly is None:
        if rng.random() < 0.03:                      # a legitimate unusually big purchase
            amount = median * float(rng.uniform(3.2, 5.0))
        elif rng.random() < 0.10:                    # a legitimate round amount
            amount = max(100.0, round(amount, -2))
    elif anomaly == "large_amount":
        amount = median * float(rng.uniform(4.0, 9.0))
        off_p, miss_p = 0.30, 0.20
    elif anomaly == "structuring":
        cat = ["Vendor Payment", "International Wire"][int(rng.integers(2))]
        amount = float(rng.integers(910, 999) * 10)  # 9,100 .. 9,980: just under 10,000
        miss_p = 0.30
    elif anomaly == "record_mismatch":
        sign = 1 if rng.random() < 0.5 else -1
        invoice_factor = 1 + sign * float(rng.uniform(0.08, 0.40))
        miss_p = 0.15
    elif anomaly == "missing_document":
        amount *= float(rng.uniform(1.0, 2.5))
        miss_p = 1.0
    elif anomaly == "foreign_offhours":
        acct_idx = int(NEW_ACCOUNTS[int(rng.integers(len(NEW_ACCOUNTS)))])
        amount *= float(rng.uniform(2.0, 4.0))
        off_p, flip_p, miss_p = 1.0, 1.0, 0.30

    acct = accounts[acct_idx]
    off_hours = rng.random() < off_p
    hour = int(rng.choice([0, 1, 2, 3, 4, 5, 22, 23])) if off_hours else int(rng.integers(8, 20))
    tx_country = acct["account_country"]
    if rng.random() < flip_p:
        tx_country = str(rng.choice([c for c in ["KZ"] + OTHER_COUNTRIES if c != acct["account_country"]]))
    doc_missing = bool(rng.random() < miss_p)

    # Hidden facts the environment reveals only when the agent investigates / requests documents.
    suspicious = anomaly is not None
    if suspicious:
        signal = "problem" if rng.random() < 0.90 else "clean"
        doc_outcome = None
        if doc_missing:
            always_refused = anomaly == "missing_document"
            doc_outcome = "refused" if (always_refused or rng.random() < 0.7) else "received"
    else:
        signal = "clean" if rng.random() < 0.95 else "problem"
        doc_outcome = "received" if doc_missing else None

    amount = round(amount, 2)
    return {
        "day": day, "hour": hour, "minute": int(rng.integers(0, 60)),
        "category": cat, "amount": amount,
        "invoice_amount": round(amount * invoice_factor, 2),
        "acct": acct, "tx_country": tx_country,
        "doc_status_initial": "missing" if doc_missing else "available",
        "label": {
            "is_suspicious": suspicious, "anomaly_type": anomaly,
            "investigation_signal": signal, "doc_request_outcome": doc_outcome,
        },
    }


def make_episode(split: str, index: int, n_tx: int = TX_PER_EPISODE) -> dict:
    """Create one reproducible episode. split is 'train' or 'dev'."""
    seed = (TRAIN_SEED_BASE if split == "train" else DEV_SEED_BASE) + index
    rng = np.random.default_rng(seed)
    accounts = build_accounts()
    start = (date(2026, 1, 5) if split == "train" else date(2026, 7, 6)) + timedelta(days=EPISODE_DAYS * index)

    # 1) decide which suspicious transactions this episode contains (6-10 of them)
    n_susp = int(rng.integers(6, 11))
    raws: list[dict] = []
    kinds = list(ANOMALY_TYPES)
    while len(raws) < n_susp:
        kind = kinds[int(rng.choice(len(kinds), p=ANOMALY_PROBS))]
        if kind == "structuring":     # structuring = a burst of similar payments on one account/day
            k = min(int(rng.integers(2, 4)), n_susp - len(raws))
            acct_idx = STRUCTURING_ACCOUNTS[int(rng.integers(len(STRUCTURING_ACCOUNTS)))]
            day = int(rng.integers(0, EPISODE_DAYS))
            raws += [_raw_tx(rng, accounts, kind, acct_idx, day) for _ in range(k)]
        else:
            raws.append(_raw_tx(rng, accounts, kind))
    # 2) fill the rest with legitimate transactions
    while len(raws) < n_tx:
        raws.append(_raw_tx(rng, accounts, None))

    # 3) sort chronologically, then compute features that depend on the whole batch
    raws.sort(key=lambda r: (r["day"], r["hour"], r["minute"]))
    per_day: dict[tuple, int] = {}
    for r in raws:
        key = (r["acct"]["account_id"], r["day"])
        per_day[key] = per_day.get(key, 0) + 1

    prefix = "T" if split == "train" else "D"
    episode_id = f"{split}-{index:02d}"
    transactions = []
    for k, r in enumerate(raws, start=1):
        acct = r["acct"]
        ratio = r["amount"] / CATEGORY_STATS[r["category"]][0]
        tx = {
            "transaction_id": f"TX-{prefix}{index:02d}-{k:03d}",
            "episode_id": episode_id,
            "date": (start + timedelta(days=r["day"])).isoformat(),
            "time": f"{r['hour']:02d}:{r['minute']:02d}",
            "amount": r["amount"],
            "invoice_amount": r["invoice_amount"],
            "category": r["category"],
            "account_id": acct["account_id"],
            "account_type": acct["account_type"],
            "account_age_days": acct["account_age_days"],
            "account_country": acct["account_country"],
            "tx_country": r["tx_country"],
            "doc_status_initial": r["doc_status_initial"],
            "velocity": per_day[(acct["account_id"], r["day"])],
            "country_mismatch": int(r["tx_country"] != acct["account_country"]),
            "off_hours": int(r["hour"] < 6 or r["hour"] >= 22),
            "round_amount": int(r["amount"] >= 1000 and abs(r["amount"] / 100 - round(r["amount"] / 100)) < 1e-9),
            "near_threshold": int(9000 <= r["amount"] < 10000),
            "new_account": int(acct["account_age_days"] < 60),
            "amount_ratio": round(ratio, 2),
            "label": r["label"],
        }
        tx["risk_score"] = compute_risk_score(tx)
        tx["risk_level"] = risk_level(tx["risk_score"])
        # the "status" column of the transactions table (simple, deterministic-by-seed)
        if tx["risk_level"] == "High":
            tx["status"] = "Under Review"
        else:
            tx["status"] = "Pending" if rng.random() < 0.15 else "Posted"
        transactions.append(tx)

    return {
        "episode_id": episode_id, "split": split, "index": index, "seed": seed,
        "n_transactions": len(transactions),
        "n_suspicious": sum(t["label"]["is_suspicious"] for t in transactions),
        "transactions": transactions,
    }
