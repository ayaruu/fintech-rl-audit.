"""
Audit report builder.

Takes one finished audit run (every step an agent took on one episode) and turns it
into a structured report: anomalies, incorrect transactions, missing documents,
risk assessment and a final summary. Can export as JSON or CSV.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from environment.config import ANOMALY_TYPES, REVIEW_THRESHOLD

# Risk rating rule (explained in the README): share of the audited money that was flagged
RATING_HIGH = 0.20
RATING_MEDIUM = 0.08


def _reasons(tx: dict, info: dict) -> list[str]:
    """Human-readable reasons, built ONLY from information an auditor could see."""
    r = []
    if tx["amount_ratio"] > 3:
        r.append(f"Amount is {tx['amount_ratio']:.1f}x the typical {tx['category']} amount")
    if tx["near_threshold"]:
        r.append(f"Amount just below the {REVIEW_THRESHOLD:,} review limit")
    if tx["country_mismatch"]:
        r.append(f"Transaction country ({tx['tx_country']}) differs from account country ({tx['account_country']})")
    if tx["off_hours"]:
        r.append(f"Booked off-hours ({tx['time']})")
    if tx["velocity"] >= 3:
        r.append(f"{tx['velocity']} transactions on this account on the same day")
    if tx["new_account"]:
        r.append(f"Young account ({tx['account_age_days']} days old)")
    if tx["round_amount"]:
        r.append("Round amount")
    if tx["doc_status_initial"] == "missing":
        outcome = info["doc_outcome"]
        r.append("Supporting document missing" + (f" ({outcome} after request)" if info["docs_requested"] else " (not requested)"))
    if info["investigated"] and info["investigation_result"] == "problem found":
        r.append("Investigation found a problem")
    return r


def build_report(run: dict, episode: dict) -> dict:
    """`run` = {policy, policy_name, split, episode_index, steps, summary}; `episode` = the episode data."""
    txs = {t["transaction_id"]: t for t in episode["transactions"]}

    # 1) collapse the step log into one record per transaction
    per_tx: dict[str, dict] = {}
    for s in run["steps"]:
        d = per_tx.setdefault(s["transaction_id"], {
            "investigated": False, "docs_requested": False, "decision": None, "outcome": None, "reward": 0.0})
        d["reward"] += s["reward"]
        if s["action_name"] == "investigate":
            d["investigated"] = True
        elif s["action_name"] == "request_docs":
            d["docs_requested"] = True
        if s["decision_made"]:
            d["decision"], d["outcome"] = s["action_name"], s["outcome"]

    # 2) one row per transaction
    rows = []
    for tid, tx in txs.items():
        info = per_tx.get(tid, {"investigated": False, "docs_requested": False, "decision": None,
                                "outcome": None, "reward": 0.0})
        label = tx["label"]
        info["investigation_result"] = (
            None if not info["investigated"]
            else ("problem found" if label["investigation_signal"] == "problem" else "looks clean"))
        if tx["doc_status_initial"] != "missing":
            info["doc_outcome"] = "n/a"
        elif info["docs_requested"]:
            info["doc_outcome"] = label["doc_request_outcome"]
        else:
            info["doc_outcome"] = "not requested"
        rows.append({
            "transaction_id": tid, "date": tx["date"], "time": tx["time"], "amount": tx["amount"],
            "invoice_amount": tx["invoice_amount"], "category": tx["category"], "account_id": tx["account_id"],
            "risk_score": tx["risk_score"], "risk_level": tx["risk_level"],
            "decision": info["decision"] or "undecided",
            "investigated": info["investigated"], "docs_requested": info["docs_requested"],
            "doc_status_initial": tx["doc_status_initial"], "doc_outcome": info["doc_outcome"],
            "reasons": _reasons(tx, info),
            "synthetic_truth": "suspicious" if label["is_suspicious"] else "legitimate",
            "anomaly_type_truth": ANOMALY_TYPES.get(label["anomaly_type"]) if label["anomaly_type"] else None,
            "outcome": info["outcome"], "reward": round(info["reward"], 2),
        })

    flagged = [r for r in rows if r["decision"] == "flag"]
    total_amount = round(sum(r["amount"] for r in rows), 2)
    flagged_amount = round(sum(r["amount"] for r in flagged), 2)
    pct = flagged_amount / total_amount if total_amount else 0.0
    rating = "High" if pct >= RATING_HIGH else "Medium" if pct >= RATING_MEDIUM else "Low"
    m = run["summary"]

    anomalies = sorted(
        [{**{k: r[k] for k in ("transaction_id", "date", "amount", "category", "account_id", "risk_score", "risk_level",
                               "reasons", "synthetic_truth", "anomaly_type_truth")},
          "correct": r["outcome"] == "true_positive"} for r in flagged],
        key=lambda a: (-a["risk_score"], -a["amount"]))

    incorrect = [{
        "transaction_id": r["transaction_id"], "date": r["date"], "category": r["category"],
        "ledger_amount": r["amount"], "invoice_amount": r["invoice_amount"],
        "difference": round(r["invoice_amount"] - r["amount"], 2),
        "difference_pct": round((r["invoice_amount"] - r["amount"]) / r["amount"] * 100, 1),
        "decision": r["decision"], "caught": r["decision"] == "flag",
    } for r in rows if abs(r["invoice_amount"] - r["amount"]) > 0.005]

    missing_docs = [{
        "transaction_id": r["transaction_id"], "date": r["date"], "amount": r["amount"], "category": r["category"],
        "requested": r["docs_requested"], "doc_outcome": r["doc_outcome"], "decision": r["decision"],
    } for r in rows if r["doc_status_initial"] == "missing"]

    missed = [{k: r[k] for k in ("transaction_id", "date", "amount", "category", "risk_level", "anomaly_type_truth")}
              for r in rows if r["outcome"] == "false_negative"]

    by_category = []
    for cat in sorted({r["category"] for r in rows}):
        cat_rows = [r for r in rows if r["category"] == cat]
        cat_flagged = [r for r in cat_rows if r["decision"] == "flag"]
        by_category.append({"category": cat, "transactions": len(cat_rows), "flagged": len(cat_flagged),
                            "flagged_amount": round(sum(r["amount"] for r in cat_flagged), 2)})

    narrative = (
        f"{run['policy_name']} audited {len(rows)} transactions in episode {run['episode_id']} "
        f"({run['split']} set) and flagged {len(flagged)} of them, worth {flagged_amount:,.2f} "
        f"({pct * 100:.1f}% of the total {total_amount:,.2f}). "
        f"Accuracy {m['accuracy'] * 100:.1f}%, precision {m['precision'] * 100:.1f}%, recall {m['recall'] * 100:.1f}%. "
        f"Against the synthetic ground truth, {m['true_positive']} suspicious transactions were caught, "
        f"{m['false_negative']} were missed and {m['false_positive']} legitimate ones were flagged by mistake. "
        f"The agent investigated {m['investigations']} times and requested documents {m['doc_requests']} times, "
        f"for a total reward of {m['total_reward']}. Overall risk rating: {rating}."
    )
    if m.get("undecided"):
        narrative += f" Note: {m['undecided']} transactions were left undecided because the episode was cut off."

    return {
        "meta": {
            "episode_id": run["episode_id"], "split": run["split"], "policy": run["policy"],
            "policy_name": run["policy_name"], "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_note": "Synthetic data only. Hackathon prototype, not a real audit.",
        },
        "summary": {
            "transactions": len(rows), "flagged": len(flagged),
            "approved": sum(r["decision"] == "approve" for r in rows),
            "undecided": sum(r["decision"] == "undecided" for r in rows),
            "total_amount": total_amount, "flagged_amount": flagged_amount,
            "flagged_amount_pct": round(pct * 100, 1), "risk_rating": rating,
            "metrics": m, "narrative": narrative,
        },
        "anomalies": anomalies,
        "incorrect_transactions": incorrect,
        "missing_documents": missing_docs,
        "missed_suspicious_transactions": missed,
        "risk_assessment": {
            "rating": rating,
            "rule": (f"High if flagged amount >= {RATING_HIGH * 100:.0f}% of the audited amount, "
                     f"Medium if >= {RATING_MEDIUM * 100:.0f}%, otherwise Low."),
            "flagged_amount": flagged_amount, "total_amount": total_amount,
            "flagged_amount_pct": round(pct * 100, 1),
            "by_category": by_category,
        },
        "decisions": rows,
    }


def report_to_json(report: dict) -> str:
    return json.dumps(report, indent=2)


def report_to_csv(report: dict) -> str:
    """CSV with one row per transaction (the full decision table)."""
    df = pd.DataFrame(report["decisions"])
    df["reasons"] = df["reasons"].apply(lambda r: "; ".join(r))
    return df.to_csv(index=False)
