"""
Metrics used everywhere (environment summary, policy comparison, reports).
All numbers shown in the website come from these functions, so they always
agree with each other and with trajectories.jsonl.
"""
from __future__ import annotations

OUTCOMES = ("true_positive", "false_positive", "true_negative", "false_negative")


def _from_counts(counts: dict, investigations: int, doc_requests: int, total_reward: float) -> dict:
    tp, fp = counts["true_positive"], counts["false_positive"]
    tn, fn = counts["true_negative"], counts["false_negative"]
    n = tp + fp + tn + fn
    return {
        "decisions": n,
        "true_positive": tp, "false_positive": fp, "true_negative": tn, "false_negative": fn,
        "accuracy": round((tp + tn) / n, 4) if n else 0.0,          # correct decisions / all decisions
        "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,  # of the flagged, how many were truly suspicious
        "recall": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,     # of the suspicious, how many were flagged
        "investigations": investigations,
        "doc_requests": doc_requests,
        "total_reward": round(total_reward, 2),
    }


def compute_metrics(decisions: list[dict], investigations: int, doc_requests: int, total_reward: float) -> dict:
    """Metrics for ONE episode. `decisions` is a list of dicts with an 'outcome' key."""
    counts = {o: 0 for o in OUTCOMES}
    for d in decisions:
        counts[d["outcome"]] += 1
    return _from_counts(counts, investigations, doc_requests, total_reward)


def aggregate(summaries: list[dict]) -> dict:
    """Combine several episode summaries (counts are summed, ratios recomputed)."""
    counts = {o: sum(s[o] for s in summaries) for o in OUTCOMES}
    out = _from_counts(
        counts,
        sum(s["investigations"] for s in summaries),
        sum(s["doc_requests"] for s in summaries),
        sum(s["total_reward"] for s in summaries),
    )
    out["episodes"] = len(summaries)
    out["mean_reward"] = round(out["total_reward"] / len(summaries), 2) if summaries else 0.0
    return out
