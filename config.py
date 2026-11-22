"""
Shared constants for the FinTech RL Audit project.

Everything that several files need to agree on (action ids, reward values,
observation layout, data-generation settings) lives here, so there is exactly
ONE place to change it.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# ACTIONS
# --------------------------------------------------------------------------
APPROVE, FLAG, REQUEST_DOCS, INVESTIGATE = 0, 1, 2, 3
ACTION_NAMES = ["approve", "flag", "request_docs", "investigate"]
ACTION_IDS = {name: i for i, name in enumerate(ACTION_NAMES)}
N_ACTIONS = len(ACTION_NAMES)
ACTION_LABELS = {
    "approve": "Approve transaction",
    "flag": "Flag transaction",
    "request_docs": "Request additional documentation",
    "investigate": "Investigate transaction",
}
# approve / flag END the current transaction (a final decision).
# request_docs / investigate only GATHER INFORMATION (the transaction stays open).
DECISION_ACTIONS = (APPROVE, FLAG)

# --------------------------------------------------------------------------
# REWARD FUNCTION  (see README.md -> "Reward function")
# --------------------------------------------------------------------------
REWARDS = {
    "true_positive": 10.0,        # flagged a transaction that really was suspicious
    "true_negative": 2.0,         # approved a legitimate transaction
    "false_positive": -3.0,       # flagged a legitimate transaction (wasted audit work)
    "false_negative": -10.0,      # approved a suspicious transaction (missed fraud / error)
    "investigate": -1.0,          # cost of the first investigation of a transaction
    "request_docs": -0.3,         # cost of asking for missing documents
    "repeat_investigate": -1.5,   # unnecessary: investigating the same transaction again
    "unnecessary_request": -0.5,  # unnecessary: documents were not missing / already requested
}
REWARD_DESCRIPTIONS = {
    "true_positive": "Flag a suspicious transaction (correct catch)",
    "true_negative": "Approve a legitimate transaction (correct approval)",
    "false_positive": "Flag a legitimate transaction (false alarm)",
    "false_negative": "Approve a suspicious transaction (missed it)",
    "investigate": "Investigate a transaction (first time)",
    "request_docs": "Request documents that really are missing",
    "repeat_investigate": "Investigate the same transaction again (unnecessary)",
    "unnecessary_request": "Request documents that are not missing / already requested",
}

# --------------------------------------------------------------------------
# OBSERVATION LAYOUT  (every value is scaled to the range 0..1)
# --------------------------------------------------------------------------
OBS_FIELDS = [
    "amount",               # log-scaled transaction amount
    "category",             # category index / (n_categories - 1)
    "account_age",          # account age in days, capped at 365, / 365
    "doc_status",           # 0 available, 1/3 missing, 2/3 received, 1 refused
    "amount_ratio",         # amount / typical amount of the category, capped at 10, / 10
    "velocity",             # transactions on the same account that day (cap 6) / 6
    "country_mismatch",     # 1 if transaction country differs from account country
    "off_hours",            # 1 if booked between 22:00 and 06:00
    "round_amount",         # 1 if amount >= 1000 and a multiple of 100
    "near_threshold",       # 1 if amount is between 9,000 and 10,000 (review limit)
    "risk_score",           # heuristic risk score 0..1 (see data_generator.compute_risk_score)
    "investigation_result", # 0 not investigated, 0.5 looks clean, 1 problem found
    "account_prior_flags",  # earlier flagged transactions on this account (cap 3) / 3
    "history_flag_rate",    # share of already-decided transactions that were flagged
    "progress",             # position in the episode (0..1)
]
OBS_DESCRIPTIONS = {
    "amount": "Transaction amount (log scale)",
    "category": "Transaction category",
    "account_age": "Account age",
    "doc_status": "Document availability",
    "amount_ratio": "Amount compared with the category norm",
    "velocity": "Same-day transactions on this account",
    "country_mismatch": "Country differs from the account's home country",
    "off_hours": "Booked outside business hours",
    "round_amount": "Suspiciously round amount",
    "near_threshold": "Just below the 10,000 review limit",
    "risk_score": "Heuristic risk score",
    "investigation_result": "Result of investigation (if done)",
    "account_prior_flags": "Earlier flags on this account (history)",
    "history_flag_rate": "Share of transactions flagged so far (history)",
    "progress": "Episode progress",
}
OBS_INDEX = {name: i for i, name in enumerate(OBS_FIELDS)}
N_OBS = len(OBS_FIELDS)

DOC_STATUS_NAMES = ["available", "missing", "received", "refused"]
MAX_STEPS_PER_TX = 6   # an episode is cut off (truncated) after n_transactions * 6 steps

# --------------------------------------------------------------------------
# SYNTHETIC DATA SETTINGS
# --------------------------------------------------------------------------
CATEGORIES = [
    "Payroll", "Vendor Payment", "Travel & Expenses", "Utilities",
    "Office Supplies", "Software Subscription", "Client Refund", "International Wire",
]
# (typical amount, spread) per category -> amounts are log-normal around the typical value
CATEGORY_STATS = {
    "Payroll": (4500.0, 0.25),
    "Vendor Payment": (2800.0, 0.60),
    "Travel & Expenses": (450.0, 0.50),
    "Utilities": (600.0, 0.35),
    "Office Supplies": (150.0, 0.50),
    "Software Subscription": (900.0, 0.40),
    "Client Refund": (350.0, 0.60),
    "International Wire": (6000.0, 0.50),
}
CATEGORY_PROBS = [0.14, 0.20, 0.16, 0.10, 0.12, 0.10, 0.08, 0.10]

ANOMALY_TYPES = {
    "large_amount": "Unusually large amount",
    "structuring": "Amounts just below the review limit",
    "record_mismatch": "Ledger amount differs from invoice",
    "missing_document": "Supporting document missing",
    "foreign_offhours": "Foreign, off-hours payment from a new account",
}
ANOMALY_PROBS = [0.25, 0.20, 0.20, 0.15, 0.20]

TX_PER_EPISODE = 40
EPISODE_DAYS = 14
N_TRAIN_EPISODES = 10
N_DEV_EPISODES = 10
TRAIN_SEED_BASE = 1000
DEV_SEED_BASE = 2000
POLICY_SEED = 42          # fixed seed used by every policy and by training
REVIEW_THRESHOLD = 10_000  # synthetic "must be reviewed" payment limit


def risk_level(score: float) -> str:
    """Convert a 0..1 risk score into a label used in the tables."""
    if score >= 0.50:
        return "High"
    if score >= 0.25:
        return "Medium"
    return "Low"


def obs_to_dict(obs) -> dict:
    """Turn an observation vector into {field_name: value} (handy for the UI)."""
    return {name: round(float(obs[i]), 4) for i, name in enumerate(OBS_FIELDS)}
