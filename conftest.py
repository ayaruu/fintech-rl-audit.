import copy
import json
from pathlib import Path

import pytest

from environment.audit_env import AuditEnv

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def episodes():
    d = ROOT / "data"
    return {"train": json.loads((d / "train_episodes.json").read_text()),
            "dev": json.loads((d / "dev_episodes.json").read_text())}


@pytest.fixture()
def tiny_episodes(episodes):
    """One episode with 2 transactions: #0 legitimate (docs available), #1 suspicious. Fully known labels."""
    ep = copy.deepcopy(episodes["dev"][0])
    txs = ep["transactions"][:2]
    txs[0]["label"].update(is_suspicious=False, anomaly_type=None)
    txs[0]["doc_status_initial"] = "available"
    txs[1]["label"].update(is_suspicious=True, anomaly_type="large_amount")
    ep["transactions"] = txs
    return [ep]


@pytest.fixture()
def tiny_env(tiny_episodes):
    env = AuditEnv(tiny_episodes)
    env.reset(seed=0)
    return env


@pytest.fixture(scope="session")
def service():
    from backend.services import AuditService
    return AuditService()
