from environment.config import N_DEV_EPISODES, N_TRAIN_EPISODES, TX_PER_EPISODE
from environment.data_generator import make_episode


def test_episode_counts(episodes):
    assert len(episodes["train"]) == N_TRAIN_EPISODES == 10
    assert len(episodes["dev"]) == N_DEV_EPISODES == 10
    for split in episodes.values():
        for ep in split:
            assert len(ep["transactions"]) == TX_PER_EPISODE


def test_data_contains_all_required_kinds(episodes):
    txs = [t for sp in episodes.values() for e in sp for t in e["transactions"]]
    assert any(t["label"]["is_suspicious"] for t in txs)
    assert any(not t["label"]["is_suspicious"] for t in txs)
    assert any(t["doc_status_initial"] == "missing" for t in txs)               # missing documents
    assert any(t["invoice_amount"] != t["amount"] for t in txs)                 # inconsistent records
    assert len({t["category"] for t in txs}) >= 6                                # different transaction types


def test_generation_is_reproducible(episodes):
    assert make_episode("dev", 3) == episodes["dev"][3]


def test_train_and_dev_differ(episodes):
    assert episodes["train"][0]["transactions"] != episodes["dev"][0]["transactions"]


def test_files_exist():
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    for f in ["trajectories.jsonl", "data/train_episodes.json", "data/dev_episodes.json", "data/q_table.json"]:
        assert (root / f).exists(), f
