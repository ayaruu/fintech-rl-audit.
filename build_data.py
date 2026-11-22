"""
Generates the synthetic datasets, trains the Q-learning agent, evaluates the four
policies and writes trajectories.jsonl.  Safe to run again at any time: the seeds
are fixed, so you always get exactly the same files.

    python build_data.py
"""
from pathlib import Path

from environment.pipeline import build_all

if __name__ == "__main__":
    build_all(Path(__file__).resolve().parent)
