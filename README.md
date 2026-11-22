# FinTech RL Environment for Audit

**A reinforcement-learning environment for financial auditing, with a Python/FastAPI backend and an interactive dashboard.**
An AI agent examines synthetic financial transactions, can investigate them or request missing documents, and then approves or flags each one. A reward function scores every decision, and four policies can be compared on the same episodes.

> **Synthetic data only.** This is a hackathon prototype, not a real banking or audit system.

## Purpose

Auditors review thousands of transactions and must balance two risks: missing real problems (costly) and raising false alarms (wasteful). We built a safe, reproducible *playground* where audit strategies, from simple rules to learning agents, can be tried, scored and compared fairly. The environment follows the standard Gymnasium interface, so other RL agents can be plugged in later.

## Main features

* **Custom Gymnasium environment** (`AuditEnv`) with `reset()`, `step()`, `observation_space`, `action_space` and a documented reward function.
* **Synthetic datasets:** 10 training + 10 development episodes (800 transactions) with normal and suspicious transactions, missing documents and inconsistent records, all generated with fixed seeds.
* **Four policies** (Random, Rule-Based, Risk-Threshold, RL agent) compared by total reward, accuracy, precision, recall and number of investigations.
* **Dashboard website** with five pages: Dashboard, Transactions (search and filters), AI Audit Agent (run an agent or play manually), Audit Reports (export JSON/CSV) and RL Environment (live state and diagram).
* **One-click demo mode** that walks through the six scenes needed for a 1-2 minute video.
* **`trajectories.jsonl`:** every step (observation, action, reward, outcome) of every policy on every episode.
* **Automated tests** (pytest) and a live API smoke-test script.

## Technologies used

| Area | Tools |
|---|---|
| Backend | Python 3.10+, FastAPI, Uvicorn |
| RL | Gymnasium, NumPy, tabular Q-learning (own implementation) |
| Data | Pandas, NumPy, JSON / JSONL / CSV files |
| Frontend | HTML, CSS, vanilla JavaScript, hand-written SVG charts (no build step, no CDN) |
| Testing | pytest, httpx (FastAPI TestClient) |

---

## 1. Quick start (5 minutes)

You need **Python 3.10 or newer** (check with `python --version`) and Git.

```bash
# 0. get the code
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# 1. make sure the terminal is in this folder (the one containing README.md)
python -m venv .venv

# 2. activate it
#    Windows:      .venv\Scripts\activate
#    Mac / Linux:  source .venv/bin/activate

# 3. install everything
pip install -r requirements.txt

# 4. start the website (backend + frontend together)
python run.py
```

Open **http://localhost:8000** in your browser. Done. The interactive API docs are at http://localhost:8000/docs.

All data files are already included. If you ever delete them, they are rebuilt automatically at start-up (takes ~5 s), or on demand with `python build_data.py`.

Run the tests: `python -m pytest`

---

## 2. Project structure

```
fintech-rl-audit/
├── README.md
├── requirements.txt
├── pytest.ini
├── run.py                    # starts the website:  python run.py
├── build_data.py             # regenerates datasets, trains the agent, writes trajectories.jsonl
├── trajectories.jsonl        # every step of every policy on all 20 episodes
├── environment/              # the RL part (no web code)
│   ├── config.py             # actions, REWARDS, observation layout, data settings
│   ├── data_generator.py     # synthetic transactions (fixed seeds)
│   ├── audit_env.py          # AuditEnv: the custom Gymnasium environment
│   ├── policies.py           # Random, Rule-Based, Risk-Threshold, RL policy
│   ├── rl_agent.py           # tabular Q-learning agent
│   ├── runner.py             # runs policies in the env, records trajectories
│   ├── metrics.py            # accuracy / precision / recall / reward
│   ├── pipeline.py           # builds all data files
│   └── _compat.py            # imports gymnasium (small fallback if it is missing)
├── backend/
│   ├── main.py               # FastAPI endpoints + serves the frontend
│   ├── services.py           # all logic behind the endpoints
│   └── reports.py            # audit report builder, JSON/CSV export
├── frontend/                 # plain HTML / CSS / JavaScript, no build step
│   ├── index.html  styles.css  app.js  charts.js
├── scripts/
│   └── api_smoke_test.py     # live API check against a running server (stdlib only)
├── data/
│   ├── train_episodes.json   dev_episodes.json   transactions.csv
│   ├── q_table.json          training_log.json
│   └── evaluation_summary.json   dataset_summary.json
└── tests/                    # pytest suite
```

**Architecture in one picture**

```
Browser (frontend/*.js) --HTTP/JSON--> FastAPI (backend/main.py) --> services.py --> environment/ (AuditEnv, policies, agent)
                                                                              \--> data/*.json, trajectories.jsonl
```

Why this design: one `python run.py` command starts everything, the frontend needs no Node/npm, and the RL code is independent of the web code, so it can be tested and reused alone.

---

## 3. The RL environment (`environment/audit_env.py`)

A custom [Gymnasium](https://gymnasium.farama.org/) environment (`reset()`, `step()`, `observation_space`, `action_space`).

* **Episode** = a batch of 40 transactions (14 days of activity). The agent handles one transaction at a time.
* **Action space** `Discrete(4)`: `0 approve`, `1 flag`, `2 request_docs`, `3 investigate`.
  Approve/flag are final decisions and move to the next transaction. Request-docs/investigate only gather information.
* **Observation space** `Box(0, 1, shape=(15,))`: amount, category, account age, document availability, amount vs. category norm, same-day velocity, country mismatch, off-hours, round amount, just-below-limit, risk score, investigation result, earlier flags on the account, share flagged so far (history), episode progress. (Table in the *RL Environment* page.)
* **Hidden truth**: each transaction has a hidden label (suspicious or not). The agent never sees it. *Investigating* reveals a noisy hint (correct 90% of the time for suspicious, 95% for legitimate transactions). *Requesting documents* turns "missing" into "received" or "refused".
* **Termination**: the episode ends after the last transaction is decided. It is *truncated* if the agent wastes more than `40 × 6` steps.
* Invalid actions raise `ValueError`.

### Reward function

| Situation | Reward |
|---|---|
| Flag a suspicious transaction (true positive) | **+10** |
| Approve a legitimate transaction (true negative) | **+2** |
| Flag a legitimate transaction (false positive) | **−3** |
| Approve a suspicious transaction (missed, false negative) | **−10** |
| Investigate (first time on a transaction) | −1.0 |
| Request documents that really are missing | −0.3 |
| Investigate again (unnecessary) | −1.5 |
| Request documents that are not missing / already requested (unnecessary) | −0.5 |

Missing a suspicious transaction is punished hardest because that is the costliest audit error. Information gathering costs a little, so the agent must learn *when it is worth it*. All values live in `environment/config.py` (`REWARDS`).

---

## 4. The four policies (`environment/policies.py`)

1. **Random**: picks any of the 4 actions at random (seeded). A floor to beat.
2. **Rule-Based**: fixed audit rules: request missing documents; flag refused documents; flag amounts above 3× the category norm; flag foreign off-hours payments; flag amounts just below the 10,000 limit; otherwise approve. Never investigates.
3. **Risk-Threshold**: risk ≥ 0.50 → flag; 0.25–0.50 → investigate once, flag only if a problem is found; < 0.25 → approve.
4. **RL Agent**: **tabular Q-learning**, trained on the 10 *training* episodes only (300 epochs, ε-greedy, fixed seed 42) and evaluated on the 10 *development* episodes it has never seen. State = risk bucket, document status, investigation result, big-amount flag, earlier-flags flag (240 states × 4 actions).

All policies and training use fixed seeds (`POLICY_SEED = 42`, data seeds 1000+i / 2000+i), so every result is reproducible.

### Results (development set, 10 unseen episodes, 400 transactions)

| Policy | Total reward | Accuracy | Precision | Recall | Investigations |
|---|---:|---:|---:|---:|---:|
| Random | −501.7 | 49.3% | 20.0% | 51.2% | 206 |
| Rule-Based | **1171.8** | 93.3% | 79.1% | 90.0% | 0 |
| Risk-Threshold | 1007.0 | **94.3%** | **90.1%** | 80.0% | 78 |
| RL Agent (Q-learning) | 919.5 | 90.8% | 69.4% | **96.3%** | 289 |

(Percentages are rounded half-up, exactly as the website shows them; exact 4-decimal values are in `data/evaluation_summary.json`, which also holds the training-set results. The website can show both sets.)

**Honest reading of these results**

* The RL agent clearly **beats random** and has the **best recall** (it misses the fewest suspicious transactions).
* It does **not** beat the hand-written rules on total reward. It investigates too often (289 times; each costs −1) and flags too many legitimate transactions (lower precision). The synthetic anomalies are fairly easy for hand-written rules, and the Q-table only sees a coarse summary of each transaction.
* We tried a richer state in cross-validation on the training episodes only; it gave no meaningful gain, so we kept the simpler one. We did not tune the rewards to make RL win.
* So: this is a **small trained Q-learning agent, not a state-of-the-art model**. The value of the project is the environment and the evaluation framework, which make that comparison possible.

Every number on the website is computed by the same code (`environment/runner.py` + `metrics.py`) that writes `trajectories.jsonl`; a test checks that they agree.

---

## 5. Datasets

Generated by `environment/data_generator.py` (fake data, fixed seeds).

| | Train | Dev |
|---|---:|---:|
| Episodes | 10 | 10 |
| Transactions | 400 | 400 |
| Suspicious | 80 | 80 |
| Missing documents | 58 | 44 |
| Inconsistent records (ledger ≠ invoice) | 11 | 11 |

* **8 transaction categories** (payroll, vendor payment, travel, utilities, office supplies, software, client refund, international wire) with realistic amount ranges.
* **5 anomaly types**: unusually large amount; structuring (several amounts just under 10,000); record mismatch (ledger vs. invoice differ 8–40%); missing document; foreign off-hours payment from a young account.
* Legitimate data contains realistic noise (big but legitimate purchases, round payroll amounts, travel abroad, missing documents that arrive on request), so risk indicators are imperfect.
* Files: `data/train_episodes.json`, `data/dev_episodes.json` (full episodes with hidden labels), `data/transactions.csv` (flat table).
* `trajectories.jsonl`: 4,934 lines. For each policy × each of the 20 episodes: one `"type":"step"` line per step (observation vector, action, reward, running total, outcome) and one `"type":"episode_result"` line with the episode metrics.

---

## 6. The website

| Page | What it does |
|---|---|
| **Dashboard** | KPIs (total / audited / suspicious transactions, accuracy, total reward), reward-per-episode chart, learning curve, policy comparison table. Choose policy and train/dev set. |
| **Transactions** | 800 synthetic transactions with search, filters (set, episode, category, risk, status), paging, optional *ground truth* column. |
| **AI Audit Agent** | Start an episode, run any of the 4 policies (animated) or **play yourself** (approve / flag / request docs / investigate). Shows decisions, rewards, flagged transactions, final result. |
| **Audit Reports** | Anomalies with reasons, incorrect records, missing documents, risk assessment, summary. **Export JSON / CSV.** |
| **RL Environment** | Live episode, step, observation, action, reward, total reward, status + a diagram of how the environment works. |

**Risk rating rule:** High if flagged amount ≥ 20% of the audited amount, Medium if ≥ 8%, otherwise Low.

### API endpoints (see `/docs`)

`GET /api/health · /api/meta · /api/dashboard · /api/compare · /api/training-log · /api/transactions · /api/episode/current · /api/report · /api/report/export`
`POST /api/episode/start · /api/episode/step · /api/episode/run`

The server keeps **one** active episode in memory (fine for a demo; restarting resets it).

---

## 7. How to test every main feature

1. **Automated tests:** `python -m pytest` runs the whole suite (environment init, reset/step, action validation, reward calculations, episode termination, truncation, the four policies, dataset checks, service logic, and API endpoints in `tests/test_api.py`).
   The API tests need `fastapi` and `httpx`; they are skipped automatically if those are missing, so after `pip install -r requirements.txt` you should see **no skipped tests**.
1. **Live API check:** start the server (`python run.py`), then in a second terminal run `python scripts/api_smoke_test.py`. It makes ~38 real HTTP requests (all endpoints, error codes 400/404/409, CSV/JSON export, and checks that the website numbers equal the batch evaluation) and prints PASS/FAIL for each.
2. **Dashboard:** open the site; 5 KPI cards and 2 charts appear; hover the chart for values; switch policy / dataset.
3. **Transactions:** type `wire` in search, set risk to *High*, tick *Show synthetic ground truth*.
4. **Agent:** pick *RL Agent*, click *Start audit episode*, then *Step once* a few times, then *Run agent*. Rewards add up; flagged transactions are listed with a ✔/✘ check.
5. **Manual play:** choose *Me (manual play)*, start, press the four action buttons. Try a bad request in `/docs` (action `"dance"`) and get a 400 error.
6. **Reports:** click *View audit report →*; export JSON and CSV.
7. **Environment page:** watch step / reward / observation change while stepping.
8. **Comparison:** Dashboard → *Policy comparison* (4 rows, best reward highlighted).

---

## 8. Recording the 1–2 minute demo video

Click **▶ Start demo mode** in the sidebar. It automatically walks through the six required scenes with on-screen captions (about 50 seconds):

1. Dashboard · 2. Transactions (search/filters) · 3. Running the audit agent · 4. Suspicious transactions detected · 5. Comparing the four policies · 6. Final audit report.

Tips: use a 1440×900 browser window, start screen recording (Windows: `Win+G`, Mac: `Shift+Cmd+5`) *then* click the demo button; narrate over it. Press **Stop demo** any time. For a longer video, add a manual-play segment (Agent page) before the demo.

---

## 9. Uploading the repository to GitHub

1. Create an **empty** repository on github.com (**New repository**; do *not* tick "Add a README" or ".gitignore").
2. In a terminal, inside the project folder:

```bash
git init -b main
git add -A
git status                      # check: no .venv, no __pycache__, no secrets
git commit -m "FinTech RL Environment for Audit"
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
git push -u origin main
```

3. If Git asks who you are: `git config --global user.name "Your Name"` and `git config --global user.email "you@example.com"`.
4. GitHub no longer accepts account passwords for `git push`. Sign in through the browser prompt, GitHub Desktop, or use a Personal Access Token as the password.
5. To check your submission works for others: `git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git`, then follow section 1.

The project contains **no secrets or API keys** and needs no `.env` file.

## 10. What was verified (and how)

Development was done in a sandbox where `fastapi`, `gymnasium` and `httpx` could not be installed, so please re-run the checks on your own machine after installing the requirements:

* `python -m pytest` and `python scripts/api_smoke_test.py` (see section 7).
* In the sandbox the environment ran on a tiny built-in stand-in for Gymnasium (same method names) and the API ran on a minimal stand-in for FastAPI. The code is written for the real libraries, but **it has not yet been executed against the real `gymnasium` / `fastapi` packages**. If something differs on your machine, tell us the error message.
* Regenerating the data with `python build_data.py` produces byte-identical files (checked), so results are reproducible.

## 11. Limitations (please be upfront about these)

* **Synthetic data only**, generated by us; real audit data is far messier. Rules and the RL agent are fitted to this generator's patterns.
* The RL agent is a **small tabular Q-learning agent** trained on only 10 episodes (400 transactions). It does not beat the rule-based baseline on total reward (see §4). We did **not** use Stable-Baselines3: for a problem this small a Q-table is simpler, trains in seconds and is easier to explain; SB3 would need PyTorch and a lot more data to be worthwhile.
* Investigation and document results are simulated lookups (noisy hints), not real document analysis.
* The risk score is a hand-made heuristic.
* Single-user server state; no login, no database, not hardened for production.
* The "ground truth" views exist only because the data is synthetic; a real audit never has them.

## 12. Ideas for improvement

* Train a neural agent (e.g. PPO from Stable-Baselines3) on many more generated episodes and compare it to the Q-table.
* Richer state/features and a tuned investigation cost.
* More anomaly types (duplicate payments, vendor collusion), and multi-day account history.
* Persist sessions in a database; support several users.
