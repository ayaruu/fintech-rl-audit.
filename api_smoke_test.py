"""
Live API smoke test. Works against a RUNNING server using only the Python standard library.

    1) in one terminal:   python run.py
    2) in another:        python scripts/api_smoke_test.py            (optional: --base http://localhost:8000)

Exits with code 1 if any check fails.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://localhost:8000")
BASE = ap.parse_args().base.rstrip("/")
results = []


def call(method, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BASE + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read().decode()


def check(name, cond, extra=""):
    results.append(bool(cond))
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


def j(txt):
    return json.loads(txt)


s, _, t = call("GET", "/api/health"); check("health", s == 200 and j(t)["status"] == "ok")
s, _, t = call("GET", "/"); check("website index served", s == 200 and "AuditRL" in t)
for f in ("app.js", "charts.js", "styles.css"):
    s, _, _ = call("GET", "/" + f); check(f"static file {f}", s == 200)
s, _, t = call("GET", "/api/meta"); m = j(t)
check("meta: 4 policies, 4 actions, 20 episodes", len(m["policies"]) == 4 and len(m["actions"]) == 4 and
      len(m["episodes"]["train"]) == 10 and len(m["episodes"]["dev"]) == 10)
for split in ("dev", "train"):
    s, _, t = call("GET", f"/api/dashboard?split={split}&policy=rl_agent"); d = j(t)
    check(f"dashboard {split}", s == 200 and d["total_transactions"] == 800 and d["audited_transactions"] == 400)
s, _, t = call("GET", "/api/compare?split=dev"); c = j(t)
check("compare: 4 policies with all metrics", s == 200 and len(c["policies"]) == 4 and
      all(k in p for p in c["policies"] for k in ("total_reward", "accuracy", "precision", "recall", "investigations")))
s, _, t = call("GET", "/api/training-log"); check("training log", s == 200 and len(j(t)["log"]) > 100)
s, _, t = call("GET", "/api/transactions?limit=5"); check("transactions list", s == 200 and j(t)["total"] == 800 and j(t)["returned"] == 5)
s, _, t = call("GET", "/api/transactions?risk=High&search=wire&split=dev")
check("transactions filters", s == 200 and all(x["risk_level"] == "High" and x["split"] == "dev" for x in j(t)["transactions"]))
check("bad split -> 400", call("GET", "/api/dashboard?split=bogus")[0] == 400)
check("bad policy -> 400", call("GET", "/api/dashboard?policy=bogus")[0] == 400)

# automatic episode
s, _, t = call("POST", "/api/episode/start", {"split": "dev", "episode_index": 0, "policy": "rl_agent"})
check("start episode", s == 200 and j(t)["active"] and j(t)["status"] == "running")
s, _, t = call("POST", "/api/episode/step", {}); check("step by policy", s == 200 and j(t)["step"]["step"] == 1)
s, _, t = call("POST", "/api/episode/step", {"action": "flag"}); check("manual action on policy episode -> 400", s == 400)
s, _, t = call("POST", "/api/episode/run"); r = j(t)
check("run to end", s == 200 and r["state"]["done"] and r["state"]["decided"] == 40)
best = next(p for p in c["policies"] if p["policy"] == "rl_agent")["per_episode_rewards"][0]
check("session reward == batch evaluation reward", r["state"]["total_reward"] == best, f'{r["state"]["total_reward"]} vs {best}')
check("step after finish -> 409", call("POST", "/api/episode/step", {})[0] == 409)
check("current episode completed", j(call("GET", "/api/episode/current")[2])["status"] == "completed")
s, _, t = call("GET", "/api/report?source=session"); rep = j(t)
check("report sections", s == 200 and all(k in rep for k in ("anomalies", "incorrect_transactions", "missing_documents", "risk_assessment", "summary")))
check("report flagged == TP+FP", rep["summary"]["flagged"] == rep["summary"]["metrics"]["true_positive"] + rep["summary"]["metrics"]["false_positive"])
s, h, t = call("GET", "/api/report/export?format=csv&source=session")
check("CSV export", s == 200 and "attachment" in str(h).lower() and t.count("\n") == 41)
s, h, t = call("GET", "/api/report/export?format=json&source=session")
check("JSON export", s == 200 and j(t)["meta"]["episode_id"] == "dev-00")
check("bad export format -> 400", call("GET", "/api/report/export?format=xml&source=session")[0] == 400)
check("report for chosen episode/policy", call("GET", "/api/report?source=episode&split=train&episode_index=3&policy=rule_based")[0] == 200)
check("report bad episode -> 400", call("GET", "/api/report?source=episode&episode_index=99")[0] == 400)

# manual play + action validation
call("POST", "/api/episode/start", {"split": "dev", "episode_index": 1, "policy": "manual"})
for bad in ("dance", 9, -1, 1.5):
    check(f"invalid action {bad!r} -> 400", call("POST", "/api/episode/step", {"action": bad})[0] == 400)
check("step without action on manual -> 400", call("POST", "/api/episode/step", {})[0] == 400)
s, _, t = call("POST", "/api/episode/step", {"action": "investigate"}); check("manual investigate", s == 200 and j(t)["step"]["reward"] == -1.0)
s, _, t = call("POST", "/api/episode/step", {"action": 1}); check("manual flag (numeric action)", s == 200 and j(t)["step"]["decision_made"])
check("run on manual episode -> 400", call("POST", "/api/episode/run")[0] == 400)
check("bad start policy -> 400", call("POST", "/api/episode/start", {"policy": "nope"})[0] == 400)
check("bad start episode -> 400", call("POST", "/api/episode/start", {"episode_index": 77})[0] == 400)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
