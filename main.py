"""
FastAPI backend.

Start it with:   python run.py        (then open http://localhost:8000)

The same server also serves the website (the files in /frontend), so beginners
only need ONE command. Interactive API docs: http://localhost:8000/docs
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .services import AuditService

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="FinTech RL Environment for Audit", version="1.0.0",
              description="Hackathon prototype. Synthetic data only.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

service = AuditService()   # loads the datasets and the trained agent once at start-up


# ---------------------------------------------------------------- helpers
def _call(fn, *args, **kwargs):
    """Run a service method and convert Python errors into proper HTTP errors."""
    try:
        return fn(*args, **kwargs)
    except LookupError as e:        # not found / nothing to report yet
        raise HTTPException(status_code=404, detail=str(e).strip("'\""))
    except RuntimeError as e:       # wrong state (e.g. no active episode, episode finished)
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:         # invalid input (e.g. invalid action)
        raise HTTPException(status_code=400, detail=str(e))


class StartRequest(BaseModel):
    split: str = "dev"
    episode_index: int = 0
    policy: str = "rl_agent"   # random | rule_based | risk_threshold | rl_agent | manual


class StepRequest(BaseModel):
    action: Optional[Any] = None   # 0-3 or a name; leave empty to let the policy decide (validated in services.parse_action)


# ---------------------------------------------------------------- endpoints
@app.get("/api/health")
def health():
    return {"status": "ok", "transactions": int(len(service.tx_df))}


@app.get("/api/meta")
def meta():
    return service.meta()


@app.get("/api/dashboard")
def dashboard(split: str = "dev", policy: str = "rl_agent"):
    return _call(service.dashboard, split, policy)


@app.get("/api/compare")
def compare(split: str = "dev"):
    return _call(service.compare, split)


@app.get("/api/training-log")
def training_log():
    return {"agent": service.agent.meta, "log": service.training_log}


@app.get("/api/transactions")
def transactions(split: str = "all", episode: str = "", search: str = "", category: str = "",
                 risk: str = "", status: str = "", suspicious: str = "all", limit: int = 1000):
    return _call(service.transactions, split=split, episode=episode, search=search, category=category,
                 risk=risk, status=status, suspicious=suspicious, limit=limit)


@app.post("/api/episode/start")
def start_episode(req: StartRequest):
    return _call(service.start_episode, req.split, req.episode_index, req.policy)


@app.post("/api/episode/step")
def step_episode(req: Optional[StepRequest] = None):
    action = req.action if req else None
    record = _call(service.step_session, action)
    return {"step": record, "state": service.session_state()}


@app.post("/api/episode/run")
def run_episode_to_end():
    return _call(service.run_session)


@app.get("/api/episode/current")
def current_episode():
    return service.session_state()


@app.get("/api/report")
def report(source: str = "episode", split: str = "dev", episode_index: int = 0, policy: str = "rl_agent"):
    return _call(service.report, source=source, split=split, episode_index=episode_index, policy=policy)


@app.get("/api/report/export")
def export_report(format: str = "json", source: str = "episode", split: str = "dev",
                  episode_index: int = 0, policy: str = "rl_agent"):
    content, media_type, filename = _call(service.export_report, format, source=source, split=split,
                                          episode_index=episode_index, policy=policy)
    return Response(content=content, media_type=media_type,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


# The website itself (must be mounted LAST so it does not hide the /api routes)
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
