"""
app.py — ClusterOrch-Gym environment server
FastAPI app with OpenEnv step/reset/state API + web dashboard.
Multi-step support: agents can investigate before submitting diagnosis.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional

from tasks import TASKS, GRADERS, get_task_list, get_investigation, get_post_fix_simulation


# ── pydantic models ─────────────────────────────────────

class Observation(BaseModel):
    task_id: str
    log: str
    description: str
    difficulty: str
    context: dict


class Action(BaseModel):
    action_type: str = "diagnose"   # "investigate", "diagnose", or "fix"
    diagnosis: str = ""
    root_cause: str = ""
    fix: str = ""
    severity: str = ""
    query: str = ""                 # for investigate actions


class Reward(BaseModel):
    score: float
    found_issue: bool
    correct_fix: bool
    feedback: str


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: dict


class ResetRequest(BaseModel):
    task_id: Optional[str] = "local_nvlink"


# ── environment class ───────────────────────────────────

class ClusterOrchEnv:
    """
    RL environment for distributed training cluster diagnosis.
    Supports multi-step: investigate first, then diagnose.
    Or single-step: just diagnose directly (backward compatible).
    """

    def __init__(self):
        self.current_task_id: Optional[str] = None
        self.current_observation: Optional[Observation] = None
        self.step_count: int = 0
        self.max_steps: int = 10

    def reset(self, task_id: str = "local_nvlink") -> Observation:
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Available: {list(TASKS.keys())}")
        task = TASKS[task_id]
        self.current_task_id = task_id
        self.step_count = 0
        self.current_observation = Observation(
            task_id=task["task_id"], log=task["log"],
            description=task["description"], difficulty=task["difficulty"],
            context=task["context"],
        )
        return self.current_observation

    def step(self, action: Action) -> StepResult:
        if self.current_task_id is None:
            raise RuntimeError("Call reset() before step()")

        self.step_count += 1

        # investigation step — returns more info, no grading yet
        if action.action_type == "investigate" and self.step_count < self.max_steps:
            query = action.query or action.diagnosis or "help"
            extra_info = get_investigation(self.current_task_id, query)
            updated_log = self.current_observation.log + f"\n\n--- Investigation: {query} ---\n{extra_info}"
            self.current_observation = Observation(
                task_id=self.current_observation.task_id,
                log=updated_log,
                description=self.current_observation.description,
                difficulty=self.current_observation.difficulty,
                context=self.current_observation.context,
            )
            return StepResult(
                observation=self.current_observation,
                reward=Reward(score=0.0, found_issue=False, correct_fix=False,
                              feedback="Investigation complete. Submit diagnosis/fix or continue investigating."),
                done=False,
                info={"step": self.step_count, "max_steps": self.max_steps,
                      "task_id": self.current_task_id, "action_type": "investigate"},
            )

        # fix step — grade + simulate applying the fix + return observable result
        if action.action_type == "fix":
            grader = GRADERS[self.current_task_id]
            result = grader(action.model_dump())
            sim = get_post_fix_simulation(self.current_task_id, action.fix)
            post_fix_log = self.current_observation.log + f"\n\n--- Fix Applied ---\n{sim['observation']}"
            post_fix_obs = Observation(
                task_id=self.current_observation.task_id,
                log=post_fix_log,
                description=self.current_observation.description,
                difficulty=self.current_observation.difficulty,
                context=self.current_observation.context,
            )
            self.current_observation = post_fix_obs
            return StepResult(
                observation=post_fix_obs,
                reward=Reward(score=result["score"], found_issue=result["found_issue"],
                              correct_fix=result["correct_fix"], feedback=result["feedback"]),
                done=True,
                info={"task_id": self.current_task_id, "max_score": 1.0,
                      "steps_taken": self.step_count, "grader_version": "2.0.0",
                      "fix_quality": sim["quality"], "action_type": "fix"},
            )

        # diagnosis step — grade the answer (backward compatible)
        grader = GRADERS[self.current_task_id]
        result = grader(action.model_dump())
        return StepResult(
            observation=self.current_observation,
            reward=Reward(score=result["score"], found_issue=result["found_issue"],
                          correct_fix=result["correct_fix"], feedback=result["feedback"]),
            done=True,
            info={"task_id": self.current_task_id, "max_score": 1.0,
                  "steps_taken": self.step_count, "grader_version": "2.0.0"},
        )

    def state(self) -> dict:
        if self.current_task_id is None:
            return {"status": "idle", "task_id": None}
        return {"status": "active", "task_id": self.current_task_id,
                "difficulty": TASKS[self.current_task_id]["difficulty"],
                "step": self.step_count}


# ── fastapi app ─────────────────────────────────────────

app = FastAPI(title="ClusterOrch-Gym", version="1.0.0",
              description="RL environment for AI agents to diagnose distributed GPU training failures")
env = ClusterOrchEnv()


@app.get("/")
def root(request: Request):
    """Serves dashboard for browsers, JSON for API clients"""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return HTMLResponse(content=DASHBOARD_HTML)
    from fastapi.responses import JSONResponse
    return JSONResponse(content={"name": "clusterorch-gym",
            "description": "RL environment for training AI agents to diagnose and remediate failures in distributed GPU training clusters",
            "version": "1.0.0", "author": "Pavitra Kushwaha", "tasks": len(TASKS)})


@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return get_task_list()


@app.get("/state")
def get_state():
    return env.state()


@app.post("/reset")
def reset(body: ResetRequest = ResetRequest()):
    """Reset environment to a task. Returns Observation with NCCL log."""
    try:
        return env.reset(task_id=body.task_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset error: {str(e)}")


@app.post("/step")
def step(action: Action):
    """Submit diagnosis or investigation. Returns StepResult with score 0.0-1.0."""
    try:
        return env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step error: {str(e)}")


# ── dashboard HTML ──────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ClusterOrch-Gym — Distributed Training Diagnosis Environment</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--bg:#0a0b14;--card:#12141f;--border:#1e2035;--text:#e2e8f0;--muted:#8892b0;
--accent:#6366f1;--cyan:#06b6d4;--green:#22c55e;--orange:#f59e0b;--red:#ef4444;
--glow:0 0 20px rgba(99,102,241,0.15)}
body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);
min-height:100vh;overflow-x:hidden}
.bg-grid{position:fixed;inset:0;background-image:
linear-gradient(rgba(99,102,241,0.03) 1px,transparent 1px),
linear-gradient(90deg,rgba(99,102,241,0.03) 1px,transparent 1px);
background-size:60px 60px;pointer-events:none;z-index:0}

/* ── top nav ─────────────────────────── */
.topnav{position:sticky;top:0;z-index:10;padding:12px 2rem;
background:rgba(10,11,20,0.85);backdrop-filter:blur(16px);border-bottom:1px solid var(--border);
display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
.topnav-brand{font-weight:700;font-size:1.1rem;display:flex;align-items:center;gap:8px}
.topnav-brand span{background:linear-gradient(135deg,#fff,var(--accent));
-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.topnav-links{display:flex;gap:6px;flex-wrap:wrap}
.nav-link{display:inline-flex;align-items:center;gap:6px;padding:7px 14px;border-radius:8px;
font-size:0.8rem;font-weight:500;text-decoration:none;transition:all 0.2s ease;
border:1px solid var(--border);color:var(--text);background:var(--card)}
.nav-link:hover{border-color:var(--accent);background:rgba(99,102,241,0.08);color:#fff;
box-shadow:var(--glow);transform:translateY(-1px)}
.nav-link .dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.dot-green{background:var(--green);box-shadow:0 0 6px var(--green)}
.dot-blue{background:var(--accent);box-shadow:0 0 6px var(--accent)}
.dot-cyan{background:var(--cyan);box-shadow:0 0 6px var(--cyan)}
.dot-orange{background:var(--orange);box-shadow:0 0 6px var(--orange)}
.nav-link code{background:transparent;padding:0;font-size:0.75rem;color:var(--cyan)}

.container{max-width:1100px;margin:0 auto;padding:2rem;position:relative;z-index:1}
.hero{text-align:center;padding:2.5rem 0 1.5rem}
.hero-badge{display:inline-flex;align-items:center;gap:8px;padding:6px 16px;
