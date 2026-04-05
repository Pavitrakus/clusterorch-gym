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

