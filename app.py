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
border-radius:20px;background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.2);
font-size:0.8rem;color:var(--accent);margin-bottom:1.5rem;font-weight:500}
.pulse{width:8px;height:8px;background:var(--green);border-radius:50%;
animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
h1{font-size:2.5rem;font-weight:700;
background:linear-gradient(135deg,#fff 0%,var(--accent) 50%,var(--cyan) 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.75rem}
.subtitle{color:var(--muted);font-size:1.05rem;max-width:650px;margin:0 auto;line-height:1.6}
.stats{display:flex;justify-content:center;gap:2rem;margin:2.5rem 0;flex-wrap:wrap}
.stat{padding:1rem 1.5rem;border-radius:12px;background:var(--card);
border:1px solid var(--border);text-align:center;min-width:120px}
.stat-value{font-size:1.5rem;font-weight:700;color:var(--accent)}
.stat-label{font-size:0.75rem;color:var(--muted);margin-top:4px;text-transform:uppercase;letter-spacing:1px}

/* ── live panels ─────────────────────── */
.live-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1rem;margin:2rem 0}
.live-panel{border-radius:12px;background:var(--card);border:1px solid var(--border);overflow:hidden;
transition:all 0.3s ease}
.live-panel:hover{border-color:var(--accent);box-shadow:var(--glow)}
.live-header{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;
border-bottom:1px solid var(--border);background:rgba(99,102,241,0.04)}
.live-title{font-weight:600;font-size:0.85rem;display:flex;align-items:center;gap:8px}
.live-btn{padding:5px 12px;border-radius:6px;border:1px solid var(--border);background:var(--card);
color:var(--cyan);font-family:'JetBrains Mono',monospace;font-size:0.7rem;cursor:pointer;
transition:all 0.2s ease}
.live-btn:hover{border-color:var(--accent);background:rgba(99,102,241,0.1)}
.live-body{padding:12px 16px;font-family:'JetBrains Mono',monospace;font-size:0.75rem;
color:var(--cyan);max-height:200px;overflow-y:auto;white-space:pre-wrap;line-height:1.5;
background:#0d0f1a}

.section{margin:3rem 0}
.section-title{font-size:1.3rem;font-weight:600;margin-bottom:1.5rem;
display:flex;align-items:center;gap:10px}
.section-title::before{content:'';width:4px;height:24px;background:var(--accent);border-radius:2px}
.tasks-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:1rem}
.task-card{padding:1.25rem;border-radius:12px;background:var(--card);
border:1px solid var(--border);transition:all 0.3s ease;position:relative;overflow:hidden}
.task-card:hover{border-color:var(--accent);box-shadow:var(--glow);transform:translateY(-2px)}
.task-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;
background:linear-gradient(90deg,var(--accent),var(--cyan))}
.task-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem}
.task-id{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:0.95rem}
.badge{padding:3px 10px;border-radius:12px;font-size:0.7rem;font-weight:600;text-transform:uppercase;letter-spacing:0.5px}
.badge-easy{background:rgba(34,197,94,0.15);color:var(--green);border:1px solid rgba(34,197,94,0.3)}
.badge-medium{background:rgba(245,158,11,0.15);color:var(--orange);border:1px solid rgba(245,158,11,0.3)}
.badge-hard{background:rgba(239,68,68,0.15);color:var(--red);border:1px solid rgba(239,68,68,0.3)}
.task-desc{color:var(--muted);font-size:0.85rem;line-height:1.5}
.api-table{width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden}
.api-table th,.api-table td{padding:12px 16px;text-align:left;border-bottom:1px solid var(--border)}
.api-table th{background:rgba(99,102,241,0.08);font-size:0.8rem;text-transform:uppercase;
letter-spacing:1px;color:var(--accent);font-weight:600}
.api-table td{font-size:0.9rem}
.api-table tr:hover td{background:rgba(99,102,241,0.03)}
.method{font-family:'JetBrains Mono',monospace;font-weight:600;font-size:0.8rem;
padding:3px 8px;border-radius:6px}
.method-get{background:rgba(34,197,94,0.15);color:var(--green)}
.method-post{background:rgba(99,102,241,0.15);color:var(--accent)}
.path{font-family:'JetBrains Mono',monospace;color:var(--cyan)}
.path a{color:var(--cyan);text-decoration:none}
.path a:hover{text-decoration:underline}
code{font-family:'JetBrains Mono',monospace;background:rgba(99,102,241,0.1);
padding:2px 6px;border-radius:4px;font-size:0.85rem}
.code-block{background:#0d0f1a;border:1px solid var(--border);border-radius:10px;
padding:1rem 1.25rem;font-family:'JetBrains Mono',monospace;font-size:0.8rem;
line-height:1.6;color:var(--cyan);overflow-x:auto;margin:1rem 0}
.footer{text-align:center;padding:2rem 0;color:var(--muted);font-size:0.8rem;
border-top:1px solid var(--border);margin-top:3rem}
.feature-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:1rem}
.feature{padding:1rem;border-radius:10px;background:var(--card);border:1px solid var(--border)}
.feature-icon{font-size:1.5rem;margin-bottom:0.5rem}
.feature-title{font-weight:600;font-size:0.9rem;margin-bottom:0.25rem}
.feature-text{color:var(--muted);font-size:0.8rem;line-height:1.4}
</style>
</head>
<body>
<div class="bg-grid"></div>

<!-- ── sticky top nav ─────────────────── -->
<nav class="topnav">
<div class="topnav-brand"><span>⚡ ClusterOrch-Gym</span></div>
<div class="topnav-links">
  <a href="/docs" class="nav-link" target="_blank"><span class="dot dot-blue"></span> API Docs <code>/docs</code></a>
  <a href="/health" class="nav-link" target="_blank"><span class="dot dot-green"></span> Health <code>/health</code></a>
  <a href="/tasks" class="nav-link" target="_blank"><span class="dot dot-cyan"></span> Tasks <code>/tasks</code></a>
  <a href="/state" class="nav-link" target="_blank"><span class="dot dot-orange"></span> State <code>/state</code></a>
</div>
</nav>

<div class="container">
<div class="hero">
<div class="hero-badge"><span class="pulse"></span> Environment Online — v1.0.0</div>
<h1>ClusterOrch-Gym</h1>
<p class="subtitle">OpenEnv RL environment where AI agents learn to diagnose and fix failures
in distributed GPU training clusters. Real NCCL logs. Deterministic grading. Partial credit.</p>
</div>
<div class="stats">
<div class="stat"><div class="stat-value">5</div><div class="stat-label">Tasks</div></div>
<div class="stat"><div class="stat-value">3</div><div class="stat-label">Difficulties</div></div>
<div class="stat"><div class="stat-value">1536</div><div class="stat-label">Max GPUs</div></div>
<div class="stat"><div class="stat-value">∞</div><div class="stat-label">Episodes</div></div>
</div>

<!-- ── live endpoint panels ─────────── -->
<div class="section">
<h2 class="section-title">Live Endpoints</h2>
<div class="live-grid">
<div class="live-panel">
  <div class="live-header">
    <span class="live-title"><span class="dot dot-green"></span> GET /health</span>
    <button class="live-btn" onclick="fetchEndpoint('/health','panel-health')">▶ Run</button>
  </div>
  <div class="live-body" id="panel-health">Click "Run" to fetch live response...</div>
</div>
<div class="live-panel">
  <div class="live-header">
    <span class="live-title"><span class="dot dot-cyan"></span> GET /tasks</span>
    <button class="live-btn" onclick="fetchEndpoint('/tasks','panel-tasks')">▶ Run</button>
  </div>
  <div class="live-body" id="panel-tasks">Click "Run" to fetch live response...</div>
</div>
<div class="live-panel">
  <div class="live-header">
    <span class="live-title"><span class="dot dot-orange"></span> GET /state</span>
    <button class="live-btn" onclick="fetchEndpoint('/state','panel-state')">▶ Run</button>
  </div>
  <div class="live-body" id="panel-state">Click "Run" to fetch live response...</div>
</div>
</div></div>

<div class="section">
<h2 class="section-title">Tasks</h2>
<div class="tasks-grid">
<div class="task-card"><div class="task-header"><span class="task-id">local_nvlink</span>
<span class="badge badge-easy">Easy</span></div>
<p class="task-desc">Single 8-GPU node at 47% NVLink bandwidth. P2P disabled + NUMA affinity mismatch.</p></div>
<div class="task-card"><div class="task-header"><span class="task-id">ring_straggler</span>
<span class="badge badge-medium">Medium</span></div>
<p class="task-desc">256-GPU ring stalled 47 min. Rank 47 has bad InfiniBand NIC blocking everyone.</p></div>
<div class="task-card"><div class="task-header"><span class="task-id">ib_link_flap</span>
<span class="badge badge-medium">Medium</span></div>
<p class="task-desc">128-GPU cluster with intermittent slowdowns. Overheating cable causing IB link flapping.</p></div>
<div class="task-card"><div class="task-header"><span class="task-id">cross_dc_deadlock</span>
<span class="badge badge-hard">Hard</span></div>
<p class="task-desc">1536 GPUs across 3 DCs. Silent deadlock — both ranks sending before receiving.</p></div>
<div class="task-card"><div class="task-header"><span class="task-id">nccl_config_drift</span>
<span class="badge badge-hard">Hard</span></div>
<p class="task-desc">512-GPU cluster at 17% efficiency. 4 node groups with conflicting NCCL configs.</p></div>
</div></div>
<div class="section">
<h2 class="section-title">Features</h2>
<div class="feature-grid">
<div class="feature"><div class="feature-icon">🔍</div><div class="feature-title">Multi-Step Investigation</div>
<div class="feature-text">Agents can investigate before diagnosing — request GPU stats, IB counters, or topology details.</div></div>
<div class="feature"><div class="feature-icon">📊</div><div class="feature-title">Partial Credit Scoring</div>
<div class="feature-text">Graders award 0.0–1.0 based on diagnosis accuracy, fix quality, and severity assessment.</div></div>
<div class="feature"><div class="feature-icon">🎯</div><div class="feature-title">Deterministic Grading</div>
<div class="feature-text">Same input always produces same score. No LLM-as-judge. Fully reproducible.</div></div>
<div class="feature"><div class="feature-icon">📋</div><div class="feature-title">Real NCCL Logs</div>
<div class="feature-text">Logs match real NCCL_DEBUG=INFO output format from production GPU clusters.</div></div>
</div></div>
<div class="section">
<h2 class="section-title">API Reference</h2>
<table class="api-table">
<tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
<tr><td><span class="method method-post">POST</span></td><td class="path">/reset</td><td>Reset environment to a task, get observation with NCCL log</td></tr>
<tr><td><span class="method method-post">POST</span></td><td class="path">/step</td><td>Submit diagnosis or investigation, get graded result</td></tr>
<tr><td><span class="method method-get">GET</span></td><td class="path"><a href="/state" target="_blank">/state</a></td><td>Current environment state</td></tr>
<tr><td><span class="method method-get">GET</span></td><td class="path"><a href="/health" target="_blank">/health</a></td><td>Health check (instant, always fast)</td></tr>
<tr><td><span class="method method-get">GET</span></td><td class="path"><a href="/tasks" target="_blank">/tasks</a></td><td>List all available tasks</td></tr>
</table></div>
<div class="section">
<h2 class="section-title">Quick Start</h2>
<div class="code-block">
# Reset to a task<br>
curl -X POST /reset -H "Content-Type: application/json" \\<br>
&nbsp;&nbsp;-d '{"task_id": "local_nvlink"}'<br><br>
# Submit a diagnosis<br>
curl -X POST /step -H "Content-Type: application/json" \\<br>
&nbsp;&nbsp;-d '{"diagnosis": "P2P disabled", "root_cause": "NCCL_P2P_DISABLE=1",<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"fix": "export NCCL_P2P_DISABLE=0", "severity": "high"}'<br><br>
# Or investigate first (multi-step)<br>
curl -X POST /step -H "Content-Type: application/json" \\<br>
&nbsp;&nbsp;-d '{"action_type": "investigate", "query": "show numa topology"}'
</div></div>
<div class="footer">
<p>ClusterOrch-Gym v1.0.0 </p>
<p style="margin-top:0.5rem">
  <a href="/docs" style="color:var(--accent);text-decoration:none;margin-right:1.5rem">📖 Interactive API Docs</a>
  <a href="/health" style="color:var(--green);text-decoration:none;margin-right:1.5rem">💚 Health</a>
  <a href="/tasks" style="color:var(--cyan);text-decoration:none">📋 Tasks</a>
</p>
</div></div>

<script>
async function fetchEndpoint(path, panelId) {
  const el = document.getElementById(panelId);
  el.textContent = 'Loading...';
  try {
    const res = await fetch(path, {headers: {'Accept': 'application/json'}});
    const data = await res.json();
    el.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    el.textContent = 'Error: ' + err.message;
  }
}
// auto-fetch health on load
window.addEventListener('load', () => fetchEndpoint('/health', 'panel-health'));
</script>
</body></html>"""
