"""
inference.py — multi-step agent for ClusterOrch-Gym.
Uses investigation → diagnosis → fix flow to demonstrate environment depth.
Official hackathon stdout format.
"""

import json
import os
import re
import sys

import requests
from openai import OpenAI

# ── config from env vars ────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
BENCHMARK = "clusterorch-gym"
MAX_INVESTIGATE = 2  # investigation steps before fix

TASKS = ["local_nvlink", "ring_straggler", "ib_link_flap", "cross_dc_deadlock",
         "nccl_config_drift", "cuda_oom_fragmentation", "checkpoint_corruption",
         "grad_accum_mismatch"]

SYSTEM_PROMPT = """You are a senior ML infrastructure engineer with deep expertise in NCCL, InfiniBand, NVLink, CUDA memory management, distributed training, and checkpoint systems.

Analyze the NCCL/training log from a distributed GPU cluster. Identify the root cause and provide a fix.

Think step by step:
1. What symptoms are visible in the log?
2. What is the specific root cause (config var, hardware, rank, software bug)?
3. What exact command or config change would fix it?
4. How severe is this (low/medium/high/critical)?

Respond ONLY with a JSON object:
{
    "diagnosis": "describe the failure mode",
    "root_cause": "specific technical root cause",
    "fix": "exact command(s) or config change(s). Explain WHY.",
    "severity": "low|medium|high|critical"
}"""

INVESTIGATE_PROMPT = """Based on the log below, what investigation would help most?
Pick ONE from the available options. Respond with ONLY the query string (no JSON, no explanation).

Log excerpt (last 500 chars):
{log_tail}

Available investigations: {available}"""


# ── stdout logging (official format) ────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action_str, reward, done, error=None):
    err = error if error else "null"
    d = "true" if done else "false"
    print(f"[STEP] step={step} action={action_str} reward={reward:.2f} done={d} error={err}", flush=True)

def log_end(success, steps, score, rewards):
    s = "true" if success else "false"
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={s} steps={steps} score={score:.2f} rewards={r}", flush=True)

def _fmt_action(action: dict, action_type: str = "fix") -> str:
    """Format action as function-style string for stdout (matches official sample)."""
    if action_type == "investigate":
        q = action.get("query", "")[:60].replace("'", "")
        return f"investigate('{q}')"
    diag = action.get("diagnosis", "")[:80].replace("'", "")
    sev = action.get("severity", "medium")
    return f"{action_type}('{diag}',severity='{sev}')"


# ── LLM interaction ─────────────────────────────────────

def call_llm(client, prompt, system=SYSTEM_PROMPT, max_tokens=1024):
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            temperature=0.1, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"# LLM error: {e}", file=sys.stderr)
        return None


def parse_diagnosis(content):
    if not content:
        return {"diagnosis": "LLM unavailable", "root_cause": "API error",
                "fix": "retry", "severity": "medium"}
    content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.MULTILINE)
    content = re.sub(r"```\s*$", "", content, flags=re.MULTILINE).strip()
    try:
        p = json.loads(content)
        return {k: str(p.get(k, "")) for k in ["diagnosis", "root_cause", "fix", "severity"]}
    except json.JSONDecodeError:
        def extract(f):
            m = re.search(rf'"{f}"\s*:\s*"([^"]*)"', content)
            return m.group(1) if m else (content[:200] if f == "diagnosis" else "")
        return {k: extract(k) or ("medium" if k == "severity" else "") for k in ["diagnosis", "root_cause", "fix", "severity"]}


# ── main loop ───────────────────────────────────────────

def run_task(client, task_id):
    # 1. Reset
    reset_resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    reset_resp.raise_for_status()
    obs = reset_resp.json()

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    step_num = 0
    rewards = []
    current_log = obs["log"]
    available_investigations = "nvidia-smi, nccl config, numa topology, bandwidth test, infiniband counters, cable diagnostics, rank status, timing analysis, memory breakdown, checkpoint log, gradient analysis, loss comparison, config dump"

    # 2. Investigate (multi-step)
    for i in range(MAX_INVESTIGATE):
        inv_prompt = INVESTIGATE_PROMPT.format(
            log_tail=current_log[-500:], available=available_investigations)
        query = call_llm(client, inv_prompt,
                         system="You pick the best investigation query. Reply with ONLY the query.",
                         max_tokens=50)
        if not query:
            query = "config"
        query = query.strip().strip('"').strip("'")[:60]

        inv_resp = requests.post(f"{ENV_URL}/step", json={
            "action_type": "investigate", "query": query})
        inv_resp.raise_for_status()
        inv_result = inv_resp.json()

        step_num += 1
        rewards.append(0.0)
        log_step(step=step_num, action_str=_fmt_action({"query": query}, "investigate"),
                 reward=0.0, done=False)

        current_log = inv_result["observation"]["log"]

