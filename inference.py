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

