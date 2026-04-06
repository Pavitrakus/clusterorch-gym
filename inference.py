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
