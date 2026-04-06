"""
validate.py — pre-submission self-check (21 checks)
spins up server, validates everything. run: python validate.py
"""

import json
import os
import subprocess
import sys
import time

import requests
import yaml

BASE = "http://localhost:7860"
RESULTS = []
TASK_IDS = ["local_nvlink", "ring_straggler", "ib_link_flap", "cross_dc_deadlock", 
            "nccl_config_drift", "cuda_oom_fragmentation", "checkpoint_corruption", "grad_accum_mismatch"]


def check(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail and not passed:
        msg += f" — {detail}"
    print(msg)
    RESULTS.append((name, passed))
    return passed


def wait_for_server(proc, timeout=15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        if proc.poll() is not None:
            return False
        time.sleep(0.5)
    return False


def main():
    print("=" * 60)
    print("ClusterOrch-Gym Pre-Submission Validator (21 checks)")
    print("=" * 60 + "\n")

    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

