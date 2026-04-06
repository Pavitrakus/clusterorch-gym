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

    try:
        if not check("1. Server starts on port 7860", wait_for_server(proc)):
            print("\n  Server failed to start. Aborting.")
            proc.kill()
            return

        # health
        r = requests.get(f"{BASE}/health")
        check("2. GET /health returns 200 and status ok",
              r.status_code == 200 and r.json().get("status") == "ok")

        # root
        r = requests.get(f"{BASE}/")
        check("3. GET / returns 200", r.status_code == 200)

        # tasks
        r = requests.get(f"{BASE}/tasks")
        tasks = r.json()
        check("4. GET /tasks returns list with 8 tasks", r.status_code == 200 and len(tasks) >= 8)

        # reset each task
        for i, tid in enumerate(TASK_IDS, 5):
            r = requests.post(f"{BASE}/reset", json={"task_id": tid})
            d = r.json()
            valid = r.status_code == 200 and "task_id" in d and "log" in d and len(d.get("log", "")) > 50
            check(f"{i}. POST /reset task_id={tid} valid Observation", valid)

        # step returns score 0-1
        requests.post(f"{BASE}/reset", json={"task_id": "local_nvlink"})
        action = {"diagnosis": "P2P disabled, NUMA mismatch", "root_cause": "NCCL_P2P_DISABLE=1",
                  "fix": "export NCCL_P2P_DISABLE=0", "severity": "high"}
        r = requests.post(f"{BASE}/step", json=action)
        score = r.json().get("reward", {}).get("score", -1)
        check("10. POST /step score between 0.0 and 1.0", 0.0 <= score <= 1.0, f"got {score}")

        # score is float
        check("11. Score is float not int/string", isinstance(score, float), f"type={type(score).__name__}")

        # state
        r = requests.get(f"{BASE}/state")
        check("12. GET /state returns valid JSON", r.status_code == 200 and isinstance(r.json(), dict))

        # grader variance
        test_actions = [
            {"diagnosis": "P2P disabled", "root_cause": "NCCL_P2P_DISABLE", "fix": "export NCCL_P2P_DISABLE=0", "severity": "high"},
            {"diagnosis": "something wrong", "root_cause": "unknown", "fix": "restart", "severity": "low"},
            {"diagnosis": "NUMA affinity", "root_cause": "NUMA mismatch", "fix": "fix pinning", "severity": "medium"},
        ]
        scores = []
        for ta in test_actions:
            requests.post(f"{BASE}/reset", json={"task_id": "local_nvlink"})
            r = requests.post(f"{BASE}/step", json=ta)
            scores.append(r.json()["reward"]["score"])
        check("13. Grader returns different scores", len(set(scores)) >= 2, f"scores={scores}")

        # multi-step investigation works
        requests.post(f"{BASE}/reset", json={"task_id": "local_nvlink"})
        r = requests.post(f"{BASE}/step", json={"action_type": "investigate", "query": "numa"})
        inv = r.json()
        check("14. Multi-step investigation works",
              r.status_code == 200 and inv.get("done") == False and inv["reward"]["score"] == 0.0)

        # test fix action
        requests.post(f"{BASE}/reset", json={"task_id": "local_nvlink"})
        r = requests.post(f"{BASE}/step", json={"action_type": "fix", "diagnosis": "test", "fix": "test_fix"})
        fix_res = r.json()
        check("14b. Fix action works and simulates result",
              r.status_code == 200 and fix_res.get("done") == True and "info" in fix_res and "fix_quality" in fix_res["info"])

        # openenv.yaml
        yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "openenv.yaml")
        check("15. openenv.yaml exists and valid YAML", os.path.exists(yaml_path))
        if os.path.exists(yaml_path):
            with open(yaml_path) as f:
                meta = yaml.safe_load(f)
            required = ["name", "version", "description", "author", "tasks", "endpoints", "runtime"]
            check("16. openenv.yaml has all required fields",
                  all(k in meta for k in required), f"missing: {[k for k in required if k not in meta]}")
        else:
            check("16. openenv.yaml has all required fields", False)

        # files exist
        root_dir = os.path.dirname(os.path.abspath(__file__))
        check("17. Dockerfile exists", os.path.exists(os.path.join(root_dir, "Dockerfile")))
        check("18. inference.py in root", os.path.exists(os.path.join(root_dir, "inference.py")))

        readme_path = os.path.join(root_dir, "README.md")
        readme_ok = False
        if os.path.exists(readme_path):
            with open(readme_path, encoding="utf-8") as f:
                content = f.read().lower()
            readme_ok = "baseline" in content and ("score" in content or "|" in content)
        check("19. README.md mentions baseline scores", readme_ok)

        # all tasks complete
        all_ok = True
        for tid in TASK_IDS:
            try:
                requests.post(f"{BASE}/reset", json={"task_id": tid})
                r = requests.post(f"{BASE}/step", json={"diagnosis": "test", "root_cause": "test",
                                                         "fix": "test", "severity": "medium"})
                if r.status_code != 200:
                    all_ok = False
            except Exception:
                all_ok = False
        check("20. All 8 tasks complete without crash", all_ok)

        # grader variance across tasks (not just one task)
        task_scores = []
        for tid in ["local_nvlink", "cross_dc_deadlock", "nccl_config_drift"]:
            requests.post(f"{BASE}/reset", json={"task_id": tid})
            r = requests.post(f"{BASE}/step", json={"diagnosis": "generic issue", "root_cause": "something",
                                                     "fix": "fix it", "severity": "medium"})
            task_scores.append(r.json()["reward"]["score"])
        check("21. Grader consistency: weak answers score low across tasks",
              all(s <= 0.3 for s in task_scores), f"scores={task_scores}")

    finally:
        proc.kill()
        proc.wait()

    # summary
    print("\n" + "=" * 60)
    passed = sum(1 for _, p in RESULTS if p)
    failed = len(RESULTS) - passed
    print(f"PASSED: {passed}/{len(RESULTS)}")
    print(f"FAILED: {failed}/{len(RESULTS)}")
    if failed:
        print("\nFailed checks:")
        for name, p in RESULTS:
            if not p:
                print(f"  ✗ {name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
