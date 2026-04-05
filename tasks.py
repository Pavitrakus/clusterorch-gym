"""
tasks.py — 8 diagnostic scenarios with graders and post-fix simulations.
Covers: networking, memory management, storage, and training correctness.
Grading: multi-signal keyword + structural analysis. Deterministic, no LLM-as-judge.
"""

import string


def _clean(text: str) -> str:
    return text.lower().translate(str.maketrans("", "", string.punctuation))


def _has_any(text: str, keywords: list[str]) -> bool:
    cleaned = _clean(text)
    return any(kw.lower() in cleaned for kw in keywords)


def _floor_score(score: float, combined_text: str) -> float:
    if score == 0.0 and len(combined_text.strip()) > 10:
        return 0.05
    return round(min(score, 1.0), 2)


def _consistency_bonus(diag: str, fix: str, keywords: list[str]) -> float:
    """Bonus if diagnosis and fix mention the same core concept."""
    d, f = _clean(diag), _clean(fix)
    matches = sum(1 for kw in keywords if kw.lower() in d and kw.lower() in f)
    return min(0.05 * matches, 0.10)


# ───────────────────────────────────────────────────────────
# TASK 1 — local_nvlink (easy)
# ───────────────────────────────────────────────────────────

TASK_LOCAL_NVLINK = {
    "task_id": "local_nvlink", "difficulty": "easy",
    "description": "Diagnose NVLink bandwidth degradation on a single 8-GPU node achieving only 46.8% of theoretical bandwidth.",
    "context": {"num_gpus": 8, "topology": "NVLink", "node_type": "DGX A100",
                "expected_bandwidth_gbps": 400.0, "observed_bandwidth_gbps": 187.3},
    "log": """[2024-03-15 02:34:11] [rank 0] NCCL INFO NCCL version 2.18.3+cuda12.2
[2024-03-15 02:34:11] [rank 0] NCCL INFO NET/Plugin: No plugin found
[2024-03-15 02:34:11] [rank 0] NCCL INFO Using network Socket
[2024-03-15 02:34:11] [rank 0] NCCL INFO NCCL_P2P_DISABLE set by environment to 1
[2024-03-15 02:34:12] [rank 0] NCCL INFO Channel 00 : 0[0] -> 1[1] -> 2[2] -> 3[3] -> 4[4] -> 5[5] -> 6[6] -> 7[7]
[2024-03-15 02:34:13] [rank 0] NCCL INFO AllReduce: algo: Ring, protocol: LL
[2024-03-15 02:34:14] allreduce_perf: 8 x 8.00 GB: Avg bus bandwidth: 187.3 GB/s (peak: 400.0 GB/s, efficiency: 46.8%)
[2024-03-15 02:34:14] WARNING: P2P disabled, falling back to SHM transport
[2024-03-15 02:34:14] NUMA node for GPU 0: 0, process affinity: 1
[2024-03-15 02:34:14] NUMA node for GPU 4: 1, process affinity: 0
NCCL WARN Abnormal bandwidth detected on ring topology""",
    "investigations": {
        "nvidia-smi": "GPU 0: A100-SXM4-80GB, 42C, 85W/400W, Util: 23%\nGPU 1: A100-SXM4-80GB, 41C, 82W/400W, Util: 22%\nGPU 2-7: similar low utilization (21-24%)\nAll GPUs healthy, no ECC errors.",
        "nccl": "NCCL_P2P_DISABLE=1\nNCCL_SHM_DISABLE=0\nNCCL_ALGO=Ring\nNCCL_PROTO=LL\nNCCL_DEBUG=INFO\nNCCL_IB_DISABLE=1\nNote: P2P is explicitly disabled, forcing shared memory transport.",
        "numa": "GPU 0 -> NUMA node 0, Process affinity: NUMA 1 (MISMATCH)\nGPU 1 -> NUMA node 0, Process affinity: NUMA 0 (OK)\nGPU 4 -> NUMA node 1, Process affinity: NUMA 0 (MISMATCH)\nGPU 5 -> NUMA node 1, Process affinity: NUMA 1 (OK)\n2 of 8 GPUs have NUMA affinity mismatch.",
        "bandwidth": "NVLink bandwidth test (nvidia-smi nvlink -s):\nGPU0<->GPU1: 42.1 GB/s (expected: 50 GB/s)\nGPU0<->GPU4: 18.3 GB/s (expected: 50 GB/s) ← DEGRADED\nP2P paths are using SHM fallback instead of NVLink direct.",
        "default": "Available: nvidia-smi, nccl config, numa topology, bandwidth test",
    },
    "post_fix": {
        "correct": "[POST-FIX] Applied: export NCCL_P2P_DISABLE=0 && numactl rebind\n[POST-FIX] Re-running allreduce benchmark...\nallreduce_perf: 8 x 8.00 GB: Avg bus bandwidth: 378.4 GB/s (peak: 400.0 GB/s, efficiency: 94.6%)\nIMPROVEMENT: 187.3 → 378.4 GB/s (+102%)\nStatus: RESOLVED",
        "partial": "[POST-FIX] Applied: export NCCL_P2P_DISABLE=0\n[POST-FIX] Re-running allreduce benchmark...\nallreduce_perf: 8 x 8.00 GB: Avg bus bandwidth: 312.1 GB/s (peak: 400.0 GB/s, efficiency: 78.0%)\nIMPROVEMENT: 187.3 → 312.1 GB/s (+67%) — NUMA mismatch still present\nStatus: PARTIALLY RESOLVED",
        "wrong": "[POST-FIX] Applied changes.\n[POST-FIX] Re-running allreduce benchmark...\nallreduce_perf: 8 x 8.00 GB: Avg bus bandwidth: 191.2 GB/s (efficiency: 47.8%)\nIMPROVEMENT: 187.3 → 191.2 GB/s (+2%)\nStatus: NOT RESOLVED — root cause still present",
    },
}

def grade_local_nvlink(action: dict) -> dict:
    diag = action.get("diagnosis", "") + " " + action.get("root_cause", "")
    fix = action.get("fix", "")
    sev = action.get("severity", "").lower().strip()
    score = 0.0
    fb = []
    if _has_any(diag, ["p2p", "nccl_p2p_disable", "peer to peer", "p2p_disable"]):
        score += 0.35; fb.append("Correctly identified P2P disabled (+0.35)")
    else: fb.append("Missed P2P/NCCL_P2P_DISABLE issue")
    if _has_any(diag, ["numa", "affinity", "memory access"]):
        score += 0.20; fb.append("NUMA mismatch found (+0.20)")
    fix_clean = _clean(fix)
    if "nccl_p2p_disable" in fix_clean and ("0" in fix_clean or "enable" in fix_clean):
        score += 0.30; fb.append("Correct fix: NCCL_P2P_DISABLE=0 (+0.30)")
    elif _has_any(fix, ["p2p", "enable p2p", "unset nccl_p2p"]):
        score += 0.15; fb.append("Partial fix: mentioned P2P (+0.15)")
    if sev in ("high", "critical"): score += 0.15; fb.append(f"Severity {sev} (+0.15)")
    score += _consistency_bonus(diag, fix, ["p2p", "numa", "disable"])
    score = _floor_score(score, diag + fix + sev)
    return {"score": score, "found_issue": score >= 0.35, "correct_fix": score >= 0.65, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# TASK 2 — ring_straggler (medium)
# ───────────────────────────────────────────────────────────

TASK_RING_STRAGGLER = {
    "task_id": "ring_straggler", "difficulty": "medium",
    "description": "Identify a straggler GPU in a 256-GPU ring. Training stalled 47 minutes. Find the bad rank and fix.",
    "context": {"num_gpus": 256, "num_nodes": 8, "gpus_per_node": 32, "topology": "Ring",
                "stall_duration_minutes": 47, "interconnect": "InfiniBand HDR"},
    "log": """[2024-03-15 03:12:01] Training step 8432 started
[2024-03-15 03:12:01] [rank 0] NCCL INFO AllReduce start, 2.4GB tensor
[2024-03-15 03:12:01] [rank 47] NCCL INFO AllReduce start, 2.4GB tensor
[2024-03-15 03:58:44] [rank 0] NCCL WARN Timeout waiting for rank 47
[2024-03-15 03:58:44] [rank 47] NCCL INFO NET : Send : 4 [ERROR 12] Connection reset by peer
[2024-03-15 03:58:44] [rank 47] NCCL WARN Timeout on send to rank 48
[2024-03-15 03:58:45] [rank 0] Collective Communication Time: 2847.3s (expected: ~45s)
[2024-03-15 03:58:45] NCCL ERROR AllReduce failed with error 5 (Internal error)
Node 5 (ranks 128-159): ib0 interface showing packet_seq_err: 847293
Topology: Ring algorithm, all-to-all pattern across 8 nodes
Current NCCL_ALGO=RING, NCCL_PROTO=Simple""",
    "investigations": {
        "rank": "[rank 47] Last successful send: step 8431 (46min ago)\n[rank 47] Send queue depth: 847 (should be 0)\n[rank 47] Retry count: 12847\n[rank 46] Waiting on recv from rank 47\n[rank 48] Waiting on send from rank 47",
        "ib": "Node 5 ib0 counters:\n  port_rcv_errors: 847293\n  packet_seq_err: 847293\n  symbol_err_count: 12841\n  link_error_recovery: 3\n  link_downed: 0\nAll other nodes: all counters at 0.",
        "topology": "Ring: 0->1->...->46->47->48->...->255->0\nRank 47 between rank 46 and 48.\nIn Ring, ALL data must pass through every rank.\nTree algorithm would create bypass paths.",
        "default": "Available: rank status, infiniband counters, topology details",
    },
    "post_fix": {
        "correct": "[POST-FIX] Applied: export NCCL_ALGO=Tree && isolated rank 47\n[POST-FIX] Re-running training step...\nAllReduce completed in 41.2s (expected: ~45s)\nTraining resumed successfully. Step 8432 → 8433 in 42s.\nStatus: RESOLVED",
        "partial": "[POST-FIX] Applied: export NCCL_ALGO=Tree\n[POST-FIX] Tree bypasses rank 47 but NIC still degraded.\nAllReduce completed in 58.3s (expected: ~45s)\nStatus: PARTIALLY RESOLVED — rank 47 NIC still needs replacement",
        "wrong": "[POST-FIX] Changes applied.\n[POST-FIX] AllReduce still timing out at 2800s.\nStatus: NOT RESOLVED",
    },
}

def grade_ring_straggler(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    combined = diag + " " + root; score = 0.0; fb = []
    if "47" in root or "47" in diag: score += 0.30; fb.append("Identified rank 47 (+0.30)")
    if _has_any(combined, ["straggler", "timeout", "nic", "infiniband", "packet", "ib0"]):
        score += 0.25; fb.append("Good diagnosis (+0.25)")
    if _has_any(fix, ["tree", "nccl_algo=tree", "isolate rank", "exclude rank"]):
        score += 0.30; fb.append("Fix: TREE or isolate rank (+0.30)")
    if any(kw in fix.lower() for kw in ["bypass", "tolerate", "single point", "bottleneck"]):
        score += 0.15; fb.append("Explained why TREE > RING (+0.15)")
    score += _consistency_bonus(diag, fix, ["rank 47", "tree", "straggler"])
    score = _floor_score(score, combined + fix)
    return {"score": score, "found_issue": score >= 0.30, "correct_fix": score >= 0.60, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# TASK 3 — ib_link_flap (medium)
# ───────────────────────────────────────────────────────────

TASK_IB_LINK_FLAP = {
    "task_id": "ib_link_flap", "difficulty": "medium",
    "description": "Diagnose intermittent training slowdowns in a 128-GPU cluster. AllReduce spikes from 12s to 85s every 5-10 minutes.",
    "context": {"num_gpus": 128, "num_nodes": 4, "gpus_per_node": 32,
                "topology": "Fat-tree", "interconnect": "InfiniBand HDR"},
    "log": """[2024-03-15 05:20:01] Training step 12001 started
[2024-03-15 05:20:13] [rank 0] AllReduce completed in 12.3s
[2024-03-15 05:24:18] Training step 12002 started
[2024-03-15 05:25:41] [rank 64] NCCL WARN NET/IB : Got completion with error 12, opcode 0, len 0, vendor err 129
[2024-03-15 05:25:41] [rank 64] NCCL INFO NET/IB : Recovering from link error
[2024-03-15 05:25:43] [rank 0] AllReduce completed in 85.2s (expected: ~12s)
[2024-03-15 05:30:14] [rank 0] AllReduce completed in 12.8s (recovered)
[2024-03-15 05:36:52] [rank 64] NCCL WARN NET/IB : Got completion with error 12, vendor err 129
[2024-03-15 05:36:53] [rank 0] AllReduce completed in 94.1s
Node 2 ib0 port counters:
  symbol_err_count: 48271
  link_downed: 17
  link_error_recovery: 17
  port_rcv_errors: 3841
All other nodes ib0: all error counters at 0
Cable: Node 2, Port 1: QSFP28, vendor: GenericFiber, temp: 72°C (threshold: 65°C)""",
    "investigations": {
        "ib": "Node 2 ib0 detailed:\n  symbol_err_count: 48271 (increasing ~500/min)\n  link_downed: 17 events in last hour\n  link_error_recovery: 17\nAll other nodes: 0 errors.",
        "cable": "Cable diagnostics Node 2 Port 1:\n  Type: QSFP28 100G\n  Temperature: 72°C (WARNING: above 65°C threshold)\n  TX Power: -2.1 dBm (marginal)\n  RX Power: -4.8 dBm (low)\nOther cables: all within normal range.",
        "timing": "AllReduce timing last 20 steps:\n  Normal: 12.1-12.8s\n  Spikes: 85-94s (7x slower)\nCorrelates with link_error_recovery events on Node 2.",
        "default": "Available: infiniband counters, cable diagnostics, timing analysis",
    },
    "post_fix": {
        "correct": "[POST-FIX] Replaced QSFP28 cable on Node 2 Port 1\n[POST-FIX] New cable temp: 38°C, all counters reset to 0\nAllReduce 20 consecutive steps: 11.8-12.4s (no spikes)\nStatus: RESOLVED",
        "partial": "[POST-FIX] Reseated cable. Temp dropped to 58°C.\nAllReduce spikes reduced but still occurring every ~30min.\nStatus: PARTIALLY RESOLVED — cable degraded, needs replacement",
        "wrong": "[POST-FIX] Changes applied.\nAllReduce still spiking to 85s+ every 5-10min.\nStatus: NOT RESOLVED",
    },
}

def grade_ib_link_flap(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    sev = action.get("severity", "").lower().strip(); combined = diag + " " + root; score = 0.0; fb = []
    if _has_any(combined, ["flap", "flapping", "link error", "intermittent", "symbol error", "link_downed"]):
        score += 0.25; fb.append("Identified link flapping (+0.25)")
    if _has_any(combined, ["node 2", "rank 64", "port 1"]):
        score += 0.25; fb.append("Identified Node 2/rank 64 (+0.25)")
    if _has_any(fix, ["replace cable", "cable", "transceiver", "temperature", "cooling", "qsfp", "swap"]):
        score += 0.30; fb.append("Correct fix: cable replacement (+0.30)")
    if sev in ("high", "critical"): score += 0.20; fb.append(f"Severity {sev} (+0.20)")
    score = _floor_score(score, combined + fix + sev)
    return {"score": score, "found_issue": score >= 0.25, "correct_fix": score >= 0.50, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# TASK 4 — cross_dc_deadlock (hard)
# ───────────────────────────────────────────────────────────

TASK_CROSS_DC_DEADLOCK = {
    "task_id": "cross_dc_deadlock", "difficulty": "hard",
    "description": "Resolve a deadlock in cross-datacenter weight sync. 1536 GPUs across 3 DCs hanging for 3 hours with no error.",
    "context": {"num_gpus": 1536, "num_datacenters": 3, "model_params": "1T", "hang_duration_hours": 3},
    "log": """[2024-03-15 11:00:01] Cross-DC weight sync initiated
[2024-03-15 11:00:01] DC1 (IB): ranks 0-511, fabric: mlx5_0, RDMA enabled
[2024-03-15 11:00:01] DC2 (TCP): ranks 512-1023, interface: eth0, RTT: 45ms
[2024-03-15 11:00:01] DC3 (TCP): ranks 1024-1535, interface: eth0, RTT: 67ms
[2024-03-15 11:00:02] [rank 0] ncclSend(rank 512, 1.2TB) queued
[2024-03-15 11:00:02] [rank 512] ncclSend(rank 0, 1.2TB) queued
[2024-03-15 11:00:02] [rank 0] waiting for recv from rank 512...
[2024-03-15 11:00:02] [rank 512] waiting for recv from rank 0...
[2024-03-15 14:03:17] NCCL_IB_TIMEOUT: 14 (current), recommended: 22
[2024-03-15 14:03:17] NCCL_SOCKET_IFNAME: not set
[2024-03-15 14:03:17] [rank 0] still waiting... (10933s elapsed)
No crash. No error. Both ranks waiting on each other forever.""",
    "investigations": {
        "rank": "[rank 0] State: BLOCKED on ncclRecv from rank 512\n[rank 512] State: BLOCKED on ncclRecv from rank 0\nBoth ranks queued a Send BEFORE posting a Recv.\nClassic circular dependency.",
        "network": "DC1<->DC2 RTT: 45ms, bandwidth: 25 Gbps\nDC1<->DC3 RTT: 67ms, bandwidth: 10 Gbps\nNCCL_IB_TIMEOUT=14 (too low for 45ms+ RTT)",
        "deadlock": "Deadlock analysis:\n1. rank 0: ncclSend(512) -> waiting ncclRecv(512)\n2. rank 512: ncclSend(0) -> waiting ncclRecv(0)\nTextbook circular wait. Fix: ensure send/recv ordering or hierarchical comm.",
        "default": "Available: rank status, network topology, deadlock analysis",
    },
    "post_fix": {
        "correct": "[POST-FIX] Applied: hierarchical comm with intra-DC reduce first\n[POST-FIX] Send/recv ordering fixed: recv posted before send\n[POST-FIX] Cross-DC sync completed in 23.4s\nStatus: RESOLVED",
        "partial": "[POST-FIX] NCCL_IB_TIMEOUT increased to 22\n[POST-FIX] Timeout no longer triggers but deadlock persists.\nStatus: NOT RESOLVED — ordering issue not addressed",
        "wrong": "[POST-FIX] Changes applied.\n[POST-FIX] Still hanging after 300s.\nStatus: NOT RESOLVED",
    },
}

def grade_cross_dc_deadlock(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    sev = action.get("severity", "").lower().strip(); combined = diag + " " + root; score = 0.0; fb = []
    if _has_any(diag, ["deadlock", "dead lock", "dead-lock"]): score += 0.25; fb.append("Identified deadlock (+0.25)")
    elif _has_any(diag, ["hang", "stuck", "blocked"]): score += 0.10; fb.append("Partial: hang not deadlock (+0.10)")
    send_recv = ["send recv", "sendrecv", "both sending", "circular wait", "send before recv", "ordering"]
    if any(kw in _clean(combined) for kw in send_recv): score += 0.25; fb.append("Send/recv ordering (+0.25)")
    if any(kw in fix.lower() for kw in ["hierarchical", "intra-dc", "two-phase", "two phase", "local first"]):
        score += 0.25; fb.append("Fix: hierarchical comm (+0.25)")
    if _has_any(fix, ["nccl_socket_ifname", "socket_ifname", "eth0", "mlx5"]):
        score += 0.15; fb.append("NCCL_SOCKET_IFNAME (+0.15)")
    if sev == "critical": score += 0.10; fb.append("Severity critical (+0.10)")
    score = _floor_score(score, combined + fix + sev)
    return {"score": score, "found_issue": score >= 0.25, "correct_fix": score >= 0.50, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# TASK 5 — nccl_config_drift (hard)
# ───────────────────────────────────────────────────────────

TASK_NCCL_CONFIG_DRIFT = {
    "task_id": "nccl_config_drift", "difficulty": "hard",
    "description": "Diagnose 17.4% efficiency in a 512-GPU cluster. Multiple node groups have conflicting NCCL configs.",
    "context": {"num_gpus": 512, "num_nodes": 16, "observed_efficiency": 17.4, "expected_efficiency": 85.0},
    "log": """[2024-03-15 08:15:01] Multi-node training initialization, 512 GPUs, 16 nodes
[2024-03-15 08:15:01] NCCL config check across nodes:
  Nodes 0-3:   NCCL_ALGO=RING,  NCCL_PROTO=LL,     NCCL_NET_GDR_LEVEL=5, NCCL_IB_HCA=mlx5_0
  Nodes 4-7:   NCCL_ALGO=TREE,  NCCL_PROTO=Simple,  NCCL_NET_GDR_LEVEL=2, NCCL_IB_HCA=mlx5_0
  Nodes 8-11:  NCCL_ALGO=RING,  NCCL_PROTO=LL128,   NCCL_NET_GDR_LEVEL=5, NCCL_IB_HCA=mlx5_1
  Nodes 12-15: NCCL_ALGO=RING,  NCCL_PROTO=Simple,  NCCL_NET_GDR_LEVEL=5, NCCL_IB_HCA=mlx5_0
[2024-03-15 08:15:02] WARNING: Mixed algorithms detected across ranks
[2024-03-15 08:15:03] allreduce_perf: 512 x 2.0 GB: efficiency: 17.4%
[2024-03-15 08:15:04] Nodes 8-11 using mlx5_1 (secondary HCA), other nodes using mlx5_0 (primary)
[2024-03-15 08:15:04] GDR Level mismatch: nodes 4-7 at level 2 (host memory copy), others at level 5""",
    "investigations": {
        "config": "Per-node NCCL config dump:\nNodes 0-3:  ALGO=RING, PROTO=LL, GDR=5, HCA=mlx5_0  (engineer A)\nNodes 4-7:  ALGO=TREE, PROTO=Simple, GDR=2, HCA=mlx5_0  (engineer B)\nNodes 8-11: ALGO=RING, PROTO=LL128, GDR=5, HCA=mlx5_1  (engineer C)\nNodes 12-15: ALGO=RING, PROTO=Simple, GDR=5, HCA=mlx5_0 (default)\n4 different configurations!",
        "gdr": "GDR levels:\n  Level 5 = NIC reads GPU memory directly (fastest)\n  Level 2 = data through host memory (2-3x slower)\nNodes 4-7 at GDR 2 bottleneck the cluster.",
        "hca": "mlx5_0: Primary port, 200 Gbps spine switch\nmlx5_1: Secondary, 25 Gbps management network\nNodes 8-11 using MANAGEMENT network for training!",
        "bandwidth": "Per-group:\n  Nodes 0-3: 178 GB/s\n  Nodes 4-7: 42 GB/s (GDR bottleneck)\n  Nodes 8-11: 22 GB/s (wrong HCA!)\n  Nodes 12-15: 165 GB/s",
        "default": "Available: config dump, GDR analysis, HCA details, bandwidth breakdown",
    },
    "post_fix": {
        "correct": "[POST-FIX] Unified config: ALGO=RING, PROTO=LL128, GDR=5, HCA=mlx5_0 on all nodes\n[POST-FIX] allreduce_perf: 512 x 2.0 GB: efficiency: 87.2%\nIMPROVEMENT: 17.4% → 87.2% (+401%)\nStatus: RESOLVED",
        "partial": "[POST-FIX] Standardized ALGO across nodes but GDR/HCA still mixed.\nallreduce_perf: efficiency: 42.1%\nIMPROVEMENT: 17.4% → 42.1% (+142%)\nStatus: PARTIALLY RESOLVED",
        "wrong": "[POST-FIX] Changes applied.\nallreduce_perf: efficiency: 19.1%\nStatus: NOT RESOLVED",
    },
}

def grade_nccl_config_drift(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    combined = diag + " " + root; score = 0.0; fb = []
    if _has_any(combined, ["mismatch", "inconsistent", "drift", "different config", "mixed"]):
        score += 0.20; fb.append("Config inconsistency (+0.20)")
    if _has_any(combined, ["algorithm", "nccl_algo", "protocol", "nccl_proto", "gdr", "hca", "mlx5"]):
        score += 0.20; fb.append("Specific variable mismatches (+0.20)")
    if _has_any(fix, ["uniform", "consistent", "same config", "standardize", "unify", "all nodes"]):
        score += 0.25; fb.append("Standardize config (+0.25)")
    fix_lower = fix.lower()
    vals = ["ring" in fix_lower, "gdr" in fix_lower and "5" in fix_lower, "mlx5_0" in fix_lower]
    if sum(vals) >= 2: score += 0.20; fb.append("Correct values (+0.20)")
    elif sum(vals) >= 1: score += 0.10; fb.append("Partial values (+0.10)")
    if _has_any(combined + " " + fix, ["gdr level 2", "host memory", "gpu direct"]):
        score += 0.15; fb.append("GDR biggest impact (+0.15)")
    elif _has_any(fix, ["mlx5_1", "wrong hca", "management network"]):
        score += 0.15; fb.append("Wrong HCA critical (+0.15)")
    score = _floor_score(score, combined + fix)
    return {"score": score, "found_issue": score >= 0.20, "correct_fix": score >= 0.45, "feedback": ". ".join(fb)}


