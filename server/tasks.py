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
    if len(combined_text.strip()) > 10:
        score = max(score, 0.05)
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
    sev = action.get("severity", "").lower().strip()
    combined = diag + " " + root; score = 0.0; fb = []
    if "47" in root or "47" in diag: score += 0.30; fb.append("Identified rank 47 (+0.30)")
    if _has_any(combined, ["straggler", "timeout", "nic", "infiniband", "packet", "ib0"]):
        score += 0.25; fb.append("Good diagnosis (+0.25)")
    elif _has_any(combined, ["slow", "stall", "wait", "block", "error", "network", "ring", "connection"]):
        score += 0.10; fb.append("Partial: generic networking terms (+0.10)")
    if _has_any(fix, ["tree", "nccl_algo=tree", "isolate rank", "exclude rank"]):
        score += 0.30; fb.append("Fix: TREE or isolate rank (+0.30)")
    elif _has_any(fix, ["nccl", "replace", "switch", "algorithm", "restart"]):
        score += 0.10; fb.append("Partial fix mention (+0.10)")
    if any(kw in fix.lower() for kw in ["bypass", "tolerate", "single point", "bottleneck"]):
        score += 0.15; fb.append("Explained why TREE > RING (+0.15)")
    if sev in ("high", "critical"): score += 0.10; fb.append(f"Severity {sev} (+0.10)")
    score += _consistency_bonus(diag, fix, ["rank 47", "tree", "straggler"])
    score = _floor_score(score, combined + fix + sev)
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
    elif _has_any(diag, ["hang", "stuck", "blocked", "wait", "timeout"]): score += 0.10; fb.append("Partial: hang/wait (+0.10)")
    elif _has_any(combined, ["network", "sync", "communication", "connection", "latency", "datacenter", "dc", "cross"]):
        score += 0.05; fb.append("Generic networking terms (+0.05)")
    send_recv = ["send recv", "sendrecv", "both sending", "circular wait", "send before recv", "ordering"]
    if any(kw in _clean(combined) for kw in send_recv): score += 0.25; fb.append("Send/recv ordering (+0.25)")
    if any(kw in fix.lower() for kw in ["hierarchical", "intra-dc", "two-phase", "two phase", "local first"]):
        score += 0.25; fb.append("Fix: hierarchical comm (+0.25)")
    elif _has_any(fix, ["barrier", "synchronize", "ordering", "sequence", "reorder"]):
        score += 0.10; fb.append("Partial: synchronization fix (+0.10)")
    elif _has_any(fix, ["nccl", "config", "export", "set", "env"]):
        score += 0.05; fb.append("Generic config fix (+0.05)")
    if _has_any(fix, ["nccl_socket_ifname", "socket_ifname", "eth0", "mlx5"]):
        score += 0.15; fb.append("NCCL_SOCKET_IFNAME (+0.15)")
    if sev == "critical": score += 0.10; fb.append("Severity critical (+0.10)")
    elif sev == "high": score += 0.08; fb.append("Severity high (+0.08)")
    elif sev == "medium": score += 0.03; fb.append("Severity medium (+0.03)")
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


# ───────────────────────────────────────────────────────────
# TASK 6 — cuda_oom_fragmentation (medium)   [MEMORY]
# OOM crash despite 53% GPU memory free
# ───────────────────────────────────────────────────────────

TASK_CUDA_OOM_FRAG = {
    "task_id": "cuda_oom_fragmentation", "difficulty": "medium",
    "description": "GPU 3 crashes with OOM despite 42.8 GB free (53% of 80 GB). The allocation request is only 2.4 GB. Find out why and fix.",
    "context": {"num_gpus": 8, "gpu_type": "A100-80GB", "crash_gpu": 3,
                "free_memory_gb": 42.8, "requested_gb": 2.4, "category": "memory_management"},
    "log": """[2024-03-16 14:22:01] Training step 4521 - forward pass complete
[2024-03-16 14:22:02] Training step 4521 - backward pass starting
[2024-03-16 14:22:02] [GPU 3] RuntimeError: CUDA out of memory.
  Tried to allocate 2.40 GiB (GPU 3; 80.00 GiB total; 37.18 GiB allocated; 42.82 GiB free)
[2024-03-16 14:22:02] [GPU 3] torch.cuda.memory_summary():
  |  Allocated:  37.18 GiB  (847 blocks)
  |  Free:       42.82 GiB  (312 fragments)
  |  Largest free block: 1.82 GiB
  |  Total fragments: 312 (avg size: 137 MB)
[2024-03-16 14:22:02] [GPU 3] Allocation pattern: dynamic shapes from variable-length sequences
  batch_size=dynamic (4-64 seqs), seq_len=128-8192 tokens
[2024-03-16 14:22:03] [GPU 0-2,4-7] No OOM - largest free block: 12.4+ GiB
[2024-03-16 14:22:03] GPU 3 receives longest sequences (8192 tokens) due to length-sorted batching
PYTORCH_CUDA_ALLOC_CONF: not set (using defaults)
torch.cuda.max_memory_allocated(3): 72.4 GiB (of 80 GiB)""",
    "investigations": {
        "memory": "GPU 3 memory breakdown:\n  Model params: 18.2 GiB (fixed)\n  Activations: 12.4 GiB (varies with seq_len)\n  Gradients: 6.6 GiB (varies)\n  312 free fragments, largest = 1.82 GiB\n  Need 2.4 GiB contiguous but no single block is large enough.",
        "config": "PYTORCH_CUDA_ALLOC_CONF: not set\nAvailable options:\n  expandable_segments:True — allows coalescing fragments\n  max_split_size_mb:512 — prevents excessive splitting\n  garbage_collection_threshold:0.6 — triggers GC earlier\nNone of these are enabled.",
        "pattern": "Allocation timeline GPU 3:\n  Step 4500: peak 71.2 GiB, 412 blocks\n  Step 4510: peak 73.8 GiB, 623 blocks\n  Step 4520: peak 72.4 GiB, 847 blocks\nBlock count increasing = fragmentation growing.\nDynamic tensor shapes create many small free gaps.",
        "default": "Available: memory breakdown, allocator config, allocation pattern",
    },
    "post_fix": {
        "correct": "[POST-FIX] Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512\n[POST-FIX] Restarted training from checkpoint...\nGPU 3: 847 blocks → 124 blocks, largest free: 28.4 GiB\nStep 4521 backward pass completed successfully.\nStatus: RESOLVED",
        "partial": "[POST-FIX] Set max_split_size_mb:512\n[POST-FIX] Fragmentation reduced but still OOM on step 4589.\nStatus: PARTIALLY RESOLVED — expandable_segments needed too",
        "wrong": "[POST-FIX] Changes applied.\n[POST-FIX] Still OOM on step 4521.\nStatus: NOT RESOLVED",
    },
}

def grade_cuda_oom_frag(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    sev = action.get("severity", "").lower().strip(); combined = diag + " " + root; score = 0.0; fb = []
    if _has_any(combined, ["fragment", "fragmented", "fragmentation"]):
        score += 0.30; fb.append("Identified memory fragmentation (+0.30)")
    elif _has_any(combined, ["oom", "out of memory", "contiguous"]):
        score += 0.10; fb.append("Identified OOM without fragmentation root cause (+0.10)")
    if _has_any(combined, ["dynamic", "variable", "seq_len", "sequence length", "varying"]):
        score += 0.15; fb.append("Identified dynamic shapes as cause (+0.15)")
    if _has_any(fix, ["expandable_segments", "expandable segments"]):
        score += 0.30; fb.append("Correct fix: expandable_segments (+0.30)")
    elif _has_any(fix, ["max_split_size", "split_size"]):
        score += 0.15; fb.append("Partial fix: max_split_size (+0.15)")
    elif _has_any(fix, ["pytorch_cuda_alloc_conf", "alloc_conf"]):
        score += 0.10; fb.append("Mentioned allocator config (+0.10)")
    if sev in ("high", "critical"): score += 0.15; fb.append(f"Severity {sev} (+0.15)")
    score += _consistency_bonus(diag, fix, ["fragment", "dynamic", "alloc"])
    score = _floor_score(score, combined + fix + sev)
    return {"score": score, "found_issue": score >= 0.30, "correct_fix": score >= 0.45, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# TASK 7 — checkpoint_corruption (medium)   [STORAGE]
# loss spikes after checkpoint resume
# ───────────────────────────────────────────────────────────

TASK_CHECKPOINT_CORRUPTION = {
    "task_id": "checkpoint_corruption", "difficulty": "medium",
    "description": "After resuming from checkpoint at step 100k, loss spikes from 2.1 to 14.7. One shard is corrupted. Find which rank and why.",
    "context": {"num_gpus": 64, "num_nodes": 2, "checkpoint_step": 100000,
                "pre_resume_loss": 2.1, "post_resume_loss": 14.7, "category": "storage"},
    "log": """[2024-03-17 09:00:01] Resuming training from checkpoint step 100000
[2024-03-17 09:00:02] Loading model shards: 64 files, 480 GiB total
[2024-03-17 09:00:05] [rank 0-2,4-63] Shard loaded, checksum OK
[2024-03-17 09:00:05] [rank 3] Shard loaded, checksum MISMATCH
  Expected: 0xa3f2c819, Got: 0xb41e7d03
  File: /shared/ckpt/step_100000/rank_003.pt (7.5 GiB)
  File mtime: 2024-03-16 23:47:12 (during active training)
[2024-03-17 09:00:06] WARNING: Proceeding despite checksum mismatch (no strict mode)
[2024-03-17 09:00:10] Step 100001: loss = 14.72 (expected: ~2.1)
[2024-03-17 09:00:15] Step 100002: loss = 13.89
[2024-03-17 09:00:20] Step 100003: loss = 14.21 (not recovering)
[2024-03-17 09:00:20] [rank 3] weight norm: 0.0023 (other ranks: 2.41-2.58)
[2024-03-17 09:00:20] [rank 3] gradient norm: 847.2 (other ranks: 1.2-1.8)
Checkpoint save log from 2024-03-16 23:47:
  23:47:10 AllReduce step 99999 in progress
  23:47:11 Checkpoint save triggered (async, no barrier)
  23:47:12 [rank 3] saving shard... (AllReduce still running)
  23:47:14 [rank 3] shard saved (weights may be mid-update)
  23:47:15 AllReduce step 99999 completed""",
    "investigations": {
        "checkpoint": "Checkpoint save sequence:\n  1. No dist.barrier() before save\n  2. rank 3 saved during active AllReduce\n  3. Weights were mid-gradient-update when serialized\n  4. Other ranks finished AllReduce before saving\nThis is a race condition in the checkpoint logic.",
        "rank": "[rank 3] weight stats after resume:\n  mean: 0.0003 (expected: 0.42)\n  std: 0.0001 (expected: 0.12)\n  weight_norm: 0.0023 (expected: ~2.5)\nRank 3 weights are essentially zeroed out.",
        "loss": "Loss trajectory:\n  Step 99998: 2.08\n  Step 99999: 2.11\n  Step 100001: 14.72 ← resumed from corrupted checkpoint\n  Step 100002: 13.89\nLoss is not recovering — corrupted weights propagating through AllReduce.",
        "default": "Available: checkpoint save log, rank 3 analysis, loss trajectory",
    },
    "post_fix": {
        "correct": "[POST-FIX] Added dist.barrier() before checkpoint save + checksum validation\n[POST-FIX] Restored from step 99500 (last clean checkpoint)\n[POST-FIX] Step 99501: loss = 2.14 (healthy)\nStatus: RESOLVED",
        "partial": "[POST-FIX] Added strict checksum validation.\n[POST-FIX] Still need to restore from earlier checkpoint.\nStatus: PARTIALLY RESOLVED",
        "wrong": "[POST-FIX] Changes applied.\n[POST-FIX] Loss still at 14.2 — corrupted weights still loaded.\nStatus: NOT RESOLVED",
    },
}

def grade_checkpoint_corruption(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    sev = action.get("severity", "").lower().strip(); combined = diag + " " + root; score = 0.0; fb = []
    if _has_any(combined, ["corrupt", "checksum", "mismatch", "stale", "mid-update"]):
        score += 0.25; fb.append("Identified checkpoint corruption (+0.25)")
    if "3" in combined or "rank 3" in combined.lower():
        score += 0.15; fb.append("Identified rank 3 (+0.15)")
    if _has_any(combined, ["barrier", "race condition", "no sync", "async save", "during allreduce"]):
        score += 0.25; fb.append("Root cause: no barrier before save (+0.25)")
    if _has_any(fix, ["barrier", "dist.barrier", "synchronize", "sync before"]):
        score += 0.20; fb.append("Fix: add barrier (+0.20)")
    if _has_any(fix, ["checksum", "validate", "verify", "strict"]):
        score += 0.10; fb.append("Checksum validation (+0.10)")
    if sev in ("high", "critical"): score += 0.10; fb.append(f"Severity {sev} (+0.10)")
    score += _consistency_bonus(diag, fix, ["barrier", "checksum", "rank 3"])
    score = _floor_score(score, combined + fix + sev)
    return {"score": score, "found_issue": score >= 0.25, "correct_fix": score >= 0.45, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# TASK 8 — gradient_accumulation_mismatch (hard)  [CORRECTNESS]
# training doesn't converge — different accum steps
# ───────────────────────────────────────────────────────────

TASK_GRAD_ACCUM_MISMATCH = {
    "task_id": "grad_accum_mismatch", "difficulty": "hard",
    "description": "Training loss plateaus at 4.2 and never converges. 256 GPUs. Effective batch size varies across nodes. Find root cause.",
    "context": {"num_gpus": 256, "num_nodes": 8, "gpus_per_node": 32,
                "expected_loss": 1.5, "observed_loss": 4.2, "category": "training_correctness"},
    "log": """[2024-03-18 10:00:01] Distributed training: 256 GPUs, 8 nodes, FSDP
[2024-03-18 10:00:01] Target learning rate: 3e-4, warmup: 2000 steps
[2024-03-18 10:00:02] Per-node config scan:
  Nodes 0-3: GRADIENT_ACCUMULATION_STEPS=4, micro_batch=16, effective_batch=256
  Nodes 4-7: GRADIENT_ACCUMULATION_STEPS=8, micro_batch=16, effective_batch=512
[2024-03-18 10:00:02] WARNING: effective_batch_size varies across nodes
[2024-03-18 10:30:01] Step 2000: loss = 3.84 (expected: ~3.2 at this step)
[2024-03-18 11:00:01] Step 4000: loss = 4.21 (expected: ~2.4)
[2024-03-18 11:30:01] Step 6000: loss = 4.18 (plateau, not converging)
[2024-03-18 11:30:02] Gradient norm analysis:
  Nodes 0-3 avg grad_norm: 1.24 (normal)
  Nodes 4-7 avg grad_norm: 0.61 (half of nodes 0-3)
[2024-03-18 11:30:02] Learning rate: 3e-4 (constant after warmup)
[2024-03-18 11:30:03] AllReduce averaging gradients across ALL 256 GPUs
  But nodes 4-7 accumulated 8 steps vs 4 steps → their gradients are double-counted
  Effective LR for nodes 4-7: 1.5e-4 (half of intended)""",
    "investigations": {
        "config": "Environment variable dump:\nNodes 0-3: GRADIENT_ACCUMULATION_STEPS=4 (set by launch script v2.1)\nNodes 4-7: GRADIENT_ACCUMULATION_STEPS=8 (set by launch script v2.0)\nNodes 4-7 still using old launch script with doubled accumulation.\nThis was not caught because training still 'runs' — it just doesn't converge.",
        "gradient": "Gradient analysis:\n  Nodes 0-3 sync every 4 micro-batches → grad_norm ~1.24\n  Nodes 4-7 sync every 8 micro-batches → grad_norm ~0.61\n  After AllReduce, gradients are AVERAGED across all ranks.\n  But nodes 4-7 contribute 'stale' gradients from more accumulation.\n  Result: gradient direction is inconsistent → loss doesn't converge.",
        "loss": "Loss comparison (isolated):\n  If all nodes used ACCUM=4: converges to 1.5 by step 10k\n  If all nodes used ACCUM=8: converges to 1.6 by step 12k\n  Mixed (current): plateaus at 4.2, never converges.\nThe mismatch is worse than either setting alone.",
        "default": "Available: config dump, gradient analysis, loss comparison",
    },
    "post_fix": {
        "correct": "[POST-FIX] Set GRADIENT_ACCUMULATION_STEPS=4 on ALL nodes\n[POST-FIX] Restarted from step 6000 checkpoint\nStep 6500: loss = 3.92\nStep 7000: loss = 3.41\nStep 8000: loss = 2.87 (converging!)\nStatus: RESOLVED",
        "partial": "[POST-FIX] Set uniform accumulation but didn't restart.\nStep 6001: loss = 4.19 (stale optimizer state)\nStatus: PARTIALLY RESOLVED — need optimizer state reset",
        "wrong": "[POST-FIX] Changes applied.\nStep 6001: loss = 4.22\nStatus: NOT RESOLVED",
    },
}

def grade_grad_accum_mismatch(action: dict) -> dict:
    diag = action.get("diagnosis", ""); root = action.get("root_cause", ""); fix = action.get("fix", "")
    combined = diag + " " + root; score = 0.0; fb = []
    if _has_any(combined, ["accumulation", "accum", "gradient_accumulation", "batch size"]):
        score += 0.25; fb.append("Identified gradient accumulation issue (+0.25)")
    if _has_any(combined, ["mismatch", "inconsistent", "different", "4 vs 8", "varies"]):
        score += 0.20; fb.append("Identified mismatch across nodes (+0.20)")
    if _has_any(combined, ["converge", "plateau", "direction", "effective lr", "learning rate"]):
        score += 0.10; fb.append("Explained convergence impact (+0.10)")
    if _has_any(fix, ["uniform", "same", "consistent", "all nodes", "standardize"]):
        score += 0.15; fb.append("Fix: unify accumulation (+0.15)")
    if _has_any(fix, ["gradient_accumulation_steps=4", "accum=4", "accumulation=4", "set to 4"]):
        score += 0.20; fb.append("Specific fix: ACCUM=4 (+0.20)")
    elif _has_any(fix, ["gradient_accumulation_steps", "launch script"]):
        score += 0.10; fb.append("Mentioned setting (+0.10)")
    if _has_any(fix, ["restart", "reset optimizer", "fresh"]):
        score += 0.10; fb.append("Restart from checkpoint (+0.10)")
    score += _consistency_bonus(diag, fix, ["accumulation", "mismatch", "uniform"])
    score = _floor_score(score, combined + fix)
    return {"score": score, "found_issue": score >= 0.25, "correct_fix": score >= 0.40, "feedback": ". ".join(fb)}


# ───────────────────────────────────────────────────────────
# registry
# ───────────────────────────────────────────────────────────

TASKS = {
    "local_nvlink": TASK_LOCAL_NVLINK,
    "ring_straggler": TASK_RING_STRAGGLER,
    "ib_link_flap": TASK_IB_LINK_FLAP,
    "cross_dc_deadlock": TASK_CROSS_DC_DEADLOCK,
    "nccl_config_drift": TASK_NCCL_CONFIG_DRIFT,
    "cuda_oom_fragmentation": TASK_CUDA_OOM_FRAG,
    "checkpoint_corruption": TASK_CHECKPOINT_CORRUPTION,
    "grad_accum_mismatch": TASK_GRAD_ACCUM_MISMATCH,
}

GRADERS = {
    "local_nvlink": grade_local_nvlink,
    "ring_straggler": grade_ring_straggler,
    "ib_link_flap": grade_ib_link_flap,
    "cross_dc_deadlock": grade_cross_dc_deadlock,
    "nccl_config_drift": grade_nccl_config_drift,
    "cuda_oom_fragmentation": grade_cuda_oom_frag,
    "checkpoint_corruption": grade_checkpoint_corruption,
    "grad_accum_mismatch": grade_grad_accum_mismatch,
}


def get_task_list() -> list[dict]:
    return [
        {"task_id": t["task_id"], "difficulty": t["difficulty"],
         "description": t["description"], "max_score": 1.0,
         "category": t["context"].get("category", "networking")}
        for t in TASKS.values()
    ]


def get_investigation(task_id: str, query: str) -> str:
    inv = TASKS.get(task_id, {}).get("investigations", {})
    if not inv:
        return "No investigation data available for this task."
    q = query.lower()
    for key, response in inv.items():
        if key == "default":
            continue
        if key in q or any(word in q for word in key.split()):
            return response
    return inv.get("default", "Try: " + ", ".join(k for k in inv.keys() if k != "default"))


def get_post_fix_simulation(task_id: str, fix_text: str) -> dict:
    """Simulate applying a fix and return post-fix observation + quality level."""
    task = TASKS.get(task_id, {})
    pf = task.get("post_fix", {})
    if not pf:
        return {"quality": "unknown", "observation": "No post-fix simulation available."}

    fix_lower = fix_text.lower()
    grader = GRADERS.get(task_id)
    if grader:
        # use grader score to determine fix quality
        result = grader({"diagnosis": "", "root_cause": "", "fix": fix_text, "severity": ""})
        fix_score = result["score"]
        if fix_score >= 0.30:
            return {"quality": "correct", "observation": pf.get("correct", "")}
        elif fix_score >= 0.10:
            return {"quality": "partial", "observation": pf.get("partial", "")}
    return {"quality": "wrong", "observation": pf.get("wrong", "")}
