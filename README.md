# ClusterOrch-Gym 

> The first reproducible RL benchmark for AI agents that debug distributed GPU training clusters.

---

I've spent too much time staring at `NCCL_DEBUG=INFO` output trying to figure out why a 1T-parameter training run stalled at step 8432. It's always something stupid — P2P disabled by some engineer who forgot to unset it, a dying QSFP28 transceiver on Node 2, or three teams who each configured their NCCL stack differently and nobody noticed until AllReduce hit 17% efficiency.

These failures follow patterns. The same patterns, over and over. So I built an environment to train agents to do the firefighting instead.

**ClusterOrch-Gym** is an [OpenEnv](https://huggingface.co/openenv)-compliant RL environment where AI agents learn to read raw `NCCL_DEBUG` telemetry, investigate cluster state like a real SRE, and prescribe exact remediations for distributed training failures — ranked and scored deterministically.

No LLM-as-judge. No vibes-based grading. Pure heuristics.

---

## Why this exists

When you're running a 2048-GPU job on InfiniBand HDR and one NIC starts throwing `packet_seq_err`, you have maybe 10 minutes before the entire ring blocks and you've burned $40k of GPU-hours. The current state of the art is a senior MLE grepping through thousands of lines of NCCL output, manually cross-referencing port counters with `ibstat` and hoping their intuition is right.

There's no reproducible sandbox to train or evaluate agents on this. **Until now.**

ClusterOrch-Gym gives you:
- Realistic `NCCL_DEBUG=INFO` logs (not toy examples — actual error codes like `ERROR 12`, `vendor err 129`, `link_downed`)
- Multi-step investigation loop (ask for `nvidia-smi`, IB port counters, NUMA topology, cable temps before diagnosing)
- Deterministic partial-credit grading (RL needs gradients, not binary pass/fail)
- 8 failure scenarios from "catch-this-in-5-minutes" to "this-took-our-team-6-hours"

---

## The 8 Failure Scenarios

### 🟢 Easy

**`local_nvlink`** — Single DGX A100, 8 GPUs, achieving 46.8% of theoretical NVLink bandwidth (187.3 GB/s vs 400 GB/s peak). Two overlapping bugs: `NCCL_P2P_DISABLE=1` forcing SHM fallback instead of NVLink direct, plus a NUMA affinity mismatch where GPU 0 (NUMA node 0) is pinned to a process on NUMA node 1. Classic misconfiguration. Should be caught in under 5 minutes.

---

### 🟡 Medium

**`ring_straggler`** — 256-GPU cluster, 8 nodes of 32 GPUs each, InfiniBand HDR interconnect. Training stalled 47 minutes at step 8432. Rank 47 is throwing `ERROR 12: Connection reset by peer` with `packet_seq_err: 847293` on `ib0`. Ring topology means all 255 other ranks are blocked waiting for the one bad link. Fix: switch `NCCL_ALGO=TREE` (tolerates single-link failures) or isolate rank 47 and restart just that process.

**`ib_link_flap`** — 128-GPU fat-tree cluster with intermittent AllReduce spikes every 3-5 steps (12s normal → 85-94s during events). Root cause: QSFP28 transceiver on Node 2 Port 1 running at 72°C (threshold: 65°C), causing thermal-induced link flapping with `link_downed: 17` events. Agents need to query cable diagnostics explicitly to find the temperature reading — it's not in the initial log.

**`cuda_oom_fragmentation`** — Single-node training crashes with `CUDA out of memory` despite `nvidia-smi` showing 53% free VRAM (42.3 GB available on an 80GB A100). Cause: CUDA memory allocator fragmentation — PyTorch's caching allocator holds 38.7 GB in small freed blocks that can't satisfy a contiguous 12.4 GB allocation. Fix: `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` plus `torch.cuda.empty_cache()` between phases.

**`checkpoint_corruption`** — Loss was stable at 2.847 for 3,000 steps, then spiked to 8.2 after resuming from checkpoint `step_15000.pt`. Root cause: checkpoint was saved mid-AllReduce sequence, creating a race condition where rank 0 wrote gradients to disk while ranks 1-7 hadn't finished their reduction. The checkpoint is internally inconsistent. Fix: add a `dist.barrier()` call before every `torch.save()`.

---

### 🔴 Hard

**`cross_dc_deadlock`** — 1536 GPUs across 3 datacenters. DC1 uses InfiniBand RDMA, DC2/DC3 use TCP (45ms and 67ms RTT). Training hung for 3 hours with **no error message, no crash, no timeout** — just two ranks waiting on each other forever. Classic circular send/recv deadlock: rank 0 queued `ncclSend(rank 512)` then blocked on `ncclRecv(rank 512)`, while rank 512 did the same in reverse. Neither can progress. Additionally: `NCCL_IB_TIMEOUT=14` is too low for cross-DC latency (should be 22+), and `NCCL_SOCKET_IFNAME` isn't set so NCCL may be routing traffic over the wrong interface.

**`nccl_config_drift`** — 512-GPU fat-tree cluster at 17.4% efficiency (34.7 GB/s vs 200 GB/s theoretical). Four engineering teams each configured their 4-node group differently:

| Nodes | NCCL_ALGO | NCCL_PROTO | NCCL_NET_GDR_LEVEL | NCCL_IB_HCA |
|-------|-----------|------------|---------------------|-------------|
| 0-3   | RING      | LL         | 5                   | mlx5_0      |
| 4-7   | TREE      | Simple     | 2 ← bottleneck      | mlx5_0      |
| 8-11  | RING      | LL128      | 5                   | mlx5_1 ← WRONG |
| 12-15 | RING      | Simple     | 5                   | mlx5_0      |

Nodes 4-7 with `GDR_LEVEL=2` are copying through host memory (45 GB/s) instead of NIC-direct (180 GB/s). Nodes 8-11 are routing training traffic over the 25 Gbps management network (`mlx5_1`) instead of the 200 Gbps spine (`mlx5_0`). Together they drag the whole cluster down.

**`grad_accum_mismatch`** — 512-GPU training run with `gradient_accumulation_steps=8` configured. Loss looks reasonable (3.2 after warmup) but hasn't moved in 2,800 steps — clear convergence failure. Root cause: nodes 8-15 have `gradient_accumulation_steps=4` (half of the rest), causing effective learning rate inconsistency across the collective. Gradients are being averaged over different numbers of micro-batches per rank before AllReduce, producing corrupted gradient tensors. Silent — no crash, no error, just a model that never learns.

---

## Multi-Step Investigation Loop

Agents aren't limited to a single observation. Before diagnosing, they can query the environment for additional diagnostic data — exactly like a real engineer would run `ibstat`, `nvidia-smi`, or `cat /proc/numactl` before making a call.

```python
# Reset to a scenario
