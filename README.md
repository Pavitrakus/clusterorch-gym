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
