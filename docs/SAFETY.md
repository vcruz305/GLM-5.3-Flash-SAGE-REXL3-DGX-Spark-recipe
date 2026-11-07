# Safety: the memory guard on a unified-memory box

## Why the guard exists

The DGX Spark is a unified-memory (UMA) machine: one 128 GB pool backs both
the CPU and the GB10 GPU, and the model server's GPU allocations are host
allocations. When a workload overcommits that pool, this platform does not
return a clean engine OOM. The OS thrashes first: sshd and the rest of the
userspace stop responding, and the box stays wedged for tens of minutes after
the offending process is gone.

This is not theoretical. The reference deployments published the same failure
on this hardware: host memory wedges during load left sshd unresponsive for
30-40 minutes, repeatable, with swap death on a fully-committed pool. The
failure signature is a host-level wedge, not a vLLM error. An unguarded box
does not fail loudly — it disappears from the network while the engine is still
nominally running.

## The guard

`lane/guard.sh` is armed by `lane/serve.sh` on **every** boot, before the
engine starts:

- **Floor:** host MemAvailable below 524,288 KiB (0.5 GiB) kills the container
  and every `vllm serve` process, immediately.
- **Swap watcher:** swap growth beyond 64 MiB above the pre-boot baseline
  kills the same way.
- **Sampling:** 0.05 s. The minimum observed MemAvailable is logged on the way
  down, so a near-miss is visible in `guard.log` even without a trip.
- **Forensics survive the kill:** the guard removes the container, but engine
  stdout is redirected into the run directory (`runs/<ts>/ctr/vllm_full.log`)
  before the container ever starts, so the boot log outlives the container.
- **State restore:** `serve.sh` records the pre-boot values of
  `vm.compaction_proactiveness`, `vm.watermark_boost_factor` and
  `vm.swappiness` in `runs/<ts>/sysctl.before` and zeroes them for the boot;
  `guard.sh stop` restores them from that file.

## What the trips taught (all measured)

1. **The floor is not negotiable and free memory is insensitive to the KV
   budget.** Two 262K spec boots tripped at 0.48 GiB and 0.499 GiB MemAvailable
   at KVB 7 GiB and 6.5 GiB respectively — the difference between the budgets
   did not move the floor. Three boots were wasted trying to thread it with KVB
   reduction. The binding consumers were the 262K-scaled non-KV overheads
   (pool metadata, capture buffers, prefill workspaces), not the pool.
2. **An unarmed guard looks identical to an armed one until the box swaps.**
   A misordered argv once left the floor un-armed on every previous boot; the
   boots looked normal. The fix is procedural: read `guard.log` back after
   every boot and confirm the recorded floor value (`floor=524288KiB`).
3. **What actually cleared the floor** was reducing the real demand: the
   disjoint-slab drafter partition (removing the linear per-token KV term),
   the drafter dummy-module shrink, MNBT 2048 (halving max_in_flight), and the
   indexer-workspace clamp — documented with numbers in
   [MEASUREMENTS.md](MEASUREMENTS.md).
4. **The concurrency quotient is one full-length session plus headroom, never
   two.** Two full-length 262K streams need more tokens than the pool holds;
   the second is refused. Treating the printed concurrency as two full windows
   is how a box ends up wedged.

## Operating rules

- Never boot this stack without the guard armed. If `guard.log` does not show
  the floor value, stop and re-arm before loading weights.
- After any trip, restore the sysctls from the run's `sysctl.before` before
  diagnosing anything else (`guard.sh stop` does this).
- Never change global swap settings to work around a trip; the trip is the
  correct outcome of an impossible configuration.
- Keep `--oom-score-adj 1000`, `--ulimit core=0` and the read-only model
  mounts exactly as the launcher sets them.
