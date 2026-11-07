# GLM-5.3-Flash SAGE-REXL3 on one NVIDIA DGX Spark (vLLM)

Serve **GLM-5.3-Flash SAGE-REXL3-2.48bpw** on a single GB10 Spark through vLLM
plus the `vllm-exl3` plugin and ExLlamaV3, with the Inco AI DFlash2 drafter
providing speculative decode at the full 262,144-token window. This recipe
publishes the measured runtime: the launcher, the memory guard, the in-container
loader patches, the disjoint-slab drafter KV planner port that made the 262K
spec window servable, and every measurement behind the headline numbers. It is
independent community work, not an endorsement by Z.ai, NVIDIA or vLLM.

> **License notice — read before using the drafter.** The DFlash2 draft model
> ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2))
> is **CC BY-NC-ND 4.0**: non-commercial research and evaluation only. This
> repository does not bundle or modify it; commercial users must contact
> Inco AI for licensing. The SAGE-REXL3 pack itself is proprietary and stays
> behind a gated Hugging Face repo — it is referenced here, never redistributed.

## Headline results (one DGX Spark, GB10, 128 GB unified)

| Measurement | Result |
|---|---|
| AR serial decode p50 | 19.2 tok/s (window-independent: identical at 8K, 64K and 262K) |
| Spec decode at 124K context depth (DFlash2 K5, 262K window) | 41.15 tok/s — 2.1x the AR rate at that depth (19.8) |
| Spec decode by class at 8K-class windows (K3/K5/K8, best per class) | prose 23.8 (K3), structured 64.1 (K8), code 29.2 (K3), math 34.0 (K5) vs AR 19.1 on every class |
| Long-window retrieval | Needle at 50% depth of a 124K-token haystack: FOUND (greedy, uncached); page-crossing generation healthy |
| Warm re-hit TTFT at 124K depth | 229.4 s cold -> 33.2 s warm (6.9x); no full target re-prefill |
| Quality (greedy, thinking-off) | GSM8K-100 94/100 (AR reference 93/100); HumanEval-50 47/50 (AR reference 49/50) — inside the parity gate |
| Uncached prefill | 496 / 552 / 559 tok/s at 11K / 48K / 140K; 544 tok/s at 124K depth on the spec config; ~444-484 tok/s at 259K |

Every number, its run directory and its date is in
[docs/MEASUREMENTS.md](docs/MEASUREMENTS.md). The prefill length-collapse and
the not-shipped fix are documented there too.

## Hardware, engine and checkpoint

| | |
|---|---|
| Hardware | One NVIDIA DGX Spark: GB10, SM121 (compute capability 12.1), aarch64, 128 GB unified CPU/GPU memory |
| Engine | Docker image `m31-flash-exl3-spark:dev-fdd7736`: vLLM main snapshot 2026-10-22 + the `vllm-exl3` plugin + ExLlamaV3 1.5.0 + torch 2.14 |
| Checkpoint | [vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw](https://huggingface.co/vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw) (gated). 97.6 GiB served footprint, 150,894 tensors — every tensor EXL3 except 22 small DSA `q_a`/`kv_a` tensors the upstream loader holds in BF16 |
| Drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2) — **CC BY-NC-ND 4.0, non-commercial research/evaluation only** |
| Base model | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) |
| Guard | 0.5 GiB MemAvailable floor + 64 MiB swap watcher, armed every boot, auto-kill on trip — **non-optional** (see [docs/SAFETY.md](docs/SAFETY.md)) |

## Quick start

Run on the Spark from a clone of this repository. The launcher is
[lane/serve.sh](lane/serve.sh); the in-container patches it mounts live in
[patches/](patches/). Everything is env-var driven — the launcher never takes
a model path or a flag on the command line.

```bash
# Stage the gated pack and the drafter (each under $HOME/models by default;
# override with MODEL_DIR / DRAFTER_DIR). Request access to the gated repo.

# AR reference server: full 262,144-token window, no speculation.
MAXLEN=262144 MAXSEQS=2 KVB=5368709120 MNBT=4096 \
GV_PARSER=glm45 \
bash lane/serve.sh
```

```bash
# Final spec server: DFlash2 K5 speculative decode at the full window.
MAXLEN=262144 MAXSEQS=1 MNBT=2048 KVB=6710886400 \
RECIPE=1 GV_DUMMY_SHRINK=1 GV_DRAFTER_SLABS=1 GV_INDEXER_WS=1 \
GV_PARSER=glm45 \
SPEC='{"method":"dflash","model":"/drafter","num_speculative_tokens":5,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' \
bash lane/serve.sh
```

The knobs, exactly as the lane uses them:

| Env | Purpose |
|---|---|
| `MAXLEN` | `--max-model-len`. 262144 is the proven window; 8192 for cheap diagnostic boots |
| `KVB` | `--kv-cache-memory-bytes`, pinned explicitly — the pool is never auto-sized on this box |
| `MNBT` | `--max-num-batched-tokens`. 4096 on AR; **2048 is the price of the 262K spec window** (see measurements) |
| `MAXSEQS` | `--max-num-seqs`. 2 on AR, 1 on spec — treat the quotient as ONE full-length session plus headroom, never two |
| `RECIPE=1` | adds `--no-enable-flashinfer-autotune --skip-mm-profiling --long-prefill-token-threshold 1024` (required on spec boots: the autotune warmup transient trips the guard once the drafter is resident) |
| `SPEC` | the speculative-config JSON, base64-safe through the payload via the `GV_*` env channel — never inline it into a `bash -c` payload |
| `GV_PARSER=glm45` | adds `--reasoning-parser glm45 --tool-call-parser glm45 --enable-auto-tool-choice` |
| `GV_DUMMY_SHRINK=1` | shrinks the drafter's dummy `embed_tokens`/`lm_head` (2.28 GiB) that the speculator replaces with the target's anyway |
| `GV_DRAFTER_SLABS=1` | the disjoint-slab drafter KV partition — the port that makes 262K spec servable ([patches/gv_drafter_slabs.py](patches/gv_drafter_slabs.py)) |
| `GV_INDEXER_WS=1` | right-sizes the sparse-indexer prefill workspace to the legal maximum |
| `KV_DTYPE` | KV cache dtype; measured: fp8 gives no decode or capacity gain here — keep `auto` |

The API listens at `http://127.0.0.1:8000/v1` (override `PORT`; the engine binds
`0.0.0.0` inside the container, so keep the box firewalled or tunnel through
SSH — do not expose an unauthenticated server).

## The guard is not optional

[lane/guard.sh](lane/guard.sh) samples `/proc/meminfo` every 0.05 s, kills the
container and every `vllm serve` process when MemAvailable drops below
**0.5 GiB** or swap grows more than **64 MiB**, and restores the boot-time
sysctls from the run's `sysctl.before` on stop. On this unified-memory box an
overcommitted host does not return a clean OOM — it wedges the OS and sshd for
30-40 minutes first. The guard is armed by `serve.sh` on every boot; read the
evidence and the trip forensics protocol in [docs/SAFETY.md](docs/SAFETY.md).

## What is in this repository

| Path | Purpose |
|---|---|
| [lane/](lane/) | Host launcher (`serve.sh`) and memory guard (`guard.sh`) with sysctl save/restore |
| [patches/](patches/) | In-container boot patches: loader fixes, drafter dummy shrink, disjoint-slab KV planner port (+ CPU-only probe), indexer workspace right-size, loader-filter hooks |
| [bench/](bench/) | Measurement protocols: decode ladder, needle retrieval, prefill ladder, quality subsets — no prompt sets are bundled |
| [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md) | Every measured table with dates and run-directory provenance |
| [docs/SAFETY.md](docs/SAFETY.md) | The guard, the UMA wedge evidence, and what each trip taught |

## Known limitations

- **Prefill is the weak spot and the collapse is real**: 496-559 tok/s at
  11K-140K depth falling to ~444-484 tok/s at 259K (~9-10 min TTFT for a full
  window). Root cause is the fp32 logits round-trip in the SM121
  sparse-indexer path; the fix — porting upstream vLLM PR #60618's
  `fused_mqa_topk` to SM120 — is **not built**. Nothing in this repo changes it.
- **MNBT 2048 is the price of the 262K spec window**: prefill 544 tok/s at
  124K depth vs 577 on the AR-proven MNBT 4096. Do not "fix" one without
  re-deriving the other against the guard floor.
- **Measured dead ends** (do not spend a boot on them): a `TRITON_ATTN`
  drafter pin is worthless — per-position acceptance is identical unpinned, and
  the cross-stack acceptance gap was prompt-mix, proven by running the
  reference deployment's own prompts. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
  is inert under a pinned `--kv-cache-memory-bytes`. MNBT 7168 gives no gain.
  The indexer-workspace right-size frees only ~127 MiB at boot.
- **Quality must be re-measured under the shipped sampling config**: the
  probabilistic/standard drafter pair is required for temp>0 parity but is NOT
  output-neutral at temp 0 on this build — old temp-0 receipts do not transfer
  across it.
- The code-class acceptance residual vs the reference receipts is unexplained
  (checkpoint vintage and runtime excluded); the measured parity test shows
  our stack at or near the reference on prose and structured prompts.
- The spec path is not bit-exact: the target's own kernels are not
  bit-reproducible under greedy on this hardware, so the gate is quality parity
  within ~2 problems, not hash equality.

## Credits

- **Z.ai / GLM-5 team** for [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash).
- **Inco AI** for the DFlash2 drafter (**CC BY-NC-ND 4.0**, non-commercial
  research/evaluation).
- **Victor Cruz** for the `vllm-exl3` plugin and
  **turboderp-org/exllamav3** for the EXL3 runtime this serves on.
- [gitcommit90/glm-5.3-one-spark](https://github.com/gitcommit90/glm-5.3-one-spark)
  and **Weschera** for the reference deployments and cross-checks this recipe's
  measurements are validated against.
- The author's own
  [GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
  lineage for the harnesses, guard patterns and drafter lessons this lane builds
  on.

## License

Recipe-authored runtime code and documentation are distributed under
**AGPL-3.0-only**; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights, the SAGE pack
and the DFlash2 drafter are separately licensed and are **not** redistributed
by this repository.
