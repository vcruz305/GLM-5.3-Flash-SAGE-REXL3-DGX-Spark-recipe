# Measurements

Every number below was measured on one NVIDIA DGX Spark (GB10, SM121, 128 GB
unified memory) serving `vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw` through the
engine stack in the README. Run directories refer to the lane's `runs/<ts>/`
tree on the Spark; the engine's own log lines are quoted where they are the
datum. Nothing here is extrapolated: each table either reports a measured
value or says it was not measured.

Campaign window: 2026-11-05 to 2026-11-06.

## 1. Stack identity

| Component | Version |
|---|---|
| Docker image | `m31-flash-exl3-spark:dev-fdd7736` |
| vLLM | main snapshot 2026-10-22 |
| Plugin | `vllm-exl3` (in-image) |
| ExLlamaV3 | 1.5.0 |
| torch | 2.14 |
| Served footprint | 97.6 GiB, 150,894 tensors; load time ~140 s |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2`, revision `bf582e4eacc1810f76656d1811693ff6c6737d2a` (2026-08-31) |

## 2. AR baseline (no speculation) — 2026-11-05

Run `20261105-134910` (boot 7), MAXLEN 65536, MNBT 4096, MAXSEQS 2, KV dtype
auto, KVB 2 GiB, guard armed.

| Measurement | Value |
|---|---|
| Serial decode p50 (320 tok, temp 0, n=5) | 19.18 tok/s (spread 19.17-19.20) |
| bench_ladder decode medians (prose / structured / code / math) | 19.10 / 19.13 / 19.08 / 19.08 tok/s |
| Prefill, 34,265-token prompt | 590.3 tok/s (TTFT 58.1 s) |
| KV pool at 2 GiB | 116,508 tokens (1.78x concurrency at 64K) |

Window-independence, re-proven at MAXLEN 8192 (run `20261105-183905`): serial
p50 19.16 tok/s — identical within spread to the 64K figure. **AR decode does
not depend on the configured window.**

Flag A/B, one change per boot (runs `20261105-140804`, `-141620`, `-142421`;
discriminators serial p50 and prefill at ~32K):

| Change | Serial p50 / prefill tok/s | Verdict |
|---|---|---|
| Grouped mixed-K mode | 19.19 / 568.8 | No gain vs unified `exl3_moe_mixedk`; reverted |
| FP8 KV cache | 19.14 / 566.0 | No decode gain, no capacity gain under a byte-capped pool; reverted |
| K2-recipe bundle (MNBT 2048, seqs 1, expandable segments, autotune off) | 19.11 / 525.7 | Worse; rejected |

## 3. AR 262,144-token window proof — 2026-11-05

Run `20261105-191756`, MAXLEN 262144, MAXSEQS 2, KVB 5 GiB, no spec. Pool line:
511,500 tokens, concurrency 1.95x for one 262,144-token request. MemAvailable
after load 6.78 GiB; guard armed, never tripped.

| Probe | Result |
|---|---|
| Needle at 50% depth, 138,780-token haystack | **FOUND** (greedy, thinking off); decode at that depth 19.80 tok/s; health 200 after |
| Needle at 50% depth, 259,052-token haystack (uncached) | **FOUND**; decode at 259K depth 12.8-14.6 tok/s |
| Uncached prefill at 259K depth | ~444-484 tok/s => ~9-10 min TTFT for a full window |
| Page-crossing (29 sparse-MLA hot pages) | Generation continued across the boundary, finish=stop, health 200 — PASS |

Decode at extreme depth degrades (19.8 at 124K-class depth to ~13-15 at 259K
depth); short-depth decode is MAXLEN-independent. The window is real; retrieval
at 50% depth works at both tested depths.

## 4. Speculative decode: K-sweep at 8K-class windows — 2026-11-05

bench_ladder protocol: 1 warmup + 3x400 tok, temperature 0, thinking off.
AR baseline 19.10-19.18 tok/s on every class. Runs `20261105-175012` (K3),
`20261105-180907` (K5), `20261105-182800` (K8).

| Class | AR | K3 | K5 | K8 | Best |
|---|---:|---:|---:|---:|---|
| Prose | 19.1 | 23.84 | 22.25 | 20.37 | K3, 1.25x |
| Structured | 19.1 | 39.16 | 48.81 | 64.11 | K8, 3.35x |
| Code | 19.1 | 29.21 | 28.22 | 26.49 | K3, 1.53x |
| Math | 19.1 | 28.10 | 33.95 | 28.07 | K5, 1.78x |

Per-class acceptance (accepted/draft steps): structured 0.91-0.96, math
0.53-0.64, code 0.32-0.64, prose 0.21-0.29. **No single K dominates** — K3 is
the most balanced, K8 wins structured only. Quote per-class numbers; there is no
honest single headline.

## 5. Cross-stack acceptance: prompt parity — 2026-11-05

Run `20261105-230555`. The apparent acceptance gap vs the reference deployment
(gitcommit90 receipts) was re-measured on the reference's **own bench prompts**
(a parity test — never compare acceptance across stacks on different prompts):

| Class | Ours (per-position acceptance, 5 positions) | Their receipt | Ours tok/s |
|---|---|---|---:|
| Prose | [0.86, 0.74, 0.60, 0.44, 0.23] | [0.83, 0.55, 0.39, 0.31, 0.21] | 35.1 vs their 29.9 |
| Structured | [1.00, 0.97, 0.97, 0.95, 0.87] | [1.00, 1.00, 1.00, 0.97, 0.97] | 52.9 |
| Code | warm [0.80, 0.63, 0.44, 0.34, 0.23]; salted [0.91, 0.71, 0.56, 0.41, 0.37] | [0.94, 0.82, 0.72, 0.60, 0.52] | 30.3-34.9 vs their 40.1 |

Prose and structured match or exceed the reference at every position. Code is
the one residual gap; the vintage/runtime isolation work excluded the drafter
checkpoint vintage and the runtime as causes, and the gap remains unexplained.

Backend A/B (run `20261105-224525` vs `20261105-180907`): a `TRITON_ATTN`
drafter pin is **worthless** — per-position acceptance is identical under the
unpinned selector (which resolves FLASH_ATTN and implements the non-causal
in-draft path). The original cross-stack gap was prompt-mix, not the backend.

## 6. Drafter KV demand — measured, pre-port — 2026-11-05

The engine's own sizing-gate lines (requirement per one full-window request):

| MAXLEN | K=3 demand | K=5 demand |
|---|---:|---:|
| 8192 | 3.78 GiB | 4.81 GiB |
| 16384 | 4.20 GiB | — |
| 32768 | 4.55 GiB | — |
| 65536 | 5.25 GiB | — |
| 262144 | 8.92 GiB | 9.95 GiB |

Law: ~3.85 GiB fixed + ~21 KB/token, plus ~0.51 GiB per extra draft token.
Root cause: the drafter's sliding-window layers were page-unified into the
target's 8960-token pool blocks, so both the shared pool's per-block bytes and
the drafter's charged page count scaled with max_model_len. Pre-port ceilings
(the engine's own estimates): 206,080 tokens at K3, 98,560 tokens at K5 — the
262K spec window was **not servable**.

## 7. The disjoint-slab drafter KV partition — 2026-11-06

`patches/gv_drafter_slabs.py` (env `GV_DRAFTER_SLABS`): partition the drafter's
layers out of every unified-page path, run the stock planner on the rest (target
layout byte-identical), re-issue the drafter as one uniform group of exact
`SlidingWindowSpec` at its real kernel block (256), appended last. Companion:
the hybrid prefix-hit fixed point skips the drafter's exact-SWA group in the
reconciled min. CPU-only probe battery: 15/15 checks pass, including pool
per-block identity under all six KVCacheLayout values.

| Datum (MAXLEN 262144, K5) | Shadow (stock planner) | Patched |
|---|---:|---:|
| Per-request KV demand | 9.95 GiB | **5.16 GiB** |
| Pool token capacity | 98,560-token ceiling | 314,572 tokens, 1.20x concurrency |
| Pool per-block bytes | 184 MB (drafter page-unified slab is the max) | 69.27 MB (target-only; identity-proven) |
| Drafter group | 3 page-unified pages | 41 window-bounded pages == cdiv(min(2047+4+in_flight, 262144), 256)+1 |

The 20.5 KB/token linear term is gone; the port changes memory retention only —
compute is untouched, and the post-port 8K bench A/B deltas fall inside the
documented run-to-run spread in both directions.

## 8. The guard floor: what actually clears it — 2026-11-06

The 262K spec boot carries ~2 GiB more non-KV overhead than 8K (pool metadata,
indexer workspace, capture buffers). Measured trip forensics:

| Boot (run) | KVB | Outcome |
|---|---:|---|
| `20261106-122851` | 7 GiB | Guard trip at 0.48 GiB during post-warmup capture |
| `20261106-123641` | 6.5 GiB | Guard trip at 0.499 GiB — **free memory is insensitive to KVB** (0.48 vs 0.51 GiB); the binding consumer is not the pool |
| `20261106-125328` | 6.25 GiB + MNBT 2048 + slabs + indexer right-size | **UP**: health 200, guard never tripped, MemAvailable after capture 2.40 GiB |

MNBT 2048 halved max_in_flight (8192 -> 4096), which cut the drafter's
window-bounded demand to 5.16 GiB and raised the pool's token capacity above
the 7 GiB boot's. The indexer-workspace right-size (`GV_INDEXER_WS`) clamps the
process-locked prefill workspace to the legal maximum (65,538 entries at this
config) — but only ~127 MiB materializes at boot; it is a minor lever, not the
solution. **MNBT 2048 is the price of the 262K spec window** (see the prefill
table below).

## 9. Final spec server at 262,144 — 2026-11-06

Final boot `20261106-132831` (identical proven config to run `20261106-125328`
with the parser flags verified present): DFlash2 K5, MAXLEN 262144, MAXSEQS 1,
MNBT 2048, KVB 6.25 GiB, probabilistic/standard, unpinned drafter backend,
`--reasoning-parser glm45 --tool-call-parser glm45 --enable-auto-tool-choice`
all present in the engine args. Pool line: 314,572 tokens, 1.20x for one
262,144-token request. Guard armed the entire session, never tripped.

| Measurement | Value |
|---|---|
| Needle at 50% depth, 124,386-token haystack (unique salt, uncached) | **FOUND** (exact retrieval); TTFT 228.6 s; health 200 after |
| Decode at ~124K depth | **41.15 tok/s** (AR at that depth: 19.8) |
| Warm re-hit TTFT (byte-identical payload) | Cold 229.4 s -> warm **33.2 s (6.9x)**; the residual is the uncached tail, not a full re-prefill — the hybrid-prefix-hit companion works as ported |
| GSM8K-100, greedy, thinking-off | **94/100** (AR reference 93/100) — +1, inside the gate |
| HumanEval-50, greedy, thinking-off | **47/50** (AR reference 49/50) — -2, at the gate boundary; HumanEval/38 fails on both configs |

Quality parity holds under the ported spec server at 262K.

## 10. Quality on the AR final server — 2026-11-06

Run `20261106-003617` (MAXLEN 262144, MAXSEQS 2, KVB 5 GiB, MNBT 4096,
parser flags active; pool line identical to the GV4 final at 511,500 tokens).
All numbers greedy; the harness sets thinking mode explicitly.

| Subset | Thinking off | Thinking on |
|---|---:|---:|
| GSM8K-100 | **93/100** | **99/100** (matches the AR-64K reference) |
| HumanEval-50 | **49/50** (98.0% pass@1) | — |

Two facts to carry with these numbers. First, an earlier 98/100 GSM8K receipt
(run `20261105-203407`) was measured with a broken template that ignored the
thinking-off flag — every earlier benchmark effectively ran with CoT, and the
93/100 thinking-off figure is the honest no-CoT behavior, not damage. Second,
thinking-off *improves* HumanEval here (49/50 vs 43/50 under the broken
template) because CoT no longer burns the token budget. Thinking-off proof,
reasoning separation and tool-call preflight all pass on this server; the
chat API's observed default is thinking-off, verified by direct probe.

## 11. Prefill: the weak spot — 2026-11-05/06

Uncached, unique-salt, one request at a time:

| Depth | tok/s | Config |
|---|---:|---|
| ~11K | 495.8 | AR, MNBT 4096 |
| ~43-48K | 552.4 | AR, MNBT 4096 |
| ~135-140K | 559.3 | AR, MNBT 4096 |
| 124K depth | 577.5 | AR, MNBT 4096 (final AR server check) |
| 124K depth | **544.2** | Spec server, MNBT 2048 — the price of the 262K spec window |
| 259K depth | ~444-484 | AR — the length-collapse |

MNBT 7168 A/B (runs `20261105-234351` vs `20261106-001102`): +3.8% / +1.5% /
-0.3% at 11K / 48K / 140K — inside run-to-run spread, zero gain where TTFT
actually hurts. `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` is inert here in
every dimension (per-request demand, pool size, boot time): this lane pins
`--kv-cache-memory-bytes`, and the published env receipt applies only to the
auto-sized pool path.

Root cause (source-level, proven): every prefill row scores the entire
compressed prefix, and the SM120 deep_gemm path fire-and-forgets every fp32
(row,key) score to a global buffer that the top-k pass then re-reads in full;
the 512 MiB logits cap forces multi-launch splits as context grows (~17.6
TFLOPS effective — SIMT/memory-bound, not tensor-core). The fix — porting
upstream vLLM PR #60618's fused `fused_mqa_topk` (single-launch fused
scoring+top-k, no logits materialization) from SM100/tcgen05 to SM120 — is
**not built**. The reference deployment's flat ~800 tok/s plateau comes from a
NoPE sparse-prefill kernel rework that is not in this snapshot and is not a
config knob.

## 12. Negative results (measured, do not repeat)

| Candidate | Result |
|---|---|
| `TRITON_ATTN` drafter pin | Acceptance identical unpinned; the cross-stack gap was prompt-mix (parity proven on the reference's own prompts) |
| FP8 KV cache | No decode gain, no capacity gain under a byte-capped pool |
| Grouped mixed-K mode | No gain vs unified `exl3_moe_mixedk` |
| K2-recipe config bundle at 64K | Worse on both discriminators |
| MNBT 7168 | No gain (see the prefill table) |
| `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` | Inert under a pinned KVB |
| Indexer-workspace right-size alone | Only ~127 MiB frees at boot |
| KVB reduction for headroom | Free memory is insensitive to KVB; three wasted boots |

## 13. Not measured / open

- The K3/K8 ladder re-runs under the port (K5 post-port measured; the port does
  not change compute).
- Temp>0 behavior of the probabilistic drafter; the temp-0 non-neutrality is
  the only measured datum, so quality must be re-measured under the shipped
  sampling config.
- The SM120 `fused_mqa_topk` port (upstream PR #60618) — the prefill fix.
- The code-class acceptance residual vs the reference receipts (vintage and
  runtime excluded; unexplained).
