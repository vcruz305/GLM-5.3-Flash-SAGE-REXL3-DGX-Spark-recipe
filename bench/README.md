# Bench

This directory documents the measurement protocols used for the numbers in
[docs/MEASUREMENTS.md](../docs/MEASUREMENTS.md). **No prompt sets are
redistributed here** — the bench prompts belong to the harness in the
[K2 recipe](https://github.com/vcruz305/GLM-5.3-Flash-EXL3-K2-DGX-Spark-recipe)
(`scripts/bench_ladder.py`) and are referenced, not copied.

## Protocols

### Decode ladder (per-class speed)

Four classes (prose, structured, code, math), 1 warmup + 3x400 generated
tokens per class, temperature 0, thinking off. Median decode tok/s per class.
Acceptance is never inferred from throughput: read the engine's `/metrics`
(`accepted_tokens`, `verified_tokens_per_step`) directly. Zero acceptance
(0 of N proposed) is a loader bug, not draft quality — diagnose the drafter
weights before judging any spec number.

### Cross-stack acceptance parity

Never compare acceptance across stacks on different prompts — acceptance is
prompt-dependent and the comparison is meaningless. A parity test runs the
reference deployment's exact prompts on both stacks and compares
per-position acceptance. This is how the apparent cross-stack acceptance gap
was closed as prompt-mix (see MEASUREMENTS section 5).

### Needle retrieval

Needle at 50% depth of a long haystack, greedy, thinking off, **unique salt
per request** so no prefix-cache hit can fake the result. Record prompt token
count, TTFT, exact-retrieval verdict, decode rate at depth, and `/health`
after the probe. A sizing probe that shares the haystack leaves prefix-cache
hits that fake a 20x prefill rate — always salt.

### Prefill ladder

Uncached cold prefill at increasing depths, one request at a time, unique salt.
The engine's own loggers provide the rate. Cold and warm are different
measurements; report both and label them.

### Warm re-hit TTFT

Byte-identical payload sent twice; the second hit measures what the hybrid
prefix-hit path actually re-prefills. The residual TTFT is the uncached tail,
not a full re-prefill — that distinction is the validation of the
prefix-hit companion.

### Quality subsets

GSM8K-100 and HumanEval-50, greedy, with the thinking mode **recorded per
run** — a benchmark that cannot show its thinking mode is invalid, and every
pre-fix receipt on this stack effectively ran with CoT. HumanEval scoring runs
in a throwaway `python:3.12-slim` container with `--network none`. The spec
gate is quality parity within ~2 problems of the AR reference, not bit
equality: the target's own kernels are not bit-reproducible under greedy on
this hardware, so a bit-exactness gate discards the fastest server for a
reason that is not the spec path.

## What is deliberately not here

- Bench prompt sets (any class) — see the K2 recipe harness.
- Run logs, raw jsonl receipts, model weights, tokens.
