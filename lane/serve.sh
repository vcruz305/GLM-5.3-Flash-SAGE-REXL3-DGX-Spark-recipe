#!/bin/bash
# Host wrapper — launch the GLM-5.3-Flash SAGE-REXL3 vLLM serve on ONE DGX Spark.
#
# Model: the gated pack vcruz305/GLM-5.3-Flash-SAGE-REXL3-2.48bpw (Hugging Face),
#        staged in a local vLLM loader overlay directory (see README).
# Image: m31-flash-exl3-spark:dev-fdd7736 (vLLM main snapshot 2026-10-22 +
#        the vllm-exl3 plugin + exllamav3 1.5.0 + torch 2.14).
# Guard policy: 0.5 GiB MemAvailable floor, 64 MiB swap-growth allowance — NOT
#        optional on this unified-memory box. See docs/SAFETY.md.
#
# The engine command is generated as a payload FILE ($RUN/ctr/engine.sh,
# bind-mounted at /mem) and executed as `bash /mem/engine.sh`. Do not inline it
# into a `docker ... bash -c` string: bash array expansion inside a quoted
# payload splits the payload into separate argv words, so flags and the log
# redirect silently leak out as positional params. All values flow via
# container env vars (GV_*), and engine stdout lands in
# $RUN/ctr/vllm_full.log so trip forensics survive container removal.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="m31-flash-exl3-spark:dev-fdd7736"
CONTAINER=glm53-vllm
MODEL_HOST="${MODEL_DIR:-$HOME/models/GLM-5.3-Flash-SAGE-REXL3-2.48bpw-vllm-overlay}"
MODEL_CTR=/models/GLM-5.3-Flash-SAGE-REXL3-2.48bpw-vllm-overlay
DRAFTER_HOST="${DRAFTER_DIR:-$HOME/models/GLM-5.3-Flash-DFlash2}"
PORT="${PORT:-8000}"
MAXLEN="${MAXLEN:-65536}"
MNBT="${MNBT:-4096}"
MAXSEQS="${MAXSEQS:-2}"
KVB="${KVB:-2147483648}"
GMU="${GMU:-0.92}"
SPEC="${SPEC:-}"     # e.g. '{"method":"dflash","model":"/drafter","num_speculative_tokens":5,...}'
RUN="$HERE/runs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$RUN/ctr"
sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1 || true

# host memory policy (restored by guard.sh stop)
: >"$RUN/sysctl.before"
for k in vm.compaction_proactiveness vm.watermark_boost_factor vm.swappiness; do
  printf '%s=%s\n' "$k" "$(cat "/proc/sys/${k//.//}")" >>"$RUN/sysctl.before"
done
sudo -n sysctl -q vm.compaction_proactiveness=0 vm.watermark_boost_factor=0 vm.swappiness=0
sudo -n swapoff -a 2>/dev/null || echo "WARNING: swapoff failed; swap base nonzero" >&2
sudo -n swapon -a 2>/dev/null || true
sync; echo 1 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null; sleep 3
BASE_SWAP=$(awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo)
echo "pre-boot: MemAvailable $(awk '/^MemAvailable:/{printf "%.1f", $2/1048576}' /proc/meminfo) GiB  swap_base ${BASE_SWAP} KiB"

RECIPEARG_TXT=""
[ "${RECIPE:-0}" = 1 ] && RECIPEARG_TXT="--no-enable-flashinfer-autotune --skip-mm-profiling --long-prefill-token-threshold 1024"
CAPS_LIST=""
[ -n "${CAPS:-}" ] && CAPS_LIST="${CAPS}"
COMPCFG="{\"cudagraph_mode\":\"FULL_DECODE_ONLY\",\"custom_ops\":[\"all\"]"
[ -n "${CAPS:-}" ] && COMPCFG="${COMPCFG},\"cudagraph_capture_sizes\":[${CAPS}]"
COMPCFG="${COMPCFG}}"

# --- engine payload (no host-side expansion: values arrive via GV_* env) ---
cat >"$RUN/ctr/engine.sh" <<'PAYLOAD'
#!/bin/bash
ulimit -l unlimited 2>/dev/null
RA=${GV_RECIPE_ARGS:-}
python3 /recipe-py/gv_boot_patch.py || exit 1
python3 /recipe-py/gv_dummy_shrink_patch.py || exit 1
python3 /recipe-py/gv_diag_patch.py || exit 1
python3 /recipe-py/gv_drafter_slabs.py || exit 1
python3 /recipe-py/gv_indexer_ws.py || exit 1
SPECARG=()
[ -n "${GV_SPEC:-}" ] && SPECARG=(--speculative-config "$GV_SPEC")
PARSERARG=()
[ -n "${GV_PARSER:-}" ] && PARSERARG=(--reasoning-parser "$GV_PARSER" --tool-call-parser "$GV_PARSER" --enable-auto-tool-choice)
exec vllm serve "$GV_MODEL" \
  --served-model-name glm53-flash --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --tensor-parallel-size 1 \
  --quantization exl3 --kv-cache-dtype "${GV_KV_DTYPE:-auto}" \
  --max-model-len "$GV_MAXLEN" --max-num-seqs "$GV_MAXSEQS" \
  --max-num-batched-tokens "$GV_MNBT" \
  --gpu-memory-utilization "$GV_GMU" \
  --kv-cache-memory-bytes "$GV_KVB" \
  --enable-prefix-caching --enable-chunked-prefill \
  --async-scheduling --mm-processor-cache-gb 0 \
  --compilation-config "$GV_COMPCFG" \
  "${SPECARG[@]}" "${PARSERARG[@]}" $RA \
  >>/mem/vllm_full.log 2>&1
PAYLOAD
chmod 640 "$RUN/ctr/engine.sh"

CG_ENV=""
[ -n "${VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS:-}" ] && CG_ENV="-e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=$VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"

sudo -n docker run -d --name "$CONTAINER" \
  --gpus all \
  --memory=112g --memory-swap=112g --oom-score-adj 1000 --ulimit core=0 \
  --shm-size=2g \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --ulimit memlock=-1:-1 \
  -v "$HOME/models:/models:ro" \
  -v "$RUN/ctr:/mem:rw" \
  -v "$DRAFTER_HOST:/drafter:ro" \
  -v "$HERE/../patches:/recipe-py:ro" \
  -v "$HERE/../patches/zz_recipe_hooks.pth:/usr/local/lib/python3.12/dist-packages/zz_recipe_hooks.pth:ro" \
  -p "$PORT":8000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  ${CG_ENV} \
  -e NVCC_APPEND_FLAGS=-DP2B_CB=2 ${ALLOC_ENV:-} ${DFLASH_ENV:-} \
  -e GV_DIAG=${GV_DIAG:-} -e GV_DUMMY_SHRINK=${GV_DUMMY_SHRINK:-} \
  -e GV_DRAFTER_SLABS=${GV_DRAFTER_SLABS:-} \
  -e GV_INDEXER_WS=${GV_INDEXER_WS:-} \
  -e GV_PARSER=${GV_PARSER:-} \
  -e GV_RECIPE_ARGS="$RECIPEARG_TXT" \
  -e GV_MODEL="$MODEL_CTR" -e GV_MAXLEN="$MAXLEN" -e GV_MNBT="$MNBT" \
  -e GV_MAXSEQS="$MAXSEQS" -e GV_KVB="$KVB" -e GV_GMU="$GMU" \
  -e GV_COMPCFG="$COMPCFG" -e GV_SPEC="$SPEC" -e GV_KV_DTYPE="${KV_DTYPE:-auto}" \
  -e VLLM_EXL3_MODEL_DIR="$MODEL_CTR" \
  --entrypoint bash "$IMAGE" /mem/engine.sh &
CTR=$!
sleep 2
echo "run dir: $RUN  container pid job: $CTR"
echo "$RUN" > "$HERE/last_run"
nohup setsid "$HERE/guard.sh" watch "$CONTAINER" "$RUN" 524288 "$BASE_SWAP" > "$RUN/guard-wrap.log" 2>&1 < /dev/null &
echo "guard armed (0.5 GiB floor, 64 MiB swap allowance); logs: $RUN"
