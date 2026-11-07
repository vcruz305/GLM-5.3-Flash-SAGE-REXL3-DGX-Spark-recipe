#!/bin/bash
# LANE GV memory guard (adapted from AgentSparko recipe B guard.sh, MIT).
#   guard.sh status | watch <ctr> <stage> <log> <floor_kib> <swap_base_kib> | stop
# The floor is not optional: 0.5 GiB MemAvailable, swap growth <= 64 MiB, 0.05 s sampling.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-status}"
CONTAINER="${2:-glm53-vllm}"
RUN="${3:-$HERE/runs/na}"
FLOOR_KIB="${4:-524288}"
SWAP_BASE_KIB="${5:-0}"
SWAP_MAX=$((64 * 1024))
PORT=8000

swap_used() { awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo; }
avail_kib() { awk '/^MemAvailable:/{printf "%d", $2}' /proc/meminfo; }

snapshot() {
  echo "date: $(date -Is)"
  echo "container: $(sudo -n docker ps -a --filter name=$CONTAINER --format '{{.Status}}' | head -1)"
  echo "endpoint: $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/v1/models 2>/dev/null)"
  awk '/^MemAvailable:/{printf "mem_available_gib: %.2f\n", $2/1048576} /^SwapTotal:/{t=$2} /^SwapFree:/{f=$2} END{printf "swap_used_kib: %d\n", t-f}' /proc/meminfo
}

kill_all() {
  echo "GUARD TRIP $(date -Is): killing $CONTAINER" >> "$RUN/guard.log"
  sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1
  sudo -n pkill -9 -f "vllm serve.*glm53" 2>/dev/null
}

case "$MODE" in
status) snapshot ;;
stop)
  sudo -n docker rm -f "$CONTAINER" >/dev/null 2>&1
  pkill -f "guard.sh watch" 2>/dev/null
  if [ -s "$RUN/sysctl.before" ]; then
    while IFS='=' read -r k v; do sudo -n sysctl -q "$k=$v" >/dev/null 2>&1 || true; done < "$RUN/sysctl.before"
    echo "sysctls restored from $RUN/sysctl.before"
  fi
  echo "stopped" ;;
watch)
  echo "guard watch start $(date -Is) floor=${FLOOR_KIB}KiB swap_base=${SWAP_BASE_KIB}KiB" >> "$RUN/guard.log"
  MIN_AVAIL=999999
  while true; do
    A=$(avail_kib); S=$(swap_used)
    [ "$A" -lt "$MIN_AVAIL" ] && MIN_AVAIL=$A && echo "$(date -Is) min_avail_kib=$A" >> "$RUN/guard.log"
    if [ "$A" -lt "$FLOOR_KIB" ]; then
      echo "$(date -Is) FLOOR TRIP avail=${A}KiB" >> "$RUN/guard.log"
      kill_all; break
    fi
    if [ $(( S - SWAP_BASE_KIB )) -gt "$SWAP_MAX" ]; then
      echo "$(date -Is) SWAP TRIP swap=${S}KiB base=${SWAP_BASE_KIB}KiB" >> "$RUN/guard.log"
      kill_all; break
    fi
    sleep 0.05
  done ;;
esac
