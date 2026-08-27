#!/bin/bash
# Run vary.bash for all 4 (prefer-smaller-side × neighbor-scope-limit) settings.
# Output dirs: results/vary_kKtTHETAi_{P?}{N?} (P/N appended when that flag is true).
#
# The 4 flag combos run in parallel (separate processes). Each still uses
# JULIA_THREADS (default 8), so wall-clock needs ~4×JULIA_THREADS cores
# without oversubscription (e.g. 32 cores for the default).
#
# Usage:
#   ./scripts/tests.bash
#   PREFIX=konect-small ./scripts/tests.bash
#   JULIA_THREADS=8 SKIP_EXISTING=1 ./scripts/tests.bash
#   PARALLEL=0 ./scripts/tests.bash          # sequential (old behavior)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PARALLEL="${PARALLEL:-0}" # Set to true if you have 8 threads across 4 cores
LOG_DIR="${LOG_DIR:-results/vary_tests_logs}"
mkdir -p "$LOG_DIR"

run_one() {
  local prefer="$1" neighbor="$2"
  local tag="prefer=${prefer}_neighbor=${neighbor}"
  local log="${LOG_DIR}/${tag}.log"
  echo "========== PREFER_SMALLER_SIDE=$prefer  ENABLE_NEIGHBOR_SCOPE_LIMIT=$neighbor → $log =========="
  PREFER_SMALLER_SIDE="$prefer" \
    ENABLE_NEIGHBOR_SCOPE_LIMIT="$neighbor" \
    ./scripts/vary.bash >"$log" 2>&1
  echo "Finished $tag (exit $?)"
}

if [[ "$PARALLEL" == "0" ]]; then
  for prefer in true false; do
    for neighbor in true false; do
      echo
      run_one "$prefer" "$neighbor"
    done
  done
else
  pids=()
  for prefer in true false; do
    for neighbor in true false; do
      run_one "$prefer" "$neighbor" &
      pids+=($!)
    done
  done

  fail=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      fail=1
    fi
  done
  if (( fail )); then
    echo "One or more vary.bash jobs failed; see logs under $LOG_DIR/" >&2
    exit 1
  fi
fi

echo
echo "Done. All 4 prefer/neighbor settings finished."
echo "Logs: $LOG_DIR/"
