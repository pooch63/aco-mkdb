#!/bin/bash
# Sweep ACO ant-count (--vary=ant-count) across indexed graphs under data/,
# ordered by edge count ascending so smaller / easier graphs finish first.
#
# Usage:
#   ./scripts/vary.bash [OUT_DIR]
#   ./scripts/vary.bash results/my-sweep
#   ./scripts/vary.bash my-sweep          # → results/my-sweep
#   PREFIX=konect-small ./scripts/vary.bash
#   JULIA_THREADS=8 ./scripts/vary.bash
#   ANTS_RANGE=10,20,50,100 ITERATIONS=100 ./scripts/vary.bash
#   ACO_RUNS=6 ./scripts/vary.bash             # 6 seeded ACO replicates per ant count
#                                              # (run 1 = JIT warmup; emit uses 5)
#   ACO_TIMEOUT=10 ./scripts/vary.bash         # soft ACO search budget in seconds
#                                              # (θ always runs; JSON + SKIP_EXISTING upgrade)
#   ACO_TIMEOUT_SHORT=10 ACO_TIMEOUT_LONG=600 ./scripts/vary.bash
#                                              # two-pass: short over ALL graphs first
#                                              # (θ everywhere), then long only for graphs
#                                              # that timed out below LONG — long never
#                                              # starts until every graph has a short result
#   ACO_PROCESS_LIMIT=190 ./scripts/vary.bash  # hard kill whole julia after this many
#                                              # wall seconds (must be > ACO_TIMEOUT;
#                                              # default = ACO_TIMEOUT + 180 for load/θ)
#   RUN_PIVOT=1 ./scripts/vary.bash             # also run branch-and-pivot for optimum (slow)
#   RESUME_FROM=13 ./scripts/vary.bash          # skip graphs 1–12; start at #13
#   SKIP_EXISTING=1 ./scripts/vary.bash         # skip graphs whose *_ants.json already exists
#   DEBUG=false ./scripts/vary.bash             # skip post-reduction plant check
#   ENABLE_NEIGHBOR_SCOPE_LIMIT=false ./scripts/vary.bash  # sample full C (not N∩C)
#   PREFER_SMALLER_SIDE=false ./scripts/vary.bash          # disable smaller-side bias
#   ELITE_PHEROMONE=true ./scripts/vary.bash    # elitist pheromone emit (ablation; default off)
#   ACO_TABU=true ./scripts/vary.bash           # tabu repair on elites / bests (ablation; default off)
#   MMAS=true ./scripts/vary.bash               # MAX-MIN Ant System bounds (ablation; default off)
#   ELITE_PHEROMONE=true ACO_TABU=true MMAS=true ./scripts/vary.bash  # all three on
#
# OUT_DIR: positional arg, else $OUT_DIR env, else results/vary_kKtTHETAi_{P?}{N?}{E?}{T?}{M?}
# (flag letter appended when that option is true: P prefer-smaller-side, N neighbor-scope,
#  E elite-pheromone, T aco-tabu, M mmas). Bare names are nested under results/.
#
# Then compare pivot time on ACO-beats-heuristic trials:
#   PREFIX=konect-small ./scripts/compare-seeds.bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.bash
source "$ROOT/scripts/common.bash"

if (( $# > 1 )); then
  echo "Usage: $0 [OUT_DIR]" >&2
  exit 1
fi
OUT_DIR_ARG="${1:-}"

THREADS="${JULIA_THREADS:-8}"
ANTS_RANGE="${ANTS_RANGE:-100}"
ITERATIONS="${ITERATIONS:-5}"
ACO_RUNS="${ACO_RUNS:-6}"
# Soft wall-clock budget for ACO search only (passed to --aco-timeout=). Empty = no limit.
# Two-pass: set ACO_TIMEOUT_SHORT + ACO_TIMEOUT_LONG (LONG > SHORT). Legacy ACO_TIMEOUT
# alone is a single pass. If SHORT+LONG are set, ACO_TIMEOUT is ignored.
ACO_TIMEOUT="${ACO_TIMEOUT:-}"
ACO_TIMEOUT_SHORT="${ACO_TIMEOUT_SHORT:-}"
ACO_TIMEOUT_LONG="${ACO_TIMEOUT_LONG:-}"
# Hard wall-clock limit for the entire julia process (GNU timeout). Must be strictly
# greater than the active ACO timeout so load + CNN + θ-heuristic can finish before
# ACO's budget binds. Default: active timeout + 180s. Cleared between two-pass phases
# so each pass gets its own default unless the user set ACO_PROCESS_LIMIT explicitly.
ACO_PROCESS_LIMIT_USER="${ACO_PROCESS_LIMIT:-}"
ACO_PROCESS_LIMIT="$ACO_PROCESS_LIMIT_USER"
SEED="${SEED:-1}"
RESUME_FROM="${RESUME_FROM:-1}"
RUN_PIVOT="${RUN_PIVOT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
# When true, vary.jl verifies the planted biclique still exists after reduction.
DEBUG="${DEBUG:-true}"
export DEBUG
# Prefer last-node neighbors ∩ C when nonempty (ACO default). Set false to always use full C.
ENABLE_NEIGHBOR_SCOPE_LIMIT="${ENABLE_NEIGHBOR_SCOPE_LIMIT:-true}"
# Bias ACO toward the smaller bipartition side (ACO default). Set false to disable.
PREFER_SMALLER_SIDE="${PREFER_SMALLER_SIDE:-true}"
# Paper ablations — default off (baseline ACO). Set true to measure quality/runtime impact.
ELITE_PHEROMONE="${ELITE_PHEROMONE:-false}"
ACO_TABU="${ACO_TABU:-false}"
MMAS="${MMAS:-false}"
PREFIX="$(normalize_prefix "${PREFIX:-}")"

# Optional inject. Plant sides default to θ so the plant is θ-feasible.
# Set INJECT=0 to disable; override with INJECT_U / INJECT_V.
INJECT="${INJECT:-1}"
K="${K:-2}"
THETA="${THETA:-5}"
INJECT_U="${INJECT_U:-$THETA}"
INJECT_V="${INJECT_V:-$THETA}"
INJECT_NAME=""
[[ "$INJECT" == "1" ]] && INJECT_NAME="i"

if ! [[ "$RESUME_FROM" =~ ^[1-9][0-9]*$ ]]; then
  echo "RESUME_FROM must be a positive integer (got: $RESUME_FROM)" >&2
  exit 1
fi
if ! [[ "$ACO_RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ACO_RUNS must be a positive integer (got: $ACO_RUNS)" >&2
  exit 1
fi

TWO_PASS=0
if [[ -n "$ACO_TIMEOUT_SHORT" || -n "$ACO_TIMEOUT_LONG" ]]; then
  if [[ -z "$ACO_TIMEOUT_SHORT" || -z "$ACO_TIMEOUT_LONG" ]]; then
    echo "Set both ACO_TIMEOUT_SHORT and ACO_TIMEOUT_LONG for two-pass, or neither" >&2
    exit 1
  fi
  if ! [[ "$ACO_TIMEOUT_SHORT" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
     ! awk -v t="$ACO_TIMEOUT_SHORT" 'BEGIN { exit !(t > 0) }'; then
    echo "ACO_TIMEOUT_SHORT must be a positive number of seconds (got: $ACO_TIMEOUT_SHORT)" >&2
    exit 1
  fi
  if ! [[ "$ACO_TIMEOUT_LONG" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
     ! awk -v t="$ACO_TIMEOUT_LONG" 'BEGIN { exit !(t > 0) }'; then
    echo "ACO_TIMEOUT_LONG must be a positive number of seconds (got: $ACO_TIMEOUT_LONG)" >&2
    exit 1
  fi
  if ! awk -v s="$ACO_TIMEOUT_SHORT" -v l="$ACO_TIMEOUT_LONG" \
       'BEGIN { exit !(l > s) }'; then
    echo "ACO_TIMEOUT_LONG ($ACO_TIMEOUT_LONG) must be strictly greater than ACO_TIMEOUT_SHORT ($ACO_TIMEOUT_SHORT)" >&2
    exit 1
  fi
  TWO_PASS=1
  if [[ -n "$ACO_TIMEOUT" ]]; then
    echo "Note: ACO_TIMEOUT=$ACO_TIMEOUT ignored (using SHORT/LONG two-pass)"
  fi
elif [[ -n "$ACO_TIMEOUT" ]]; then
  if ! [[ "$ACO_TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
     ! awk -v t="$ACO_TIMEOUT" 'BEGIN { exit !(t > 0) }'; then
    echo "ACO_TIMEOUT must be a positive number of seconds (got: $ACO_TIMEOUT)" >&2
    exit 1
  fi
fi

validate_process_limit() {
  local soft="$1"
  ACO_PROCESS_LIMIT="$ACO_PROCESS_LIMIT_USER"
  if [[ -z "$ACO_PROCESS_LIMIT" ]]; then
    ACO_PROCESS_LIMIT="$(python3 -c "print(float('$soft') + 180.0)")"
  fi
  if ! [[ "$ACO_PROCESS_LIMIT" =~ ^[0-9]+([.][0-9]+)?$ ]] ||
     ! awk -v t="$ACO_PROCESS_LIMIT" 'BEGIN { exit !(t > 0) }'; then
    echo "ACO_PROCESS_LIMIT must be a positive number of seconds (got: $ACO_PROCESS_LIMIT)" >&2
    exit 1
  fi
  if ! awk -v soft="$soft" -v hard="$ACO_PROCESS_LIMIT" \
       'BEGIN { exit !(hard > soft) }'; then
    echo "ACO_PROCESS_LIMIT ($ACO_PROCESS_LIMIT) must be strictly greater than ACO timeout ($soft)" >&2
    echo "(process limit covers load + reduce + θ-heuristic + ACO; search budget is ACO only)" >&2
    exit 1
  fi
}

DIR_SUFFIX="$(dir_suffix_for_prefix "$PREFIX")"
# OUT_DIR: results/vary_kKtTHETAi_ then P/N/E/T/M for each enabled flag.
FLAGS=""
[[ "$PREFER_SMALLER_SIDE" == "true" ]] && FLAGS+="P"
[[ "$ENABLE_NEIGHBOR_SCOPE_LIMIT" == "true" ]] && FLAGS+="N"
[[ "$ELITE_PHEROMONE" == "true" ]] && FLAGS+="E"
[[ "$ACO_TABU" == "true" ]] && FLAGS+="T"
[[ "$MMAS" == "true" ]] && FLAGS+="M"
if [[ -n "$OUT_DIR_ARG" ]]; then
  OUT_DIR="$OUT_DIR_ARG"
else
  OUT_DIR="${OUT_DIR:-results/vary_k${K}t${THETA}${INJECT_NAME}_${FLAGS}${DIR_SUFFIX}}"
fi
OUT_DIR="$(nest_under_results "$OUT_DIR")"
mkdir -p "$OUT_DIR"

echo "Discovering graphs (ascending by edges)…"
if [[ -n "$PREFIX" ]]; then
  echo "Prefix filter: $PREFIX"
fi
mapfile -t DATASETS < <(order_graph_keys "$PREFIX")
n="${#DATASETS[@]}"
echo "Found $n graphs"
echo "ACO replicates per ant count: $ACO_RUNS (run 1 = JIT warmup; emit uses the rest)"
if [[ "$TWO_PASS" == "1" ]]; then
  echo "ACO two-pass: short=${ACO_TIMEOUT_SHORT}s then long=${ACO_TIMEOUT_LONG}s"
  echo "  (long starts only after every graph has a short-pass JSON)"
elif [[ -n "$ACO_TIMEOUT" ]]; then
  validate_process_limit "$ACO_TIMEOUT"
  echo "ACO search budget (soft): ${ACO_TIMEOUT}s"
  echo "Julia process limit (hard kill): ${ACO_PROCESS_LIMIT}s"
else
  echo "ACO timeout: none"
fi
echo "Run pivot for optimum: $RUN_PIVOT"
echo "Neighbor scope limit: $ENABLE_NEIGHBOR_SCOPE_LIMIT"
echo "Prefer smaller side: $PREFER_SMALLER_SIDE"
echo "Elite pheromone (emit): $ELITE_PHEROMONE"
echo "ACO tabu repair: $ACO_TABU"
echo "MMAS bounds: $MMAS"
echo "DEBUG (post-reduction plant check): $DEBUG"
echo "Writing vary results under $OUT_DIR/"

if (( n == 0 )); then
  echo "No graphs to run." >&2
  exit 1
fi
if (( RESUME_FROM > n )); then
  echo "RESUME_FROM=$RESUME_FROM is past the last graph ($n)" >&2
  exit 1
fi
if (( RESUME_FROM > 1 )); then
  echo "Resuming from graph #$RESUME_FROM (${DATASETS[RESUME_FROM-1]})"
fi

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed="$SEED")
fi

INJECT_ARGS=()
if [[ "$INJECT" == "1" ]]; then
  INJECT_ARGS=(--inject --u="$INJECT_U" --v="$INJECT_V")
fi

PIVOT_ARGS=(--vary-pivot=false)
if [[ "$RUN_PIVOT" == "1" ]]; then
  PIVOT_ARGS=(--vary-pivot=true)
fi

# Run one ACO-timeout pass over DATASETS. Sets ACO_TIMEOUT for skip/julia.
# Accumulates skipped_existing and timed_out_count into caller-visible names.
run_vary_pass() {
  local pass_label="$1"
  local budget="$2"
  local i=0
  local pass_skipped=0
  local pass_timed_out=0

  ACO_TIMEOUT="$budget"
  local timeout_args=()
  local use_hard_kill=0
  if [[ -n "$ACO_TIMEOUT" ]]; then
    validate_process_limit "$ACO_TIMEOUT"
    timeout_args=(--aco-timeout="$ACO_TIMEOUT")
    use_hard_kill=1
    echo
    echo "── Pass: $pass_label (ACO_TIMEOUT=${ACO_TIMEOUT}s, process limit=${ACO_PROCESS_LIMIT}s) ──"
  else
    echo
    echo "── Pass: $pass_label (no ACO timeout) ──"
  fi

  for key in "${DATASETS[@]}"; do
    i=$((i + 1))
    if (( i < RESUME_FROM )); then
      continue
    fi

    local name out
    name="$(basename "$key")"
    out="${OUT_DIR}/${name}_ants.json"

    echo
    echo "[$i/$n] $key → $out"

    if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
      if should_skip_existing_vary "$out" "$ACO_TIMEOUT"; then
        echo "Skipping (exists): $out"
        pass_skipped=$((pass_skipped + 1))
        continue
      fi
      echo "Re-running (timeout upgrade or incomplete): $out"
    fi

    local julia_cmd=(
      julia -t "$THREADS" bin/load.jl "$key"
      --prefer-smaller-side="$PREFER_SMALLER_SIDE"
      --neighbor-scope-limit="$ENABLE_NEIGHBOR_SCOPE_LIMIT"
      --elite-pheromone="$ELITE_PHEROMONE"
      --aco-tabu="$ACO_TABU"
      --mmas="$MMAS"
      --reduce=lo
      "${INJECT_ARGS[@]}"
      --k="$K" --theta="$THETA"
      --vary=ant-count
      --ants-range="$ANTS_RANGE"
      --iterations="$ITERATIONS"
      --aco-runs="$ACO_RUNS"
      "${timeout_args[@]}"
      "${PIVOT_ARGS[@]}"
      "${SEED_ARGS[@]}"
      --save="$out"
    )

    if [[ "$use_hard_kill" == "1" ]]; then
      # Soft deadline inside Julia (ACO only); hard kill if a single epoch never returns.
      set +e
      timeout -k 15 "$ACO_PROCESS_LIMIT" "${julia_cmd[@]}"
      local ec=$?
      set -e
      if [[ $ec -eq 124 ]]; then
        echo "Hard process kill after ${ACO_PROCESS_LIMIT}s; patching $out (ACO budget was ${ACO_TIMEOUT}s)"
        mark_vary_aco_timeout "$out" "$ACO_TIMEOUT"
        pass_timed_out=$((pass_timed_out + 1))
      elif [[ $ec -ne 0 ]]; then
        exit "$ec"
      elif [[ -f "$out" ]] && python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get('aco_timed_out') or d.get('aco_status')=='timeout' else 1)" "$out"; then
        pass_timed_out=$((pass_timed_out + 1))
      fi
    else
      "${julia_cmd[@]}"
    fi
  done

  skipped_existing=$((skipped_existing + pass_skipped))
  timed_out_count=$((timed_out_count + pass_timed_out))
  echo "Pass $pass_label done (skipped=$pass_skipped, timed_out=$pass_timed_out)"
}

skipped_existing=0
timed_out_count=0

if [[ "$TWO_PASS" == "1" ]]; then
  # Short over the full suite first — θ for every graph. Long only after that.
  run_vary_pass "short" "$ACO_TIMEOUT_SHORT"

  if ! vary_short_pass_complete "$OUT_DIR" "${DATASETS[@]}"; then
    echo
    echo "Short pass incomplete (missing/running JSON under $OUT_DIR/)."
    echo "Not starting long pass — re-run to finish short, then long will start."
  else
    need_long="$(count_vary_need_long "$OUT_DIR" "$ACO_TIMEOUT_LONG" "${DATASETS[@]}")"
    if [[ "$need_long" == "0" ]]; then
      echo
      echo "Short pass complete; no graphs need long budget (ACO_TIMEOUT_LONG=${ACO_TIMEOUT_LONG}s)."
    else
      echo
      echo "Short pass complete; $need_long graph(s) timed out below ${ACO_TIMEOUT_LONG}s — starting long pass."
      run_vary_pass "long" "$ACO_TIMEOUT_LONG"
    fi
  fi
else
  run_vary_pass "single" "$ACO_TIMEOUT"
fi

echo
echo "Done. JSON results under ${OUT_DIR}/"
if (( skipped_existing > 0 )); then
  echo "Skipped existing: $skipped_existing"
fi
if (( timed_out_count > 0 )); then
  echo "ACO timed out: $timed_out_count (re-run with larger ACO_TIMEOUT / ACO_TIMEOUT_LONG to retry)"
fi
echo "Next: PREFIX=${PREFIX:-} ./scripts/compare-seeds.bash ${OUT_DIR}"
