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

i=0
skipped_existing=0
for key in "${DATASETS[@]}"; do
  i=$((i + 1))
  if (( i < RESUME_FROM )); then
    continue
  fi

  name="$(basename "$key")"
  out="${OUT_DIR}/${name}_ants.json"

  echo
  echo "[$i/$n] $key → $out"

  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "Skipping (exists): $out"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  julia -t "$THREADS" bin/load.jl "$key" \
    --prefer-smaller-side="$PREFER_SMALLER_SIDE" \
    --neighbor-scope-limit="$ENABLE_NEIGHBOR_SCOPE_LIMIT" \
    --elite-pheromone="$ELITE_PHEROMONE" \
    --aco-tabu="$ACO_TABU" \
    --mmas="$MMAS" \
    --reduce=lo \
    "${INJECT_ARGS[@]}" \
    --k="$K" --theta="$THETA" \
    --vary=ant-count \
    --ants-range="$ANTS_RANGE" \
    --iterations="$ITERATIONS" \
    --aco-runs="$ACO_RUNS" \
    "${PIVOT_ARGS[@]}" \
    "${SEED_ARGS[@]}" \
    --save="$out"
done

echo
echo "Done. JSON results under ${OUT_DIR}/"
if (( skipped_existing > 0 )); then
  echo "Skipped existing: $skipped_existing"
fi
echo "Next: PREFIX=${PREFIX:-} ./scripts/compare-seeds.bash ${OUT_DIR}"
