#!/bin/bash
# Sweep ACO ant-count (--vary=ant-count) across indexed graphs under data/,
# ordered by edge count ascending so smaller / easier graphs finish first.
#
# Usage:
#   ./scripts/vary.bash
#   PREFIX=konect-small ./scripts/vary.bash
#   JULIA_THREADS=8 ./scripts/vary.bash
#   ANTS_RANGE=10,20,50,100 ITERATIONS=100 ./scripts/vary.bash
#   ACO_RUNS=10 ./scripts/vary.bash             # 10 seeded ACO replicates per ant count
#   RUN_PIVOT=1 ./scripts/vary.bash             # also run branch-and-pivot for optimum (slow)
#   RESUME_FROM=13 ./scripts/vary.bash          # skip graphs 1–12; start at #13
#   SKIP_EXISTING=1 ./scripts/vary.bash         # skip graphs whose *_ants.json already exists
#
# Then compare pivot time on ACO-beats-heuristic trials:
#   PREFIX=konect-small ./scripts/compare-seeds.bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.bash
source "$ROOT/scripts/common.bash"

THREADS="${JULIA_THREADS:-8}"
ANTS_RANGE="${ANTS_RANGE:-100}"
ITERATIONS="${ITERATIONS:-5}"
ACO_RUNS="${ACO_RUNS:-5}"
SEED="${SEED:-1}"
RESUME_FROM="${RESUME_FROM:-1}"
RUN_PIVOT="${RUN_PIVOT:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PREFIX="$(normalize_prefix "${PREFIX:-}")"

# Optional inject (same defaults as test.bash k2t5i). Set INJECT=0 to disable.
INJECT="${INJECT:-1}"
K="${K:-2}"
THETA="${THETA:-5}"
INJECT_U="${INJECT_U:-5}"
INJECT_V="${INJECT_V:-5}"
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
OUT_DIR="${OUT_DIR:-vary_k${K}t${THETA}${INJECT_NAME}${DIR_SUFFIX}}"
mkdir -p "$OUT_DIR"

echo "Discovering graphs (ascending by edges)…"
if [[ -n "$PREFIX" ]]; then
  echo "Prefix filter: $PREFIX"
fi
mapfile -t DATASETS < <(order_graph_keys "$PREFIX")
n="${#DATASETS[@]}"
echo "Found $n graphs"
echo "ACO replicates per ant count: $ACO_RUNS"
echo "Run pivot for optimum: $RUN_PIVOT"
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

  julia -t "$THREADS" load.jl "$key" \
    --prefer-smaller-side=false \
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
