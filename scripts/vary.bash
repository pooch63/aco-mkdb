#!/bin/bash
# Sweep ACO ant-count (--vary=ant-count) across every indexed graph under data/,
# ordered by edge count ascending so smaller / easier graphs finish first.
#
# Usage:
#   ./vary.bash
#   JULIA_THREADS=8 ./vary.bash
#   ANTS_RANGE=10,20,50,100 ITERATIONS=100 ./vary.bash
#   ACO_RUNS=10 ./vary.bash             # 10 seeded ACO replicates per ant count
#   RESUME_FROM=13 ./vary.bash          # skip graphs 1–12; start at #13

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

THREADS="${JULIA_THREADS:-8}"
ANTS_RANGE="${ANTS_RANGE:-1,2,5,10,20,50,100}"
ITERATIONS="${ITERATIONS:-3}"
ACO_RUNS="${ACO_RUNS:-20}"
SEED="${SEED:-1}"
RESUME_FROM="${RESUME_FROM:-1}"

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

OUT_DIR="vary_k${K}t${THETA}${INJECT_NAME}"
mkdir -p "$OUT_DIR"

echo "Discovering graphs (ascending by edges)…"
mapfile -t DATASETS < <(julia order_graphs.jl --keys-only)
n="${#DATASETS[@]}"
echo "Found $n graphs"
echo "ACO replicates per ant count: $ACO_RUNS"

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

i=0
for key in "${DATASETS[@]}"; do
  i=$((i + 1))
  if (( i < RESUME_FROM )); then
    continue
  fi

  name="$(basename "$key")"
  out="${OUT_DIR}/${name}_ants.json"

  echo
  echo "[$i/$n] $key → $out"

  julia -t "$THREADS" load.jl "$key" \
    --prefer-smaller-side=false \
    "${INJECT_ARGS[@]}" \
    --k="$K" --theta="$THETA" \
    --vary=ant-count \
    --ants-range="$ANTS_RANGE" \
    --iterations="$ITERATIONS" \
    --aco-runs="$ACO_RUNS" \
    "${SEED_ARGS[@]}" \
    --save="${OUT_DIR}/${name}_ants.json"
done

echo
echo "Done. JSON results under ${OUT_DIR}/"
