#!/bin/bash
# ACO vs θ-heuristic on indexed graphs (no ant-count sweep, no pivot).
# Ordered by edge count ascending so smaller / easier graphs finish first.
#
# Usage:
#   ./scripts/test.bash
#   PREFIX=konect-small ./scripts/test.bash
#   JULIA_THREADS=8 PREFIX=konect-small ./scripts/test.bash
#   SKIP_EXISTING=1 PREFIX=konect-small ./scripts/test.bash
#   RUN_K5T6=1 PREFIX=konect-small ./scripts/test.bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.bash
source "$ROOT/scripts/common.bash"

THREADS="${JULIA_THREADS:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PREFIX="$(normalize_prefix "${PREFIX:-}")"
RUN_K5T6="${RUN_K5T6:-0}"
SEED="${SEED:-1}"

INJECT="${INJECT:-1}"
K="${K:-2}"
THETA="${THETA:-5}"
INJECT_U="${INJECT_U:-5}"
INJECT_V="${INJECT_V:-5}"
INJECT_NAME=""
[[ "$INJECT" == "1" ]] && INJECT_NAME="i"

DIR_SUFFIX="$(dir_suffix_for_prefix "$PREFIX")"
OUT_DIR="${OUT_DIR:-k${K}t${THETA}${INJECT_NAME}${DIR_SUFFIX}}"
mkdir -p "$OUT_DIR"

echo "Discovering graphs (ascending by edges)…"
if [[ -n "$PREFIX" ]]; then
  echo "Prefix filter: $PREFIX"
fi
mapfile -t DATASETS < <(order_graph_keys "$PREFIX")
n="${#DATASETS[@]}"
echo "Found $n graphs"
echo "Writing ACO vs heuristic logs under $OUT_DIR/"

if (( n == 0 )); then
  echo "No graphs to run." >&2
  exit 1
fi

SEED_ARGS=()
if [[ -n "$SEED" ]]; then
  SEED_ARGS=(--seed="$SEED")
fi

run_one() {
  local key="$1"
  local out_dir="$2"
  local k="$3"
  local theta="$4"
  local u="$5"
  local v="$6"
  local name out
  name="$(basename "$key")"
  out="${out_dir}/${name}.txt"

  echo "Processing $key → $out"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo "Skipping (exists): $out"
    return 0
  fi

  local inject_args=()
  if [[ "$INJECT" == "1" ]]; then
    inject_args=(--inject --u="$u" --v="$v")
  fi

  julia -t "$THREADS" load.jl "$key" \
    --prefer-smaller-side=false \
    "${inject_args[@]}" \
    --k="$k" --theta="$theta" \
    "${SEED_ARGS[@]}" \
    --benchmark=aco,heuristic > "$out"
}

i=0
for key in "${DATASETS[@]}"; do
  i=$((i + 1))
  echo
  echo "[$i/$n] $key"
  run_one "$key" "$OUT_DIR" "$K" "$THETA" "$INJECT_U" "$INJECT_V"
done

if [[ "$RUN_K5T6" == "1" || "$RUN_K5T6" == "true" ]]; then
  OUT_K5="k5t6i${DIR_SUFFIX}"
  mkdir -p "$OUT_K5"
  echo
  echo "Running k=5 θ=6 inject u=v=6 under $OUT_K5/"
  i=0
  for key in "${DATASETS[@]}"; do
    i=$((i + 1))
    echo
    echo "[$i/$n] $key (k5t6i)"
    run_one "$key" "$OUT_K5" 5 6 6 6
  done
fi

echo
echo "Done. Logs under ${OUT_DIR}/"
