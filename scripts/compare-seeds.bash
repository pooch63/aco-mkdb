#!/bin/bash
# For each vary.jl ant-count JSON in a folder, run compare-seeds.jl when ACO beat
# the θ-heuristic. Uses the best beating ACO subgraph (max edges; min time_to_best
# on ties) as an extra branch-and-pivot seed and records the wall-time reduction.
#
# Usage:
#   ./scripts/compare-seeds.bash
#   ./scripts/compare-seeds.bash vary_k2t5i
#   JULIA_THREADS=8 INJECT=1 ./scripts/compare-seeds.bash vary_k2t5i
#   RESUME_FROM=3 ./scripts/compare-seeds.bash vary_k2t5i

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VARY_DIR="${1:-vary_k2t5i}"
THREADS="${JULIA_THREADS:-8}"
SEED="${SEED:-1}"
RESUME_FROM="${RESUME_FROM:-1}"

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

if [[ ! -d "$VARY_DIR" ]]; then
  echo "Vary directory not found: $VARY_DIR" >&2
  exit 1
fi

OUT_DIR="compare_k${K}t${THETA}${INJECT_NAME}"
mkdir -p "$OUT_DIR"

mapfile -t FILES < <(find "$VARY_DIR" -maxdepth 1 -type f -name '*.json' | sort)
n="${#FILES[@]}"
echo "Found $n vary JSON file(s) in $VARY_DIR"
echo "Writing compare results under $OUT_DIR/"

if (( RESUME_FROM > n )); then
  echo "RESUME_FROM=$RESUME_FROM is past the last file ($n)" >&2
  exit 1
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
ran=0
skipped=0
for path in "${FILES[@]}"; do
  i=$((i + 1))
  if (( i < RESUME_FROM )); then
    continue
  fi

  base="$(basename "$path" .json)"
  # boxes_ants.json → boxes.json
  name="${base%_ants}"
  out="${OUT_DIR}/${name}.json"

  echo
  echo "[$i/$n] $path → $out"

  set +e
  julia -t "$THREADS" compare-seeds.jl "$path" \
      "${INJECT_ARGS[@]}" \
      --k="$K" --theta="$THETA" \
      "${SEED_ARGS[@]}" \
      --save="$out"
  status=$?
  set -e

  if (( status == 0 )); then
    ran=$((ran + 1))
  elif (( status == 2 )); then
    skipped=$((skipped + 1))
    rm -f "$out"
  else
    echo "compare-seeds.jl failed on $path (exit $status)" >&2
    exit 1
  fi
done

echo
echo "Done. Compared=$ran  skipped=$skipped  results under ${OUT_DIR}/"
echo "Plot with: python optimum-plot.py ${OUT_DIR} --mode=seed-compare -o seed-compare.tex"
