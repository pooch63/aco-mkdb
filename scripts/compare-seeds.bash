#!/bin/bash
# For each vary.jl ant-count JSON in a folder, run compare-seeds.jl.
# When ACO beat the θ-heuristic, uses the best beating ACO subgraph (max edges;
# min time_to_best on ties) as an extra branch-and-pivot seed and records the
# wall-time reduction. When ACO never beat θ, still writes a compact marker JSON
# (beat_heuristic=false) so SKIP_EXISTING can skip without re-parsing the vary
# file. Pivot timeouts are written into the compare JSON (timed_out_pivots +
# pivot_timeout_s); SKIP_EXISTING re-runs a timeout result only when TIMEOUT is
# strictly larger than the previous limit. Graphs are ordered by edge count
# ascending (same as vary.bash).
#
# Usage:
#   ./scripts/compare-seeds.bash
#   PREFIX=konect-small ./scripts/compare-seeds.bash
#   ./scripts/compare-seeds.bash vary_k2t5i
#   JULIA_THREADS=8 INJECT=1 ./scripts/compare-seeds.bash vary_k2t5i
#   RESUME_FROM=3 ./scripts/compare-seeds.bash vary_k2t5i
#   SKIP_EXISTING=1 PREFIX=konect-small ./scripts/compare-seeds.bash
#   TIMEOUT=2000 ./scripts/compare-seeds.bash vary_k2t5i
#   TIMEOUT=4000 ./scripts/compare-seeds.bash vary_k2t5i  # redo prior timeouts < 4000s

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.bash
source "$ROOT/scripts/common.bash"

THREADS="${JULIA_THREADS:-8}"
SEED="${SEED:-1}"
RESUME_FROM="${RESUME_FROM:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
TIMEOUT="${TIMEOUT:-2000}"
PREFIX="$(normalize_prefix "${PREFIX:-}")"

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

# Accept integer or decimal seconds (compare-seeds.jl --timeout=).
if ! [[ "$TIMEOUT" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v t="$TIMEOUT" 'BEGIN { exit !(t > 0) }'; then
  echo "TIMEOUT must be a positive number of seconds (got: $TIMEOUT)" >&2
  exit 1
fi

DIR_SUFFIX="$(dir_suffix_for_prefix "$PREFIX")"
DEFAULT_VARY="vary_k${K}t${THETA}${INJECT_NAME}${DIR_SUFFIX}"
VARY_DIR="${1:-$DEFAULT_VARY}"
OUT_DIR="${OUT_DIR:-compare_k${K}t${THETA}${INJECT_NAME}${DIR_SUFFIX}}"

if [[ ! -d "$VARY_DIR" ]]; then
  echo "Vary directory not found: $VARY_DIR" >&2
  echo "Run PREFIX=${PREFIX:-} ./scripts/vary.bash first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "Discovering vary JSONs (ascending by edges)…"
if [[ -n "$PREFIX" ]]; then
  echo "Prefix filter: $PREFIX"
fi
echo "Pivot timeout: ${TIMEOUT}s"
FILES=()
while IFS= read -r key; do
  name="$(basename "$key")"
  path="${VARY_DIR}/${name}_ants.json"
  [[ -f "$path" ]] && FILES+=("$path")
done < <(order_graph_keys "$PREFIX")

n="${#FILES[@]}"
echo "Found $n vary JSON file(s) in $VARY_DIR"
echo "Writing compare results under $OUT_DIR/"

if (( n == 0 )); then
  echo "No matching vary JSONs. Run PREFIX=${PREFIX:-} ./scripts/vary.bash first." >&2
  exit 1
fi
if (( RESUME_FROM > n )); then
  echo "RESUME_FROM=$RESUME_FROM is past the last file ($n)" >&2
  exit 1
fi
if (( RESUME_FROM > 1 )); then
  echo "Resuming from file #$RESUME_FROM (${FILES[RESUME_FROM-1]})"
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
markers=0
skipped_existing=0
for path in "${FILES[@]}"; do
  i=$((i + 1))
  if (( i < RESUME_FROM )); then
    continue
  fi

  base="$(basename "$path" .json)"
  # boxes_ants.json → boxes.json
  name="${base%_ants}"
  out="${OUT_DIR}/${name}.json"

  # Check exists *before* any Julia startup / vary-JSON parse.
  # Timeout results are re-run only when TIMEOUT > prior pivot_timeout_s.
  if [[ "$SKIP_EXISTING" == "1" && -f "$out" ]]; then
    echo
    echo "[$i/$n] $path → $out"
    if should_skip_existing_compare "$out" "$TIMEOUT"; then
      echo "Skipping (exists): $out"
      skipped_existing=$((skipped_existing + 1))
      continue
    fi
    echo "Re-running (larger timeout than prior timeout result): $out"
  fi

  echo
  echo "[$i/$n] $path → $out"

  set +e
  julia -t "$THREADS" compare-seeds.jl "$path" \
      "${INJECT_ARGS[@]}" \
      --k="$K" --theta="$THETA" \
      --timeout="$TIMEOUT" \
      "${SEED_ARGS[@]}" \
      --save="$out"
  status=$?
  set -e

  if (( status == 0 )); then
    if [[ -f "$out" ]] && grep -Eq '"beat_heuristic"[[:space:]]*:[[:space:]]*false' "$out"; then
      markers=$((markers + 1))
    else
      ran=$((ran + 1))
    fi
  elif (( status == 2 )); then
    # Legacy skip without a marker (should be rare after compare-seeds.jl update).
    echo "compare-seeds.jl skipped without writing $out" >&2
    rm -f "$out"
  else
    echo "compare-seeds.jl failed on $path (exit $status)" >&2
    exit 1
  fi
done

echo
echo "Done. Compared=$ran  no-beat markers=$markers  results under ${OUT_DIR}/"
if (( skipped_existing > 0 )); then
  echo "Skipped existing: $skipped_existing"
fi
echo "Plot with: python -m emit seed-compare ${OUT_DIR} -o seed-compare.tex"
