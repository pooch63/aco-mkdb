#!/bin/bash
# Convert every extracted KONECT download under data/raw/download.tsv.* into
# indexed graphs under data/konect-small/<name>/ via process.jl.
#
# Layout expected per download:
#   data/raw/download.tsv.<name>/<name>/out.*
#
# Usage:
#   ./scripts/extract-konect-small.bash
#   SKIP_EXISTING=1 ./scripts/extract-konect-small.bash
#   JULIA_THREADS=8 ./scripts/extract-konect-small.bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

THREADS="${JULIA_THREADS:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
RAW_ROOT="${RAW_ROOT:-data/raw}"
PROVIDER="${PROVIDER:-konect-small}"

shopt -s nullglob
DIRS=("$RAW_ROOT"/download.tsv.*)
n="${#DIRS[@]}"

echo "Found $n download.tsv.* folder(s) under $RAW_ROOT/"
if (( n == 0 )); then
  echo "Nothing to extract." >&2
  exit 1
fi

i=0
ran=0
skipped_existing=0
for dir in "${DIRS[@]}"; do
  [[ -d "$dir" ]] || continue
  i=$((i + 1))

  mapfile -t inners < <(find "$dir" -mindepth 1 -maxdepth 1 -type d | sort)
  if (( ${#inners[@]} != 1 )); then
    echo "[$i/$n] Expected exactly one subdirectory in $dir (found ${#inners[@]})" >&2
    exit 1
  fi
  inner="${inners[0]}"
  name="$(basename "$inner")"

  mapfile -t outs < <(find "$inner" -maxdepth 1 -type f -name 'out.*' | sort)
  if (( ${#outs[@]} == 0 )); then
    echo "[$i/$n] No out.* file in $inner" >&2
    exit 1
  fi
  if (( ${#outs[@]} > 1 )); then
    echo "[$i/$n] Multiple out.* files in $inner; using ${outs[0]}" >&2
  fi
  source_path="${outs[0]}"
  out_dir="data/${PROVIDER}/${name}"

  echo
  echo "[$i/$n] $dir → ${PROVIDER}/${name}"
  echo "  source: $source_path"

  if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/indexed_interactions.csv" ]]; then
    echo "Skipping (exists): ${out_dir}/indexed_interactions.csv"
    skipped_existing=$((skipped_existing + 1))
    continue
  fi

  julia -t "$THREADS" process.jl "${PROVIDER}/${name}" --source="$source_path"
  ran=$((ran + 1))
done

echo
echo "Done. Processed=$ran  under data/${PROVIDER}/"
if (( skipped_existing > 0 )); then
  echo "Skipped existing: $skipped_existing"
fi
