#!/bin/bash
# Convert every KONECT download under data/raw/download.tsv.* into
# indexed graphs under data/konect-small/<name>/ via process.jl.
#
# If download.tsv.*.tar.bz2 (or .bz2) archives are present, extracts them
# with tar -xf first, then processes the resulting folders.
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

# Extract any download.tsv.*.tar.bz2 (or *.bz2) archives first, then process folders.
ARCHIVES=("$RAW_ROOT"/download.tsv.*.tar.bz2 "$RAW_ROOT"/download.tsv.*.bz2)
# Deduplicate: *.bz2 also matches *.tar.bz2; keep unique paths.
declare -A seen_archive=()
UNIQUE_ARCHIVES=()
for archive in "${ARCHIVES[@]}"; do
  [[ -f "$archive" ]] || continue
  [[ -n "${seen_archive[$archive]+x}" ]] && continue
  seen_archive[$archive]=1
  UNIQUE_ARCHIVES+=("$archive")
done

if (( ${#UNIQUE_ARCHIVES[@]} > 0 )); then
  echo "Found ${#UNIQUE_ARCHIVES[@]} .bz2 archive(s) under $RAW_ROOT/; extracting with tar -xf"
  for archive in "${UNIQUE_ARCHIVES[@]}"; do
    base="$(basename "$archive")"
    # download.tsv.<name>.tar.bz2 → download.tsv.<name>
    if [[ "$base" == *.tar.bz2 ]]; then
      dest_name="${base%.tar.bz2}"
    else
      dest_name="${base%.bz2}"
    fi
    dest="$RAW_ROOT/$dest_name"
    mkdir -p "$dest"
    echo "  tar -xf $archive → $dest/"
    tar -xf "$archive" -C "$dest"
  done
fi

DIRS=()
for path in "$RAW_ROOT"/download.tsv.*; do
  [[ -d "$path" ]] || continue
  DIRS+=("$path")
done
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
