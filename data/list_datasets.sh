#!/usr/bin/env bash
# Write data/datasets.txt: one indexed dataset key per line.
# Honors .dataignore and always skips raw/ (via bin/order_graphs.jl).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUT="${SCRIPT_DIR}/datasets.txt"

julia "$ROOT/bin/order_graphs.jl" --keys-only > "$OUT"
echo "Wrote $(wc -l < "$OUT") datasets to $OUT"
