#!/bin/bash
# Rebuild graph datasets listed in data/datasets.txt (or another manifest).
#
# Usage:
#   ./scripts/fetch-datasets.bash
#   ./scripts/fetch-datasets.bash data/datasets.txt
#   SKIP_EXISTING=1 ./scripts/fetch-datasets.bash
#   PREFIX=konect-small ./scripts/fetch-datasets.bash
#   DRY_RUN=1 ./scripts/fetch-datasets.bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MANIFEST="${1:-data/datasets.txt}"
ARGS=("$MANIFEST")

if [[ "${SKIP_EXISTING:-1}" == "1" ]]; then
  ARGS+=(--skip-existing)
fi

if [[ -n "${PREFIX:-}" ]]; then
  ARGS+=(--prefix="$PREFIX")
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

exec julia bin/fetch_datasets.jl "${ARGS[@]}"
