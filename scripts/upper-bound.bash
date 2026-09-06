#!/usr/bin/env bash
# Upper bound on k-MDB edge count after CNN reduction for a vary.jl JSON.
# See scripts/upper-bound.jl (calls upper_bound from src/search.jl).
#
# Usage:
#   ./scripts/upper-bound.bash results/vary_k2t5i_PN/magazine_ants.json
#   VERBOSE=1 ./scripts/upper-bound.bash results/vary_k2t5i_PN/magazine_ants.json

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 <vary_*_ants.json> [--verbose]" >&2
  exit 1
fi

JSON_PATH="$1"
EXTRA=()
if (( $# == 2 )); then
  EXTRA+=("$2")
elif [[ "${VERBOSE:-0}" =~ ^(1|true|yes|on)$ ]]; then
  EXTRA+=(--verbose)
fi

exec julia --project="$ROOT" "$ROOT/scripts/upper-bound.jl" "$JSON_PATH" "${EXTRA[@]}"
