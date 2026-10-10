#!/usr/bin/env bash
# Reduction suite on the (k, θ) pairs in scripts/kt-sweep.bash.
# Writes results/reduce_sweep/k<k>t<theta>.json (one file per pair).
#
#   ./scripts/reduce-sweep.bash
#   ./scripts/reduce-sweep.bash konect-small/moreno   # subset of datasets
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec julia src/super-reduce.jl --sweep "$@"
