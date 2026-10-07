#!/bin/bash
# Median-deviation ablations of evaporation, pheromone deposit, and the η exponent.
#
# Each sweep holds the colony at one ant count (default A=100) and varies one
# knob. Results use the same JSON shape as an ant-count sweep, with
# param_value on every trial. emit/quality.py plots the cross-graph median
# percent deviation from the θ-heuristic beside the ant-count curve.
#
# Defaults match the ant-count figure: k=2, θ=5, planted biclique, F_P and
# F_N, T=5 epochs, ACO_RUNS=6 (run 1 is JIT; emit keeps 5).
#
#   ρ (paper evaporation; ACO multiplies τ by 1-ρ):  0.01, 0.05, 0.1, 0.2, 0.5, 0.8
#   Δτ:                                                0, 1, 2, 5, 10
#   exponent on η:                                     1, 2, 3, 4, 5
#
# Usage:
#   ./scripts/param-ablation.bash
#   PREFIX=konect-small ./scripts/param-ablation.bash
#   ACO_TIMEOUT=600 ./scripts/param-ablation.bash
#
# Outputs (under results/ unless the flag letters change):
#   ablate_rho_k2t5i_PN
#   ablate_pheromone_k2t5i_PN
#   ablate_eta-exp_k2t5i_PN
#
# SKIP_EXISTING, ACO_TIMEOUT, RESUME_FROM, SEED, and PREFIX are forwarded
# to vary.bash. Do not set OUT_DIR; each sweep names its own directory.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RHO_RANGE="${RHO_RANGE:-0.01,0.05,0.1,0.2,0.5,0.8}"
PHEROMONE_RANGE="${PHEROMONE_RANGE:-0,1,2,5,10}"
ETA_RANGE="${ETA_RANGE:-1,2,3,4,5}"

export ANTS_RANGE="${ANTS_RANGE:-100}"
export ITERATIONS="${ITERATIONS:-5}"
export ACO_RUNS="${ACO_RUNS:-6}"
export K="${K:-2}"
export THETA="${THETA:-5}"
export PREFER_SMALLER_SIDE="${PREFER_SMALLER_SIDE:-true}"
export ENABLE_NEIGHBOR_SCOPE_LIMIT="${ENABLE_NEIGHBOR_SCOPE_LIMIT:-true}"

if [[ -n "${OUT_DIR:-}" ]]; then
  echo "Unset OUT_DIR; param-ablation.bash writes one directory per sweep." >&2
  exit 1
fi

run_one() {
  local param="$1"
  local range="$2"
  echo
  echo "======== $param  range=$range  ants=$ANTS_RANGE ========"
  VARY_MODE=param PARAM="$param" PARAM_RANGE="$range" \
    "$ROOT/scripts/vary.bash"
}

run_one rho "$RHO_RANGE"
run_one pheromone "$PHEROMONE_RANGE"
run_one eta-exp "$ETA_RANGE"

echo
echo "Param ablations finished."
echo "  results/ablate_rho_k${K}t${THETA}i_PN"
echo "  results/ablate_pheromone_k${K}t${THETA}i_PN"
echo "  results/ablate_eta-exp_k${K}t${THETA}i_PN"
