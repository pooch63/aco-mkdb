#!/bin/bash
# Epoch-trace suite for the anytime, best-of-R, allocation, and pheromone figures.
#
# Each counted replicate runs T epochs and logs the incumbent after every
# epoch (see vary.jl epoch_trace). Prefixes of that trace are the shorter
# budgets. R_max counted replicates support without-replacement best-of-R.
#
# Defaults match the paper setting except for the budgets:
#   A=100, T=100, ACO_RUNS=31 (run 1 is JIT; 30 counted replicates),
#   k=2, θ=5, F_P and F_N on, Δτ=1, ρ=0.05 (retention 0.95).
#
# Usage:
#   ./scripts/budget-trace.bash
#   ABLATION=1 ./scripts/budget-trace.bash          # also Δτ=0 (τ stays at 1)
#   RHO_SWEEP=1 ./scripts/budget-trace.bash         # also ρ ∈ {0.01, 0.2, 0.5}
#   ABLATION=1 RHO_SWEEP=1 ./scripts/budget-trace.bash
#   ACO_RUNS=101 ./scripts/budget-trace.bash        # 100 counted replicates
#   PREFIX=konect-small ./scripts/budget-trace.bash
#
# Outputs (under results/ unless OUT_DIR is set for the baseline only):
#   budget_k2t5i_PN            pheromone on, ρ=0.05
#   budget_k2t5i_PN_dt0        Δτ=0, evaporation off
#   budget_k2t5i_PN_rho001     ρ=0.01
#   budget_k2t5i_PN_rho020     ρ=0.2
#   budget_k2t5i_PN_rho050     ρ=0.5
#
# SKIP_EXISTING, ACO_TIMEOUT, RESUME_FROM, and SEED are forwarded to vary.bash.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ANTS_RANGE="${ANTS_RANGE:-100}"
ITERATIONS="${ITERATIONS:-100}"
# 31 = 1 JIT warmup + 30 counted replicates.
ACO_RUNS="${ACO_RUNS:-31}"
K="${K:-2}"
THETA="${THETA:-5}"
PREFER_SMALLER_SIDE="${PREFER_SMALLER_SIDE:-true}"
ENABLE_NEIGHBOR_SCOPE_LIMIT="${ENABLE_NEIGHBOR_SCOPE_LIMIT:-true}"
ABLATION="${ABLATION:-0}"
RHO_SWEEP="${RHO_SWEEP:-0}"
BASE_OUT="${OUT_DIR:-results/budget_k${K}t${THETA}i_PN}"

if ! [[ "$ACO_RUNS" =~ ^[1-9][0-9]*$ ]] || (( ACO_RUNS < 2 )); then
  echo "ACO_RUNS must be >= 2 (1 JIT + at least 1 counted replicate; got: $ACO_RUNS)" >&2
  exit 1
fi

run_one() {
  local out="$1"
  local phero="$2"
  local evap="$3"
  local label="$4"
  echo
  echo "======== $label ========"
  echo "  dir=$out  Δτ=$phero  retention(1-ρ)=$evap  T=$ITERATIONS  runs=$ACO_RUNS  A=$ANTS_RANGE"
  ANTS_RANGE="$ANTS_RANGE" \
  ITERATIONS="$ITERATIONS" \
  ACO_RUNS="$ACO_RUNS" \
  K="$K" \
  THETA="$THETA" \
  PREFER_SMALLER_SIDE="$PREFER_SMALLER_SIDE" \
  ENABLE_NEIGHBOR_SCOPE_LIMIT="$ENABLE_NEIGHBOR_SCOPE_LIMIT" \
  PHEROMONE="$phero" \
  EVAPORATION="$evap" \
  OUT_DIR="$out" \
    "$ROOT/scripts/vary.bash" "$out"
}

# Paper ρ=0.05 → multiply τ by 0.95. Δτ=1.
run_one "$BASE_OUT" "1" "0.95" "pheromone on (ρ=0.05)"

if [[ "$ABLATION" == "1" ]]; then
  # No deposit and no evaporation, so every trail stays at its initial 1.
  run_one "${BASE_OUT}_dt0" "0" "1" "pheromone ablation (Δτ=0)"
fi

if [[ "$RHO_SWEEP" == "1" ]]; then
  # ρ=0.05 is the baseline directory above.
  run_one "${BASE_OUT}_rho001" "1" "0.99" "ρ=0.01"
  run_one "${BASE_OUT}_rho020" "1" "0.80" "ρ=0.2"
  run_one "${BASE_OUT}_rho050" "1" "0.50" "ρ=0.5"
fi

echo
echo "Budget-trace suite finished."
echo "Baseline: $BASE_OUT"
