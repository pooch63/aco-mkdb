#!/bin/bash
# Regenerate every experiment JSON the paper emit pipeline needs, with
# post-JIT timing (ACO_RUNS=6 → run 1 discarded; θ-heuristic timed on 2nd solve).
#
# Covers:
#   1. Flag ablation (P×N) at k=2, θ=5 — 100 ants only for ACO / ACO-P / ACO-N / ACO-PN
#   2. Ant-count sweep with ACO-PN (P=N=true) → results/vary_k2t5i_PN
#   3. (k,θ) PN table runs: k∈{2,3,4} θ=5 and k=3 θ∈{5,6,7} at ants=100
#   4. Quality ant-count sweep (2..200) for the groupplot (ACO-PN)
#   5. Pivot seed comparison (konect-small by default)
#   6. Quick ACO vs θ-heuristic evaluate logs (optional)
#
# Usage:
#   ./scripts/regenerate-paper-data.bash              # all phases
#   PHASES=kt,flags ./scripts/regenerate-paper-data.bash
#   PHASES=quality,compare PREFIX=konect-small ./scripts/regenerate-paper-data.bash
#   DRY_RUN=1 ./scripts/regenerate-paper-data.bash    # print commands only
#   SKIP_EXISTING=0 ./scripts/regenerate-paper-data.bash  # overwrite JSON
#
# Phases (comma-separated via PHASES=…; default=all):
#   flags     — 4 P×N combos at k=2 θ=5, ants=100 (build.json flag_dirs)
#   sweep     — ACO-PN ant-count sweep (ANTS_SWEEP) → vary_k2t5i_PN
#   kt        — PN @ ants=100 for (2,5)(3,5)(4,5)(3,6)(3,7)
#   quality   — PN ants 2,5,10,20,50,100,200 (quality figure; ACO_RUNS=6)
#   compare   — compare-seeds on vary_k2t5i_PN (PREFIX=konect-small default)
#   evaluate  — scripts/evaluate.bash ACO vs θ logs
#
# Shared env (forwarded to vary.bash / compare-seeds / evaluate):
#   JULIA_THREADS  SEED  SKIP_EXISTING  PREFIX  INJECT  ITERATIONS  ACO_RUNS

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=common.bash
source "$ROOT/scripts/common.bash"

PHASES_RAW="${PHASES:-all}"
DRY_RUN="${DRY_RUN:-0}"
THREADS="${JULIA_THREADS:-8}"
SEED="${SEED:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
PREFIX="$(normalize_prefix "${PREFIX:-}")"
INJECT="${INJECT:-1}"
ITERATIONS="${ITERATIONS:-5}"
# 6 recorded runs → emit / best_trial discard run 1 → 5 counted replicates.
ACO_RUNS="${ACO_RUNS:-6}"
COMPARE_PREFIX="${COMPARE_PREFIX:-konect-small}"
COMPARE_TIMEOUT="${TIMEOUT:-2000}"

ANTS_TABLE="${ANTS_TABLE:-100}"
ANTS_QUALITY="${ANTS_QUALITY:-2,5,10,20,50,100,200}"

LOG_DIR="${LOG_DIR:-results/regenerate_paper_logs}"
mkdir -p "$LOG_DIR"

# Unique (k,θ) pairs: k=2,3,4 @ θ=5 and k=3 @ θ=5,6,7
KT_PAIRS=(
  "2:5"
  "3:5"
  "4:5"
  "3:6"
  "3:7"
)

want_phase() {
  local name="$1"
  if [[ "$PHASES_RAW" == "all" ]]; then
    return 0
  fi
  # ants100 is a legacy alias for the 100-ant flag ablation dirs.
  if [[ "$name" == "flags" ]]; then
    local IFS=','
    local p
    for p in $PHASES_RAW; do
      [[ "$p" == "flags" || "$p" == "ants100" ]] && return 0
    done
    return 1
  fi
  local IFS=','
  local p
  for p in $PHASES_RAW; do
    [[ "$p" == "$name" ]] && return 0
  done
  return 1
}

run_cmd() {
  local desc="$1"
  shift
  echo
  echo "========== $desc =========="
  echo "+ $*"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_vary() {
  local desc="$1"
  shift
  # Remaining args are env assignments then optional OUT_DIR positional.
  run_cmd "$desc" env \
    JULIA_THREADS="$THREADS" \
    SEED="$SEED" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    PREFIX="$PREFIX" \
    INJECT="$INJECT" \
    ITERATIONS="$ITERATIONS" \
    ACO_RUNS="$ACO_RUNS" \
    "$@" \
    ./scripts/vary.bash
}

echo "Paper data regeneration"
echo "  PHASES=$PHASES_RAW  DRY_RUN=$DRY_RUN  ACO_RUNS=$ACO_RUNS (run 1 = JIT)"
echo "  PREFIX=${PREFIX:-<all>}  SKIP_EXISTING=$SKIP_EXISTING  threads=$THREADS"
echo "  Logs under $LOG_DIR/"

# ---------------------------------------------------------------------------
# 1. Flag ablation (prefer-smaller-side × neighbor-scope) at k=2, θ=5, 100 ants
#    OUT_DIRs match paper/build.json flag_dirs / missing_at_base.
#    When the sweep phase also runs, skip ACO-PN here so vary_k2t5i_PN gets the
#    full ant-count range (emit still reads ants=100 for the ablation panel).
# ---------------------------------------------------------------------------
if want_phase flags; then
  echo
  echo "### Phase: flags (PxN ablation @ k=2 θ=5, ants=$ANTS_TABLE)"
  skip_pn=0
  if want_phase sweep; then
    skip_pn=1
    echo "(sweep phase also selected → ACO-PN deferred to ant-count sweep)"
  fi
  for prefer in true false; do
    for neighbor in true false; do
      if [[ "$skip_pn" == "1" && "$prefer" == "true" && "$neighbor" == "true" ]]; then
        continue
      fi
      # Explicit dirs for plain / ACO-N so they do not collide with sweep naming.
      out=""
      if [[ "$prefer" == "false" && "$neighbor" == "false" ]]; then
        out="results/vary_k2t5i_100_"
      elif [[ "$prefer" == "false" && "$neighbor" == "true" ]]; then
        out="results/vary_k2t5i_100_N"
      fi
      tag="prefer=${prefer}_neighbor=${neighbor}"
      log="${LOG_DIR}/flags_${tag}.log"
      echo "→ $tag  (log: $log${out:+, OUT_DIR=$out})"
      if [[ "$DRY_RUN" == "1" ]]; then
        if [[ -n "$out" ]]; then
          echo "+ K=2 THETA=5 ANTS_RANGE=$ANTS_TABLE PREFER_SMALLER_SIDE=$prefer ENABLE_NEIGHBOR_SCOPE_LIMIT=$neighbor OUT_DIR=$out ./scripts/vary.bash"
        else
          echo "+ K=2 THETA=5 ANTS_RANGE=$ANTS_TABLE PREFER_SMALLER_SIDE=$prefer ENABLE_NEIGHBOR_SCOPE_LIMIT=$neighbor ./scripts/vary.bash"
        fi
        continue
      fi
      if [[ -n "$out" ]]; then
        K=2 THETA=5 \
          ANTS_RANGE="$ANTS_TABLE" \
          PREFER_SMALLER_SIDE="$prefer" \
          ENABLE_NEIGHBOR_SCOPE_LIMIT="$neighbor" \
          OUT_DIR="$out" \
          JULIA_THREADS="$THREADS" SEED="$SEED" \
          SKIP_EXISTING="$SKIP_EXISTING" PREFIX="$PREFIX" \
          INJECT="$INJECT" ITERATIONS="$ITERATIONS" ACO_RUNS="$ACO_RUNS" \
          ./scripts/vary.bash >"$log" 2>&1
      else
        K=2 THETA=5 \
          ANTS_RANGE="$ANTS_TABLE" \
          PREFER_SMALLER_SIDE="$prefer" \
          ENABLE_NEIGHBOR_SCOPE_LIMIT="$neighbor" \
          JULIA_THREADS="$THREADS" SEED="$SEED" \
          SKIP_EXISTING="$SKIP_EXISTING" PREFIX="$PREFIX" \
          INJECT="$INJECT" ITERATIONS="$ITERATIONS" ACO_RUNS="$ACO_RUNS" \
          ./scripts/vary.bash >"$log" 2>&1
      fi
    done
  done
fi

# ---------------------------------------------------------------------------
# 3. (k,θ) PN runs for ACO vs θ tables / statistics (ants=100)
# ---------------------------------------------------------------------------
if want_phase kt; then
  echo
  echo "### Phase: kt (PN @ ants=$ANTS_TABLE for each (k,θ))"
  for pair in "${KT_PAIRS[@]}"; do
    k="${pair%%:*}"
    theta="${pair##*:}"
    # (2,5) is already produced by the sweep phase (full ant range) or by flags
    # (ants=100). Skip the redundant 100-ant-only rewrite when either ran.
    if [[ "$k" == "2" && "$theta" == "5" ]] && { want_phase sweep || want_phase flags; }; then
      echo "→ skip k=2 θ=5 (already covered by sweep/flags → vary_k2t5i_PN)"
      continue
    fi
    run_vary "ACO-PN k=$k θ=$theta ants=$ANTS_TABLE" \
      K="$k" THETA="$theta" ANTS_RANGE="$ANTS_TABLE" \
      PREFER_SMALLER_SIDE=true ENABLE_NEIGHBOR_SCOPE_LIMIT=true
  done
fi

# ---------------------------------------------------------------------------
# 4. Quality figure ant-count sweep (ACO-PN, ants 2..200)
# ---------------------------------------------------------------------------
if want_phase quality; then
  echo
  echo "### Phase: quality (PN ants=$ANTS_QUALITY → vary_k2t5i_2_200_PN*)"
  # Matches paper/build.json QUALITY input naming when ACO_RUNS=20 was used;
  # with ACO_RUNS=6 write to the canonical 2_200_PN dir (update build.json if needed).
  out="results/vary_k2t5i_2_200_PN"
  if [[ "$ACO_RUNS" != "6" ]]; then
    out="results/vary_k2t5i_2_200_PN_acoruns${ACO_RUNS}"
  fi
  run_vary "quality sweep → $out" \
    K=2 THETA=5 ANTS_RANGE="$ANTS_QUALITY" \
    PREFER_SMALLER_SIDE=true ENABLE_NEIGHBOR_SCOPE_LIMIT=true \
    OUT_DIR="$out"
fi

# ---------------------------------------------------------------------------
# 5. Pivot seed comparison
# ---------------------------------------------------------------------------
if want_phase compare; then
  echo
  echo "### Phase: compare (pivot seeding on vary_k2t5i_PN)"
  cprefix="${COMPARE_PREFIX}"
  if [[ -n "$PREFIX" ]]; then
    cprefix="$PREFIX"
  fi
  run_cmd "compare-seeds PREFIX=$cprefix TIMEOUT=$COMPARE_TIMEOUT" \
    env \
    JULIA_THREADS="$THREADS" \
    SEED="$SEED" \
    SKIP_EXISTING="$SKIP_EXISTING" \
    PREFIX="$cprefix" \
    TIMEOUT="$COMPARE_TIMEOUT" \
    INJECT="$INJECT" \
    K=2 THETA=5 \
    ./scripts/compare-seeds.bash results/vary_k2t5i_PN
fi

# ---------------------------------------------------------------------------
# 6. Evaluate (ACO vs θ text logs) — optional quick checks
# ---------------------------------------------------------------------------
if want_phase evaluate; then
  echo
  echo "### Phase: evaluate (ACO vs θ-heuristic .txt logs)"
  for pair in "${KT_PAIRS[@]}"; do
    k="${pair%%:*}"
    theta="${pair##*:}"
    run_cmd "evaluate k=$k θ=$theta" \
      env \
      JULIA_THREADS="$THREADS" \
      SEED="$SEED" \
      SKIP_EXISTING="$SKIP_EXISTING" \
      PREFIX="$PREFIX" \
      INJECT="$INJECT" \
      K="$k" THETA="$theta" \
      INJECT_U="$theta" INJECT_V="$theta" \
      ./scripts/evaluate.bash
  done
fi

echo
echo "Done."
echo "Next: point paper/build.json QUALITY.input at the quality OUT_DIR if needed,"
echo "      then: make paper"
echo "JIT note: ACO_RUNS=$ACO_RUNS records that many replicates; emit discards run 1."
