#!/bin/bash
# Record methods on the paper (k, θ) sweeps.
#
# ACO and the θ-heuristic already live in results/vary_k*t*i_PN. This
# runs whatever METHODS names through compare-methods, one JSON per
# (k, θ): results/method_sweep/k{k}t{theta}.json. Same graphs, seed,
# reduction (lo), and planted biclique as that vary suite.
#
#   ./scripts/method-sweep.bash
#   PAIR=2:5 ./scripts/method-sweep.bash
#   DATASET=amazon/boxes PAIR=2:5 ./scripts/method-sweep.bash
#   METHODS=double-greedy ./scripts/method-sweep.bash
#   DRY_RUN=1 ./scripts/method-sweep.bash
#
# METHODS is the list to run together on each graph (default the two
# that are not already in the vary JSON). A graph already in the output
# is skipped once every requested id is present, and re-run — all of
# those methods together — when one is missing. Leave aco and heuristic
# out of METHODS: the win-rate and runtime figures read those from the
# vary JSON. Graphs whose vary file has aco_timed_out are skipped too:
# those plots already drop them, and reduction often leaves them too
# large for a second full pass.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

METHODS="${METHODS:-double-greedy,clique-order-edge}"
PAIR="${PAIR:-}"
DATASET="${DATASET:-}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-1}"
THREADS="${JULIA_THREADS:-8}"
OUT_ROOT="results/method_sweep"

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/config"

mapfile -t JOBS < <(python3 - "$ROOT" "$METHODS" "$PAIR" "$DATASET" "$SEED" "$OUT_ROOT" <<'PY'
import json, os, re, sys

root, methods_raw, pair, dataset, seed, out_root = sys.argv[1:]
methods = [m.strip() for m in methods_raw.split(",") if m.strip()]
if not methods:
    sys.exit("METHODS is empty")
cfg = json.load(open(os.path.join(root, "paper", "build.json")))
dirs = list(cfg["fragments"]["COMPARE"]["param_dirs"].values())
seen = set()
want = None
if pair:
    k_s, _, t_s = pair.partition(":")
    want = (int(k_s), int(t_s))

def read_head(path):
    with open(path, "rb") as f:
        return f.read(262144).decode("utf-8", "ignore")

def dataset_of(path, blob=None):
    blob = read_head(path) if blob is None else blob
    m = re.search(r'"dataset"\s*:\s*"([^"]+)"', blob)
    if m:
        return m.group(1)
    with open(path) as f:
        data = json.load(f)
    return data.get("dataset")

def aco_timed_out(blob):
    return (
        '"aco_timed_out": true' in blob
        or '"aco_timed_out":true' in blob
    )

for rel in dirs:
    name = os.path.basename(str(rel).rstrip("/"))
    m = re.fullmatch(r"vary_k(\d+)t(\d+).*", name)
    if not m:
        print(f"skip {rel}: directory name has no k/theta", file=sys.stderr)
        continue
    k, theta = int(m.group(1)), int(m.group(2))
    if (k, theta) in seen:
        continue
    seen.add((k, theta))
    if want is not None and (k, theta) != want:
        continue
    if theta <= k:
        print(f"skip k={k} θ={theta}: compare-methods requires θ > k", file=sys.stderr)
        continue
    vary_dir = rel if os.path.isabs(rel) else os.path.join(root, "results", rel)
    if not os.path.isdir(vary_dir):
        print(f"skip k={k} θ={theta}: missing {vary_dir}", file=sys.stderr)
        continue
    datasets = []
    n_timeout = 0
    if dataset:
        datasets = [dataset]
    else:
        for fn in sorted(os.listdir(vary_dir)):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(vary_dir, fn)
            blob = read_head(path)
            if aco_timed_out(blob):
                n_timeout += 1
                continue
            key = dataset_of(path, blob)
            if key:
                datasets.append(key)
        if n_timeout:
            print(
                f"k={k} θ={theta}: skip {n_timeout} ACO timeout graph(s)",
                file=sys.stderr,
            )
    if not datasets:
        print(f"skip k={k} θ={theta}: no datasets in {vary_dir}", file=sys.stderr)
        continue
    config_path = os.path.join(out_root, "config", f"k{k}t{theta}.json")
    payload = {
        "name": f"method_sweep_k{k}t{theta}",
        "output": os.path.join(out_root, f"k{k}t{theta}.json"),
        "k": k,
        "theta": theta,
        "seed": int(seed),
        "reduction": "lo",
        "inject": True,
        "jit_warmup": True,
        "skip_completed": True,
        "datasets": datasets,
        "methods": [
            {"id": mid, "method": mid, "label": mid} for mid in methods
        ],
    }
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"{k}\t{theta}\t{config_path}\t{len(datasets)}")
PY
)

if ((${#JOBS[@]} == 0)); then
  echo "No (k, θ) suites to run." >&2
  exit 1
fi

echo "Method sweep: METHODS=$METHODS  seed=$SEED  threads=$THREADS"
echo "Output: $OUT_ROOT/k{k}t{theta}.json"
failed=0
for job in "${JOBS[@]}"; do
  IFS=$'\t' read -r k theta config ngraphs <<<"$job"
  log="$OUT_ROOT/logs/k${k}t${theta}.log"
  echo
  echo "=== k=$k θ=$theta  ($ngraphs graph(s))  → $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "+ julia -t $THREADS bin/compare-methods.jl --config=$config"
    continue
  fi
  set +e
  julia -t "$THREADS" bin/compare-methods.jl --config="$config" 2>&1 | tee "$log"
  ec=${PIPESTATUS[0]}
  set -e
  if [[ "$ec" -ne 0 ]]; then
    echo "k=$k θ=$theta failed (exit $ec); see $log" >&2
    failed=$((failed + 1))
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "$failed suite(s) failed." >&2
  exit 1
fi
echo
echo "Done. Figures read $OUT_ROOT once each k{k}t{theta}.json covers its vary suite."
