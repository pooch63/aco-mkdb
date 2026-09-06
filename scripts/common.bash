# Shared helpers for test.bash / vary.bash / compare-seeds.bash.
# PREFIX=konect-small  restrict discovery to data/<prefix>/...
# SKIP_EXISTING=1      skip a graph when its output JSON already exists
#                      (compare-seeds: timeout results are re-run only when
#                       TIMEOUT is strictly larger than the prior pivot_timeout_s;
#                       vary: ACO timeout results are re-run only when
#                       ACO_TIMEOUT is strictly larger than prior aco_timeout_s)

normalize_prefix() {
  local p="${1:-}"
  p="${p%/}"
  printf '%s' "$p"
}

dir_suffix_for_prefix() {
  local p
  p="$(normalize_prefix "$1")"
  if [[ -n "$p" ]]; then
    printf '_%s' "${p//\//_}"
  fi
}

order_graph_keys() {
  local prefix
  prefix="$(normalize_prefix "${1:-}")"
  local args=(--keys-only)
  if [[ -n "$prefix" ]]; then
    args+=(--prefix="$prefix")
  fi
  julia bin/order_graphs.jl "${args[@]}"
}

# Nest a bare directory name under results/ so sweeps don't litter the repo root.
# Absolute paths and paths that already contain a slash are left unchanged.
nest_under_results() {
  local d="$1"
  if [[ -z "$d" || "$d" == /* || "$d" == */* ]]; then
    printf '%s' "$d"
  else
    printf 'results/%s' "$d"
  fi
}

# Prefer an existing directory; if missing, also try results/<name>.
resolve_data_dir() {
  local d="$1"
  if [[ -d "$d" ]]; then
    printf '%s' "$d"
  elif [[ "$d" != /* && "$d" != results/* && -d "results/$d" ]]; then
    printf 'results/%s' "$d"
  else
    printf '%s' "$d"
  fi
}

# Exit 0 → skip existing compare JSON; exit 1 → re-run.
# Timeout results are skipped unless new_timeout > previous pivot_timeout_s.
should_skip_existing_compare() {
  local out="$1"
  local timeout="$2"
  python3 - "$out" "$timeout" <<'PY'
import json, sys

path, timeout = sys.argv[1], float(sys.argv[2])
try:
    with open(path) as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

if data.get("beat_heuristic") is False:
    sys.exit(0)

def timed_out(d):
    any_to = d.get("any_timed_out")
    if any_to is True:
        return True
    if any_to is False:
        return False
    tops = d.get("timed_out_pivots") or []
    if tops:
        return True
    for key in ("pivot_theta", "pivot_aco_seed"):
        p = d.get(key) or {}
        if p.get("timed_out") or p.get("status") == "timeout":
            return True
    return False

if not timed_out(data):
    sys.exit(0)

prev = data.get("pivot_timeout_s")
if prev is None:
    for key in ("pivot_theta", "pivot_aco_seed"):
        p = data.get(key) or {}
        if not (p.get("timed_out") or p.get("status") == "timeout"):
            continue
        wt = p.get("wall_time_s")
        if wt is None:
            continue
        prev = float(wt) if prev is None else max(float(prev), float(wt))

if prev is not None and timeout > float(prev):
    sys.exit(1)
sys.exit(0)
PY
}

# Exit 0 when every listed graph under OUT_DIR has a usable θ checkpoint
# (short pass complete for the suite). Exit 1 if any graph is still missing.
vary_short_pass_complete() {
  local out_dir="$1"
  shift
  python3 - "$out_dir" "$@" <<'PY'
import json, os, sys

out_dir = sys.argv[1]
keys = sys.argv[2:]
for key in keys:
    name = os.path.basename(key)
    path = os.path.join(out_dir, f"{name}_ants.json")
    if not os.path.isfile(path):
        sys.exit(1)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        sys.exit(1)
    heur = data.get("heuristic") or {}
    if heur.get("final_edges") is None and "wall_time_s" not in heur:
        sys.exit(1)
    if data.get("aco_status") == "running":
        sys.exit(1)
sys.exit(0)
PY
}

# Print count of graphs in OUT_DIR (from key list) that timed out with
# aco_timeout_s strictly below LONG — these need the long-budget retry.
# Echoes the integer count on stdout.
count_vary_need_long() {
  local out_dir="$1"
  local long_timeout="$2"
  shift 2
  python3 - "$out_dir" "$long_timeout" "$@" <<'PY'
import json, os, sys

out_dir, long_raw = sys.argv[1], sys.argv[2]
long = float(long_raw)
keys = sys.argv[3:]
n = 0
for key in keys:
    name = os.path.basename(key)
    path = os.path.join(out_dir, f"{name}_ants.json")
    if not os.path.isfile(path):
        continue
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        continue
    timed_out = data.get("aco_timed_out") is True or data.get("aco_status") == "timeout"
    if not timed_out:
        continue
    prev = data.get("aco_timeout_s")
    if prev is None or long > float(prev):
        n += 1
print(n)
PY
}

# Exit 0 → skip existing vary *_ants.json; exit 1 → re-run.
# Complete ACO results (or legacy JSON without timeout fields) are skipped.
# Timed-out / still-running checkpoints are re-run when new ACO_TIMEOUT is
# strictly larger than the previous aco_timeout_s (or when status is running).
# When ACO_TIMEOUT is empty/unset, only skip finished non-timeout results;
# timed-out files are re-run (caller wants an unlimited finish).
should_skip_existing_vary() {
  local out="$1"
  local timeout="${2:-}"
  python3 - "$out" "$timeout" <<'PY'
import json, sys

path, timeout_raw = sys.argv[1], sys.argv[2]
try:
    with open(path) as f:
        data = json.load(f)
except Exception:
    sys.exit(1)

# Must have a θ-heuristic block to count as a usable checkpoint/result.
heur = data.get("heuristic") or {}
if heur.get("final_edges") is None and "wall_time_s" not in heur:
    sys.exit(1)

status = data.get("aco_status")
timed_out = data.get("aco_timed_out") is True or status == "timeout"
running = status == "running"

if running:
    # Incomplete checkpoint from a killed process — always redo.
    sys.exit(1)

if not timed_out:
    # Finished ACO (or legacy JSON without timeout fields).
    sys.exit(0)

# Timed out previously.
if not timeout_raw:
    # No new timeout budget → re-run without a limit.
    sys.exit(1)

try:
    timeout = float(timeout_raw)
except ValueError:
    sys.exit(1)

prev = data.get("aco_timeout_s")
if prev is not None and timeout > float(prev):
    sys.exit(1)
sys.exit(0)
PY
}

# After a hard process kill (GNU timeout exit 124), mark checkpoint JSON as
# ACO timed out while preserving the θ-heuristic block.
mark_vary_aco_timeout() {
  local out="$1"
  local timeout="$2"
  python3 - "$out" "$timeout" <<'PY'
import json, sys

path, timeout = sys.argv[1], float(sys.argv[2])
try:
    with open(path) as f:
        data = json.load(f)
except Exception as e:
    print(f"mark_vary_aco_timeout: cannot read {path}: {e}", file=sys.stderr)
    sys.exit(1)

if data.get("aco_status") == "ok" and data.get("aco_timed_out") is not True:
    # Soft timeout already finished cleanly; leave alone.
    if data.get("trials"):
        sys.exit(0)

data["aco_timed_out"] = True
data["aco_timeout_s"] = timeout
data["aco_status"] = "timeout"
data.setdefault("trials", [])
# Incomplete budget: do not keep a best_trial from a partial run.
data.pop("best_trial", None)
data.pop("aco_discovery_s", None)

with open(path, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")
print(f"Patched ACO timeout → {path} (limit={timeout}s)")
PY
}
