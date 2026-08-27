# Shared helpers for test.bash / vary.bash / compare-seeds.bash.
# PREFIX=konect-small  restrict discovery to data/<prefix>/...
# SKIP_EXISTING=1      skip a graph when its output JSON already exists
#                      (compare-seeds: timeout results are re-run only when
#                       TIMEOUT is strictly larger than the prior pivot_timeout_s)

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
