# Shared helpers for test.bash / vary.bash / compare-seeds.bash.
# PREFIX=konect-small  restrict discovery to data/<prefix>/...
# SKIP_EXISTING=1      skip a graph when its output JSON already exists

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
  julia order_graphs.jl "${args[@]}"
}
