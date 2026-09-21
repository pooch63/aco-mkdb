"""
Shared graph-selection logic for quartile tests.

Exports:
  `selected_graphs()` → `Vector{@NamedTuple{key::String, path::String, edges::Int}}`

CLI flags (all optional):
  positional arg      – a single dataset key, e.g. `amazon/boxes`
  --N=<int>           – take the first N graphs (ascending by |E|); default 5
  --prefix=<str>      – restrict discovery to this provider/key prefix
"""

isdefined(@__MODULE__, :__PATHS_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))

const _QUARTILE_DATA_ROOT = joinpath(@__DIR__, "..", "..", "data")

function _parse_quartile_N(default::Int = 5)
    for arg in ARGS
        startswith(arg, "--N=") && return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function _parse_quartile_prefix()
    for arg in ARGS
        startswith(arg, "--prefix=") && return split(arg, "=", limit=2)[2]
    end
    return nothing
end

function _parse_quartile_single_graph()
    for arg in ARGS
        startswith(arg, "-") && continue
        return arg          # first non-flag positional arg
    end
    return nothing
end

"""
    selected_graphs() -> Vector{NamedTuple}

Return up to N real-world graphs sorted ascending by edge count.

If a bare positional argument (non-flag) is passed on the CLI, only that one
dataset is returned (regardless of --N). Otherwise every indexed graph under
`data/` is discovered, optionally filtered by `--prefix=`, and the first N
(ascending |E|) are returned.
"""
function selected_graphs()
    single = _parse_quartile_single_graph()
    if single !== nothing
        path = resolve_graph_path(single; data_root=_QUARTILE_DATA_ROOT)
        edges = isfile(path) ? count_indexed_edges(path) : 0
        return [(key=single, path=path, edges=edges)]
    end

    N      = _parse_quartile_N()
    prefix = _parse_quartile_prefix()
    all    = order_graphs_by_edges(; data_root=_QUARTILE_DATA_ROOT,
                                     ascending=true,
                                     prefix=prefix)
    return all[1:min(N, length(all))]
end
