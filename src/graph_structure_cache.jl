#=
=================================================================================
Graph structure cache (per dataset under data/)
=================================================================================
Stores reduction-related graph metrics once per (k, θ, reduction, inject config)
so vary.jl and emit backfills do not reload graphs to recompute degrees.

File: data/<dataset>/graph_structure.json (tracked in git; CSVs stay ignored).
=#

const __GRAPH_STRUCTURE_CACHE_JL__ = true

using JSON3

const GRAPH_STRUCTURE_CACHE_FILE = "graph_structure.json"
const GRAPH_STRUCTURE_CACHE_VERSION = 1

"""
Max and mean degree over all vertices in a reduced frozen bipartite graph.
Average is `2|E|/(|U|+|V|)` (each edge contributes to one U and one V degree).
"""
function reduced_degree_stats(fg::FrozenBipartite)
    n = length(fg.u_ids) + length(fg.v_ids)
    n == 0 && return (0, 0.0)
    max_deg = 0
    for u in fg.u_ids
        max_deg = max(max_deg, degree_u(fg, u))
    end
    for v in fg.v_ids
        max_deg = max(max_deg, degree_v(fg, v))
    end
    avg_deg = (2 * length(fg.v_adj)) / n
    return (max_deg, avg_deg)
end

"""
Path to `graph_structure.json` for a dataset key (e.g. `amazon/boxes`).
"""
function graph_structure_cache_path(dataset_name::AbstractString;
    data_root::AbstractString="data")
    return joinpath(dataset_dir(dataset_name; data_root=data_root), GRAPH_STRUCTURE_CACHE_FILE)
end

"""
Stable cache key for one reduction setup.
`reduction` is the mode used for the one-shot peel (vary forces CNN → `simple`).
"""
function graph_structure_cache_key(k::Int, θ::Int, reduction::ReductionMode.T,
    inject; seed=nothing)
    red = reduction == ReductionMode.none ? "none" : "simple"
    parts = ["k=$k", "theta=$θ", "reduction=$red"]
    if inject.enabled
        seed === nothing && throw(ArgumentError(
            "graph structure cache requires seed when inject is enabled"))
        push!(parts, "inject=1", "u=$(inject.nU)", "v=$(inject.nV)",
            "seed=$(string(seed))")
    else
        push!(parts, "inject=0")
    end
    return join(parts, ",")
end

function _empty_graph_structure_cache()
    return Dict{String,Any}(
        "version" => GRAPH_STRUCTURE_CACHE_VERSION,
        "entries" => Dict{String,Any}(),
    )
end

function load_graph_structure_cache(path::AbstractString)
    if !isfile(path)
        return _empty_graph_structure_cache()
    end
    raw = JSON3.read(read(path, String), Dict{String,Any})
    entries = get(raw, "entries", Dict{String,Any}())
    if !(entries isa Dict)
        entries = Dict{String,Any}()
    end
    return Dict{String,Any}(
        "version" => get(raw, "version", GRAPH_STRUCTURE_CACHE_VERSION),
        "entries" => entries,
    )
end

function save_graph_structure_cache(path::AbstractString, cache::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, cache)
        println(io)
    end
    return path
end

function _entry_matches_graph(entry::AbstractDict, nU::Int, nV::Int, edges::Int,
    reduced_nU::Int, reduced_nV::Int, reduced_edges::Int)
    return Int(get(entry, "nU", -1)) == nU &&
        Int(get(entry, "nV", -1)) == nV &&
        Int(get(entry, "edges", -1)) == edges &&
        Int(get(entry, "reduced_nU", -1)) == reduced_nU &&
        Int(get(entry, "reduced_nV", -1)) == reduced_nV &&
        Int(get(entry, "reduced_edges", -1)) == reduced_edges
end

function _entry_has_degree_stats(entry::AbstractDict)
    return get(entry, "reduced_max_degree", nothing) !== nothing &&
        get(entry, "reduced_avg_degree", nothing) !== nothing
end

"""
Look up cached structure metrics. Returns `nothing` if missing or fingerprint mismatch.
"""
function lookup_graph_structure_cache(dataset_name::AbstractString, key::AbstractString,
    nU::Int, nV::Int, edges::Int, reduced_nU::Int, reduced_nV::Int, reduced_edges::Int;
    data_root::AbstractString="data")
    path = graph_structure_cache_path(dataset_name; data_root=data_root)
    cache = load_graph_structure_cache(path)
    entry = get(cache["entries"], String(key), nothing)
    entry === nothing && return nothing
    if !_entry_matches_graph(entry, nU, nV, edges, reduced_nU, reduced_nV, reduced_edges)
        return nothing
    end
    !_entry_has_degree_stats(entry) && return nothing
    return entry
end

"""
Write structure metrics to the dataset cache (creates/updates one entry).
"""
function store_graph_structure_cache!(dataset_name::AbstractString, key::AbstractString,
    nU::Int, nV::Int, edges::Int, reduced_nU::Int, reduced_nV::Int, reduced_edges::Int,
    reduced_max_degree::Int, reduced_avg_degree::Real;
    data_root::AbstractString="data")
    path = graph_structure_cache_path(dataset_name; data_root=data_root)
    cache = load_graph_structure_cache(path)
    entries = cache["entries"]
    entries[String(key)] = Dict{String,Any}(
        "nU" => nU,
        "nV" => nV,
        "edges" => edges,
        "reduced_nU" => reduced_nU,
        "reduced_nV" => reduced_nV,
        "reduced_edges" => reduced_edges,
        "reduced_max_degree" => reduced_max_degree,
        "reduced_avg_degree" => Float64(reduced_avg_degree),
    )
    save_graph_structure_cache(path, cache)
    println("Cached graph structure → $path [$key]")
    return path
end

"""
Return `(reduced_max_degree, reduced_avg_degree)` from cache or by computing once
from `fg`, then persist to `data/<dataset>/graph_structure.json`.
"""
function resolve_reduced_degree_stats!(dataset_name::AbstractString, key::AbstractString,
    nU::Int, nV::Int, edges::Int, fg::FrozenBipartite;
    data_root::AbstractString="data")
    reduced_nU = length(fg.u_ids)
    reduced_nV = length(fg.v_ids)
    reduced_edges = length(fg.v_adj)
    cached = lookup_graph_structure_cache(dataset_name, key, nU, nV, edges,
        reduced_nU, reduced_nV, reduced_edges; data_root=data_root)
    if cached !== nothing
        println("Graph structure from cache ($key)")
        return Int(cached["reduced_max_degree"]), Float64(cached["reduced_avg_degree"])
    end
    max_deg, avg_deg = reduced_degree_stats(fg)
    store_graph_structure_cache!(dataset_name, key, nU, nV, edges,
        reduced_nU, reduced_nV, reduced_edges, max_deg, avg_deg; data_root=data_root)
    return max_deg, avg_deg
end
