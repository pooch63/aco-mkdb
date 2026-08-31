#!/usr/bin/env julia
#=
Measure mean missing edges when an ant's subgraph first reaches a target size.

Usage:
  julia scripts/missing-at-size.jl <vary_dir_or_json> [--size=5] [--ants=100]

When given a directory, processes every *.json and prints:
  MEAN,<pooled mean over all ant samples>
  GRAPHS,<number of JSON files processed>
  SAMPLES,<total ant samples>

When given a single JSON file, prints MEAN,<value> only.
=#

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "bin", "load.jl"))

using JSON3

function parse_int_arg(prefix, default)
    for arg in ARGS
        if startswith(arg, prefix)
            return parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return default
end

function input_path()
    for arg in ARGS
        if !startswith(arg, "--")
            return arg
        end
    end
    return nothing
end

function folder_uses_inject(json_path::AbstractString)
    dir = basename(dirname(json_path))
    return occursin(r"k\d+t\d+i", dir)
end

function load_vary_meta(json_path)
    data = JSON3.read(read(json_path, String), Dict{String, Any})
    dataset = String(get(data, "dataset", ""))
    k = Int(get(data, "k", 2))
    θ = Int(get(data, "theta", 5))
    seed = get(data, "base_seed", get(data, "seed", "1"))
    seed = parse(UInt64, string(seed))
    prefer = get(data, "prefer_smaller_side", true) in (true, 1)
    neighbor = get(data, "neighbor_scope_limit", true) in (true, 1)
    graph = get(data, "graph", Dict{String, Any}())
    edge_count = get(data, "edge_count", get(graph, "edges", nothing))
    edge_count = edge_count === nothing ? nothing : Int(edge_count)
    return (; dataset, k, θ, seed, prefer, neighbor, edge_count)
end

function measure_missing_at_size(fg::FrozenBipartite, k::Int, θ::Int, num_ants::Int,
    target_size::Int; prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true)
    num_subspecies = 1
    pheromones = ColonyPheromones(fg, num_subspecies)
    ants = new_ants(fg, num_ants, num_subspecies)
    recorded = Int[]
    seen = falses(num_ants)
    active = collect(1:num_ants)

    while !isempty(active)
        additions = ColonyPheromones(fg, num_subspecies)
        invalids = Int[]
        for i in active
            if seen[i]
                continue
            end
            ant = ants[i]
            if advance_ant!(fg, pheromones, additions, 1, ant, k, θ;
                    prefer_smaller_side=prefer_smaller_side,
                    neighbor_scope_limit=neighbor_scope_limit,
                    ant_id=i)
                depth = Subgraph.vertex_count(ant.explored)
                if depth == target_size
                    push!(recorded, ant.missing)
                    seen[i] = true
                end
            else
                push!(invalids, i)
            end
        end
        merge_pheromones!(pheromones, additions)
        setdiff!(active, invalids)
        setdiff!(active, findall(seen))
    end
    return recorded
end

const MAX_EDGE_COUNT = parse(Int, get(ENV, "MISSING_AT_MAX_EDGES", "100000"))

function measure_file(json_path::AbstractString, target_size::Int, num_ants::Int)
    meta = load_vary_meta(json_path)
    if meta.edge_count !== nothing && meta.edge_count > MAX_EDGE_COUNT
        return Int[]
    end
    inject = if folder_uses_inject(json_path)
        (; enabled=true, nU=5, nV=5, attempts=20)
    else
        (; enabled=false, nU=0, nV=0, attempts=0)
    end

    Random.seed!(meta.seed)
    graph_path = resolve_graph_path(meta.dataset)
    isfile(graph_path) || return Int[]

    g, _edges, _plant = load_graph_maybe_inject(graph_path, inject, meta.k, Random.default_rng())
    g_red = deepcopy(g)
    fg = apply_graph_reductions!(g_red, meta.k, meta.θ, nothing, nothing, true, ReductionMode.simple)
    compact_fg, _remapping = compact_frozen(fg)

    return measure_missing_at_size(compact_fg, meta.k, meta.θ, num_ants, target_size;
        prefer_smaller_side=meta.prefer,
        neighbor_scope_limit=meta.neighbor)
end

function json_paths(path::AbstractString)
    if isfile(path)
        return [path]
    end
    isdir(path) || error("path not found: $path")
    files = filter(f -> endswith(f, ".json"), readdir(path; join=true))
    return sort(files)
end

function main()
    path = input_path()
    path === nothing && error("usage: missing-at-size.jl <vary_dir_or_json> [--size=5] [--ants=100]")

    target_size = parse_int_arg("--size=", 5)
    num_ants = parse_int_arg("--ants=", 100)

    all_samples = Int[]
    n_files = 0
    for json_path in json_paths(path)
        samples = measure_file(json_path, target_size, num_ants)
        isempty(samples) && continue
        append!(all_samples, samples)
        n_files += 1
    end

    if isempty(all_samples)
        println("MEAN,")
        isdir(path) && (println("GRAPHS,0"); println("SAMPLES,0"))
    else
        mean_val = sum(all_samples) / length(all_samples)
        println("MEAN,", mean_val)
        if isdir(path)
            println("GRAPHS,", n_files)
            println("SAMPLES,", length(all_samples))
        end
    end
end

main()
