#!/usr/bin/env julia
#=
Measure ant-construction statistics over a single walk (fresh ants, no epochs).

Usage:
  julia scripts/ant-walk-stats.jl <vary_dir_or_json> [--ants=100]

Prints:
  FEAS_RATE,<fraction of ants whose final subgraph is θ-feasible>
  MEAN_VERTICES,<mean final |U|+|V|>
  MEAN_EXAMINED,<mean candidate nodes scored per ant (sum of |pool| per step)>
  GRAPHS,<number of JSON files processed>
  SAMPLES,<total ant samples>
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

function θ_feasible(sg::SubGraph, θ::Int)
    return length(sg.U) ≥ θ && length(sg.V) ≥ θ
end

function pool_size(fg::FrozenBipartite, ant::Ant, neighbor_scope_limit::Bool)
    if neighbor_scope_limit
        pool = neighbor_restricted_candidates(fg, ant.last_visited, ant.candidates)
        return isempty(pool) ? length(ant.candidates) : length(pool)
    end
    return length(ant.candidates)
end

const MAX_WALK_STEPS = parse(Int, get(ENV, "ANT_WALK_MAX_STEPS", "150"))

function measure_ant_walks(fg::FrozenBipartite, k::Int, θ::Int, num_ants::Int;
    prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true)
    num_subspecies = 1
    pheromones = ColonyPheromones(fg, num_subspecies)
    ants = new_ants(fg, num_ants, num_subspecies)
    feasible = Bool[]
    vertices = Int[]
    examined = Int[]
    examined_sum = zeros(Int, num_ants)
    steps = zeros(Int, num_ants)
    active = collect(1:num_ants)

    while !isempty(active)
        additions = ColonyPheromones(fg, num_subspecies)
        invalids = Int[]
        for i in active
            ant = ants[i]
            if steps[i] >= MAX_WALK_STEPS
                sg = ant.explored
                push!(feasible, θ_feasible(sg, θ))
                push!(vertices, Subgraph.vertex_count(sg))
                push!(examined, examined_sum[i])
                push!(invalids, i)
                continue
            end
            examined_sum[i] += pool_size(fg, ant, neighbor_scope_limit)
            if advance_ant!(fg, pheromones, additions, 1, ant, k, θ;
                    prefer_smaller_side=prefer_smaller_side,
                    neighbor_scope_limit=neighbor_scope_limit,
                    ant_id=i)
                steps[i] += 1
                continue
            end
            sg = ant.explored
            push!(feasible, θ_feasible(sg, θ))
            push!(vertices, Subgraph.vertex_count(sg))
            push!(examined, examined_sum[i])
            push!(invalids, i)
        end
        merge_pheromones!(pheromones, additions)
        setdiff!(active, invalids)
    end

    return feasible, vertices, examined
end

const MAX_EDGE_COUNT = parse(Int, get(ENV, "ANT_WALK_MAX_EDGES", "100000"))

function measure_file(json_path::AbstractString, num_ants::Int)
    meta = load_vary_meta(json_path)
    if meta.edge_count !== nothing && meta.edge_count > MAX_EDGE_COUNT
        return (Bool[], Int[], Int[])
    end
    inject = if folder_uses_inject(json_path)
        (; enabled=true, nU=5, nV=5, attempts=20)
    else
        (; enabled=false, nU=0, nV=0, attempts=0)
    end

    Random.seed!(meta.seed)
    graph_path = resolve_graph_path(meta.dataset)
    isfile(graph_path) || return (Bool[], Int[], Int[])

    g, _edges, _plant = load_graph_maybe_inject(graph_path, inject, meta.k, Random.default_rng())
    g_red = deepcopy(g)
    fg = apply_graph_reductions!(g_red, meta.k, meta.θ, nothing, nothing, true, ReductionMode.simple)
    compact_fg, _remapping = compact_frozen(fg)

    return measure_ant_walks(compact_fg, meta.k, meta.θ, num_ants;
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

function pooled_mean(values)
    isempty(values) && return nothing
    return sum(values) / length(values)
end

function main()
    path = input_path()
    path === nothing &&
        error("usage: ant-walk-stats.jl <vary_dir_or_json> [--ants=100]")

    num_ants = parse_int_arg("--ants=", 100)

    all_feas = Bool[]
    all_verts = Int[]
    all_exam = Int[]
    n_files = 0
    for json_path in json_paths(path)
        feas, verts, exam = measure_file(json_path, num_ants)
        isempty(feas) && continue
        append!(all_feas, feas)
        append!(all_verts, verts)
        append!(all_exam, exam)
        n_files += 1
    end

    if isempty(all_feas)
        println("FEAS_RATE,")
        println("MEAN_VERTICES,")
        println("MEAN_EXAMINED,")
        isdir(path) && (println("GRAPHS,0"); println("SAMPLES,0"))
    else
        feas_rate = 100.0 * count(identity, all_feas) / length(all_feas)
        println("FEAS_RATE,", feas_rate)
        println("MEAN_VERTICES,", pooled_mean(all_verts))
        println("MEAN_EXAMINED,", pooled_mean(all_exam))
        if isdir(path)
            println("GRAPHS,", n_files)
            println("SAMPLES,", length(all_feas))
        end
    end
end

main()
