#=
=================================================================================
Graph Loader
=================================================================================
This script loads a saved indexed graph from disk and runs the graph analysis
workflow on it. It accepts either:

  1. No argument, in which case it loads data/indexed_interactions.csv
  2. A dataset name, in which case it looks for data/<dataset_name>/indexed_interactions.csv
  3. An explicit file path to a CSV graph file

Solver flags (independent of branch mode):
  --ga         use the genetic algorithm
  --heuristic  use only the initial heuristic (no search)
  (default)    use branch-and-bound (find_kmdb!)

Branch mode flags (only used by branch-and-bound):
  --pivot    pivot branching (default)
  --binary   binary branching

GA-only flags:
  --seed=S   random seed for reproducible GA runs (default: time_ns())

Prerequisites:
  Ensure you have Julia and the required packages installed.

How to Run from the Command Line:
  Format:
    julia load.jl [dataset_name_or_path] [--ga|--heuristic|--binary|--pivot] [--reduce=...] [--seed=...]

  Examples:
    julia load.jl
    julia load.jl Grocery_and_Gourmet_Food --binary
    julia load.jl Grocery_and_Gourmet_Food --ga
    julia load.jl Grocery_and_Gourmet_Food --ga --seed=12345
    julia load.jl Grocery_and_Gourmet_Food --heuristic
    julia load.jl /path/to/indexed_interactions.csv --pivot
=================================================================================
=#

using Profile
using ProfileCanvas
using Random
using EnumX

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__OPPONENT_JL__) || include("opponent.jl")
isdefined(@__MODULE__, :__GA_JL__) || include("ga.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")

global const DEBUG = true

# Placeholder for ga()'s subgraph-split parameter until the GA is fully wired up.
const GA_N = 10

@enumx Solver ga_solver branch_solver heuristic_solver

"""
    load_bipartite_graph(filepath::String) -> BipartiteGraph{Int}

Reads a `user_id,item_id,timestamp` CSV (with header) and builds a mutable
bipartite graph where U = user_id, V = item_id, and edge data = timestamp.
The graph is reduced and then frozen inside the search pipeline.
"""
function load_bipartite_graph(filepath::String; max_lines::Union{Int,Nothing}=nothing)
    g = BipartiteGraph{Int}()
    open(filepath, "r") do io
        readline(io)  # skip header line
        count = 0
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ",")
            u  = parse(Int, parts[1])
            v  = parse(Int, parts[2])
            ts = parse(Int, parts[3])
            add_edge!(g, u, v, ts)
            count += 1
            max_lines !== nothing && count >= max_lines && break
        end
    end
    return g
end

function resolve_graph_path(dataset_name::Union{String,Nothing}=nothing)
    if dataset_name === nothing || isempty(dataset_name)
        return "data/indexed_interactions.csv"
    end

    if isfile(dataset_name)
        return dataset_name
    end

    return joinpath("data", dataset_name, "indexed_interactions.csv")
end

function parse_reduction()
    for arg in ARGS
        if startswith(arg, "--reduce=")
            value = split(arg, "=", limit=2)[2]
            if value == "lo"
                return ReductionMode.simple
            elseif value == "hi"
                return ReductionMode.all_reductions
            elseif value == "none"
                return ReductionMode.none
            else
                throw(ArgumentError("Unsupported reduction setting: $value"))
            end
        end
    end

    return ReductionMode.all_reductions
end

function parse_seed()
    for arg in ARGS
        if startswith(arg, "--seed=")
            return parse(UInt64, split(arg, "=", limit=2)[2])
        end
    end
    return nothing
end

function parse_args()
    dataset_name = nothing
    solver = Solver.branch_solver
    mode = BranchMode.pivot
    profile = false
    reduction = parse_reduction()
    seed = parse_seed()

    for arg in ARGS
        if arg == "--ga"
            solver = Solver.ga_solver
        elseif arg == "--heuristic"
            solver = Solver.heuristic_solver
        elseif arg == "--pivot"
            mode = BranchMode.pivot
        elseif arg == "--binary"
            mode = BranchMode.binary
        elseif arg == "--profile"
            profile = true
        elseif startswith(arg, "--reduce=") || startswith(arg, "--seed=")
            continue
        elseif dataset_name === nothing
            dataset_name = arg
        else
            throw(ArgumentError("Unexpected argument: $arg"))
        end
    end

    if dataset_name === nothing
        dataset_name = DEBUG ? "boxes" : nothing
    end

    return dataset_name, solver, mode, profile, reduction, seed
end

"""
Run the selected solver in-place on `g`, returning the best SubGraph found.
Branch-and-bound uses `mode`; GA and heuristic ignore it.
"""
function solve!(g::BipartiteGraph, solver::Solver.T, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T)
    if solver == Solver.ga_solver
        return ga(g, k, θ, GA_N, 2, 0.02, 500; repair=RepairMode.mixed)
    elseif solver == Solver.heuristic_solver
        fg = if reduction == ReductionMode.none
            freeze(g)
        else
            apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
        end
        if length(fg.u_ids) < θ || length(fg.v_ids) < θ
            return SubGraph(Set(), Set())
        end
        return theta_based_heuristic(fg, k, θ; return_invalid=false)
    else
        return find_kmdb!(g, true, mode, k, θ, reduction)
    end
end

function solve(g::BipartiteGraph, solver::Solver.T, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T)
    return solve!(deepcopy(g), solver, mode, k, θ, reduction)
end

# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

function main()
    dataset_name, solver, mode, profile, reduction, seed = parse_args()
    graph_path = resolve_graph_path(dataset_name)

    k, θ = 3, 5

    if !isfile(graph_path)
        println(stderr, "Error: Could not find a saved graph at '$graph_path'.")
        exit(1)
    end

    println("Loading graph from: $graph_path")
    if solver == Solver.ga_solver
        # Default to a fresh seed when none was provided so the run is printable/replicable.
        seed = seed === nothing ? UInt64(time_ns()) : seed
        Random.seed!(seed)
        println("Solver: genetic algorithm")
    elseif solver == Solver.heuristic_solver
        println("Solver: initial heuristic only")
    else
        println("Solver: branch-and-bound ($(mode == BranchMode.pivot ? "pivot" : "binary"))")
    end
    println("Reduction: $(reduction == ReductionMode.simple ? "simple" : reduction == ReductionMode.none ? "none" : "progressive")")

    with_stacksize(2_000_000_000) do
        if profile
            println("Profile mode enabled: warming up compilation on a small graph...")
            # Warm up: run the search once on a very small slice to compile methods
            gw = load_bipartite_graph(graph_path; max_lines = 50)
            Dw = solve!(gw, solver, mode, k, θ, reduction)

            # Load full graph for the actual profiled run
            g = load_bipartite_graph(graph_path)

            println("Starting profiling run — this may take a while...")
            Profile.clear()
            @profile begin
                D = solve(g, solver, mode, k, θ, reduction)
            end

            # Display profile using ProfileCanvas
            try
                ProfileCanvas.canvas()
            catch e
                @warn "ProfileCanvas failed to display:" exception=(e, catch_backtrace())
            end
            @show D
        else
            g = load_bipartite_graph(graph_path)
            D = solve!(g, solver, mode, k, θ, reduction)

            @show D

            # D = sa(freeze(g), D, k, 0.99, 0.95, 20, 300)

            # @show D
            println(Subgraph.missing_edges(freeze(g), D))
        end
    end

    if solver == Solver.ga_solver
        println("Random seed: --seed=$seed")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = @allocated begin main() end
    a > 0 && @show a
end