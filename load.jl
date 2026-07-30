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
  --tabu       use parallel tabu search over an N-sized population
  --aco        use ant colony optimization
  --heuristic  use only the initial heuristic (no search)
  (default)    use branch-and-bound (find_kmdb!)

Branch mode flags (only used by branch-and-bound):
  --pivot    pivot branching (default)
  --binary   binary branching

Benchmarking:
  --benchmark=aco,pivot   measure graph / pivot / ACO memory & time (see benchmark.jl)
  --benchmark=pivot       pivot + graph memory only
  --benchmark=aco         ACO + graph memory only (no iterations-to-optimal without pivot)

GA / tabu flags:
  --seed=S   random seed for reproducible runs (default: time_ns())

ACO flags:
  --ants=N          number of ants per iteration (default: 10)
  --iterations=N    number of ACO iterations (default: 100)
  --pheremone=N     pheromone deposit per step (default: 1)
  --evaporation=X   pheromone evaporation rate in (0, 1] (default: 0.9)

Prerequisites:
  Ensure you have Julia and the required packages installed.

How to Run from the Command Line:
  Format:
    julia load.jl [dataset_name_or_path] [--ga|--tabu|--aco|--heuristic|--binary|--pivot] [--benchmark=...] [--reduce=...] [--seed=...] [--ants=...] [--iterations=...] [--pheremone=...] [--evaporation=...]

  Examples:
    julia load.jl
    julia load.jl Grocery_and_Gourmet_Food --binary
    julia load.jl Grocery_and_Gourmet_Food --ga
    julia load.jl Grocery_and_Gourmet_Food --ga --seed=12345
    julia load.jl Grocery_and_Gourmet_Food --tabu
    julia load.jl Grocery_and_Gourmet_Food --aco
    julia load.jl Grocery_and_Gourmet_Food --aco --ants=20 --iterations=200 --evaporation=0.85
    julia load.jl Grocery_and_Gourmet_Food --heuristic
    julia load.jl Grocery_and_Gourmet_Food --benchmark=aco,pivot
    julia load.jl Grocery_and_Gourmet_Food --benchmark=aco,pivot --ants=20 --iterations=200 --seed=1
    julia load.jl /path/to/indexed_interactions.csv --pivot
=================================================================================
=#

using Profile
using ProfileCanvas
using Random
using EnumX

const SRC = joinpath(@__DIR__, "src")
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__OPPONENT_JL__) || include(joinpath(SRC, "opponent.jl"))
isdefined(@__MODULE__, :__GA_JL__) || include(joinpath(SRC, "ga.jl"))
isdefined(@__MODULE__, :__PARALLEL_TABU_JL__) || include(joinpath(SRC, "parallel_tabu.jl"))
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath(SRC, "aco.jl"))
isdefined(@__MODULE__, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))
isdefined(@__MODULE__, :__BENCHMARK_JL__) || include(joinpath(@__DIR__, "benchmark.jl"))

global const DEBUG = true

# Placeholder for ga()'s subgraph-split parameter until the GA is fully wired up.
const GA_N = 10

const ACO_PHEREMONE = 1
const ACO_NUM_ANTS = 10
const ACO_NUM_ITERATIONS = 100
const ACO_EVAPORATION = 0.9

@enumx Solver ga_solver branch_solver heuristic_solver tabu_solver aco_solver

"""
    load_bipartite_graph(filepath::String) -> BipartiteGraph{Int}

Reads a `user_id,item_id,timestamp` CSV (with header) and builds a mutable
bipartite graph where U = user_id, V = item_id, and edge data = timestamp.
The graph is reduced and then frozen inside the search pipeline.
"""
# Room for improvement: can mmap file, or chunk them, create multiple bipartite graphs,
# then merge them.
function load_bipartite_graph(filepath::String; max_lines::Union{Int,Nothing}=nothing)
    g = BipartiteGraph{Int}()
    edge_count = 0
    open(filepath, "r") do io
        readline(io)  # skip header line
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ",")
            u  = parse(Int, parts[1])
            v  = parse(Int, parts[2])
            ts = parse(Int, parts[3])
            add_edge!(g, u, v, ts)
            edge_count += 1
            max_lines !== nothing && edge_count >= max_lines && break
        end
        println("Graph edges: $(edge_count)")
    end
    return g, edge_count
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

function parse_aco_options()
    pheremone = ACO_PHEREMONE
    num_ants = ACO_NUM_ANTS
    num_iterations = ACO_NUM_ITERATIONS
    evaporation = ACO_EVAPORATION

    for arg in ARGS
        if startswith(arg, "--ants=")
            num_ants = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--iterations=")
            num_iterations = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--pheremone=")
            pheremone = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--evaporation=")
            evaporation = parse(Float64, split(arg, "=", limit=2)[2])
        end
    end

    if !(0.0 < evaporation <= 1.0)
        throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    end

    return pheremone, num_ants, num_iterations, evaporation
end

function parse_benchmark()
    for arg in ARGS
        if startswith(arg, "--benchmark=")
            return parse_benchmark_targets(split(arg, "=", limit=2)[2])
        elseif arg == "--benchmark"
            throw(ArgumentError("--benchmark requires a value, e.g. --benchmark=aco,pivot"))
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
    aco_options = parse_aco_options()
    benchmark = parse_benchmark()

    for arg in ARGS
        if arg == "--ga"
            solver = Solver.ga_solver
        elseif arg == "--tabu"
            solver = Solver.tabu_solver
        elseif arg == "--aco"
            solver = Solver.aco_solver
        elseif arg == "--heuristic"
            solver = Solver.heuristic_solver
        elseif arg == "--pivot"
            mode = BranchMode.pivot
        elseif arg == "--binary"
            mode = BranchMode.binary
        elseif arg == "--profile"
            profile = true
        elseif startswith(arg, "--reduce=") || startswith(arg, "--seed=") ||
               startswith(arg, "--ants=") || startswith(arg, "--iterations=") ||
               startswith(arg, "--pheremone=") || startswith(arg, "--evaporation=") ||
               startswith(arg, "--benchmark=")
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

    return dataset_name, solver, mode, profile, reduction, seed, aco_options, benchmark
end

"""
Run the selected solver in-place on `g`, returning the best SubGraph found.
Branch-and-bound uses `mode`; GA, tabu, and heuristic ignore it.
"""
function solve!(g::BipartiteGraph, solver::Solver.T, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T, aco_options)
    pheremone, num_ants, num_iterations, evaporation = aco_options

    if solver == Solver.ga_solver
        return ga(g, k, θ, GA_N, 2, 0.02, 500; repair=RepairMode.mixed)
    elseif solver == Solver.tabu_solver
        result = parallel_tabu(g, k, θ, GA_N; reduction=reduction)
        return result.best_fitness
    elseif solver == Solver.aco_solver
        return aco(g, pheremone, num_ants, num_iterations, evaporation, k, θ; parallelize=false)
    elseif solver == Solver.heuristic_solver
        fg = if reduction == ReductionMode.none
            freeze(g)
        else
            apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
        end
        if length(fg.u_ids) < θ || length(fg.v_ids) < θ
            return SubGraph(Set(), Set())
        end
        return theta_based_heuristic(fg, k, θ; return_invalid=true)
    else
        return find_kmdb!(g, true, mode, k, θ, reduction)
    end
end

function solve(g::BipartiteGraph, solver::Solver.T, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T, aco_options)
    return solve!(deepcopy(g), solver, mode, k, θ, reduction, aco_options)
end

# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

function main()
    dataset_name, solver, mode, profile, reduction, seed, aco_options, benchmark = parse_args()
    graph_path = resolve_graph_path(dataset_name)
    pheremone, num_ants, num_iterations, evaporation = aco_options

    # k, θ = 4, 8
    k, θ = 6, 7

    if !isfile(graph_path)
        println(stderr, "Error: Could not find a saved graph at '$graph_path'.")
        exit(1)
    end

    println("Loading graph from: $graph_path")
    if benchmark !== nothing
        println("Mode: benchmark ($(join(sort!(collect(String(t) for t in benchmark)), ",")))")
        if :aco in benchmark
            seed = seed === nothing ? UInt64(time_ns()) : seed
            Random.seed!(seed)
            println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation")
        end
    elseif solver == Solver.ga_solver || solver == Solver.tabu_solver || solver == Solver.aco_solver
        # Default to a fresh seed when none was provided so the run is printable/replicable.
        seed = seed === nothing ? UInt64(time_ns()) : seed
        Random.seed!(seed)
        if solver == Solver.ga_solver
            println("Solver: genetic algorithm")
        elseif solver == Solver.tabu_solver
            println("Solver: parallel tabu")
        else
            println("Solver: ant colony optimization")
            println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation")
        end
    elseif solver == Solver.heuristic_solver
        println("Solver: initial heuristic only")
    else
        println("Solver: branch-and-bound ($(mode == BranchMode.pivot ? "pivot" : "binary"))")
    end
    println("Reduction: $(reduction == ReductionMode.simple ? "simple" : reduction == ReductionMode.none ? "none" : "progressive")")

    with_stacksize(2_000_000_000) do
        if benchmark !== nothing
            g, edges = load_bipartite_graph(graph_path)
            println("nU=$(length(g.adjU)), nV=$(length(g.adjV)), |E|=$(edges)")
            run_benchmarks!(g, edges, benchmark, k, θ, reduction, aco_options)
        elseif profile
            println("Profile mode enabled: warming up compilation on a small graph...")
            # Warm up: run the search once on a very small slice to compile methods
            gw, edges = load_bipartite_graph(graph_path; max_lines = 50)
            Dw = solve!(gw, solver, mode, k, θ, reduction, aco_options)

            # Load full graph for the actual profiled run
            g, edges = load_bipartite_graph(graph_path)

            println("Starting profiling run — this may take a while...")
            Profile.clear()
            @profile begin
                D = solve(g, solver, mode, k, θ, reduction, aco_options)
            end

            # Display profile using ProfileCanvas
            try
                ProfileCanvas.canvas()
            catch e
                @warn "ProfileCanvas failed to display:" exception=(e, catch_backtrace())
            end
            @show D
        else
            g, edges = load_bipartite_graph(graph_path)

            println("nU=$(length(g.adjU)), nV=$(length(g.adjV)), |E|=$(edges)")
            
            D = solve!(g, solver, mode, k, θ, reduction, aco_options)

            @show D

            # D = sa(freeze(g), D, k, 0.99, 0.95, 20, 300)

            # @show D
            println(Subgraph.missing_edges(freeze(g), D))
        end
    end

    if benchmark !== nothing && :aco in benchmark
        println("Random seed: --seed=$seed")
    elseif solver == Solver.ga_solver || solver == Solver.tabu_solver || solver == Solver.aco_solver
        println("Random seed: --seed=$seed")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = @allocated begin main() end
    a > 0 && @show a
end