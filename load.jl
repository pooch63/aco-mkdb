#=
=================================================================================
Graph Loader
=================================================================================
This script loads a saved indexed graph from disk and runs the graph analysis
workflow on it. It accepts either:

  1. No argument, in which case it loads data/indexed_interactions.csv
  2. A dataset key (flat or nested), e.g. boxes or amazon/boxes →
     data/<dataset_key>/indexed_interactions.csv
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
  --benchmark=heuristic   θ-heuristic + graph memory only
  --benchmark=ga          genetic algorithm + graph memory only
  --benchmark=aco,pivot,heuristic,ga   any comma-separated mix of the above
  --save=NAME.json        write benchmark JSON under results/ (or an explicit path)

GA / tabu flags:
  --seed=S   random seed for reproducible runs (default: time_ns())

ACO flags:
  --ants=N          number of ants per iteration (default: 10)
  --iterations=N    number of ACO iterations (default: 100)
  --pheremone=N     pheromone deposit per step (default: 1)
  --evaporation=X   pheromone evaporation rate in (0, 1] (default: 0.9)
  --subspecies=N    number of ant subspecies (default: 1)
  --prefer-smaller-side=true|false   bias adds onto the smaller side while min < θ (default: true)
  --elite-seed=true|false            seed some ants from best − nodes (default: true)
  --elite-seed-ants=N                ants seeded from elite each iteration (default: 3)
  --elite-seed-remove=N              nodes stripped from elite seed (default: 2)

Prerequisites:
  Ensure you have Julia and the required packages installed.

How to Run from the Command Line:
  Format:
    julia load.jl [dataset_name_or_path] [--ga|--tabu|--aco|--heuristic|--binary|--pivot] [--benchmark=...] [--save=...] [--reduce=...] [--seed=...] [--ants=...] [--iterations=...] [--pheremone=...] [--evaporation=...] [--subspecies=...]

  Examples:
    julia load.jl
    julia load.jl amazon/boxes --binary
    julia load.jl amazon/grocery --ga
    julia load.jl amazon/grocery --ga --seed=12345
    julia load.jl amazon/grocery --tabu
    julia load.jl amazon/grocery --aco
    julia load.jl amazon/grocery --aco --ants=20 --iterations=200 --evaporation=0.85
    julia load.jl amazon/grocery --heuristic
    julia load.jl amazon/boxes --benchmark=aco,pivot
    julia load.jl amazon/boxes --benchmark=aco,pivot --ants=20 --iterations=200 --seed=1
    julia load.jl amazon/boxes --benchmark=heuristic,ga,aco --save=boxes_compare.json
    julia load.jl /path/to/indexed_interactions.csv --pivot
=================================================================================
=#

using Profile
using ProfileCanvas
using Random
using EnumX

const SRC = joinpath(@__DIR__, "src")
isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))
isdefined(@__MODULE__, :__OPPONENT_JL__) || include(joinpath(SRC, "opponent.jl"))
isdefined(@__MODULE__, :__GA_JL__) || include(joinpath(SRC, "ga.jl"))
isdefined(@__MODULE__, :__PARALLEL_TABU_JL__) || include(joinpath(SRC, "parallel_tabu.jl"))
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath(SRC, "aco", "algorithm.jl"))
isdefined(@__MODULE__, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))
isdefined(@__MODULE__, :__BENCHMARK_JL__) || include(joinpath(@__DIR__, "benchmark.jl"))

global const DEBUG = true

# Placeholder for ga()'s subgraph-split parameter until the GA is fully wired up.
const GA_N = 10

const ACO_PHEREMONE = 1
const ACO_NUM_ANTS = 50
const ACO_NUM_ITERATIONS = 3
const ACO_EVAPORATION = 0.95
const ACO_NUM_SUBSPECIES = 1

@enumx Solver ga_solver branch_solver heuristic_solver tabu_solver aco_solver

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

function parse_bool_eq(flag::String, default::Bool)
    prefix = "--$flag="
    for arg in ARGS
        if startswith(arg, prefix)
            val = lowercase(split(arg, "=", limit=2)[2])
            if val in ("true", "1", "yes", "on")
                return true
            elseif val in ("false", "0", "no", "off")
                return false
            else
                throw(ArgumentError("Bad boolean for --$flag: $val (expected true/false)"))
            end
        end
    end
    return default
end

function parse_aco_options()
    pheremone = ACO_PHEREMONE
    num_ants = ACO_NUM_ANTS
    num_iterations = ACO_NUM_ITERATIONS
    evaporation = ACO_EVAPORATION
    num_subspecies = ACO_NUM_SUBSPECIES
    elite_seed_ants = 3
    elite_seed_remove = 2

    for arg in ARGS
        if startswith(arg, "--ants=")
            num_ants = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--iterations=")
            num_iterations = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--pheremone=")
            pheremone = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--evaporation=")
            evaporation = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--subspecies=")
            num_subspecies = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--elite-seed-ants=")
            elite_seed_ants = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--elite-seed-remove=")
            elite_seed_remove = parse(Int, split(arg, "=", limit=2)[2])
        end
    end

    prefer_smaller_side = parse_bool_eq("prefer-smaller-side", true)
    elite_seed = parse_bool_eq("elite-seed", true)

    if !(0.0 < evaporation <= 1.0)
        throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    end
    if num_subspecies < 1
        throw(ArgumentError("subspecies must be >= 1, got $num_subspecies"))
    end
    if elite_seed_ants < 0
        throw(ArgumentError("elite-seed-ants must be >= 0, got $elite_seed_ants"))
    end
    if elite_seed_remove < 0
        throw(ArgumentError("elite-seed-remove must be >= 0, got $elite_seed_remove"))
    end

    # NamedTuple: first five fields stay positionally destructurable.
    return (; pheremone, num_ants, num_iterations, evaporation, num_subspecies,
        prefer_smaller_side, elite_seed, elite_seed_ants, elite_seed_remove)
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
    save_path = parse_benchmark_save()

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
               startswith(arg, "--subspecies=") || startswith(arg, "--benchmark=") ||
               startswith(arg, "--save=") ||
               startswith(arg, "--prefer-smaller-side=") || startswith(arg, "--elite-seed=") ||
               startswith(arg, "--elite-seed-ants=") || startswith(arg, "--elite-seed-remove=")
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

    return dataset_name, solver, mode, profile, reduction, seed, aco_options, benchmark, save_path
end

"""
Run the selected solver in-place on `g`, returning the best SubGraph found
(or a `Vector{SubGraph}` for ACO / branch-and-bound multi-solution modes).
Branch-and-bound uses `mode`; GA, tabu, and heuristic ignore it.
"""
function solve!(g::BipartiteGraph, solver::Solver.T, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T, aco_options; num_solutions::Int=1)
    pheremone, num_ants, num_iterations, evaporation, num_subspecies = aco_options

    if solver == Solver.ga_solver
        return ga(g, k, θ, GA_N, 2, 0.02, 500; repair=RepairMode.mixed)
    elseif solver == Solver.tabu_solver
        result = parallel_tabu(g, k, θ, GA_N; reduction=reduction)
        return result.best_fitness
    elseif solver == Solver.aco_solver
        remapped, _iterations, _times = aco(g, pheremone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
            parallelize=false,
            prefer_smaller_side=aco_options.prefer_smaller_side,
            elite_seed=aco_options.elite_seed,
            elite_seed_ants=aco_options.elite_seed_ants,
            elite_seed_remove=aco_options.elite_seed_remove)
        return remapped
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
        return find_kmdb!(g, true, mode, k, θ, reduction; num_solutions=num_solutions)
    end
end

function solve(g::BipartiteGraph, solver::Solver.T, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T, aco_options; num_solutions::Int=1)
    return solve!(deepcopy(g), solver, mode, k, θ, reduction, aco_options; num_solutions=num_solutions)
end

# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

function main()
    dataset_name, solver, mode, profile, reduction, seed, aco_options, benchmark, save_path =
        parse_args()
    graph_path = resolve_graph_path(dataset_name)
    pheremone, num_ants, num_iterations, evaporation, num_subspecies = aco_options
    ga_options = (; N=GA_N, O=2, k_mutate=0.02, generations=500)

    # Room for algorithmic improvement: So same problem where k being overshot leads to the ants not finding it
    # I suppose we could have incrementally larger k, but is there a better solution?
    # k, θ = 3, 6
    k, θ = 2, 5

    if !isfile(graph_path)
        println(stderr, "Error: Could not find a saved graph at '$graph_path'.")
        exit(1)
    end

    if save_path !== nothing && benchmark === nothing
        throw(ArgumentError("--save= is only valid with --benchmark=..."))
    end

    println("Loading graph from: $graph_path")
    if benchmark !== nothing
        println("Mode: benchmark ($(join(sort!(collect(String(t) for t in benchmark)), ",")))")
        needs_seed = :aco in benchmark || :ga in benchmark
        if needs_seed
            seed = seed === nothing ? UInt64(time_ns()) : seed
            Random.seed!(seed)
        end
        if :aco in benchmark
            println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies")
        end
        if :ga in benchmark
            println("GA: N=$(ga_options.N) O=$(ga_options.O) generations=$(ga_options.generations) k_mutate=$(ga_options.k_mutate)")
        end
        if save_path !== nothing
            println("Save: $(resolve_benchmark_save_path(save_path))")
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
            println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies")
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
            results = run_benchmarks!(g, edges, benchmark, k, θ, reduction, aco_options;
                ga_options=ga_options)
            if save_path !== nothing
                payload = benchmark_results_to_dict(results;
                    k=k, θ=θ, dataset=String(dataset_name), targets=benchmark,
                    seed=seed, reduction=reduction, edge_count=edges)
                save_benchmark_json(resolve_benchmark_save_path(save_path), payload)
            end
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

            # Always write a stable HTML path; view() alone only opens a temp file.
            out = joinpath(@__DIR__, "results", "profile_aco.html")
            try
                ProfileCanvas.html_file(out)
                println("Wrote profile flamegraph to $out")
                try
                    run(`xdg-open $out`; wait=false)
                catch
                end
            catch e
                @warn "ProfileCanvas.html_file failed:" exception=(e, catch_backtrace())
                println("\n── Profile.print (flat, top frames) ──")
                Profile.print(; C=false, maxdepth=25)
            end
            @show D
        else
            g, edges = load_bipartite_graph(graph_path)

            println("nU=$(length(g.adjU)), nV=$(length(g.adjV)), |E|=$(edges)")
            
            D = solve!(g, solver, mode, k, θ, reduction, aco_options)

            if D isa AbstractVector
                fg = freeze(g)
                for (s, d) in enumerate(D)
                    println("Solution $s:")
                    @show (d.U, d.V)
                    println(Subgraph.missing_edges(fg, d))
                end
            else
                @show D

                # D = sa(freeze(g), D, k, 0.99, 0.95, 20, 300)

                # @show D
                println(Subgraph.missing_edges(freeze(g), D))
            end
        end
    end

    if benchmark !== nothing && (:aco in benchmark || :ga in benchmark)
        println("Random seed: --seed=$seed")
    elseif solver == Solver.ga_solver || solver == Solver.tabu_solver || solver == Solver.aco_solver
        println("Random seed: --seed=$seed")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = @allocated begin main() end
    a > 0 && @show a
end