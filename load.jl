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

Parameter sweeps (see vary.jl):
  --vary=ant-count            sweep `--ants-range` and record time / iters / quality as JSON
  --ants-range=5,10,20,50     ant counts to try (default: 1,2,5,10,20,50,100)
  --aco-runs=N                stochastic ACO replicates per ant count (default: 1; distinct seeds)
  --vary-pivot=true           run pivot once for an optimum edge count (default: false)

Problem parameters:
  --k=N       maximum missing edges (default: 2)
  --theta=N   minimum side size θ (default: 5)

In-memory biclique injection (does not modify the CSV on disk):
  --inject              plant a nearly-complete biclique before solving
  --u=N                 |U| side of the planted biclique (required with --inject)
  --v=M                 |V| side of the planted biclique (required with --inject)
  --k=K                 also the number of intentionally missing edges in the plant
  --inject-attempts=N   max random tries to find a valid plant (default: 20)

GA / tabu flags:
  --seed=S   random seed for reproducible runs (default: time_ns())

ACO flags:
  --ants=N          number of ants per iteration (default: 10)
  --iterations=N    number of ACO iterations (default: 100)
  --pheremone=N     pheromone deposit per step (default: 1)
  --evaporation=X   pheromone evaporation rate in (0, 1] (default: 0.95)
  --subspecies=N    number of ant subspecies (default: 1)
  --prefer-smaller-side=true|false   bias adds onto the smaller side while min < θ (default: true)
  --elite-seed=true|false            seed some ants from best − nodes (default: true)
  --elite-seed-ants=N                ants seeded from elite each iteration (default: 3)
  --elite-seed-remove=N              nodes stripped from elite seed (default: 2)
  --aco-reduce=true|false            run pivot + ACO, then score how many nodes
                                     sit below the min pheromone on the pivot optimum
                                     (default: false). ACO is not told the optimum;
                                     pivot is only used afterward to score the cut.

Prerequisites:
  Ensure you have Julia and the required packages installed.

How to Run from the Command Line:
  Format:
    julia load.jl [dataset_name_or_path] [--ga|--tabu|--aco|--heuristic|--binary|--pivot] [--k=...] [--theta=...] [--inject --u=... --v=...] [--benchmark=...] [--save=...] [--reduce=...] [--seed=...] [--ants=...] [--iterations=...] [--pheremone=...] [--evaporation=...] [--subspecies=...] [--aco-reduce=...]

  Examples:
    julia load.jl
    julia load.jl amazon/boxes --binary
    julia load.jl amazon/grocery --ga
    julia load.jl amazon/grocery --ga --seed=12345
    julia load.jl amazon/grocery --tabu
    julia load.jl amazon/grocery --aco
    julia load.jl amazon/grocery --aco --ants=20 --iterations=200 --evaporation=0.85
    julia load.jl amazon/boxes --aco-reduce=true --ants=50 --iterations=3
    julia load.jl amazon/grocery --heuristic
    julia load.jl amazon/boxes --k=3 --theta=6
    julia load.jl amazon/boxes --inject --u=8 --v=7 --k=3 --theta=6
    julia load.jl amazon/boxes --benchmark=aco,pivot
    julia load.jl amazon/boxes --benchmark=aco,pivot --ants=20 --iterations=200 --seed=1
    julia load.jl amazon/boxes --benchmark=heuristic,ga,aco --save=boxes_compare.json
    julia load.jl konect/bitcoin --vary=ant-count --ants-range=10,20,50,100 --save=bitcoin_ants.json
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
isdefined(@__MODULE__, :__VARY_JL__) || include(joinpath(@__DIR__, "vary.jl"))

global const DEBUG = true

# Placeholder for ga()'s subgraph-split parameter until the GA is fully wired up.
const GA_N = 10

const ACO_PHEREMONE = 1
const ACO_NUM_ANTS = 100
const ACO_NUM_ITERATIONS = 5
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

const DEFAULT_K = 2
const DEFAULT_THETA = 5

function parse_k_theta()
    k = DEFAULT_K
    θ = DEFAULT_THETA
    for arg in ARGS
        if startswith(arg, "--k=")
            k = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--theta=")
            θ = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    if k < 0
        throw(ArgumentError("k must be >= 0, got $k"))
    end
    if θ < 1
        throw(ArgumentError("theta must be >= 1, got $θ"))
    end
    return k, θ
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

"""
Parse `--inject` / `--u=` / `--v=` / `--inject-attempts=`.
`--k` from `parse_k_theta` is the planted missing-edge count when injection is on.
"""
function parse_inject()
    enabled = false
    nU = nothing
    nV = nothing
    attempts = 20

    for arg in ARGS
        if arg == "--inject"
            enabled = true
        elseif startswith(arg, "--u=")
            nU = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--v=")
            nV = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--inject-attempts=")
            attempts = parse(Int, split(arg, "=", limit=2)[2])
        end
    end

    if !enabled
        if nU !== nothing || nV !== nothing
            throw(ArgumentError("--u= and --v= require --inject"))
        end
        return (; enabled=false, nU=0, nV=0, attempts=0)
    end

    if nU === nothing || nV === nothing
        throw(ArgumentError("--inject requires --u=N and --v=M (planted biclique sides)"))
    end
    if nU <= 0 || nV <= 0
        throw(ArgumentError("--u and --v must be positive, got u=$nU v=$nV"))
    end
    if attempts <= 0
        throw(ArgumentError("--inject-attempts must be >= 1, got $attempts"))
    end

    return (; enabled=true, nU, nV, attempts)
end

function parse_args()
    dataset_name = nothing
    solver = Solver.branch_solver
    mode = BranchMode.pivot
    profile = false
    reduction = parse_reduction()
    seed = parse_seed()
    k, θ = parse_k_theta()
    aco_options = parse_aco_options()
    benchmark = parse_benchmark()
    vary = parse_vary()
    save_path = parse_benchmark_save()
    inject = parse_inject()
    aco_reduce = parse_bool_eq("aco-reduce", false)
    ants_range = parse_ants_range()
    vary_run_pivot = parse_vary_run_pivot()
    aco_runs = parse_aco_runs()

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
        elseif arg == "--inject" || startswith(arg, "--u=") || startswith(arg, "--v=") ||
               startswith(arg, "--inject-attempts=") ||
               startswith(arg, "--reduce=") || startswith(arg, "--seed=") ||
               startswith(arg, "--k=") || startswith(arg, "--theta=") ||
               startswith(arg, "--ants=") || startswith(arg, "--iterations=") ||
               startswith(arg, "--pheremone=") || startswith(arg, "--evaporation=") ||
               startswith(arg, "--subspecies=") || startswith(arg, "--benchmark=") ||
               startswith(arg, "--save=") ||
               startswith(arg, "--prefer-smaller-side=") || startswith(arg, "--elite-seed=") ||
               startswith(arg, "--elite-seed-ants=") || startswith(arg, "--elite-seed-remove=") ||
               startswith(arg, "--aco-reduce=") ||
               startswith(arg, "--vary=") || startswith(arg, "--ants-range=") ||
               startswith(arg, "--vary-pivot=") || startswith(arg, "--aco-runs=")
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

    if inject.enabled && k >= inject.nU * inject.nV
        throw(ArgumentError("--k must be < u*v for injection; got k=$k, u=$(inject.nU), v=$(inject.nV)"))
    end

    if aco_reduce && benchmark !== nothing
        throw(ArgumentError("--aco-reduce cannot be combined with --benchmark"))
    end
    if vary !== nothing && benchmark !== nothing
        throw(ArgumentError("--vary cannot be combined with --benchmark"))
    end
    if vary !== nothing && aco_reduce
        throw(ArgumentError("--vary cannot be combined with --aco-reduce"))
    end

    return dataset_name, solver, mode, profile, reduction, seed, k, θ, aco_options, benchmark, vary, save_path, inject, aco_reduce, ants_range, vary_run_pivot, aco_runs
end

"""
Plant an `nU × nV` biclique with exactly `k` missing edges into `g` in-memory.
Samples existing U/V vertices, keeps `k` absent pairs missing, and inserts the rest.
Returns `(chosen_U, chosen_V, inserted_count, missing_edges, existing_count)`.
"""
function inject_biclique!(g::BipartiteGraph, nU::Int, nV::Int, k::Int, rng::AbstractRNG;
    attempts::Int=20)
    users = collect(keys(g.adjU))
    items = collect(keys(g.adjV))

    if length(users) < nU
        throw(ArgumentError("Cannot inject |U|=$nU: graph only has $(length(users)) U vertices"))
    end
    if length(items) < nV
        throw(ArgumentError("Cannot inject |V|=$nV: graph only has $(length(items)) V vertices"))
    end
    if k < 0
        throw(ArgumentError("k must be >= 0, got $k"))
    end
    if k >= nU * nV
        throw(ArgumentError("k must be < u*v; got k=$k, u=$nU, v=$nV"))
    end

    for attempt in 1:attempts
        shuffle!(rng, users)
        shuffle!(rng, items)
        chosen_U = users[1:nU]
        chosen_V = items[1:nV]

        absent = Tuple{Int,Int}[]
        existing = 0
        for u in chosen_U, v in chosen_V
            if has_edge(g, u, v)
                existing += 1
            else
                push!(absent, (u, v))
            end
        end

        if length(absent) < k
            continue
        end

        shuffle!(rng, absent)
        missing_edges = Set(absent[1:k])
        to_insert = @view absent[k+1:end]
        for (u, v) in to_insert
            add_edge!(g, u, v, nothing)
        end

        return sort(chosen_U), sort(chosen_V), length(to_insert), missing_edges, existing
    end

    throw(ArgumentError(
        "Could not plant a $nU×$nV biclique with k=$k missing edges after $attempts attempts. " *
        "Try a smaller k, larger graph, or --inject-attempts=N."))
end

"""
Load the CSV graph and optionally plant a biclique in-memory (CSV unchanged).
"""
function load_graph_maybe_inject(graph_path::String, inject, k::Int, rng::AbstractRNG;
    max_lines::Union{Int,Nothing}=nothing)
    g, edges = load_bipartite_graph(graph_path; max_lines=max_lines)
    if inject.enabled
        chosen_U, chosen_V, inserted, missing, existing =
            inject_biclique!(g, inject.nU, inject.nV, k, rng; attempts=inject.attempts)
        edges += inserted
        println("Injected biclique: u=$(inject.nU) v=$(inject.nV) k=$k " *
                "(existing=$existing inserted=$inserted missing=$(length(missing)))")
        println("  planted U: $chosen_U")
        println("  planted V: $chosen_V")
        println("  reserved missing: $(sort!(collect(missing)))")
    end
    return g, edges
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
        remapped, _iterations, _times, _pheromones, _remapping = aco(g, pheremone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
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

"""
Run pivot for the true optimum, then ACO (blind — not given the optimum) for
pheromone trails, and report how many nodes sit strictly below the minimum
trail on the pivot solution. Pivot is evaluation-only.
"""
function run_aco_reduce!(g::BipartiteGraph, k::Int, θ::Int, reduction::ReductionMode.T, aco_options)
    pheremone, num_ants, num_iterations, evaporation, num_subspecies = aco_options

    println()
    println("── Pivot (branch-and-bound) ──")
    g_pivot = deepcopy(g)
    pivot_sols = find_kmdb!(g_pivot, true, BranchMode.pivot, k, θ, reduction)
    pivot_sol = isempty(pivot_sols) ? SubGraph() : first(pivot_sols)
    fg_pivot = freeze(g_pivot)
    pivot_edges = Subgraph.edge_count(fg_pivot, pivot_sol)
    println("Pivot solution: |U|=$(length(pivot_sol.U)) |V|=$(length(pivot_sol.V)) " *
            "edges=$pivot_edges missing=$(Subgraph.missing_edges(fg_pivot, pivot_sol))")
    @show (pivot_sol.U, pivot_sol.V)

    println()
    println("── ACO (blind; optimum not provided) ──")
    g_aco = deepcopy(g)
    aco_sols, _iterations, _times, pheromones, remapping = aco(
        g_aco, pheremone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
        parallelize=false,
        prefer_smaller_side=aco_options.prefer_smaller_side,
        elite_seed=aco_options.elite_seed,
        elite_seed_ants=aco_options.elite_seed_ants,
        elite_seed_remove=aco_options.elite_seed_remove,
        reduction=reduction)

    fg_aco = freeze(g_aco)
    best_idx = argmax(i -> Subgraph.edge_count(fg_aco, aco_sols[i]), eachindex(aco_sols))
    aco_sol = aco_sols[best_idx]
    println("ACO best subspecies=$best_idx: |U|=$(length(aco_sol.U)) |V|=$(length(aco_sol.V)) " *
            "edges=$(Subgraph.edge_count(fg_aco, aco_sol)) " *
            "missing=$(Subgraph.missing_edges(fg_aco, aco_sol))")
    @show (aco_sol.U, aco_sol.V)

    pivot_stats = pheromone_cut_stats(pheromones, remapping, pivot_sol; species=best_idx)
    print_pheromone_cut_stats(pivot_stats; label="pivot optimum")
    aco_stats = pheromone_cut_stats(pheromones, remapping, aco_sol; species=best_idx)
    print_pheromone_cut_stats(aco_stats; label="ACO best")
    return (; pivot_sol, aco_sol, pivot_stats, aco_stats, pheromones, remapping)
end

# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

function main()
    dataset_name, solver, mode, profile, reduction, seed, k, θ, aco_options, benchmark, vary, save_path, inject, aco_reduce, ants_range, vary_run_pivot, aco_runs =
        parse_args()
    graph_path = resolve_graph_path(dataset_name)
    pheremone, num_ants, num_iterations, evaporation, num_subspecies = aco_options
    ga_options = (; N=GA_N, O=2, k_mutate=0.02, generations=500)

    if !isfile(graph_path)
        println(stderr, "Error: Could not find a saved graph at '$graph_path'.")
        exit(1)
    end

    if save_path !== nothing && benchmark === nothing && vary === nothing
        throw(ArgumentError("--save= requires --benchmark=... or --vary=..."))
    end

    # Seed early when injection or a stochastic solver needs reproducibility.
    needs_seed = inject.enabled || aco_reduce || vary !== nothing ||
        (benchmark !== nothing && (:aco in benchmark || :ga in benchmark)) ||
        solver == Solver.ga_solver || solver == Solver.tabu_solver || solver == Solver.aco_solver
    if needs_seed
        seed = seed === nothing ? UInt64(time_ns()) : seed
        Random.seed!(seed)
    end
    inject_rng = Random.default_rng()

    println("Loading graph from: $graph_path")
    println("Parameters: k=$k θ=$θ")
    if inject.enabled
        println("Inject: u=$(inject.nU) v=$(inject.nV) k=$k (in-memory; CSV unchanged)")
        if θ > min(inject.nU, inject.nV)
            @warn "θ=$θ exceeds min(u,v)=$(min(inject.nU, inject.nV)); planted biclique is not θ-feasible"
        end
    end
    if aco_reduce
        println("Mode: ACO-reduce (blind ACO, then score cut vs pivot optimum)")
        println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies")
    elseif vary == :ant_count
        println("Mode: vary ant-count (range=$(join(ants_range, ",")), aco_runs=$aco_runs)")
        println("ACO: iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies")
        println("Pivot for quality baseline: $(vary_run_pivot ? "yes" : "no")")
        if save_path !== nothing
            println("Save: $(resolve_benchmark_save_path(save_path))")
        end
    elseif benchmark !== nothing
        println("Mode: benchmark ($(join(sort!(collect(String(t) for t in benchmark)), ",")))")
        if :aco in benchmark
            println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies")
        end
        if :ga in benchmark
            println("GA: N=$(ga_options.N) O=$(ga_options.O) generations=$(ga_options.generations) k_mutate=$(ga_options.k_mutate)")
        end
        if save_path !== nothing
            println("Save: $(resolve_benchmark_save_path(save_path))")
        end
    elseif solver == Solver.ga_solver
        println("Solver: genetic algorithm")
    elseif solver == Solver.tabu_solver
        println("Solver: parallel tabu")
    elseif solver == Solver.aco_solver
        println("Solver: ant colony optimization")
        println("ACO: ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies")
    elseif solver == Solver.heuristic_solver
        println("Solver: initial heuristic only")
    else
        println("Solver: branch-and-bound ($(mode == BranchMode.pivot ? "pivot" : "binary"))")
    end
    println("Reduction: $(reduction == ReductionMode.simple ? "simple" : reduction == ReductionMode.none ? "none" : "progressive")")

    with_stacksize(2_000_000_000) do
        if aco_reduce
            g, edges = load_graph_maybe_inject(graph_path, inject, k, inject_rng)
            println("nU=$(length(g.adjU)), nV=$(length(g.adjV)), |E|=$(edges)")
            run_aco_reduce!(g, k, θ, reduction, aco_options)
        elseif vary == :ant_count
            g, edges = load_graph_maybe_inject(graph_path, inject, k, inject_rng)
            println("nU=$(length(g.adjU)), nV=$(length(g.adjV)), |E|=$(edges)")
            results = run_vary_ant_count!(g, edges, k, θ, reduction, aco_options;
                ant_counts=ants_range, run_pivot=vary_run_pivot,
                seed=seed, dataset=String(dataset_name), n_runs=aco_runs)
            if save_path !== nothing
                payload = vary_results_to_dict(results;
                    k=k, θ=θ, dataset=String(dataset_name),
                    seed=seed, reduction=reduction, edge_count=edges,
                    run_pivot=vary_run_pivot)
                save_vary_json(resolve_benchmark_save_path(save_path), payload)
            end
        elseif benchmark !== nothing
            g, edges = load_graph_maybe_inject(graph_path, inject, k, inject_rng)
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
            # (skip inject — the truncated graph may be too small to plant into).
            gw, edges = load_bipartite_graph(graph_path; max_lines = 50)
            Dw = solve!(gw, solver, mode, k, θ, reduction, aco_options)

            # Load full graph for the actual profiled run
            g, edges = load_graph_maybe_inject(graph_path, inject, k, inject_rng)

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
            g, edges = load_graph_maybe_inject(graph_path, inject, k, inject_rng)

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

    if needs_seed
        println("Random seed: --seed=$seed")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = @allocated begin main() end
    a > 0 && @show a
end