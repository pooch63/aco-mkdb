include(joinpath(@__DIR__, "..", "graph.jl"))
include(joinpath(@__DIR__, "..", "opponent.jl"))
include(joinpath(@__DIR__, "generate.jl"))

using Random

function parse_seed()
    for arg in ARGS
        if startswith(arg, "--seed=")
            return parse(UInt64, split(arg, "=", limit=2)[2])
        end
    end
    return UInt64(time_ns())
end

function parse_N(default::Int=20)
    for arg in ARGS
        if startswith(arg, "--N=")
            return parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return default
end

function brute_force_mdb(g::FrozenBipartite, k::Int, θ::Int)
    Ulist = collect(g.u_ids)
    Vlist = collect(g.v_ids)

    best = SubGraph(Set{Int}(), Set{Int}())
    best_edges = 0

    nU = length(Ulist)
    nV = length(Vlist)

    for umask in 0:(2^nU - 1), vmask in 0:(2^nV - 1)
        U = Set(Ulist[i] for i in 1:nU if (umask >> (i - 1)) & 1 == 1)
        V = Set(Vlist[i] for i in 1:nV if (vmask >> (i - 1)) & 1 == 1)

        if length(U) < θ || length(V) < θ
            continue
        end

        D = SubGraph(U, V)

        if Subgraph.missing_edges(g, D) <= k
            e = Subgraph.edge_count(g, D)

            if e > best_edges
                best = D
                best_edges = e
            end
        end
    end

    return best, best_edges
end

"""
Shared graph / instance info for one suite trial.
"""
struct TrialInfo
    trial::Int
    nU::Int
    nV::Int
    n_edges::Int
    k::Int
    θ::Int
    opt_edges::Int
    opt_U::Set{Int}
    opt_V::Set{Int}
    edges::Vector{Tuple{Int,Int}}
    graph_seed::UInt64
end

"""
Per-solver outcome for one trial.
"""
struct TrialResult
    edges::Int
    ratio::Float64
    missing::Int
    valid::Bool
    U::Set{Int}
    V::Set{Int}
end

"""
Per-solver accumulators updated by `record_trial!`.
"""
mutable struct SolverStats
    valid::Int
    optimal::Int
    total_ratio::Float64
    worst_ratio::Float64
    results::Vector{TrialResult}
end

SolverStats() = SolverStats(0, 0, 0.0, 1.0, TrialResult[])

"""
Suite-wide summary: shared trial count/seed, per-trial graph info, and per-solver stats.
"""
struct SuiteSummary
    N::Int
    seed::Any
    trials::Vector{TrialInfo}
    stats::Dict{String,SolverStats}
end

function record_trial!(stats::SolverStats, g::FrozenBipartite, sol::SubGraph, k::Int, θ::Int,
    opt_edges::Int; seed=nothing, graph_seed::UInt64=UInt64(0), trial::Int=0, label::AbstractString="")
    # SubGraph may carry a cache from a different FrozenBipartite inside the solver.
    sol.edge_count_cache = nothing

    solution_edges = Subgraph.edge_count(g, sol)
    missing_edges = Subgraph.missing_edges(g, sol)
    # Either the subgraph is big enough, or there was no optimal solutions
    sufficient = (length(sol.U) ≥ θ && length(sol.V) ≥ θ) || (solution_edges == opt_edges == 0)

    if sufficient
        stats.valid += 1
    end

    if missing_edges > k
        push!(stats.results, TrialResult(0, -Inf, missing_edges, sufficient, copy(sol.U), copy(sol.V)))
        return
    end

    if solution_edges >= opt_edges && sufficient
        stats.optimal += 1
    end

    ratio = opt_edges == 0 ? 1.0 : solution_edges / opt_edges
    stats.total_ratio += ratio
    stats.worst_ratio = min(stats.worst_ratio, ratio)
    push!(stats.results, TrialResult(solution_edges, ratio, missing_edges, sufficient, copy(sol.U), copy(sol.V)))

    if ratio < 0.5
        prefix = isempty(label) ? "" : "[$label] "
        println("$(prefix)Poor solution on trial $trial")
        println("Graph seed: --seed=$graph_seed --N=1")
        println("solution edges = $solution_edges")
        println("optimal edges   = $opt_edges")
        println("ratio           = $ratio")
    end
end

"""
Generate the graph produced by `--seed=<graph_seed> --N=1`.
"""
function random_graph_from_seed(graph_seed::Integer)
    Random.seed!(UInt64(graph_seed))
    return random_graph()
end

"""
Run `N` random graphs against one or more labeled solvers.

`solvers` maps a display label to a `(g, k, θ) -> SubGraph` function.
A single `solve_fn` may be passed instead for the common one-solver case.

Each trial gets its own integer `graph_seed`. Reproduce one graph with:
`julia <test>.jl --seed=<graph_seed> --N=1`
(when `N == 1`, `--seed` is used directly as that graph's seed).
"""
function run_graph_suite(; N=10, seed=nothing, solve_fn::Union{Function,Nothing}=nothing,
    solvers::Union{AbstractDict,Nothing}=nothing)
    if solvers === nothing
        solve_fn === nothing && throw(ArgumentError("pass solve_fn or solvers"))
        solvers = Dict("default" => solve_fn)
    elseif solve_fn !== nothing
        throw(ArgumentError("pass solve_fn or solvers, not both"))
    end

    if seed !== nothing
        Random.seed!(seed)
    end

    stats = Dict{String,SolverStats}(label => SolverStats() for label in keys(solvers))
    trials = Vector{TrialInfo}(undef, N)

    for trial in 1:N
        # N=1 + --seed=X means X is the graph seed (for single-graph reproduction).
        # Otherwise draw a fresh per-graph seed from the suite RNG stream.
        graph_seed = (N == 1 && seed !== nothing) ? UInt64(seed) : rand(UInt64)
        suite_state = Random.getstate(Random.default_rng())
        Random.seed!(graph_seed)
        edges, nU, nV, k, θ = random_graph()
        Random.setstate!(Random.default_rng(), suite_state)

        g = build_frozen(edges, nU, nV)
        opt_sol, opt_edges = brute_force_mdb(g, k, θ)
        trials[trial] = TrialInfo(trial, nU, nV, length(edges), k, θ, opt_edges,
            copy(opt_sol.U), copy(opt_sol.V), sort!(collect(edges)), graph_seed)

        for (label, solver) in solvers
            sol = solver(g, k, θ)
            display_label = label == "default" ? "" : label
            record_trial!(stats[label], g, sol, k, θ, opt_edges;
                seed=seed, graph_seed=graph_seed, trial=trial, label=display_label)
        end
    end

    return SuiteSummary(N, seed, trials, stats)
end

function _print_solver_block(label::AbstractString, N::Int, s::SolverStats)
    println("[$label]")
    println("  Valid solutions      : $(s.valid) / $N")
    println("  Optimal solutions    : $(s.optimal) / $N")
    println("  Optimality rate      : $(100 * s.optimal / N)%")
    println("  Average ratio        : $(s.total_ratio / N)")
    println("  Worst ratio          : $(s.worst_ratio)")
end

"""
Return the sole `SolverStats` when the suite ran one solver.
"""
function only_stats(summary::SuiteSummary)
    length(summary.stats) == 1 ||
        throw(ArgumentError("only_stats requires a single-solver suite summary"))
    return only(values(summary.stats))
end

function trial_is_nonoptimal(result::TrialResult)
    return !result.valid || result.ratio < 1.0
end

"""
Log each non-optimal trial with seed, graph, brute-force optimum, and per-solver solutions.
"""
function print_nonoptimal_trial_seeds(summary::SuiteSummary; labels=nothing)
    ordered = labels === nothing ? collect(keys(summary.stats)) : collect(labels)
    any_bad = false

    for trial in 1:summary.N
        bad_labels = String[label for label in ordered
                            if trial_is_nonoptimal(summary.stats[label].results[trial])]
        isempty(bad_labels) && continue

        if !any_bad
            println("Non-optimal trials:")
            any_bad = true
        end

        print_trial_detail(summary, trial; labels=ordered, highlight=bad_labels)
    end
end

function print_suite_summary(summary::SuiteSummary; labels=nothing)
    println()
    println("==================== RESULTS ====================")
    println("Trials               : $(summary.N)")
    println("Seed                 : --seed=$(summary.seed)")

    ordered = labels === nothing ? collect(keys(summary.stats)) : collect(labels)
    if length(ordered) == 1
        s = summary.stats[ordered[1]]
        println("Valid solutions      : $(s.valid) / $(summary.N)")
        println("Optimal solutions    : $(s.optimal) / $(summary.N)")
        println("Optimality rate      : $(100 * s.optimal / summary.N)%")
        println("Average ratio        : $(s.total_ratio / summary.N)")
        println("Worst ratio          : $(s.worst_ratio)")
    else
        println("Optimal solutions:")
        for label in ordered
            println("  $(rpad(label, 8)): $(summary.stats[label].optimal) / $(summary.N)")
        end
        for label in ordered
            _print_solver_block(label, summary.N, summary.stats[label])
        end
    end

    print_nonoptimal_trial_seeds(summary; labels=ordered)
    println("=================================================")
end

function print_trial_detail(summary::SuiteSummary, trial::Int; labels=nothing, highlight=nothing)
    info = summary.trials[trial]
    ordered = labels === nothing ? sort!(collect(keys(summary.stats))) : collect(labels)

    println()
    println("---------- trial $trial ----------")
    println("Reproduce suite:  julia $(PROGRAM_FILE) --seed=$(summary.seed)")
    println("Reproduce graph:  julia $(PROGRAM_FILE) --seed=$(info.graph_seed) --N=1")
    println("Graph: nU=$(info.nU)  nV=$(info.nV)  |E|=$(info.n_edges)  k=$(info.k)  θ=$(info.θ)")
    println("Edges: $(info.edges)")
    println("Brute-force optimum: edges=$(info.opt_edges) missing=$(Subgraph.missing_edges(fg, info.opt_sol)) U=$(sort!(collect(info.opt_U)))  V=$(sort!(collect(info.opt_V)))")
    println()

    for label in ordered
        r = summary.stats[label].results[trial]
        status = r.valid ? "valid" : "INVALID (θ not met)"
        mark = highlight !== nothing && label in highlight ? " *" : ""
        println("  $label$mark")
        println("    edges=$(r.edges)  missing=$(r.missing)  ratio=$(r.ratio)  ($status)")
        println("    U=$(sort!(collect(r.U)))")
        println("    V=$(sort!(collect(r.V)))")
    end
    println("----------------------------------------------")
end

"""
Print a detailed per-trial comparison when labeled solvers disagree on edge count.
"""
function print_solver_mismatch(summary::SuiteSummary, trial::Int; labels=nothing)
    print_trial_detail(summary, trial; labels=labels)
end
