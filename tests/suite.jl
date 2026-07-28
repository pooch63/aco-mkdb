const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__OPPONENT_JL__) || include(joinpath(SRC, "opponent.jl"))
include(joinpath(@__DIR__, "generate.jl"))

using EnumX
using Random

# Named branch_oracle (not `branch`) so we do not shadow opponent.jl's `branch` function.
# `none` skips the reference optimum (--no-optimum).
@enumx OracleMode brute_force branch_oracle none

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

function parse_no_optimum()
    return "--no-optimum" in ARGS
end

function parse_oracle(default::OracleMode.T=OracleMode.brute_force)
    parse_no_optimum() && return OracleMode.none
    for arg in ARGS
        if startswith(arg, "--oracle=")
            val = split(arg, "=", limit=2)[2]
            if val == "brute" || val == "brute_force"
                return OracleMode.brute_force
            elseif val == "branch" || val == "branch_oracle"
                return OracleMode.branch_oracle
            elseif val == "none"
                return OracleMode.none
            else
                error("Unknown oracle: $val (expected brute, brute_force, branch, or none)")
            end
        end
    end
    return default
end

"""
Return `--save=<path>` if present, otherwise `nothing`.
"""
function parse_save()
    for arg in ARGS
        if startswith(arg, "--save=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return nothing
end

function parse_int_range(flag::String, default::UnitRange{Int})
    prefix = "--$flag="
    for arg in ARGS
        if startswith(arg, prefix)
            val = split(arg, "=", limit=2)[2]
            if occursin(':', val)
                parts = split(val, ':')
                length(parts) == 2 || error("Bad range for --$flag: $val (expected lo:hi)")
                lo, hi = parse(Int, parts[1]), parse(Int, parts[2])
                lo <= hi || error("Empty range for --$flag=$val (need lo ≤ hi; got $lo > $hi). Check the arg wasn't line-wrapped.")
                return lo:hi
            else
                n = parse(Int, val)
                return n:n
            end
        end
    end
    return default
end

function parse_float_flag(flag::String, default::Float64)
    prefix = "--$flag="
    for arg in ARGS
        if startswith(arg, prefix)
            return parse(Float64, split(arg, "=", limit=2)[2])
        end
    end
    return default
end

function parse_int_flag(flag::String, default::Int)
    prefix = "--$flag="
    for arg in ARGS
        if startswith(arg, prefix)
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

function build_mutable_graph(g::FrozenBipartite)
    mutable_graph = BipartiteGraph{Nothing}()
    for u in g.u_ids
        add_u!(mutable_graph, u)
    end
    for v in g.v_ids
        add_v!(mutable_graph, v)
    end
    for u_idx in eachindex(g.u_ids)
        u = g.u_ids[u_idx]
        for k in neighbor_range_u(g, u_idx)
            v = g.v_ids[g.v_adj[k]]
            add_edge!(mutable_graph, u, v, nothing)
        end
    end
    return mutable_graph
end

"""
Exact optimum via branch-and-bound with maximum settings (pivot + all_reductions).
"""
function branch_oracle_mdb(g::FrozenBipartite, k::Int, θ::Int)
    mutable_graph = build_mutable_graph(g)
    sol = find_kmdb!(mutable_graph, true, BranchMode.pivot, k, θ, ReductionMode.all_reductions)
    return sol, Subgraph.edge_count(g, sol)
end

function resolve_optimum(g::FrozenBipartite, k::Int, θ::Int, oracle::OracleMode.T)
    if oracle == OracleMode.brute_force
        return brute_force_mdb(g, k, θ)
    elseif oracle == OracleMode.branch_oracle
        return branch_oracle_mdb(g, k, θ)
    elseif oracle == OracleMode.none
        return nothing, nothing
    else
        error("Unknown oracle: $oracle")
    end
end

"""
Shared graph / instance info for one suite trial.

`opt_edges` is `nothing` (and `opt_U` / `opt_V` empty) when the suite ran with
`--no-optimum` (oracle `none`).
"""
struct TrialInfo
    trial::Int
    nU::Int
    nV::Int
    n_edges::Int
    k::Int
    θ::Int
    opt_edges::Union{Int,Nothing}
    opt_U::Set{Int}
    opt_V::Set{Int}
    edges::Vector{Tuple{Int,Int}}
    graph_seed::UInt64
end

has_optimum(info::TrialInfo) = info.opt_edges !== nothing

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
    total_edges::Int
    results::Vector{TrialResult}
end

SolverStats() = SolverStats(0, 0, 0.0, 1.0, 0, TrialResult[])

"""
Suite-wide summary: shared trial count/seed, per-trial graph info, and per-solver stats.
"""
struct SuiteSummary
    N::Int
    seed::Any
    oracle::OracleMode.T
    algorithm::String
    trials::Vector{TrialInfo}
    stats::Dict{String,SolverStats}
end

has_optimum(summary::SuiteSummary) = summary.oracle != OracleMode.none

function record_trial!(stats::SolverStats, g::FrozenBipartite, sol::SubGraph, k::Int, θ::Int,
    opt_edges::Union{Int,Nothing}; seed=nothing, graph_seed::UInt64=UInt64(0), trial::Int=0,
    label::AbstractString="", log_result::Bool=false)
    solution_edges = Subgraph.edge_count(g, sol)
    missing_edges = Subgraph.missing_edges(g, sol)
    # Either the subgraph is big enough, or there was no optimal solution
    sufficient = (length(sol.U) ≥ θ && length(sol.V) ≥ θ) ||
        (opt_edges !== nothing && solution_edges == opt_edges == 0)

    if sufficient
        stats.valid += 1
    end

    if missing_edges > k
        push!(stats.results, TrialResult(0, -Inf, missing_edges, sufficient, copy(sol.U), copy(sol.V)))
        if log_result
            prefix = isempty(label) ? "" : "[$label] "
            println("$(prefix)trial $trial  edges=0  missing=$missing_edges  INVALID (k exceeded)  seed=$graph_seed")
        end
        return
    end

    if opt_edges !== nothing && solution_edges >= opt_edges && sufficient
        stats.optimal += 1
    end

    ratio = opt_edges === nothing ? NaN : (opt_edges == 0 ? 1.0 : solution_edges / opt_edges)
    stats.total_edges += solution_edges
    if isfinite(ratio)
        stats.total_ratio += ratio
        stats.worst_ratio = min(stats.worst_ratio, ratio)
    end
    push!(stats.results, TrialResult(solution_edges, ratio, missing_edges, sufficient, copy(sol.U), copy(sol.V)))

    if log_result
        prefix = isempty(label) ? "" : "[$label] "
        status = sufficient ? "valid" : "INVALID (θ not met)"
        println("$(prefix)trial $trial  edges=$solution_edges  missing=$missing_edges  ($status)  |U|=$(length(sol.U)) |V|=$(length(sol.V))  seed=$graph_seed")
    elseif opt_edges !== nothing && ratio < 0.5
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
function random_graph_from_seed(graph_seed::Integer; graph_kwargs...)
    Random.seed!(UInt64(graph_seed))
    return random_graph(; graph_kwargs...)
end

"""
CLI / suite defaults used when `run_graph_suite` kwargs are omitted.

Override any of these by passing the keyword explicitly, or via FLAGS:
  --N=  --seed=  --oracle=  --no-optimum  --nU=  --nV=  --edge-prob=  --theta-max=  --k-max=
"""
function suite_cli_defaults()
    return (
        N = parse_N(5),
        seed = parse_seed(),
        oracle = parse_oracle(OracleMode.branch_oracle),
        nU_range = parse_int_range("nU", 100:500),
        nV_range = parse_int_range("nV", 100:500),
        edge_prob = parse_float_flag("edge-prob", 0.3),
        θ_max = parse_int_flag("theta-max", 20),
        k_max = parse_int_flag("k-max", 4),
    )
end

"""
Run `N` random graphs against one or more labeled solvers.

`solvers` maps a display label to a `(g, k, θ) -> SubGraph` function.
A single `solve_fn` may be passed instead for the common one-solver case.

When `N`, `seed`, `oracle`, or graph kwargs are omitted, they are taken from
CLI flags with suite defaults (see `suite_cli_defaults`).

`oracle` selects how the reference optimum is computed:
  - `brute_force` — enumerate all θ-feasible subgraphs (small graphs only)
  - `branch_oracle` — `find_kmdb!` with pivot + all_reductions
  - `none` — skip the oracle (`--no-optimum`); only log solver results

Extra kwargs are forwarded to `random_graph` (e.g. `nU_range`, `nV_range`,
`edge_prob`, `θ_max`, `k_max`).

Each trial gets its own integer `graph_seed`. Reproduce one graph with:
`julia <test>.jl --seed=<graph_seed> --N=1`
(when `N == 1`, `--seed` is used directly as that graph's seed).
"""
function run_graph_suite(; N::Union{Int,Nothing}=nothing, seed=nothing,
    solve_fn::Union{Function,Nothing}=nothing,
    solvers::Union{AbstractDict,Nothing}=nothing,
    oracle::Union{OracleMode.T,Nothing}=nothing,
    algorithm::AbstractString="", graph_kwargs...)
    defaults = suite_cli_defaults()
    N = N === nothing ? defaults.N : N
    seed = seed === nothing ? defaults.seed : seed
    oracle = oracle === nothing ? defaults.oracle : oracle

    # Fill graph generation kwargs from CLI defaults when not overridden.
    graph = Dict{Symbol,Any}(
        :nU_range => defaults.nU_range,
        :nV_range => defaults.nV_range,
        :edge_prob => defaults.edge_prob,
        :θ_max => defaults.θ_max,
        :k_max => defaults.k_max,
    )
    for (k, v) in pairs(graph_kwargs)
        graph[k] = v
    end

    if solvers === nothing
        solve_fn === nothing && throw(ArgumentError("pass solve_fn or solvers"))
        label = isempty(algorithm) ? "default" : String(algorithm)
        solvers = Dict(label => solve_fn)
    elseif solve_fn !== nothing
        throw(ArgumentError("pass solve_fn or solvers, not both"))
    end

    Random.seed!(seed)

    stats = Dict{String,SolverStats}(label => SolverStats() for label in keys(solvers))
    trials = Vector{TrialInfo}(undef, N)

    algo_label = if !isempty(algorithm)
        algorithm
    elseif length(solvers) == 1
        only(keys(solvers))
    else
        join(sort!(collect(keys(solvers))), ",")
    end

    find_opt = oracle != OracleMode.none
    println("  oracle=$(oracle)  nU=$(graph[:nU_range])  nV=$(graph[:nV_range])  edge_prob=$(graph[:edge_prob])  θ_max=$(graph[:θ_max])  k_max=$(graph[:k_max])")

    # Draw every graph_seed up front so oracle/solver RNG use cannot desync
    # sequences across algorithms that share --seed and --N.
    graph_seeds = Vector{UInt64}(undef, N)
    for trial in 1:N
        graph_seeds[trial] = N == 1 ? UInt64(seed) : rand(UInt64)
    end

    for trial in 1:N
        graph_seed = graph_seeds[trial]
        Random.seed!(graph_seed)
        edges, nU, nV, k, θ = random_graph(; graph...)

        g = build_frozen(edges, nU, nV)
        if find_opt
            opt_sol, opt_edges = resolve_optimum(g, k, θ, oracle)
            opt_U, opt_V = copy(opt_sol.U), copy(opt_sol.V)
        else
            opt_edges = nothing
            opt_U, opt_V = Set{Int}(), Set{Int}()
        end
        # Skip storing huge edge lists; graph_seed is enough to reproduce.
        stored_edges = length(edges) <= 200 ? sort!(collect(edges)) : Tuple{Int,Int}[]
        trials[trial] = TrialInfo(trial, nU, nV, length(edges), k, θ, opt_edges,
            opt_U, opt_V, stored_edges, graph_seed)

        for (label, solver) in solvers
            sol = solver(g, k, θ)
            display_label = label == "default" ? "" : label
            record_trial!(stats[label], g, sol, k, θ, opt_edges;
                seed=seed, graph_seed=graph_seed, trial=trial, label=display_label,
                log_result=!find_opt)
        end
    end

    return SuiteSummary(N, seed, oracle, String(algo_label), trials, stats)
end

function _print_solver_block(label::AbstractString, N::Int, s::SolverStats; with_optimum::Bool=true)
    println("[$label]")
    println("  Valid solutions      : $(s.valid) / $N")
    if with_optimum
        println("  Optimal solutions    : $(s.optimal) / $N")
        println("  Optimality rate      : $(100 * s.optimal / N)%")
        println("  Average ratio        : $(s.total_ratio / N)")
        println("  Worst ratio          : $(s.worst_ratio)")
    else
        println("  Average edges        : $(s.total_edges / N)")
    end
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
    return !result.valid || (isfinite(result.ratio) && result.ratio < 1.0)
end

function oracle_label(oracle::OracleMode.T)
    if oracle == OracleMode.brute_force
        return "Brute-force"
    elseif oracle == OracleMode.branch_oracle
        return "Branch"
    else
        return "None"
    end
end

"""
Log each non-optimal trial with seed, graph, oracle optimum, and per-solver solutions.
Skipped when the suite ran without an oracle.
"""
function print_nonoptimal_trial_seeds(summary::SuiteSummary; labels=nothing)
    has_optimum(summary) || return
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
    with_opt = has_optimum(summary)
    println()
    println("==================== RESULTS ====================")
    println("Trials               : $(summary.N)")
    println("Seed                 : --seed=$(summary.seed)")
    println("Oracle               : $(summary.oracle)")
    if !isempty(summary.algorithm)
        println("Algorithm            : $(summary.algorithm)")
    end

    ordered = labels === nothing ? collect(keys(summary.stats)) : collect(labels)
    if length(ordered) == 1
        s = summary.stats[ordered[1]]
        println("Valid solutions      : $(s.valid) / $(summary.N)")
        if with_opt
            println("Optimal solutions    : $(s.optimal) / $(summary.N)")
            println("Optimality rate      : $(100 * s.optimal / summary.N)%")
            println("Average ratio        : $(s.total_ratio / summary.N)")
            println("Worst ratio          : $(s.worst_ratio)")
        else
            println("Average edges        : $(s.total_edges / summary.N)")
        end
    else
        if with_opt
            println("Optimal solutions:")
            for label in ordered
                println("  $(rpad(label, 8)): $(summary.stats[label].optimal) / $(summary.N)")
            end
        end
        for label in ordered
            _print_solver_block(label, summary.N, summary.stats[label]; with_optimum=with_opt)
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
    if !isempty(info.edges)
        println("Edges: $(info.edges)")
    end
    if has_optimum(info)
        println("$(oracle_label(summary.oracle)) optimum: edges=$(info.opt_edges) missing=$(length(info.opt_U) * length(info.opt_V) - info.opt_edges) U=$(sort!(collect(info.opt_U)))  V=$(sort!(collect(info.opt_V)))")
    else
        println("Optimum: (not computed)")
    end
    println()

    for label in ordered
        r = summary.stats[label].results[trial]
        status = r.valid ? "valid" : "INVALID (θ not met)"
        mark = highlight !== nothing && label in highlight ? " *" : ""
        ratio_str = isfinite(r.ratio) ? string(r.ratio) : "n/a"
        println("  $label$mark")
        println("    edges=$(r.edges)  missing=$(r.missing)  ratio=$ratio_str  ($status)")
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

# ---- JSON serialization (no external deps; UInt64 seeds as strings) ----

function _json_escape(s::AbstractString)
    buf = IOBuffer()
    for c in s
        if c == '"'
            print(buf, "\\\"")
        elseif c == '\\'
            print(buf, "\\\\")
        elseif c == '\n'
            print(buf, "\\n")
        elseif c == '\r'
            print(buf, "\\r")
        elseif c == '\t'
            print(buf, "\\t")
        else
            print(buf, c)
        end
    end
    return String(take!(buf))
end

_json_val(x::Bool) = x ? "true" : "false"
_json_val(x::Integer) = string(x)
_json_val(x::AbstractFloat) = isfinite(x) ? string(x) : (x > 0 ? "null" : (x < 0 ? "null" : "null"))
_json_val(::Nothing) = "null"
_json_val(x::AbstractString) = "\"" * _json_escape(x) * "\""
_json_val(x::Symbol) = _json_val(string(x))
_json_val(x::OracleMode.T) = _json_val(string(x))
function _json_val(xs::AbstractVector)
    return "[" * join((_json_val(x) for x in xs), ",") * "]"
end
_json_val(xs::Set) = _json_val(sort!(collect(xs)))

function trial_to_json_dict(info::TrialInfo, results::Dict{String,TrialResult})
    solutions = Dict{String,Any}()
    for (label, r) in results
        solutions[label] = Dict(
            "edges" => r.edges,
            "missing" => r.missing,
            "ratio" => isfinite(r.ratio) ? r.ratio : nothing,
            "valid" => r.valid,
            "U" => sort!(collect(r.U)),
            "V" => sort!(collect(r.V)),
        )
    end
    return Dict{String,Any}(
        "trial" => info.trial,
        "graph_seed" => string(info.graph_seed),
        "nU" => info.nU,
        "nV" => info.nV,
        "n_edges" => info.n_edges,
        "k" => info.k,
        "theta" => info.θ,
        "opt_edges" => info.opt_edges,
        "opt_U" => sort!(collect(info.opt_U)),
        "opt_V" => sort!(collect(info.opt_V)),
        "solutions" => solutions,
    )
end

function _json_val(d::AbstractDict)
    return "{" * join(("\"$(_json_escape(string(k)))\":$(_json_val(v))" for (k, v) in d), ",") * "}"
end

"""
Write a suite summary to JSON for later comparison across algorithms.

Seeds are stored as strings so UInt64 values survive JS / JSON tooling.
"""
function save_suite_json(path::AbstractString, summary::SuiteSummary)
    trials_json = Any[]
    labels = sort!(collect(keys(summary.stats)))
    for trial in 1:summary.N
        results = Dict{String,TrialResult}(
            label => summary.stats[label].results[trial] for label in labels
        )
        push!(trials_json, trial_to_json_dict(summary.trials[trial], results))
    end

    payload = Dict{String,Any}(
        "algorithm" => summary.algorithm,
        "oracle" => string(summary.oracle),
        "N" => summary.N,
        "seed" => summary.seed === nothing ? nothing : string(summary.seed),
        "trials" => trials_json,
    )

    open(path, "w") do io
        print(io, _json_val(payload))
    end
    println("Wrote suite results to $path")
    return path
end
