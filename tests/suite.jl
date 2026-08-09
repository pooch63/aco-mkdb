const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__OPPONENT_JL__) || include(joinpath(SRC, "opponent.jl"))
include(joinpath(@__DIR__, "generate.jl"))

using EnumX
using Random

# Named branch_oracle (not `branch`) so we do not shadow opponent.jl's `branch` function.
# `none` skips the reference optimum (--no-optimum).
@enumx OracleMode brute_force branch_oracle none

"""
Parse `--seed=`. If the flag appears more than once (e.g. leftover suite
`--seed=1` plus a reproduce `--seed=<graph_seed>`), the **last** wins and a
warning is printed — matching usual CLI override behavior.
"""
function parse_seed()
    seeds = UInt64[]
    for arg in ARGS
        if startswith(arg, "--seed=")
            push!(seeds, parse(UInt64, split(arg, "=", limit=2)[2]))
        end
    end
    isempty(seeds) && return UInt64(time_ns())
    if length(seeds) > 1
        @warn "Multiple --seed= flags; using the last" ignored=seeds[1:end-1] seed=seeds[end]
    end
    return seeds[end]
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

function parse_bool_flag(flag::String, default::Bool)
    prefix = "--$flag="
    for arg in ARGS
        if startswith(arg, prefix)
            val = lowercase(split(arg, "=", limit=2)[2])
            if val in ("true", "1", "yes", "on")
                return true
            elseif val in ("false", "0", "no", "off")
                return false
            else
                error("Bad boolean for --$flag: $val (expected true/false)")
            end
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
    sols = find_kmdb!(mutable_graph, true, BranchMode.pivot, k, θ, ReductionMode.all_reductions)
    sol = isempty(sols) ? SubGraph() : first(sols)
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

`plant_U` / `plant_V` are the planted θ_plant×θ_plant k-MDB vertex sets from
`random_graph` (always present).
"""
struct TrialInfo
    trial::Int
    nU::Int
    nV::Int
    n_edges::Int
    k::Int
    θ::Int          # θ passed to solvers / oracle (may be jittered below θ_plant)
    θ_plant::Int    # θ of the planted k-MDB in the generated graph
    opt_edges::Union{Int,Nothing}
    opt_U::Set{Int}
    opt_V::Set{Int}
    plant_U::Set{Int}
    plant_V::Set{Int}
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

"""
Print the reference solution (oracle if present, else planted) and the solver's return.
"""
function log_solution_mismatch(; prefix::AbstractString="", trial::Int=0,
    graph_seed::UInt64=UInt64(0), status::AbstractString="",
    sol_edges::Int, sol_missing::Int, sol_U::AbstractSet{<:Integer}, sol_V::AbstractSet{<:Integer},
    ref_label::AbstractString, ref_edges::Union{Int,Nothing}, ref_U::AbstractSet{<:Integer},
    ref_V::AbstractSet{<:Integer}, ratio::Float64=NaN)
    println("$(prefix)Wrong answer on trial $trial  ($status)  seed=$graph_seed")
    println("  Reproduce graph: --seed=$graph_seed --N=1")
    println("  (pass only that --seed; do not also keep the suite --seed)")
    if ref_edges !== nothing
        ratio_str = isfinite(ratio) ? "  ratio=$ratio" : ""
        println("  $ref_label: edges=$ref_edges  |U|=$(length(ref_U)) |V|=$(length(ref_V))$ratio_str")
    else
        println("  $ref_label: |U|=$(length(ref_U)) |V|=$(length(ref_V))")
    end
    println("    U=$(sort!(collect(ref_U)))")
    println("    V=$(sort!(collect(ref_V)))")
    println("  Algorithm returned: edges=$sol_edges  missing=$sol_missing  |U|=$(length(sol_U)) |V|=$(length(sol_V))")
    println("    U=$(sort!(collect(sol_U)))")
    println("    V=$(sort!(collect(sol_V)))")
end

function record_trial!(stats::SolverStats, g::FrozenBipartite, sol::SubGraph, k::Int, θ::Int,
    opt_edges::Union{Int,Nothing}; seed=nothing, graph_seed::UInt64=UInt64(0), trial::Int=0,
    label::AbstractString="", log_result::Bool=false,
    opt_U::AbstractSet{<:Integer}=Set{Int}(), opt_V::AbstractSet{<:Integer}=Set{Int}(),
    plant_U::AbstractSet{<:Integer}=Set{Int}(), plant_V::AbstractSet{<:Integer}=Set{Int}())
    solution_edges = Subgraph.edge_count(g, sol)
    missing_edges = Subgraph.missing_edges(g, sol)
    # Either the subgraph is big enough, or there was no optimal solution
    sufficient = (length(sol.U) ≥ θ && length(sol.V) ≥ θ) ||
        (opt_edges !== nothing && solution_edges == opt_edges == 0)
    prefix = isempty(label) ? "" : "[$label] "
    has_opt = opt_edges !== nothing
    ref_label = has_opt ? "Oracle optimum" : "Planted solution"
    ref_U = has_opt ? opt_U : plant_U
    ref_V = has_opt ? opt_V : plant_V
    plant_edges = length(plant_U) * length(plant_V) - k  # planted has exactly k missing pairs
    ref_edges = has_opt ? opt_edges : plant_edges

    if missing_edges > k
        # Keep the returned size (edges / |U| / |V|) even though the trial is invalid.
        push!(stats.results, TrialResult(solution_edges, -Inf, missing_edges, false, copy(sol.U), copy(sol.V)))
        if log_result
            println("$(prefix)trial $trial  edges=$solution_edges  missing=$missing_edges  INVALID (k exceeded)  |U|=$(length(sol.U)) |V|=$(length(sol.V))  seed=$graph_seed")
        end
        log_solution_mismatch(; prefix, trial, graph_seed, status="INVALID (k exceeded)",
            sol_edges=solution_edges, sol_missing=missing_edges, sol_U=sol.U, sol_V=sol.V,
            ref_label, ref_edges, ref_U, ref_V, ratio=-Inf)
        return
    end

    if sufficient
        stats.valid += 1
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

    wrong = !sufficient || (has_opt && isfinite(ratio) && ratio < 1.0)
    if log_result
        status = sufficient ? "valid" : "INVALID (θ not met)"
        println("$(prefix)trial $trial  edges=$solution_edges  missing=$missing_edges  ($status)  |U|=$(length(sol.U)) |V|=$(length(sol.V))  seed=$graph_seed")
    end
    if wrong
        status = sufficient ? "suboptimal" : "INVALID (θ not met)"
        log_solution_mismatch(; prefix, trial, graph_seed, status,
            sol_edges=solution_edges, sol_missing=missing_edges, sol_U=sol.U, sol_V=sol.V,
            ref_label, ref_edges, ref_U, ref_V, ratio)
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
  --jitter=

`--jitter=J` draws the θ passed to solvers/oracle uniformly from
`max(1, θ_plant - J):θ_plant`, where `θ_plant` is the planted k-MDB size.
Never exceeds `θ_plant`. Default 0 (pass `θ_plant` unchanged; no extra RNG draw).
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
        jitter = parse_int_flag("jitter", 0),
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
`edge_prob`, `θ_max`, `k_max`). Pass `jitter` (or `--jitter=`) to randomize
the θ handed to solvers below the planted value (see `suite_cli_defaults`).

Each trial gets its own integer `graph_seed`. Reproduce one graph with:
`julia <test>.jl --seed=<graph_seed> --N=1`
(when `N == 1`, `--seed` is used directly as that graph's seed).
Do not also pass the suite seed (e.g. `--seed=1 --seed=<graph_seed>`); if both
appear, the last `--seed=` wins. For bit-identical ACO answers use
`--parallelize=none` (threaded ACO shares the global RNG across tasks).
"""
function run_graph_suite(; N::Union{Int,Nothing}=nothing, seed=nothing,
    solve_fn::Union{Function,Nothing}=nothing,
    solvers::Union{AbstractDict,Nothing}=nothing,
    oracle::Union{OracleMode.T,Nothing}=nothing,
    algorithm::AbstractString="",
    jitter::Union{Int,Nothing}=nothing, graph_kwargs...)
    defaults = suite_cli_defaults()
    N = N === nothing ? defaults.N : N
    seed = seed === nothing ? defaults.seed : seed
    oracle = oracle === nothing ? defaults.oracle : oracle
    jitter = jitter === nothing ? defaults.jitter : jitter
    jitter >= 0 || error("--jitter must be ≥ 0 (got $jitter)")

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
    println("  oracle=$(oracle)  nU=$(graph[:nU_range])  nV=$(graph[:nV_range])  edge_prob=$(graph[:edge_prob])  θ_max=$(graph[:θ_max])  k_max=$(graph[:k_max])  jitter=$jitter")

    # Draw every graph_seed up front so oracle/solver RNG use cannot desync
    # sequences across algorithms that share --seed and --N.
    graph_seeds = Vector{UInt64}(undef, N)
    for trial in 1:N
        graph_seeds[trial] = N == 1 ? UInt64(seed) : rand(UInt64)
    end

    for trial in 1:N
        graph_seed = graph_seeds[trial]
        Random.seed!(graph_seed)
        edges, nU, nV, k, θ_plant, plant_U, plant_V = random_graph(; graph...)
        # Query θ ≤ planted θ; default jitter=0 skips rand to preserve seed sequences.
        θ = if jitter == 0
            θ_plant
        else
            rand(max(1, θ_plant - jitter):θ_plant)
        end

        # Log before oracle/solver so a crash still prints the reproduce seed.
        prefix = isempty(algo_label) ? "" : "[$algo_label] "
        θ_note = θ == θ_plant ? "θ=$θ" : "θ=$θ (planted $θ_plant)"
        println("$(prefix)trial $trial / $N  nU=$nU nV=$nV |E|=$(length(edges)) k=$k $θ_note  seed=$graph_seed")
        flush(stdout)

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
        trials[trial] = TrialInfo(trial, nU, nV, length(edges), k, θ, θ_plant, opt_edges,
            opt_U, opt_V, plant_U, plant_V, stored_edges, graph_seed)

        for (label, solver) in solvers
            sol = solver(g, k, θ)
            display_label = label == "default" ? "" : label
            record_trial!(stats[label], g, sol, k, θ, opt_edges;
                seed=seed, graph_seed=graph_seed, trial=trial, label=display_label,
                log_result=!find_opt, opt_U=opt_U, opt_V=opt_V,
                plant_U=plant_U, plant_V=plant_V)
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

function _print_solver_block(summary::SuiteSummary, label::AbstractString; with_optimum::Bool=true)
    s = summary.stats[label]
    _print_solver_block(label, summary.N, s; with_optimum=with_optimum)
    if !with_optimum
        print_planted_hit_menu(summary, label; indent="  ")
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
    # ratio == -Inf marks k-exceeded invalids (see record_trial!).
    return !result.valid || result.ratio == -Inf || (isfinite(result.ratio) && result.ratio < 1.0)
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

planted_edge_count(info::TrialInfo) = length(info.plant_U) * length(info.plant_V) - info.k

"""
True when the solver returned a θ/k-valid solution with at least as many edges
as the planted k-MDB (the injected biclique).
"""
function trial_matched_planted(info::TrialInfo, result::TrialResult)
    return result.valid && result.ratio != -Inf && result.edges >= planted_edge_count(info)
end

"""
Print planted-hit totals and a largest-θ-first menu (for `--no-optimum` runs).
"""
function print_planted_hit_menu(summary::SuiteSummary, label::AbstractString;
    indent::AbstractString="")
    s = summary.stats[label]
    hits = 0
    # θ_plant => (hits, total)
    by_θ = Dict{Int,Tuple{Int,Int}}()
    for trial in 1:summary.N
        info = summary.trials[trial]
        r = s.results[trial]
        matched = trial_matched_planted(info, r)
        matched && (hits += 1)
        h, t = get(by_θ, info.θ_plant, (0, 0))
        by_θ[info.θ_plant] = (h + Int(matched), t + 1)
    end

    N = summary.N
    println("$(indent)Planted matches      : $hits / $N")
    println("$(indent)Planted match rate   : $(round(100 * hits / N; digits=1))%")
    println()
    println("$(indent)Planted hits by θ (largest first):")
    for θ in sort!(collect(keys(by_θ)); rev=true)
        h, t = by_θ[θ]
        bar = "█"^h * "·"^(t - h)
        println("$(indent)  θ=$(lpad(θ, 2))  $(lpad(h, 3))/$t  $bar")
    end
end

"""
Log each non-optimal / invalid trial with seed, graph, reference solution, and
per-solver returns. With an oracle, uses opt_U/opt_V; otherwise uses the planted
k-MDB and flags invalid (θ/k) trials.
"""
function print_nonoptimal_trial_seeds(summary::SuiteSummary; labels=nothing)
    ordered = labels === nothing ? collect(keys(summary.stats)) : collect(labels)
    any_bad = false
    with_opt = has_optimum(summary)

    for trial in 1:summary.N
        bad_labels = if with_opt
            String[label for label in ordered
                   if trial_is_nonoptimal(summary.stats[label].results[trial])]
        else
            # Without an oracle, only invalid returns are "wrong".
            String[label for label in ordered
                   if !summary.stats[label].results[trial].valid ||
                      summary.stats[label].results[trial].ratio == -Inf]
        end
        isempty(bad_labels) && continue

        if !any_bad
            println(with_opt ? "Non-optimal trials:" : "Invalid trials:")
            any_bad = true
        end

        print_trial_detail(summary, trial; labels=ordered, highlight=bad_labels)
    end
end

function print_suite_summary(summary::SuiteSummary; labels=nothing)
    with_opt = has_optimum(summary)
    ordered = labels === nothing ? collect(keys(summary.stats)) : collect(labels)

    println()
    println("==================== RESULTS ====================")
    println("Trials               : $(summary.N)")
    println("Seed                 : --seed=$(summary.seed)")
    println("Oracle               : $(summary.oracle)")
    if !isempty(summary.algorithm)
        println("Algorithm            : $(summary.algorithm)")
    end

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
            print_planted_hit_menu(summary, ordered[1])
        end
    else
        if with_opt
            println("Optimal solutions:")
            for label in ordered
                println("  $(rpad(label, 8)): $(summary.stats[label].optimal) / $(summary.N)")
            end
        end
        for label in ordered
            _print_solver_block(summary, label; with_optimum=with_opt)
        end
    end
    println("=================================================")
end

function print_trial_detail(summary::SuiteSummary, trial::Int; labels=nothing, highlight=nothing)
    info = summary.trials[trial]
    ordered = labels === nothing ? sort!(collect(keys(summary.stats))) : collect(labels)

    println()
    println("---------- trial $trial ----------")
    println("Reproduce suite:  julia $(PROGRAM_FILE) --seed=$(summary.seed)")
    println("Reproduce graph:  julia $(PROGRAM_FILE) --seed=$(info.graph_seed) --N=1")
    θ_str = info.θ == info.θ_plant ? "θ=$(info.θ)" : "θ=$(info.θ) (planted $(info.θ_plant))"
    println("Graph: nU=$(info.nU)  nV=$(info.nV)  |E|=$(info.n_edges)  k=$(info.k)  $θ_str")
    if !isempty(info.edges)
        println("Edges: $(info.edges)")
    end
    if has_optimum(info)
        println("$(oracle_label(summary.oracle)) optimum: edges=$(info.opt_edges) missing=$(length(info.opt_U) * length(info.opt_V) - info.opt_edges) U=$(sort!(collect(info.opt_U)))  V=$(sort!(collect(info.opt_V)))")
    else
        plant_edges = length(info.plant_U) * length(info.plant_V) - info.k
        println("Optimum: (not computed)")
        println("Planted: edges=$plant_edges  |U|=$(length(info.plant_U)) |V|=$(length(info.plant_V))  U=$(sort!(collect(info.plant_U)))  V=$(sort!(collect(info.plant_V)))")
    end
    println()

    for label in ordered
        r = summary.stats[label].results[trial]
        status = if r.ratio == -Inf || r.missing > info.k
            "INVALID (k exceeded)"
        elseif r.valid
            "valid"
        else
            "INVALID (θ not met)"
        end
        mark = highlight !== nothing && label in highlight ? " *" : ""
        ratio_str = isfinite(r.ratio) ? string(r.ratio) : "n/a"
        println("  $label$mark")
        println("    edges=$(r.edges)  missing=$(r.missing)  ratio=$ratio_str  ($status)  |U|=$(length(r.U)) |V|=$(length(r.V))")
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
        "theta_plant" => info.θ_plant,
        "opt_edges" => info.opt_edges,
        "opt_U" => sort!(collect(info.opt_U)),
        "opt_V" => sort!(collect(info.opt_V)),
        "plant_U" => sort!(collect(info.plant_U)),
        "plant_V" => sort!(collect(info.plant_V)),
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
