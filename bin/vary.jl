#=
=================================================================================
Parameter sweeps for load.jl.

Invoked via:
  julia bin/load.jl <dataset> --vary=ant-count [--ants-range=5,10,20,50]
      [--aco-runs=N] [--save=...]

For `--vary=ant-count`, runs ACO once per ant count (fixed iteration budget from
`--iterations`) and records iterations-to-best, solution quality, and wall time.
With `--aco-runs=N` (N>1), each ant count is repeated N times with a distinct
seed derived from `--seed`, and every replicate is written into the JSON.
Each trial also logs the returned subgraph vertex ids (`U` / `V`) so downstream
tools (compare-seeds.jl) can re-seed branch-and-pivot.
The JSON also records `best_trial` (max edges) and `aco_discovery_s`: sum of
`wall_time_s` over every replicate at that trial's ant count (full budget,
not truncated at the winning run).
Top-level keys `prefer_smaller_side` and `neighbor_scope_limit` mirror the
CLI flags used for the ACO trials.
Pivot is off by default (it is too slow on large graphs); enable with
`--vary-pivot=true` to supply an optimum edge count for quality comparison.
The θ-heuristic also runs once (same reduced graph) and is logged in the JSON
so ACO quality can be compared against it.

Graph reduction is always common-neighbor (CNN / `ReductionMode.simple`).
Progressive / `all_reductions` must not run as a one-shot peel here: that path
tightens θ_eff without interleaved search and can delete the whole graph
(e.g. kuwiki → 0×0) before ACO or the θ-heuristic run.

Reduced-graph degree stats (`reduced_max_degree`, `reduced_avg_degree`, etc.)
are cached per dataset in `data/<dataset>/graph_structure.json` (tracked in
git). vary.jl reads that file and only computes + writes when the entry for
the current (k, θ, reduction, inject) setup is missing.

When `DEBUG` is true (env; `scripts/vary.bash` defaults it on) and `--inject`
planted a biclique, each run logs whether that plant still exists after the
CNN reduction.
=================================================================================
=#

const __VARY_JL__ = true

"""
Parse `--vary=<mode>` if present; otherwise `nothing`.
"""
function parse_vary()
    for arg in ARGS
        if startswith(arg, "--vary=")
            mode = lowercase(strip(split(arg, "=", limit=2)[2]))
            if mode == "ant-count"
                return :ant_count
            else
                throw(ArgumentError("Unknown --vary mode '$mode' (expected ant-count)"))
            end
        elseif arg == "--vary"
            throw(ArgumentError("--vary requires a value, e.g. --vary=ant-count"))
        end
    end
    return nothing
end

"""
Parse `--ants-range=a,b,c` into a sorted unique vector of positive integers.
Default sweep: 1, 2, 5, 10, 20, 50, 100.
"""
function parse_ants_range()
    for arg in ARGS
        if startswith(arg, "--ants-range=")
            raw = split(arg, "=", limit=2)[2]
            counts = Int[]
            for part in split(raw, ',')
                s = strip(part)
                isempty(s) && continue
                n = parse(Int, s)
                n >= 1 || throw(ArgumentError("ants-range values must be >= 1, got $n"))
                push!(counts, n)
            end
            isempty(counts) && throw(ArgumentError("--ants-range= needs at least one integer"))
            return sort!(unique!(counts))
        end
    end
    return [1, 2, 5, 10, 20, 50, 100]
end

"""
Parse `--aco-runs=N`: number of stochastic ACO replicates per ant count (default 1).
"""
function parse_aco_runs()
    for arg in ARGS
        if startswith(arg, "--aco-runs=")
            n = parse(Int, split(arg, "=", limit=2)[2])
            n >= 1 || throw(ArgumentError("--aco-runs must be >= 1, got $n"))
            return n
        elseif arg == "--aco-runs"
            throw(ArgumentError("--aco-runs requires a value, e.g. --aco-runs=10"))
        end
    end
    return 1
end

"""
Whether to run pivot once for an optimum edge count (default: false).
Enable with `--vary-pivot=true`.
"""
function parse_vary_run_pivot()
    for arg in ARGS
        if startswith(arg, "--vary-pivot=")
            val = lowercase(split(arg, "=", limit=2)[2])
            if val in ("true", "1", "yes", "on")
                return true
            elseif val in ("false", "0", "no", "off")
                return false
            else
                throw(ArgumentError("Bad boolean for --vary-pivot: $val"))
            end
        elseif arg == "--vary-pivot"
            throw(ArgumentError("--vary-pivot requires a value, e.g. --vary-pivot=true"))
        end
    end
    return false
end

"""
Stable per-replicate seed: `base + (ant_index-1)*n_runs + (run-1)`.
"""
function vary_run_seed(base_seed::UInt64, ant_index::Int, run::Int, n_runs::Int)
    offset = UInt64((ant_index - 1) * n_runs + (run - 1))
    return base_seed + offset
end

"""Best trial by edges even if it did not beat the heuristic."""
function select_best_trial_any(trials)
    usable = [t for t in trials if t.final_edges !== nothing]
    isempty(usable) && return nothing
    best_edges = maximum(t.final_edges for t in usable)
    tied = [t for t in usable if t.final_edges == best_edges]
    trial_time(t) = t.time_to_best_s !== nothing ? Float64(t.time_to_best_s) :
        Float64(t.wall_time_s)
    return argmin(trial_time, tied)
end

"""
Full ACO wall time at the winning ant count: sum `wall_time_s` over every
same-ants replicate (not truncated at the winning run / time-to-best).
"""
function aco_discovery_until(trials, best)
    best === nothing && return nothing
    win_ants = best.num_ants
    total = 0.0
    n = 0
    for t in trials
        t.num_ants != win_ants && continue
        total += Float64(t.wall_time_s)
        n += 1
    end
    return n > 0 ? total : nothing
end

function best_trial_summary_dict(t)
    t === nothing && return nothing
    return Dict{String,Any}(
        "run" => get(t, :run, nothing),
        "seed" => get(t, :seed, nothing),
        "ants" => t.num_ants,
        "final_edges" => t.final_edges,
        "time_to_best_s" => t.time_to_best_s,
        "wall_time_s" => t.wall_time_s,
        "iterations_to_best" => t.iterations_to_best,
        "beats_heuristic" => get(t, :beats_heuristic, nothing),
        "theta_feasible" => t.theta_feasible,
        "nU" => t.nU,
        "nV" => t.nV,
    )
end

"""Map a compact-id addition trace to original vertex ids for JSON export."""
function remap_addition_order(remapping::GraphRemapping, order::Vector{Tuple{Bool,Int}})
    uo, vo = remapping.u_original, remapping.v_original
    return [
        is_u ? ["u", uo[id]] : ["v", vo[id]]
        for (is_u, id) in order
    ]
end

function _env_flag(name::AbstractString, default::Bool=false)
    v = lowercase(get(ENV, name, default ? "true" : "false"))
    return v in ("true", "1", "yes", "on")
end

"""Include per-ant raw samples in JSON (large). Default: summary mean/n only."""
record_construction_samples() = _env_flag("RECORD_CONSTRUCTION_SAMPLES", false)

"""Include last-iteration vertex addition traces in JSON (very large)."""
record_construction_orders() = _env_flag("RECORD_CONSTRUCTION_ORDERS", false)

function construction_stats_dict(stats, remapping::GraphRemapping)
    stats === nothing && return nothing
    missing_at_size = Dict{String,Any}()
    for (size, samples) in stats.missing_at_size
        n = length(samples)
        entry = Dict{String,Any}(
            "mean" => n > 0 ? sum(samples) / n : nothing,
            "n" => n,
        )
        record_construction_samples() && (entry["samples"] = samples)
        missing_at_size[string(size)] = entry
    end
    out = Dict{String,Any}("missing_at_size" => missing_at_size)
    if record_construction_orders()
        orders = [
            remap_addition_order(remapping, order)
            for order in stats.last_iteration_orders
        ]
        out["last_iteration_orders"] = orders
    end
    return out
end

"""Pooled missing-at-size means across trials, keyed by ant count then |S|."""
function pool_missing_at_size_summary(trials)
    pooled_n = Dict{String, Dict{String, Int}}()
    pooled_sum = Dict{String, Dict{String, Float64}}()
    for t in trials
        c = get(t, :construction, nothing)
        c === nothing && continue
        mas = get(c, "missing_at_size", nothing)
        mas === nothing && continue
        ants_key = string(t.num_ants)
        n_bucket = get!(pooled_n, ants_key, Dict{String, Int}())
        s_bucket = get!(pooled_sum, ants_key, Dict{String, Float64}())
        for (size_str, entry) in mas
            n = Int(get(entry, "n", 0))
            mean = get(entry, "mean", nothing)
            n == 0 || mean === nothing && continue
            n_bucket[size_str] = get(n_bucket, size_str, 0) + n
            s_bucket[size_str] = get(s_bucket, size_str, 0.0) + Float64(mean) * n
        end
    end
    out = Dict{String, Any}()
    for (ants_key, sizes) in pooled_n
        size_out = Dict{String, Any}()
        for (size_str, n) in sizes
            n == 0 && continue
            size_out[size_str] = Dict(
                "mean" => pooled_sum[ants_key][size_str] / n,
                "n" => n,
            )
        end
        isempty(size_out) || (out[ants_key] = size_out)
    end
    return out
end

"""
Run ACO for a single ant count; return metrics without verbose printing.
"""
function vary_aco_trial!(g::BipartiteGraph, k::Int, θ::Int, aco_options;
    num_ants::Int, opt_edges::Union{Nothing,Int}=nothing,
    reduction::ReductionMode.T=ReductionMode.none)
    pheremone, _na, num_iterations, evaporation, num_subspecies = aco_options
    prefer_smaller_side = get(aco_options, :prefer_smaller_side, true)
    neighbor_scope_limit = get(aco_options, :neighbor_scope_limit, true)
    elite_seed = get(aco_options, :elite_seed, true)
    elite_seed_ants = get(aco_options, :elite_seed_ants, 3)
    elite_seed_remove = get(aco_options, :elite_seed_remove, 2)

    first_hit = Ref{Union{Nothing,Int}}(nothing)
    construction_stats = Ref{Any}(nothing)

    g_run = deepcopy(g)
    m = measure_call() do
        aco(g_run, pheremone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
            parallelize=false,
            force_gc=false,
            prefer_smaller_side=prefer_smaller_side,
            neighbor_scope_limit=neighbor_scope_limit,
            elite_seed=elite_seed,
            elite_seed_ants=elite_seed_ants,
            elite_seed_remove=elite_seed_remove,
            reduction=reduction,
            construction_stats=construction_stats,
            iteration_callback = (iter, best_compact, compact_fg, _remapping, _elapsed_s) -> begin
                if opt_edges === nothing || first_hit[] !== nothing
                    return true
                end
                edges = Subgraph.edge_count(compact_fg, best_compact)
                missing = Subgraph.missing_edges(compact_fg, best_compact)
                θ_ok = (length(best_compact.U) ≥ θ && length(best_compact.V) ≥ θ) ||
                    (opt_edges == 0 && edges == 0)
                if missing <= k && θ_ok && edges >= opt_edges
                    first_hit[] = iter
                end
                return true
            end)
    end

    sols, best_iterations, best_times, _pheromones, remapping = m.value
    g_eval = deepcopy(g)
    fg_eval = if reduction == ReductionMode.none
        freeze(g_eval)
    else
        apply_graph_reductions!(g_eval, k, θ, nothing, nothing, true, reduction)
    end
    best_idx = argmax(i -> Subgraph.edge_count(fg_eval, sols[i]), eachindex(sols))
    sol = sols[best_idx]
    iterations_to_best = best_iterations[best_idx]
    time_to_best = best_times[best_idx]
    final_edges = Subgraph.edge_count(fg_eval, sol)
    missing = Subgraph.missing_edges(fg_eval, sol)
    θ_feasible = (length(sol.U) ≥ θ && length(sol.V) ≥ θ) ||
        (final_edges == 0 && Subgraph.vertex_count(sol) == 0)
    matched_optimal = opt_edges !== nothing &&
        is_optimal_solution(fg_eval, sol, k, θ, opt_edges)

    return (; num_ants,
        wall_time_s = m.time,
        time_to_best_s = time_to_best,
        allocated_bytes = m.allocated,
        rss_delta_bytes = m.rss_delta,
        iterations_budget = num_iterations,
        iterations_to_best,
        iterations_to_optimal = first_hit[],
        final_edges,
        optimal_edges = opt_edges,
        matched_optimal,
        nU = length(sol.U),
        nV = length(sol.V),
        U = sort!(collect(sol.U)),
        V = sort!(collect(sol.V)),
        missing,
        theta_feasible = θ_feasible,
        construction = construction_stats_dict(construction_stats[], remapping))
end

"""
Whether terminal `DEBUG` is on (`true`/`1`/`yes`/`on`).
`scripts/vary.bash` exports `DEBUG=true` by default.
"""
function vary_debug_enabled()
    v = lowercase(get(ENV, "DEBUG", "true"))
    return v in ("true", "1", "yes", "on")
end

"""
After reduction, check that every planted U/V vertex still exists on `fg`.
Logs present / absent (and which vertices were deleted when absent).
"""
function debug_check_plant_after_reduction!(fg::FrozenBipartite, plant;
    dataset::AbstractString="")
    plant === nothing && return nothing
    plant_U = plant.U
    plant_V = plant.V
    gone_U = [u for u in plant_U if !haskey(fg.u_index, u)]
    gone_V = [v for v in plant_V if !haskey(fg.v_index, v)]
    present = isempty(gone_U) && isempty(gone_V)
    label = isempty(dataset) ? "" : " dataset=$dataset"
    if present
        println("DEBUG: planted solution still in graph after reduction: yes$label")
        println("  plant U=$(collect(plant_U))  V=$(collect(plant_V))")
    else
        println("DEBUG: planted solution still in graph after reduction: no$label")
        println("  plant U=$(collect(plant_U))  V=$(collect(plant_V))")
        !isempty(gone_U) && println("  deleted U: $gone_U")
        !isempty(gone_V) && println("  deleted V: $gone_V")
    end
    return present
end

"""
Sweep ant counts on an already-reduced graph.

When `n_runs > 1`, each ant count is repeated with a distinct seed derived from
`seed` (required). Every replicate is appended to `trials`.
When `DEBUG` is set and `plant` is provided (from `--inject`), verifies the
planted biclique vertices survive the one-shot CNN reduction and logs the result.
"""
function run_vary_ant_count!(g::BipartiteGraph, edge_count::Int, k::Int, θ::Int,
    reduction::ReductionMode.T, aco_options; ant_counts::Vector{Int},
    run_pivot::Bool=false, seed=nothing, dataset::AbstractString="",
    n_runs::Int=1, plant=nothing, inject=nothing)
    n_runs >= 1 || throw(ArgumentError("n_runs must be >= 1, got $n_runs"))
    _, _, num_iterations, _, _ = aco_options

    # One-shot peel: CNN only. Ignore progressive / all_reductions from CLI —
    # those require interleaved search (find_kmdb!) and can empty the graph.
    if reduction != ReductionMode.simple && reduction != ReductionMode.none
        @warn "vary ant-count forces ReductionMode.simple (CNN); got $reduction"
    end
    reduction = reduction == ReductionMode.none ? ReductionMode.none : ReductionMode.simple

    base_seed = seed === nothing ? UInt64(time_ns()) : UInt64(seed)

    println()
    prefer_smaller_side = get(aco_options, :prefer_smaller_side, true)
    neighbor_scope_limit = get(aco_options, :neighbor_scope_limit, true)

    println("==================== VARY ant-count ====================")
    println("dataset=$dataset  k=$k  θ=$θ  iterations=$num_iterations")
    println("ants-range=$(join(ant_counts, ","))  run_pivot=$run_pivot  aco_runs=$n_runs")
    println("base_seed=$base_seed")
    println("prefer_smaller_side=$prefer_smaller_side  neighbor_scope_limit=$neighbor_scope_limit")
    println("reduction=$reduction (CNN / one-shot; progressive disabled)")
    if vary_debug_enabled()
        println("DEBUG=$(vary_debug_enabled())  plant_check=$(plant !== nothing)")
    end

    println()
    println("Warming up (excluded from timings)…")
    warmup_benchmarks!(k, θ, reduction, aco_options; targets=Set([:aco, :heuristic]))

    mutable_bytes = Base.summarysize(g)
    nU = length(g.adjU)
    nV = length(g.adjV)
    g_reduced = deepcopy(g)
    println()
    println("Reducing graph once (CNN)…")
    m_red = measure_call() do
        apply_graph_reductions!(g_reduced, k, θ, nothing, nothing, true, reduction)
    end
    fg = m_red.value
    if vary_debug_enabled()
        if plant === nothing
            println("DEBUG: no planted solution to verify after reduction")
        else
            debug_check_plant_after_reduction!(fg, plant; dataset=dataset)
        end
    end
    frozen_bytes = Base.summarysize(fg)
    reduced_nU = length(fg.u_ids)
    reduced_nV = length(fg.v_ids)
    reduced_edges = length(fg.v_adj)
    inject_cfg = inject === nothing ? (; enabled=false, nU=0, nV=0, attempts=0) : inject
    structure_key = graph_structure_cache_key(k, θ, reduction, inject_cfg; seed=seed)
    reduced_max_degree, reduced_avg_degree = resolve_reduced_degree_stats!(
        dataset, structure_key, nU, nV, edge_count, fg)
    print_metric_block("Graph reduction";
        wall_time_s = m_red.time,
        allocated_bytes = m_red.allocated,
        reduced_nU,
        reduced_nV,
        reduced_edges,
        reduced_max_degree,
        reduced_avg_degree,
    )
    print_metric_block("Graph structure";
        nU,
        nV,
        edges = edge_count,
        reduced_nU,
        reduced_nV,
        reduced_edges,
        reduced_max_degree,
        reduced_avg_degree,
        mutable_graph_bytes = mutable_bytes,
        frozen_after_reduction_bytes = frozen_bytes,
    )
    graph_stats = (; nU, nV, edges = edge_count,
        reduced_nU, reduced_nV, reduced_edges,
        reduced_max_degree, reduced_avg_degree,
        mutable_bytes, frozen_bytes, fg,
        reduction_time = m_red.time, reduction_allocated = m_red.allocated,
        reduction_rss_delta = m_red.rss_delta)

    solver_reduction = ReductionMode.none

    opt_edges = nothing
    pivot_stats = nothing
    if run_pivot
        pivot_stats = benchmark_pivot!(g_reduced, k, θ, solver_reduction)
        opt_edges = pivot_stats.opt_edges
    end

    # θ-heuristic once per graph — baseline for "does ACO beat the heuristic?"
    heuristic_stats = benchmark_heuristic!(g_reduced, k, θ, solver_reduction; opt_edges=opt_edges)
    heur_missing = Subgraph.missing_edges(heuristic_stats.fg_eval, heuristic_stats.sol)
    heur_θ_ok = (length(heuristic_stats.sol.U) ≥ θ && length(heuristic_stats.sol.V) ≥ θ) ||
        (heuristic_stats.final_edges == 0 && Subgraph.vertex_count(heuristic_stats.sol) == 0)
    heuristic_stats = merge(heuristic_stats, (;
        nU = length(heuristic_stats.sol.U),
        nV = length(heuristic_stats.sol.V),
        U = sort!(collect(heuristic_stats.sol.U)),
        V = sort!(collect(heuristic_stats.sol.V)),
        missing = heur_missing,
        theta_feasible = heur_θ_ok,
    ))

    trials = NamedTuple[]
    for (ai, n_ants) in enumerate(ant_counts)
        for run in 1:n_runs
            run_seed = vary_run_seed(base_seed, ai, run, n_runs)
            Random.seed!(run_seed)

            println()
            if n_runs == 1
                println("── ACO $(ai)/$(length(ant_counts)): ants=$n_ants ──")
            else
                println("── ACO ants=$n_ants  run $run/$n_runs  seed=$run_seed ──")
            end
            trial = vary_aco_trial!(g_reduced, k, θ, aco_options;
                num_ants=n_ants, opt_edges=opt_edges, reduction=solver_reduction)
            beats_heuristic = trial.final_edges > heuristic_stats.final_edges &&
                trial.theta_feasible
            trial = merge(trial, (; run, seed=string(run_seed), beats_heuristic))
            println("  wall=$(format_seconds(trial.wall_time_s))s  " *
                    "to_best=$(format_seconds(trial.time_to_best_s))s  " *
                    "iters→best=$(trial.iterations_to_best)  " *
                    "edges=$(trial.final_edges)" *
                    (opt_edges === nothing ? "" : " / $opt_edges optimal") *
                    "  vs heur=$(heuristic_stats.final_edges)" *
                    (beats_heuristic ? " (beats)" : ""))
            push!(trials, trial)
        end
    end

    println()
    println("==================== SUMMARY ======================")
    println("θ-heuristic edges=$(heuristic_stats.final_edges)" *
            (opt_edges === nothing ? "" : " / $opt_edges optimal") *
            "  θ-feasible=$(heuristic_stats.theta_feasible)" *
            "  |U|=$(heuristic_stats.nU) |V|=$(heuristic_stats.nV)")
    if n_runs == 1
        println(rpad("ants", 8), rpad("time", 12), rpad("→best", 12),
                rpad("iters", 8), rpad("edges", 10), rpad(">heur", 8), "optimal")
        for t in trials
            opt_str = t.optimal_edges === nothing ? "—" :
                (t.matched_optimal ? "yes" : "no ($(t.final_edges)/$(t.optimal_edges))")
            println(rpad(t.num_ants, 8),
                    rpad(format_seconds(t.wall_time_s) * "s", 12),
                    rpad(format_seconds(t.time_to_best_s) * "s", 12),
                    rpad(string(t.iterations_to_best), 8),
                    rpad(string(t.final_edges), 10),
                    rpad(t.beats_heuristic ? "yes" : "no", 8),
                    opt_str)
        end
    else
        println(rpad("ants", 8), rpad("run", 6), rpad("time", 12),
                rpad("iters", 8), rpad("edges", 10), rpad(">heur", 8), "optimal")
        for t in trials
            opt_str = t.optimal_edges === nothing ? "—" :
                (t.matched_optimal ? "yes" : "no ($(t.final_edges)/$(t.optimal_edges))")
            println(rpad(t.num_ants, 8),
                    rpad(t.run, 6),
                    rpad(format_seconds(t.wall_time_s) * "s", 12),
                    rpad(string(t.iterations_to_best), 8),
                    rpad(string(t.final_edges), 10),
                    rpad(t.beats_heuristic ? "yes" : "no", 8),
                    opt_str)
        end
    end
    println("===================================================")

    best_trial = select_best_trial_any(trials)
    aco_discovery_s = aco_discovery_until(trials, best_trial)
    if best_trial !== nothing
        println()
        println("Best ACO trial (by edges): ants=$(best_trial.num_ants)  " *
                "run=$(get(best_trial, :run, "?"))  " *
                "edges=$(best_trial.final_edges)  " *
                "discovery=$(format_seconds(aco_discovery_s))s  " *
                "ITB=$(best_trial.iterations_to_best)  " *
                "TTB=$(format_seconds(best_trial.time_to_best_s))s  " *
                "θ-feasible=$(best_trial.theta_feasible)")
    end

    return (; graph_stats, pivot_stats, heuristic_stats, trials, ant_counts, opt_edges,
        n_runs, base_seed=string(base_seed), best_trial, aco_discovery_s,
        prefer_smaller_side, neighbor_scope_limit)
end

function vary_results_to_dict(results; k::Int=0, θ::Int=0,
    dataset::AbstractString="", seed=nothing, reduction=nothing,
    edge_count::Union{Nothing,Int}=nothing, run_pivot::Bool=false,
    prefer_smaller_side::Union{Nothing,Bool}=nothing,
    neighbor_scope_limit::Union{Nothing,Bool}=nothing)
    pss = prefer_smaller_side === nothing ?
        get(results, :prefer_smaller_side, nothing) : prefer_smaller_side
    nsl = neighbor_scope_limit === nothing ?
        get(results, :neighbor_scope_limit, nothing) : neighbor_scope_limit
    out = Dict{String,Any}(
        "vary" => "ant-count",
        "dataset" => String(dataset),
        "k" => k,
        "theta" => θ,
        "seed" => seed === nothing ? nothing : string(seed),
        "base_seed" => get(results, :base_seed, seed === nothing ? nothing : string(seed)),
        "aco_runs" => get(results, :n_runs, 1),
        "reduction" => reduction === nothing ? nothing : string(reduction),
        "edge_count" => edge_count,
        "run_pivot" => run_pivot,
        "prefer_smaller_side" => pss,
        "neighbor_scope_limit" => nsl,
        "ants_range" => results.ant_counts,
    )

    gs = results.graph_stats
    out["graph"] = Dict(
        "nU" => get(gs, :nU, nothing),
        "nV" => get(gs, :nV, nothing),
        "edges" => get(gs, :edges, edge_count),
        "reduced_nU" => get(gs, :reduced_nU, length(gs.fg.u_ids)),
        "reduced_nV" => get(gs, :reduced_nV, length(gs.fg.v_ids)),
        "reduced_edges" => get(gs, :reduced_edges, length(gs.fg.v_adj)),
        "reduced_max_degree" => get(gs, :reduced_max_degree, nothing),
        "reduced_avg_degree" => get(gs, :reduced_avg_degree, nothing),
        "mutable_bytes" => gs.mutable_bytes,
        "frozen_bytes" => gs.frozen_bytes,
        "reduction_time_s" => gs.reduction_time,
    )

    if results.pivot_stats !== nothing
        ps = results.pivot_stats
        out["pivot"] = Dict(
            "wall_time_s" => ps.time,
            "optimal_edges" => ps.opt_edges,
        )
    end

    if results.heuristic_stats !== nothing
        hs = results.heuristic_stats
        out["heuristic"] = Dict(
            "wall_time_s" => hs.time,
            "allocated_bytes" => hs.allocated,
            "rss_delta_bytes" => hs.rss_delta,
            "final_edges" => hs.final_edges,
            "optimal_edges" => hs.opt_edges,
            "matched_optimal" => hs.matched_optimal,
            "nU" => get(hs, :nU, length(hs.sol.U)),
            "nV" => get(hs, :nV, length(hs.sol.V)),
            "U" => get(hs, :U, sort!(collect(hs.sol.U))),
            "V" => get(hs, :V, sort!(collect(hs.sol.V))),
            "missing" => get(hs, :missing, nothing),
            "theta_feasible" => get(hs, :theta_feasible, nothing),
        )
    end

    out["trials"] = [
        Dict(
            "run" => get(t, :run, 1),
            "seed" => get(t, :seed, nothing),
            "ants" => t.num_ants,
            "wall_time_s" => t.wall_time_s,
            "time_to_best_s" => t.time_to_best_s,
            "allocated_bytes" => t.allocated_bytes,
            "rss_delta_bytes" => t.rss_delta_bytes,
            "iterations_budget" => t.iterations_budget,
            "iterations_to_best" => t.iterations_to_best,
            "iterations_to_optimal" => t.iterations_to_optimal,
            "final_edges" => t.final_edges,
            "optimal_edges" => t.optimal_edges,
            "matched_optimal" => t.matched_optimal,
            "beats_heuristic" => get(t, :beats_heuristic, nothing),
            "nU" => t.nU,
            "nV" => t.nV,
            "U" => get(t, :U, Int[]),
            "V" => get(t, :V, Int[]),
            "missing" => t.missing,
            "theta_feasible" => t.theta_feasible,
            "construction" => get(t, :construction, nothing),
        ) for t in results.trials
    ]

    best = get(results, :best_trial, nothing)
    if best !== nothing
        out["best_trial"] = best_trial_summary_dict(best)
        out["aco_discovery_s"] = get(results, :aco_discovery_s, nothing)
    end

    mas_summary = pool_missing_at_size_summary(results.trials)
    isempty(mas_summary) || (out["missing_at_size"] = mas_summary)

    return out
end

function save_vary_json(path::AbstractString, payload::AbstractDict)
    return save_benchmark_json(path, payload)
end
