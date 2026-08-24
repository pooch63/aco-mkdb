#=
=================================================================================
Compare branch-and-pivot wall time when seeded with the θ-heuristic alone
versus θ-heuristic + an external seed subgraph (typically an ACO solution).

Modes
-----
1. From a vary.jl ant-count JSON (batch helper for scripts/compare-seeds.bash):

     julia -t 8 compare-seeds.jl vary_k2t5i/boxes_ants.json \
         --inject --u=5 --v=5 --save=compare_k2t5i/boxes.json

   Picks the ACO trial that beat the θ-heuristic with the most edges; on ties,
   the replicate with the smallest time_to_best_s. If ACO never beat the
   heuristic (or the best beating trial lacks U/V), still writes a compact
   marker JSON under --save= so re-runs can SKIP_EXISTING without re-parsing
   the vary file. load.jl is only included when a real pivot comparison runs.

2. Explicit seed subgraph JSON `{"U":[...],"V":[...]}`:

     julia -t 8 compare-seeds.jl --dataset=amazon/boxes --k=2 --theta=5 \
         --seed-json=seed.json --inject --u=5 --v=5 --seed=1 \
         --save=compare_k2t5i/boxes.json

Both modes time full pivot runs (default progressive / all_reductions) on a
fresh copy of the loaded graph each time — same path as a normal load.jl pivot,
so progressive reduction stays interleaved with branching. (a) θ-heuristic seed
only, (b) best of θ-heuristic and the external seed.
=================================================================================
=#

using Random
using JSON3

function usage_and_exit()
    println(stderr, """
Usage:
  julia compare-seeds.jl <vary.json> [--inject --u=N --v=M] [--seed=S] [--save=out.json]
  julia compare-seeds.jl --dataset=KEY --seed-json=seed.json --k=K --theta=T
      [--inject --u=N --v=M] [--seed=S] [--save=out.json] [--reduce=hi|lo|none]
""")
    exit(1)
end

function parse_int_eq(flag::AbstractString, default)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function parse_string_eq(flag::AbstractString, default=nothing)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return split(arg, "=", limit=2)[2]
    end
    return default
end

function json_get(obj, key::String, default=nothing)
    if obj isa AbstractDict
        return get(obj, key, get(obj, Symbol(key), default))
    end
    sym = Symbol(key)
    return haskey(obj, sym) ? obj[sym] : (haskey(obj, key) ? obj[key] : default)
end

function json_int_vec(x)
    x === nothing && return Int[]
    return Int[Int(v) for v in x]
end

"""
Among trials with beats_heuristic==true, pick max final_edges; break ties by
minimum time_to_best_s (then wall_time_s).
"""
function select_best_beating_trial(trials)
    beating = Any[]
    for t in trials
        beats = json_get(t, "beats_heuristic", false)
        beats === true || beats === 1 || continue
        push!(beating, t)
    end
    isempty(beating) && return nothing

    best_edges = maximum(Int(json_get(t, "final_edges", 0)) for t in beating)
    tied = [t for t in beating if Int(json_get(t, "final_edges", 0)) == best_edges]

    function trial_time(t)
        tb = json_get(t, "time_to_best_s", nothing)
        tb !== nothing && return Float64(tb)
        wt = json_get(t, "wall_time_s", Inf)
        return Float64(wt)
    end

    return argmin(trial_time, tied)
end

"""Best trial by edges even if it did not beat the heuristic."""
function select_best_trial_any(trials)
    usable = Any[]
    for t in trials
        json_get(t, "final_edges", nothing) === nothing && continue
        push!(usable, t)
    end
    isempty(usable) && return nothing

    best_edges = maximum(Int(json_get(t, "final_edges", 0)) for t in usable)
    tied = [t for t in usable if Int(json_get(t, "final_edges", 0)) == best_edges]

    function trial_time(t)
        tb = json_get(t, "time_to_best_s", nothing)
        tb !== nothing && return Float64(tb)
        wt = json_get(t, "wall_time_s", Inf)
        return Float64(wt)
    end

    return argmin(trial_time, tied)
end

function trial_has_subgraph(t)
    U = json_get(t, "U", nothing)
    V = json_get(t, "V", nothing)
    return U !== nothing && V !== nothing
end

function trial_summary_dict(t)
    t === nothing && return nothing
    return Dict{String,Any}(
        "run" => json_get(t, "run", nothing),
        "seed" => json_get(t, "seed", nothing),
        "ants" => json_get(t, "ants", nothing),
        "final_edges" => json_get(t, "final_edges", nothing),
        "time_to_best_s" => json_get(t, "time_to_best_s", nothing),
        "wall_time_s" => json_get(t, "wall_time_s", nothing),
        "beats_heuristic" => json_get(t, "beats_heuristic", nothing),
        "theta_feasible" => json_get(t, "theta_feasible", nothing),
        "nU" => json_get(t, "nU", nothing),
        "nV" => json_get(t, "nV", nothing),
    )
end

function aco_trial_time_stats(trials)
    total = 0.0
    n_timed = 0
    for t in trials
        wt = json_get(t, "wall_time_s", nothing)
        wt === nothing && continue
        total += Float64(wt)
        n_timed += 1
    end
    mean_wct = n_timed > 0 ? total / n_timed : nothing
    return total, mean_wct, n_timed
end

function heuristic_summary(heur)
    heur === nothing && return nothing
    return Dict{String,Any}(
        "final_edges" => json_get(heur, "final_edges", nothing),
        "wall_time_s" => json_get(heur, "wall_time_s", nothing),
        "optimal_edges" => json_get(heur, "optimal_edges", nothing),
        "matched_optimal" => json_get(heur, "matched_optimal", nothing),
        "theta_feasible" => json_get(heur, "theta_feasible", nothing),
        "nU" => json_get(heur, "nU", nothing),
        "nV" => json_get(heur, "nV", nothing),
    )
end

function graph_summary(data)
    g = json_get(data, "graph", nothing)
    if g !== nothing
        return Dict{String,Any}(
            "nU" => json_get(g, "nU", nothing),
            "nV" => json_get(g, "nV", nothing),
            "mutable_bytes" => json_get(g, "mutable_bytes", nothing),
            "frozen_bytes" => json_get(g, "frozen_bytes", nothing),
        )
    end
    return Dict{String,Any}(
        "edge_count" => json_get(data, "edge_count", nothing),
    )
end

"""
Compact compare-seeds JSON when pivot timing is skipped (ACO never beat θ,
or best beating trial lacks U/V). Enough for: python -m emit seed-compare
without re-reading the vary file.
"""
function build_skip_payload(data, vary_path::AbstractString; reason::AbstractString)
    trials = json_get(data, "trials", [])
    heur = json_get(data, "heuristic", nothing)
    heur_edges = heur === nothing ? nothing : json_get(heur, "final_edges", nothing)
    best_any = select_best_trial_any(trials)
    discovery_s, mean_wct, n_timed = aco_trial_time_stats(trials)

    payload = Dict{String,Any}(
        "compare" => "seeds",
        "beat_heuristic" => false,
        "skip_reason" => String(reason),
        "dataset" => String(json_get(data, "dataset", "")),
        "k" => Int(json_get(data, "k", 2)),
        "theta" => Int(json_get(data, "theta", 5)),
        "reduction" => json_get(data, "reduction", nothing),
        "source_vary" => vary_path,
        "graph" => graph_summary(data),
        "heuristic" => heuristic_summary(heur),
        "heuristic_edges" => heur_edges,
        "n_trials" => length(trials),
        "n_timed_trials" => n_timed,
        "aco_discovery_s" => n_timed > 0 ? discovery_s : nothing,
        "aco_mean_wct_s" => mean_wct,
        # No pivot speedup: entire ACO search is overhead.
        "inclusive_reduction_s" => n_timed > 0 ? -discovery_s : nothing,
        "time_reduction_s" => nothing,
        "time_reduction_pct" => 0.0,
        "best_trial" => trial_summary_dict(best_any),
    )
    return payload
end

function save_compare_json_light(path::AbstractString, payload)
    dir = dirname(path)
    if !isempty(dir)
        mkpath(dir)
    end
    open(path, "w") do io
        JSON3.write(io, payload)
        println(io)
    end
    println("Wrote compare results → $path")
    return path
end

function maybe_skip_existing(save_path)
    save_path === nothing && return false
    get(ENV, "SKIP_EXISTING", "1") == "1" || return false
    isfile(save_path) || return false
    println("Skipping (exists): $save_path")
    return true
end

# -----------------------------------------------------------------------------
# Heavy path (pivot timing) — load.jl only pulled in when needed.
# -----------------------------------------------------------------------------

const _LOAD_JL_INCLUDED = Ref(false)

function ensure_load_jl!()
    _LOAD_JL_INCLUDED[] && return
    include(joinpath(@__DIR__, "load.jl"))
    _LOAD_JL_INCLUDED[] = true
    return
end

function subgraph_from_uv(U, V)
    return SubGraph(Set{Int}(json_int_vec(U)), Set{Int}(json_int_vec(V)))
end

function parse_reduction_flag()
    for arg in ARGS
        startswith(arg, "--reduce=") || continue
        value = split(arg, "=", limit=2)[2]
        value == "lo" && return ReductionMode.simple
        value == "hi" && return ReductionMode.all_reductions
        value == "none" && return ReductionMode.none
        throw(ArgumentError("Unsupported --reduce=$value (expected lo|hi|none)"))
    end
    return ReductionMode.all_reductions
end

# Untyped signatures so this file parses before load.jl defines BipartiteGraph / SubGraph.
function time_pivot_seeded!(g, k::Int, θ::Int, reduction;
    use_heuristic::Bool, initial_seed)
    g_run = deepcopy(g)
    m = measure_call() do
        find_kmdb!(g_run, use_heuristic, BranchMode.pivot, k, θ, reduction;
            initial_seed=initial_seed)
    end
    sols = m.value
    sol = isempty(sols) ? SubGraph() : first(sols)
    g_eval = deepcopy(g)
    fg_eval = if reduction == ReductionMode.none
        freeze(g_eval)
    else
        apply_graph_reductions!(g_eval, k, θ, nothing, nothing, true, reduction)
    end
    edges = Subgraph.edge_count(fg_eval, sol)
    return (; sol, edges, time = m.time, allocated = m.allocated, rss_delta = m.rss_delta)
end

function run_seed_comparison!(g, k::Int, θ::Int, reduction, aco_seed;
    dataset::AbstractString="", aco_meta=nothing)
    println()
    println("==================== COMPARE SEEDS ====================")
    println("dataset=$dataset  k=$k  θ=$θ  reduction=$reduction")
    println("ACO seed: |U|=$(length(aco_seed.U)) |V|=$(length(aco_seed.V))  " *
            "U=$(sorted_str(aco_seed.U)) V=$(sorted_str(aco_seed.V))")

    # Tiny synthetic graph only — JIT is type-based, so a full-instance warmup
    # would be two extra complete pivot solves (often as expensive as the timed runs).
    println()
    println("Warming up (tiny graph, excluded from timings)…")
    aco_nt = (ACO_PHEREMONE, ACO_NUM_ANTS, ACO_NUM_ITERATIONS, ACO_EVAPORATION, ACO_NUM_SUBSPECIES)
    warmup_benchmarks!(k, θ, reduction, aco_nt; targets=Set([:pivot, :heuristic]))

    println()
    println("── Pivot seeded with θ-heuristic only ──")
    theta_run = time_pivot_seeded!(g, k, θ, reduction;
        use_heuristic=true, initial_seed=nothing)
    print_metric_block("Pivot (θ seed)";
        wall_time_s = theta_run.time,
        allocated_bytes = theta_run.allocated,
        optimal_edges = theta_run.edges,
    )

    println()
    println("── Pivot seeded with θ-heuristic + ACO subgraph ──")
    aco_run = time_pivot_seeded!(g, k, θ, reduction;
        use_heuristic=true, initial_seed=aco_seed)
    print_metric_block("Pivot (θ + ACO seed)";
        wall_time_s = aco_run.time,
        allocated_bytes = aco_run.allocated,
        optimal_edges = aco_run.edges,
    )

    time_reduction_s = theta_run.time - aco_run.time
    time_reduction_pct = theta_run.time > 0 ?
        100.0 * time_reduction_s / theta_run.time : 0.0

    println()
    println("==================== SUMMARY ======================")
    println("θ-seed pivot:     $(format_seconds(theta_run.time))s  edges=$(theta_run.edges)")
    println("θ+ACO-seed pivot: $(format_seconds(aco_run.time))s  edges=$(aco_run.edges)")
    println("time reduction:   $(format_seconds(time_reduction_s))s  ($(round(time_reduction_pct; digits=2))%)")
    println("===================================================")

    payload = Dict{String,Any}(
        "compare" => "seeds",
        "beat_heuristic" => true,
        "dataset" => String(dataset),
        "k" => k,
        "theta" => θ,
        "reduction" => string(reduction),
        "graph" => Dict(
            "nU" => length(g.adjU),
            "nV" => length(g.adjV),
        ),
        "aco_seed" => Dict(
            "U" => sort!(collect(aco_seed.U)),
            "V" => sort!(collect(aco_seed.V)),
            "nU" => length(aco_seed.U),
            "nV" => length(aco_seed.V),
        ),
        "pivot_theta" => Dict(
            "wall_time_s" => theta_run.time,
            "allocated_bytes" => theta_run.allocated,
            "rss_delta_bytes" => theta_run.rss_delta,
            "final_edges" => theta_run.edges,
        ),
        "pivot_aco_seed" => Dict(
            "wall_time_s" => aco_run.time,
            "allocated_bytes" => aco_run.allocated,
            "rss_delta_bytes" => aco_run.rss_delta,
            "final_edges" => aco_run.edges,
        ),
        "time_reduction_s" => time_reduction_s,
        "time_reduction_pct" => time_reduction_pct,
    )

    if aco_meta !== nothing
        payload["aco_trial"] = aco_meta
    end

    return payload
end

function load_graph_for_compare(dataset::AbstractString, inject, k::Int, seed)
    graph_path = resolve_graph_path(dataset)
    if !isfile(graph_path)
        error("Could not find graph for dataset '$dataset' at '$graph_path'")
    end
    if inject.enabled || seed !== nothing
        s = seed === nothing ? UInt64(time_ns()) : UInt64(seed)
        Random.seed!(s)
    end
    rng = Random.default_rng()
    println("Loading graph from: $graph_path")
    g, edges = load_graph_maybe_inject(graph_path, inject, k, rng)
    println("nU=$(length(g.adjU)), nV=$(length(g.adjV)), |E|=$edges")
    return g, edges
end

function compare_from_vary_json(vary_path::AbstractString; inject_raw, seed_override, save_path)
    if maybe_skip_existing(save_path)
        return Dict{String,Any}("compare" => "seeds", "skipped_existing" => true)
    end

    data = JSON3.read(read(vary_path, String))
    dataset = String(json_get(data, "dataset", ""))
    isempty(dataset) && error("vary JSON missing dataset: $vary_path")
    k = Int(json_get(data, "k", 2))
    θ = Int(json_get(data, "theta", 5))

    heur = json_get(data, "heuristic", nothing)
    heur_edges = heur === nothing ? nothing : json_get(heur, "final_edges", nothing)

    trials = json_get(data, "trials", [])
    best = select_best_beating_trial(trials)

    if best === nothing
        println("No ACO trial beat the θ-heuristic" *
                (heur_edges === nothing ? "" : " (heur_edges=$heur_edges)") *
                ". Writing skip marker")
        payload = build_skip_payload(data, vary_path; reason="no_aco_beat")
        if save_path !== nothing
            save_compare_json_light(save_path, payload)
        end
        return payload
    end

    if !trial_has_subgraph(best)
        println("Best beating trial has no U/V arrays. Writing skip marker " *
                "(re-run vary.jl to log subgraphs)")
        payload = build_skip_payload(data, vary_path; reason="missing_uv")
        if save_path !== nothing
            save_compare_json_light(save_path, payload)
        end
        return payload
    end

    # Real pivot comparison — pay for load.jl only now.
    # invokelatest: load.jl defines parse_inject / SubGraph / etc. in a newer
    # world age than this function (Julia 1.12+ binding world-age rules).
    ensure_load_jl!()
    return Base.invokelatest(() -> _compare_from_vary_json_loaded(
        vary_path, data, best, heur_edges, trials, dataset, k, θ;
        inject_raw, seed_override, save_path))
end

function _compare_from_vary_json_loaded(vary_path, data, best, heur_edges, trials,
    dataset, k, θ; inject_raw, seed_override, save_path)
    inject = inject_raw === nothing ? parse_inject() : inject_raw
    reduction = parse_reduction_flag()
    seed = seed_override !== nothing ? seed_override :
        begin
            raw = json_get(data, "base_seed", json_get(data, "seed", nothing))
            raw === nothing ? nothing : parse(UInt64, string(raw))
        end

    aco_seed = subgraph_from_uv(json_get(best, "U"), json_get(best, "V"))
    _discovery_all, mean_wct, _n_timed = aco_trial_time_stats(trials)
    # Discovery until the winning replicate (same ants), not the full sweep.
    win_ants = json_get(best, "ants", nothing)
    win_run = json_get(best, "run", nothing)
    discovery_until = nothing
    if win_ants !== nothing && win_run !== nothing
        total = 0.0
        saw = false
        for t in trials
            json_get(t, "ants", nothing) == win_ants || continue
            r = json_get(t, "run", nothing)
            r === nothing && continue
            r = Int(r)
            wr = Int(win_run)
            if r < wr
                wt = json_get(t, "wall_time_s", nothing)
                wt !== nothing && (total += Float64(wt))
            elseif r == wr
                saw = true
                ttb = json_get(t, "time_to_best_s", nothing)
                if ttb !== nothing
                    total += Float64(ttb)
                else
                    wt = json_get(t, "wall_time_s", nothing)
                    wt !== nothing && (total += Float64(wt))
                end
            end
        end
        discovery_until = saw ? total : nothing
    end

    aco_meta = Dict{String,Any}(
        "run" => json_get(best, "run", nothing),
        "seed" => json_get(best, "seed", nothing),
        "ants" => json_get(best, "ants", nothing),
        "final_edges" => json_get(best, "final_edges", nothing),
        "time_to_best_s" => json_get(best, "time_to_best_s", nothing),
        "wall_time_s" => json_get(best, "wall_time_s", nothing),
        "heuristic_edges" => heur_edges,
        "source_vary" => vary_path,
        "aco_discovery_s" => discovery_until,
        "aco_mean_wct_s" => mean_wct,
    )

    println("Selected ACO trial: ants=$(aco_meta["ants"]) run=$(aco_meta["run"]) " *
            "edges=$(aco_meta["final_edges"]) time_to_best=$(aco_meta["time_to_best_s"])s " *
            "(θ-heur edges=$heur_edges)")

    g, _edges = load_graph_for_compare(dataset, inject, k, seed)
    payload = run_seed_comparison!(g, k, θ, reduction, aco_seed;
        dataset=dataset, aco_meta=aco_meta)

    if discovery_until !== nothing
        payload["aco_discovery_s"] = discovery_until
        theta_t = payload["pivot_theta"]["wall_time_s"]
        aco_t = payload["pivot_aco_seed"]["wall_time_s"]
        inclusive = Float64(theta_t) - Float64(discovery_until) - Float64(aco_t)
        payload["inclusive_reduction_s"] = inclusive
        if Float64(theta_t) > 0
            payload["inclusive_reduction_pct"] = 100.0 * inclusive / Float64(theta_t)
        end
    end
    if mean_wct !== nothing
        payload["aco_mean_wct_s"] = mean_wct
    end

    if save_path !== nothing
        save_benchmark_json(resolve_benchmark_save_path(save_path), payload)
    end
    return payload
end

function compare_from_seed_json(; dataset, seed_json_path, k, θ, seed, save_path)
    if maybe_skip_existing(save_path)
        return Dict{String,Any}("compare" => "seeds", "skipped_existing" => true)
    end

    ensure_load_jl!()
    return Base.invokelatest(() -> _compare_from_seed_json_loaded(;
        dataset, seed_json_path, k, θ, seed, save_path))
end

function _compare_from_seed_json_loaded(; dataset, seed_json_path, k, θ, seed, save_path)
    inject = parse_inject()
    reduction = parse_reduction_flag()

    seed_data = JSON3.read(read(seed_json_path, String))
    # Accept {"U":...,"V":...} or {"aco_seed":{"U":...,"V":...}}.
    U = json_get(seed_data, "U", nothing)
    V = json_get(seed_data, "V", nothing)
    if U === nothing || V === nothing
        nested = json_get(seed_data, "aco_seed", nothing)
        if nested !== nothing
            U = json_get(nested, "U", nothing)
            V = json_get(nested, "V", nothing)
        end
    end
    (U === nothing || V === nothing) && error("seed JSON must contain U and V arrays: $seed_json_path")

    aco_seed = subgraph_from_uv(U, V)
    g, _edges = load_graph_for_compare(dataset, inject, k, seed)
    payload = run_seed_comparison!(g, k, θ, reduction, aco_seed; dataset=dataset)

    if save_path !== nothing
        save_benchmark_json(resolve_benchmark_save_path(save_path), payload)
    end
    return payload
end

function compare_seeds_main()
    positional = String[]
    for arg in ARGS
        startswith(arg, "-") || push!(positional, arg)
    end

    seed_json = parse_string_eq("seed-json", nothing)
    dataset = parse_string_eq("dataset", nothing)
    save_path = parse_string_eq("save", nothing)
    seed_raw = parse_string_eq("seed", nothing)
    seed_override = seed_raw === nothing ? nothing : parse(UInt64, seed_raw)
    k = parse_int_eq("k", 2)
    θ = parse_int_eq("theta", 5)

    result = nothing
    if seed_json !== nothing
        dataset === nothing && error("--seed-json= requires --dataset=")
        result = compare_from_seed_json(;
            dataset, seed_json_path=seed_json, k, θ,
            seed=seed_override, save_path)
    elseif length(positional) == 1
        # inject / reduction parsed only after ensure_load_jl! when needed
        result = compare_from_vary_json(positional[1];
            inject_raw=nothing, seed_override, save_path)
    else
        usage_and_exit()
    end

    if result === nothing
        # Should not happen for vary mode anymore (markers are returned).
        exit(2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_seeds_main()
end
