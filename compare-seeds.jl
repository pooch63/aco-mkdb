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
   the replicate with the smallest time_to_best_s. Skips the file if ACO never
   beat the heuristic, or if trials lack logged U/V (re-run vary.jl).

2. Explicit seed subgraph JSON `{"U":[...],"V":[...]}`:

     julia -t 8 compare-seeds.jl --dataset=amazon/boxes --k=2 --theta=5 \
         --seed-json=seed.json --inject --u=5 --v=5 --seed=1 \
         --save=compare_k2t5i/boxes.json

Both modes reduce the graph once (matching vary.jl), then time pivot with
ReductionMode.none: (a) θ-heuristic seed only, (b) best of θ-heuristic and the
external seed.
=================================================================================
=#

using Random
using JSON3

# Pull load helpers, find_kmdb!, measure_call, inject, etc. (main is PROGRAM_FILE-guarded).
include(joinpath(@__DIR__, "load.jl"))

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

function subgraph_from_uv(U, V)
    return SubGraph(Set{Int}(json_int_vec(U)), Set{Int}(json_int_vec(V)))
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

function trial_has_subgraph(t)
    U = json_get(t, "U", nothing)
    V = json_get(t, "V", nothing)
    return U !== nothing && V !== nothing
end

function time_pivot_seeded!(g::BipartiteGraph, k::Int, θ::Int, reduction::ReductionMode.T;
    use_heuristic::Bool, initial_seed::Union{Nothing,SubGraph})
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

function run_seed_comparison!(g::BipartiteGraph, k::Int, θ::Int, reduction::ReductionMode.T,
    aco_seed::SubGraph; dataset::AbstractString="", aco_meta=nothing)
    println()
    println("==================== COMPARE SEEDS ====================")
    println("dataset=$dataset  k=$k  θ=$θ")
    println("ACO seed: |U|=$(length(aco_seed.U)) |V|=$(length(aco_seed.V))  " *
            "U=$(sorted_str(aco_seed.U)) V=$(sorted_str(aco_seed.V))")

    println()
    println("Warming up (excluded from timings)…")
    aco_nt = (ACO_PHEREMONE, ACO_NUM_ANTS, ACO_NUM_ITERATIONS, ACO_EVAPORATION, ACO_NUM_SUBSPECIES)
    warmup_benchmarks!(k, θ, reduction, aco_nt; targets=Set([:pivot, :heuristic]))

    g_reduced = deepcopy(g)
    println()
    println("Reducing graph once…")
    m_red = measure_call() do
        apply_graph_reductions!(g_reduced, k, θ, nothing, nothing, true, reduction)
    end
    fg = m_red.value
    print_metric_block("Graph reduction";
        wall_time_s = m_red.time,
        reduced_nU = length(fg.u_ids),
        reduced_nV = length(fg.v_ids),
    )

    solver_reduction = ReductionMode.none

    # Discarded warm-up on the *real* reduced graph so both timed runs exclude JIT.
    println()
    println("Warming up pivot on reduced graph (excluded)…")
    time_pivot_seeded!(g_reduced, k, θ, solver_reduction;
        use_heuristic=true, initial_seed=nothing)
    time_pivot_seeded!(g_reduced, k, θ, solver_reduction;
        use_heuristic=true, initial_seed=aco_seed)

    println()
    println("── Pivot seeded with θ-heuristic only ──")
    theta_run = time_pivot_seeded!(g_reduced, k, θ, solver_reduction;
        use_heuristic=true, initial_seed=nothing)
    print_metric_block("Pivot (θ seed)";
        wall_time_s = theta_run.time,
        allocated_bytes = theta_run.allocated,
        optimal_edges = theta_run.edges,
    )

    println()
    println("── Pivot seeded with θ-heuristic + ACO subgraph ──")
    aco_run = time_pivot_seeded!(g_reduced, k, θ, solver_reduction;
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
        "dataset" => String(dataset),
        "k" => k,
        "theta" => θ,
        "reduction" => string(reduction),
        "graph" => Dict(
            "reduced_nU" => length(fg.u_ids),
            "reduced_nV" => length(fg.v_ids),
            "reduction_time_s" => m_red.time,
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

function compare_from_vary_json(vary_path::AbstractString; inject, seed_override, reduction, save_path)
    data = JSON3.read(read(vary_path, String))
    dataset = String(json_get(data, "dataset", ""))
    isempty(dataset) && error("vary JSON missing dataset: $vary_path")
    k = Int(json_get(data, "k", 2))
    θ = Int(json_get(data, "theta", 5))
    seed = seed_override !== nothing ? seed_override :
        begin
            raw = json_get(data, "base_seed", json_get(data, "seed", nothing))
            raw === nothing ? nothing : parse(UInt64, string(raw))
        end

    heur = json_get(data, "heuristic", nothing)
    heur_edges = heur === nothing ? nothing : json_get(heur, "final_edges", nothing)

    trials = json_get(data, "trials", [])
    best = select_best_beating_trial(trials)
    if best === nothing
        println("Skipping $vary_path: no ACO trial beat the θ-heuristic" *
                (heur_edges === nothing ? "" : " (heur_edges=$heur_edges)"))
        return nothing
    end
    if !trial_has_subgraph(best)
        println("Skipping $vary_path: best beating trial has no U/V arrays — re-run vary.jl to log subgraphs")
        return nothing
    end

    aco_seed = subgraph_from_uv(json_get(best, "U"), json_get(best, "V"))
    aco_meta = Dict{String,Any}(
        "run" => json_get(best, "run", nothing),
        "seed" => json_get(best, "seed", nothing),
        "ants" => json_get(best, "ants", nothing),
        "final_edges" => json_get(best, "final_edges", nothing),
        "time_to_best_s" => json_get(best, "time_to_best_s", nothing),
        "wall_time_s" => json_get(best, "wall_time_s", nothing),
        "heuristic_edges" => heur_edges,
        "source_vary" => vary_path,
    )

    println("Selected ACO trial: ants=$(aco_meta["ants"]) run=$(aco_meta["run"]) " *
            "edges=$(aco_meta["final_edges"]) time_to_best=$(aco_meta["time_to_best_s"])s " *
            "(θ-heur edges=$heur_edges)")

    g, _edges = load_graph_for_compare(dataset, inject, k, seed)
    payload = run_seed_comparison!(g, k, θ, reduction, aco_seed;
        dataset=dataset, aco_meta=aco_meta)

    if save_path !== nothing
        save_benchmark_json(resolve_benchmark_save_path(save_path), payload)
    end
    return payload
end

function compare_from_seed_json(; dataset, seed_json_path, k, θ, inject, seed, reduction, save_path)
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
    reduction = parse_reduction_flag()
    inject = parse_inject()

    result = nothing
    if seed_json !== nothing
        dataset === nothing && error("--seed-json= requires --dataset=")
        result = compare_from_seed_json(;
            dataset, seed_json_path=seed_json, k, θ, inject,
            seed=seed_override, reduction, save_path)
    elseif length(positional) == 1
        result = compare_from_vary_json(positional[1];
            inject, seed_override, reduction, save_path)
    else
        usage_and_exit()
    end

    # Exit 2 = skipped (no beating ACO trial / missing U/V) so bash can count skips.
    if result === nothing
        exit(2)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    compare_seeds_main()
end
