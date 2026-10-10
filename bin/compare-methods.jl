#=
=================================================================================
Head-to-head method comparison on one graph or a suite ordered by |E|.

Runs two registered SolveMethods (same or different; arbitrary kwargs) on each
reduced graph and reports whether the challenger beats the baseline under paper
semantics (θ-feasible + more edges).

JSON config (recommended for A/B / ablations):
  julia bin/compare-methods.jl --config=configs/aco_pheromone_ablation.json

Single graph (CLI overrides for ants / gens / kmax / beta / p / max-steps /
n-seeds / local-reduction / prune-by-bound / one-colony / greedy-fill /
eta-exp / best-only-pheromone still work):
  julia bin/compare-methods.jl amazon/boxes --baseline=heuristic --challenger=aco
  julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=aco \\
      --baseline-ants=10 --challenger-ants=50 --save=aco_ants_ab.json
  julia bin/compare-methods.jl --prefix=konect-small --baseline=aco --challenger=aco \\
      --challenger-greedy-fill=end --save=results/aco_fill_end_vs_aco.json
  julia bin/compare-methods.jl --baseline=aco --challenger=aco \\
      --challenger-eta-exp=1 --challenger-best-only-pheromone=true \\
      --save=results/aco_best_only_vs_aco.json
  julia bin/compare-methods.jl --baseline=aco --challenger=aco \\
      --challenger-pheromone=0 --challenger-evaporation=1 \\
      --save=results/aco_no_pheromone_vs_aco.json
  julia bin/compare-methods.jl --baseline=aco --challenger=ga \\
      --save=results/ga_vs_aco.json
  julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=split_aco \\
      --challenger-n-seeds=50 --challenger-local-reduction=false \\
      --challenger-prune-by-bound=false --challenger-one-colony=false \\
      --save=split_vs_aco.json
  julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=one_side_aco \\
      --save=one_side_vs_aco.json
  julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=oracle_aco \\
      --challenger-discrepancy=1 --challenger-decompose=false --save=oracle_vs_aco.json
  julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=retry \\
      --challenger-beta=0.04 --challenger-p=1 --challenger-max-steps=10000 --save=retry_ab.json

Multi-graph without a full config (uses CLI method names + optional --prefix=
like order_graphs.jl; checkpoints to --save=). Omit dataset and --prefix= to
run every indexed graph under data/, ascending by |E|:
  julia bin/compare-methods.jl --prefix=konect-small --baseline=aco --challenger=vns \\
      --save=vns_ab.json
  julia bin/compare-methods.jl --baseline=aco --challenger=vns --save=vns_vs_aco.json

Config schema (all but baseline/challenger optional when using CLI sides):
  {
    "name": "aco_pheromone_ablation",
    "output": "results/aco_pheromone_ab.json",
    "k": 2, "theta": 5, "seed": 1, "reduction": "lo",
    "prefix": "konect-small",          // OR "datasets": ["amazon/boxes", …]
                                       // omit both → all indexed graphs
    "skip_completed": true,            // resume: skip datasets already in output
    "inject": true,                    // or { "enabled": true, "u": 5, "v": 5 }
    "baseline": {
      "method": "aco",
      "label": "aco+pheromone",
      "params": { "ants": 50, "pheromone": 1, "evaporation": 0.95, … }
    },
    "challenger": {
      "method": "aco",
      "label": "aco-no-pheromone",
      "params": { "ants": 50, "pheromone": 0, "evaporation": 1.0, … }
    }
  }

After every graph the output JSON is rewritten with all completed results so a
crash mid-suite keeps prior checkpoints.
=================================================================================
=#

const ROOT = dirname(@__DIR__)
const SRC = joinpath(ROOT, "src")
const BIN = @__DIR__

isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))
isdefined(@__MODULE__, :__BENCHMARK_JL__) || include(joinpath(BIN, "benchmark.jl"))
# Inject helpers (`parse_inject`, `load_graph_maybe_inject`) live in load.jl.
isdefined(@__MODULE__, :__LOAD_JL__) || include(joinpath(BIN, "load.jl"))

using Random
using JSON3

function usage_and_exit(code::Int=1)
    println(stderr, """
Usage:
  julia bin/compare-methods.jl --config=PATH.json
  julia bin/compare-methods.jl <dataset> --baseline=NAME --challenger=NAME [options]
  julia bin/compare-methods.jl --prefix=PREFIX --baseline=NAME --challenger=NAME --save=OUT.json
  julia bin/compare-methods.jl --baseline=NAME --challenger=NAME --save=OUT.json
      (no dataset / --prefix= → all indexed graphs under data/)
  julia bin/compare-methods.jl --list

Injection is on by default (θ×θ, k missing); pass --no-inject to skip.
Multi-graph runs require --save= for checkpoints.
Registered methods: $(join(list_methods(), ", "))
""")
    exit(code)
end

# ── CLI flag helpers ──────────────────────────────────────────────────────────

function parse_named_method(flag::AbstractString, default::Union{Nothing,String}=nothing)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return lowercase(strip(split(arg, "=", limit=2)[2]))
    end
    return default
end

function parse_int_override(flag::AbstractString)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return parse(Int, split(arg, "=", limit=2)[2])
    end
    return nothing
end

function parse_float_override(flag::AbstractString)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return parse(Float64, split(arg, "=", limit=2)[2])
    end
    return nothing
end

"""`--flag=true|false`. `nothing` when the flag is omitted."""
function parse_bool_override(flag::AbstractString)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        val = lowercase(strip(split(arg, "=", limit=2)[2]))
        val in ("true", "1", "yes", "on") && return true
        val in ("false", "0", "no", "off") && return false
        throw(ArgumentError("Bad boolean for --$flag: $val (expected true/false)"))
    end
    return nothing
end

function parse_string_flag(flag::AbstractString, default::Union{Nothing,String}=nothing)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return strip(split(arg, "=", limit=2)[2])
    end
    return default
end

function parse_reduction_flag(default::ReductionMode.T=ReductionMode.simple)
    for arg in ARGS
        if startswith(arg, "--reduce=")
            value = split(arg, "=", limit=2)[2]
            value == "lo" && return ReductionMode.simple
            value == "hi" && return ReductionMode.all_reductions
            value == "none" && return ReductionMode.none
            throw(ArgumentError("Unsupported --reduce=$value (lo|hi|none)"))
        end
    end
    return default
end

function parse_reduction_name(value::AbstractString)
    value == "lo" && return ReductionMode.simple
    value == "hi" && return ReductionMode.all_reductions
    value == "none" && return ReductionMode.none
    value == "simple" && return ReductionMode.simple
    value == "all_reductions" && return ReductionMode.all_reductions
    throw(ArgumentError("Unsupported reduction '$value' (lo|hi|none)"))
end

reduction_name(r::ReductionMode.T) =
    r == ReductionMode.simple ? "lo" :
    r == ReductionMode.all_reductions ? "hi" : "none"

function parse_k_theta_defaults()
    k, θ = 2, 5
    for arg in ARGS
        if startswith(arg, "--k=")
            k = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--theta=")
            θ = parse(Int, split(arg, "=", limit=2)[2])
        end
    end
    return k, θ
end

function parse_seed_flag()
    for arg in ARGS
        startswith(arg, "--seed=") || continue
        return parse(UInt64, split(arg, "=", limit=2)[2])
    end
    return UInt64(time_ns())
end

function resolve_dataset_arg()
    for arg in ARGS
        startswith(arg, "-") && continue
        return arg
    end
    return nothing
end

# ── JSON helpers ──────────────────────────────────────────────────────────────

function json_to_dict(obj)
    if obj isa AbstractDict
        return Dict{String,Any}(string(k) => json_to_dict(v) for (k, v) in obj)
    elseif obj isa AbstractVector
        return Any[json_to_dict(v) for v in obj]
    elseif obj isa JSON3.Object
        return Dict{String,Any}(string(k) => json_to_dict(obj[k]) for k in keys(obj))
    elseif obj isa JSON3.Array
        return Any[json_to_dict(v) for v in obj]
    else
        return obj
    end
end

function resolve_config_path(raw::AbstractString)
    path = abspath(expanduser(raw))
    isfile(path) && return path
    alt = joinpath(ROOT, raw)
    isfile(alt) && return abspath(alt)
    error("Config not found: $raw")
end

"""
Serialize a method side-spec for the results JSON (method + params + label).
"""
function side_spec_to_dict(spec::Dict{String,Any})
    method = String(spec["method"])
    label = String(get(spec, "label", method))
    params = if haskey(spec, "params") && spec["params"] isa AbstractDict
        Dict{String,Any}(string(k) => v for (k, v) in spec["params"])
    else
        Dict{String,Any}(string(k) => v for (k, v) in spec
            if !(string(k) in ("method", "label", "params")))
    end
    out = Dict{String,Any}("method" => method, "label" => label, "params" => params)
    if haskey(spec, "id")
        out["id"] = String(spec["id"])
    end
    return out
end

function build_side_from_spec(spec::Dict{String,Any})
    return make_method_from_dict(spec)
end

"""
Build a method from CLI name + optional ants/gens/kmax overrides (legacy path).
"""
function build_side_cli(name::AbstractString, side::AbstractString; kwargs...)
    ants = parse_int_override("$side-ants")
    gens = parse_int_override("$side-gens")
    kmax = parse_int_override("$side-kmax")
    beta = parse_float_override("$side-beta")
    retry_p = parse_float_override("$side-p")
    max_steps = parse_int_override("$side-max-steps")
    n_seeds = parse_int_override("$side-n-seeds")
    local_reduction = parse_bool_override("$side-local-reduction")
    prune_by_bound = parse_bool_override("$side-prune-by-bound")
    one_colony = parse_bool_override("$side-one-colony")
    greedy_fill = parse_string_flag("$side-greedy-fill")
    eta_exp = parse_float_override("$side-eta-exp")
    best_only_pheromone = parse_bool_override("$side-best-only-pheromone")
    pheromone = parse_int_override("$side-pheromone")
    evaporation = parse_float_override("$side-evaporation")
    discrepancy = parse_int_override("$side-discrepancy")
    decompose = parse_bool_override("$side-decompose")
    branch = parse_int_override("$side-branch")
    seed_size = parse_int_override("$side-seed-size")
    pool = parse_int_override("$side-pool")
    max_nodes = parse_int_override("$side-max-nodes")
    params = Dict{String,Any}()
    for (k, v) in kwargs
        params[string(k)] = v
    end
    is_one_side = name in ("one_side_aco", "one-side-aco")
    is_oracle = name in ("oracle_aco", "oracle-aco")
    if name == "aco" || is_one_side || is_oracle
        params["ants"] = ants === nothing ? get(params, "ants", 100) : ants
        get!(params, "iterations", 5)
        get!(params, "parallelize", false)
        eta_exp !== nothing && (params["eta_exp"] = eta_exp)
        pheromone !== nothing && (params["pheromone"] = pheromone)
        evaporation !== nothing && (params["evaporation"] = evaporation)
        if name == "aco"
            get!(params, "prefer_smaller_side", true)
            get!(params, "neighbor_scope_limit", true)
            greedy_fill !== nothing && (params["greedy_fill"] = greedy_fill)
            best_only_pheromone !== nothing && (params["best_only_pheromone"] = best_only_pheromone)
        elseif is_oracle
            discrepancy !== nothing && (params["discrepancy"] = discrepancy)
            decompose !== nothing && (params["decompose"] = decompose)
            branch !== nothing && (params["branch"] = branch)
            seed_size !== nothing && (params["seed_size"] = seed_size)
            pool !== nothing && (params["pool"] = pool)
            max_nodes !== nothing && (params["max_nodes"] = max_nodes)
            n_seeds !== nothing && (params["n_seeds"] = n_seeds)
        end
    elseif name in ("split_aco", "split-aco")
        params["ants"] = ants === nothing ? get(params, "ants", 10) : ants
        get!(params, "iterations", 5)
        # Top-k S_C seeds. Omitted flag keeps SplitACOMethod's default (50).
        n_seeds !== nothing && (params["n_seeds"] = n_seeds)
        # 3-hop common-neighbor reduction before each local ACO. Default on.
        local_reduction !== nothing && (params["local_reduction"] = local_reduction)
        # Upper-bound ant halt + pheromone rollback in each local ACO. Default on.
        prune_by_bound !== nothing && (params["prune_by_bound"] = prune_by_bound)
        # One shared colony with seeds round-robined across ants. Default on.
        one_colony !== nothing && (params["one_colony"] = one_colony)
    elseif name == "ga"
        params["generations"] = gens === nothing ? get(params, "generations", 500) : gens
        get!(params, "N", 10)
    elseif name == "vns"
        params["kmax"] = kmax === nothing ? get(params, "kmax", 10) : kmax
    elseif name == "retry"
        params["beta"] = beta === nothing ? get(params, "beta", 0.04) : beta
        params["p"] = retry_p === nothing ? get(params, "p", 1.0) : retry_p
        params["max_steps"] = max_steps === nothing ? get(params, "max_steps", 10_000) : max_steps
    end
    if n_seeds !== nothing && !(name in ("split_aco", "split-aco", "oracle_aco", "oracle-aco"))
        throw(ArgumentError("--$side-n-seeds applies only to split_aco or oracle_aco"))
    end
    if local_reduction !== nothing && !(name in ("split_aco", "split-aco"))
        throw(ArgumentError("--$side-local-reduction applies only to split_aco"))
    end
    if prune_by_bound !== nothing && !(name in ("split_aco", "split-aco"))
        throw(ArgumentError("--$side-prune-by-bound applies only to split_aco"))
    end
    if one_colony !== nothing && !(name in ("split_aco", "split-aco"))
        throw(ArgumentError("--$side-one-colony applies only to split_aco"))
    end
    if greedy_fill !== nothing && name != "aco"
        throw(ArgumentError("--$side-greedy-fill applies only to aco"))
    end
    if eta_exp !== nothing && !(name == "aco" || is_one_side)
        throw(ArgumentError("--$side-eta-exp applies only to aco or one_side_aco"))
    end
    if best_only_pheromone !== nothing && name != "aco"
        throw(ArgumentError("--$side-best-only-pheromone applies only to aco"))
    end
    if pheromone !== nothing && !(name == "aco" || is_one_side || is_oracle)
        throw(ArgumentError("--$side-pheromone applies only to aco, one_side_aco, or oracle_aco"))
    end
    if evaporation !== nothing && !(name == "aco" || is_one_side || is_oracle)
        throw(ArgumentError("--$side-evaporation applies only to aco, one_side_aco, or oracle_aco"))
    end
    if discrepancy !== nothing && !is_oracle
        throw(ArgumentError("--$side-discrepancy applies only to oracle_aco"))
    end
    if decompose !== nothing && !is_oracle
        throw(ArgumentError("--$side-decompose applies only to oracle_aco"))
    end
    if branch !== nothing && !is_oracle
        throw(ArgumentError("--$side-branch applies only to oracle_aco"))
    end
    if seed_size !== nothing && !is_oracle
        throw(ArgumentError("--$side-seed-size applies only to oracle_aco"))
    end
    if pool !== nothing && !is_oracle
        throw(ArgumentError("--$side-pool applies only to oracle_aco"))
    end
    if max_nodes !== nothing && !is_oracle
        throw(ArgumentError("--$side-max-nodes applies only to oracle_aco"))
    end
    return make_method_from_dict(name, params)
end

# ── Inject from config ────────────────────────────────────────────────────────

function inject_from_config(cfg::Dict{String,Any}, θ::Int)
    raw = get(cfg, "inject", true)
    if raw isa Bool
        raw || return (; enabled=false, nU=0, nV=0, attempts=0)
        return (; enabled=true, nU=θ, nV=θ, attempts=20)
    elseif raw isa AbstractDict
        enabled = Bool(get(raw, "enabled", true))
        enabled || return (; enabled=false, nU=0, nV=0, attempts=0)
        nU = Int(get(raw, "u", get(raw, "nU", θ)))
        nV = Int(get(raw, "v", get(raw, "nV", θ)))
        attempts = Int(get(raw, "attempts", 20))
        return (; enabled=true, nU=nU, nV=nV, attempts=attempts)
    else
        throw(ArgumentError("\"inject\" must be bool or object, got $(typeof(raw))"))
    end
end

# ── Graph selection (ascending |E|) ───────────────────────────────────────────

"""
Resolve the ordered list of `(key, edges)` to run.

Priority: explicit `datasets` in config → `--prefix=` / config `prefix` discovery
→ single CLI dataset arg → all indexed graphs under `data/`. Always sorted by
edge count ascending.
"""
function resolve_graph_queue(; datasets=nothing, prefix=nothing,
    single::Union{Nothing,AbstractString}=nothing)
    data_root = joinpath(ROOT, "data")
    indexed = Dict(e.key => e.edges for e in discover_indexed_graphs(; data_root=data_root))

    keys = String[]
    if datasets !== nothing && !isempty(datasets)
        for d in datasets
            push!(keys, String(d))
        end
    elseif prefix !== nothing && !isempty(strip(String(prefix)))
        for e in order_graphs_by_edges(; data_root=data_root, ascending=true, prefix=prefix)
            push!(keys, e.key)
        end
    elseif single !== nothing
        push!(keys, String(single))
    else
        # No dataset / prefix filter: every indexed graph, ascending by |E|.
        for e in order_graphs_by_edges(; data_root=data_root, ascending=true)
            push!(keys, e.key)
        end
    end

    entries = NamedTuple{(:key, :edges), Tuple{String,Int}}[]
    for key in keys
        edges = get(indexed, key, nothing)
        if edges === nothing
            # Fall back to counting the CSV if present but not yet indexed in walk.
            path = resolve_graph_path(key)
            if isfile(path)
                edges = count_indexed_edges(path)
            else
                @warn "Skipping missing dataset $key"
                continue
            end
        end
        push!(entries, (key=key, edges=Int(edges)))
    end
    return sort!(entries; by=(e -> e.edges))
end

# ── Per-graph comparison ──────────────────────────────────────────────────────

function compare_one_graph!(dataset::AbstractString, edge_hint::Int,
    baseline_method::SolveMethod, challenger_method::SolveMethod,
    baseline_spec::Dict{String,Any}, challenger_spec::Dict{String,Any},
    k::Int, θ::Int, reduction::ReductionMode.T, seed::UInt64, inject)
    graph_path = resolve_graph_path(dataset)
    isfile(graph_path) || error("Could not find graph for dataset '$dataset' at $graph_path")

    g, edges, plant = load_graph_maybe_inject(graph_path, inject, k, Random.default_rng())
    edges_eff = edges === nothing ? edge_hint : Int(edges)

    baseline_label = String(get(baseline_spec, "label", method_id(baseline_method)))
    challenger_label = String(get(challenger_spec, "label", method_id(challenger_method)))

    println()
    println("══════════════════════════════════════════════════")
    println("dataset=$dataset  |E|=$edges_eff  nU=$(length(g.adjU)) nV=$(length(g.adjV))")
    println("k=$k  θ=$θ  reduction=$(reduction_name(reduction))  seed=$seed")
    println("baseline=$baseline_label ($(method_id(baseline_method)))  " *
            "challenger=$challenger_label ($(method_id(challenger_method)))")
    if inject.enabled
        println("inject: u=$(inject.nU) v=$(inject.nV) k=$k")
    end

    g_reduced = deepcopy(g)
    fg = apply_graph_reductions!(g_reduced, k, θ, nothing, nothing, true, reduction)
    n_u_r = length(fg.u_ids)
    n_v_r = length(fg.v_ids)
    m_r = length(fg.v_adj)
    dens_r = (n_u_r * n_v_r > 0) ? Float64(m_r) / (n_u_r * n_v_r) : 0.0
    println("reduced: nU=$n_u_r nV=$n_v_r |E_R|=$m_r (density=$(round(dens_r; digits=4)))")
    solver_reduction = ReductionMode.none

    println()
    println("── baseline ($baseline_label) ──")
    Random.seed!(seed)
    base_stats = benchmark_method!(baseline_method, g_reduced, k, θ, solver_reduction;
        label=baseline_label, injected_biclique=plant)

    println()
    println("── challenger ($challenger_label) ──")
    Random.seed!(seed)
    chal_stats = benchmark_method!(challenger_method, g_reduced, k, θ, solver_reduction;
        label=challenger_label, injected_biclique=plant)

    verdict = compare_results(base_stats.result, chal_stats.result, fg, k, θ)
    chal_beats = beats(chal_stats.result, base_stats.result, fg, k, θ)
    edge_delta = chal_stats.final_edges - base_stats.final_edges
    pct = base_stats.final_edges == 0 ? nothing :
        100.0 * edge_delta / base_stats.final_edges

    best_sol = chal_stats.final_edges >= base_stats.final_edges ? chal_stats.sol : base_stats.sol
    heat_stats = get_diffusion_heat_stats(fg;
        iterations=20,
        injected_biclique=plant,
        solution_biclique=best_sol,
        rng=Random.MersenneTwister(seed))

    println()
    println("──────────────── VERDICT ($dataset) ────────────────")
    println("  baseline edges       : $(base_stats.final_edges)" *
            "  θ-feasible=$(base_stats.theta_feasible)")
    println("  challenger edges     : $(chal_stats.final_edges)" *
            "  θ-feasible=$(chal_stats.theta_feasible)")
    println("  compare_results      : $verdict")
    println("  challenger beats?    : $chal_beats")
    println("  edge Δ               : $edge_delta" *
            (pct === nothing ? "" : "  ($(round(pct; digits=2))%)"))
    if get(heat_stats, "biclique_mean", nothing) !== nothing
        b_mean = round(heat_stats["biclique_mean"]; digits=3)
        g_mean = round(heat_stats["total_graph_mean"]; digits=3)
        h_pct = round(heat_stats["percent_change"]; digits=2)
        println("  heat (biclique/total): $b_mean vs $g_mean ($h_pct%)")
    end
    println("══════════════════════════════════════════════════")

    base_dict = method_result_to_dict(base_stats.result, fg; k=k, θ=θ)
    chal_dict = method_result_to_dict(chal_stats.result, fg; k=k, θ=θ)
    base_dict["wall_time_s"] = base_stats.time
    chal_dict["wall_time_s"] = chal_stats.time
    base_dict["label"] = baseline_label
    chal_dict["label"] = challenger_label

    return Dict{String,Any}(
        "dataset" => String(dataset),
        "edge_count" => edges_eff,
        "nU" => length(g.adjU),
        "nV" => length(g.adjV),
        "reduced_nU" => n_u_r,
        "reduced_nV" => n_v_r,
        "reduced_edges" => m_r,
        "reduced_density" => dens_r,
        "heat" => heat_stats,
        "status" => "ok",
        "inject" => Dict{String,Any}(
            "enabled" => inject.enabled,
            "u" => inject.enabled ? inject.nU : nothing,
            "v" => inject.enabled ? inject.nV : nothing,
            "attempts" => inject.enabled ? inject.attempts : nothing,
            "plant_U" => plant === nothing ? nothing : collect(plant.U),
            "plant_V" => plant === nothing ? nothing : collect(plant.V),
        ),
        "baseline" => base_dict,
        "challenger" => chal_dict,
        "verdict" => string(verdict),
        "challenger_beats_baseline" => chal_beats,
        "edge_delta" => edge_delta,
        "edge_pct_increase" => pct,
        "baseline_wall_time_s" => base_stats.time,
        "challenger_wall_time_s" => chal_stats.time,
    )
end

"""
Run every method in `specs` on one injected, reduced graph.

The first spec is the reference. Each other spec is scored against it with
`compare_results` (`versus[id]` is `challenger` when that method wins).
"""
function compare_multi_one_graph!(dataset::AbstractString, edge_hint::Int,
    specs::Vector{Dict{String,Any}},
    k::Int, θ::Int, reduction::ReductionMode.T, seed::UInt64, inject)
    graph_path = resolve_graph_path(dataset)
    isfile(graph_path) || error("Could not find graph for dataset '$dataset' at $graph_path")

    g, edges, plant = load_graph_maybe_inject(graph_path, inject, k, Random.default_rng())
    edges_eff = edges === nothing ? edge_hint : Int(edges)

    println()
    println("══════════════════════════════════════════════════")
    println("dataset=$dataset  |E|=$edges_eff  nU=$(length(g.adjU)) nV=$(length(g.adjV))")
    println("k=$k  θ=$θ  reduction=$(reduction_name(reduction))  seed=$seed")
    println("methods=$(join((String(get(s, "id", get(s, "label", "?"))) for s in specs), ", "))")
    if inject.enabled
        println("inject: u=$(inject.nU) v=$(inject.nV) k=$k")
    end

    g_reduced = deepcopy(g)
    fg = apply_graph_reductions!(g_reduced, k, θ, nothing, nothing, true, reduction)
    n_u_r = length(fg.u_ids)
    n_v_r = length(fg.v_ids)
    m_r = length(fg.v_adj)
    dens_r = (n_u_r * n_v_r > 0) ? Float64(m_r) / (n_u_r * n_v_r) : 0.0
    println("reduced: nU=$n_u_r nV=$n_v_r |E_R|=$m_r (density=$(round(dens_r; digits=4)))")
    solver_reduction = ReductionMode.none

    runs = Dict{String,Any}()
    results_by_id = Dict{String,Any}()
    ids = String[]
    for spec in specs
        id = String(get(spec, "id", get(spec, "label", get(spec, "method", "method"))))
        push!(ids, id)
        label = String(get(spec, "label", id))
        method = build_side_from_spec(spec)
        println()
        println("── $id ($label) ──")
        Random.seed!(seed)
        try
            stats = benchmark_method!(method, g_reduced, k, θ, solver_reduction;
                label=label, injected_biclique=plant)
            d = method_result_to_dict(stats.result, fg; k=k, θ=θ)
            d["wall_time_s"] = stats.time
            d["label"] = label
            d["id"] = id
            d["status"] = "ok"
            runs[id] = d
            results_by_id[id] = stats.result
            println("  $id edges=$(stats.final_edges)  θ-feasible=$(stats.theta_feasible)  " *
                    "time=$(round(stats.time; digits=3))s")
        catch e
            @error "Method $id failed on $dataset" exception=(e, catch_backtrace())
            runs[id] = Dict{String,Any}(
                "id" => id,
                "label" => label,
                "method" => String(get(spec, "method", "")),
                "status" => "error",
                "error" => sprint(showerror, e),
                "final_edges" => 0,
                "theta_feasible" => false,
                "k_valid" => false,
                "beats_valid" => false,
                "wall_time_s" => nothing,
                "nU" => 0,
                "nV" => 0,
            )
            results_by_id[id] = nothing
        end
    end

    ref = ids[1]
    versus = Dict{String,Any}()
    for id in ids
        id == ref && continue
        ref_res = results_by_id[ref]
        chal_res = results_by_id[id]
        verdict = if ref_res === nothing && chal_res === nothing
            "tie"
        elseif chal_res === nothing
            "baseline"
        elseif ref_res === nothing
            "challenger"
        else
            string(compare_results(ref_res, chal_res, fg, k, θ))
        end
        versus[id] = verdict
        println("  versus $ref → $id : $verdict")
    end
    println("══════════════════════════════════════════════════")

    return Dict{String,Any}(
        "dataset" => String(dataset),
        "edge_count" => edges_eff,
        "nU" => length(g.adjU),
        "nV" => length(g.adjV),
        "reduced_nU" => n_u_r,
        "reduced_nV" => n_v_r,
        "reduced_edges" => m_r,
        "reduced_density" => dens_r,
        "status" => "ok",
        "inject" => Dict{String,Any}(
            "enabled" => inject.enabled,
            "u" => inject.enabled ? inject.nU : nothing,
            "v" => inject.enabled ? inject.nV : nothing,
            "attempts" => inject.enabled ? inject.attempts : nothing,
            "plant_U" => plant === nothing ? nothing : collect(plant.U),
            "plant_V" => plant === nothing ? nothing : collect(plant.V),
        ),
        "runs" => runs,
        "versus" => versus,
        "reference" => ref,
    )
end

# ── Checkpoint payload ────────────────────────────────────────────────────────

function summarize_results(results::Vector)
    chal_wins = count(r -> get(r, "challenger_beats_baseline", false) === true, results)
    baseline_wins = 0
    ties = 0
    for r in results
        get(r, "status", "ok") == "ok" || continue
        v = string(get(r, "verdict", "tie"))
        if v == "baseline"
            baseline_wins += 1
        elseif v == "tie"
            ties += 1
        elseif v == "challenger"
            # counted in chal_wins via beats flag; also count here for symmetry
        end
    end
    # Prefer verdict counts for summary (beats is stricter paper win).
    chal_verdict = count(r -> string(get(r, "verdict", "")) == "challenger", results)
    return Dict{String,Any}(
        "completed" => count(r -> get(r, "status", "") == "ok", results),
        "challenger_wins" => chal_verdict,
        "baseline_wins" => baseline_wins,
        "ties" => ties,
        "paper_beats" => chal_wins,
    )
end

function build_suite_payload(cfg_meta::Dict{String,Any}, results::Vector,
    baseline_spec::Dict{String,Any}, challenger_spec::Dict{String,Any};
    planned::Int=length(results))
    payload = Dict{String,Any}(
        "name" => get(cfg_meta, "name", nothing),
        "config_path" => get(cfg_meta, "config_path", nothing),
        "k" => cfg_meta["k"],
        "theta" => cfg_meta["theta"],
        "seed" => string(cfg_meta["seed"]),
        "reduction" => cfg_meta["reduction"],
        "prefix" => get(cfg_meta, "prefix", nothing),
        "baseline" => side_spec_to_dict(baseline_spec),
        "challenger" => side_spec_to_dict(challenger_spec),
        "planned" => planned,
        "results" => results,
        "summary" => summarize_results(results),
    )
    return payload
end

function load_existing_results(path::AbstractString)
    isfile(path) || return Dict{String,Any}[], nothing
    try
        raw = json_to_dict(JSON3.read(read(path, String)))
        results = get(raw, "results", Any[])
        results isa AbstractVector || return Dict{String,Any}[], raw
        out = Dict{String,Any}[]
        for r in results
            r isa AbstractDict || continue
            push!(out, Dict{String,Any}(string(k) => v for (k, v) in r))
        end
        return out, raw
    catch e
        @warn "Could not resume from $path; starting fresh" exception=e
        return Dict{String,Any}[], nothing
    end
end

function checkpoint_suite!(path::AbstractString, payload::AbstractDict)
    save_benchmark_json(path, payload)
end

# ── Config loading ────────────────────────────────────────────────────────────

function load_compare_config(path::AbstractString)
    raw = json_to_dict(JSON3.read(read(path, String)))
    has_methods = haskey(raw, "methods")
    if !has_methods
        haskey(raw, "baseline") || throw(ArgumentError("Config needs \"baseline\" or \"methods\""))
        haskey(raw, "challenger") || throw(ArgumentError("Config needs \"challenger\" or \"methods\""))
    end

    function normalize_side(side_raw, side_name::AbstractString)
        if side_raw isa AbstractString
            m = lowercase(strip(String(side_raw)))
            return Dict{String,Any}("method" => m, "label" => m, "params" => Dict{String,Any}())
        end
        side_raw isa AbstractDict || throw(ArgumentError("\"$side_name\" must be an object or method name"))
        d = Dict{String,Any}(string(k) => v for (k, v) in side_raw)
        if !haskey(d, "method")
            throw(ArgumentError("\"$side_name\" needs \"method\""))
        end
        get!(d, "label", String(d["method"]))
        if !haskey(d, "id")
            d["id"] = String(d["label"])
        else
            d["id"] = String(d["id"])
        end
        if !haskey(d, "params")
            d["params"] = Dict{String,Any}(string(k) => v for (k, v) in d
                if !(string(k) in ("method", "label", "params", "id")))
        elseif !(d["params"] isa AbstractDict)
            throw(ArgumentError("\"$side_name.params\" must be an object"))
        else
            d["params"] = Dict{String,Any}(string(k) => v for (k, v) in d["params"])
        end
        return d
    end

    methods = nothing
    if has_methods
        raw_methods = raw["methods"]
        raw_methods isa AbstractVector || throw(ArgumentError("\"methods\" must be an array"))
        isempty(raw_methods) && throw(ArgumentError("\"methods\" must be non-empty"))
        methods = Dict{String,Any}[]
        seen = Set{String}()
        for (i, side_raw) in enumerate(raw_methods)
            spec = normalize_side(side_raw, "methods[$i]")
            id = String(spec["id"])
            id in seen && throw(ArgumentError("Duplicate method id \"$id\""))
            push!(seen, id)
            push!(methods, spec)
        end
    end

    baseline = haskey(raw, "baseline") ? normalize_side(raw["baseline"], "baseline") : nothing
    challenger = haskey(raw, "challenger") ? normalize_side(raw["challenger"], "challenger") : nothing

    k = Int(get(raw, "k", 2))
    θ = Int(get(raw, "theta", 5))
    seed = UInt64(get(raw, "seed", 1))
    reduction = parse_reduction_name(String(get(raw, "reduction", "lo")))
    skip_completed = Bool(get(raw, "skip_completed", true))
    prefix = get(raw, "prefix", nothing)
    prefix = prefix === nothing ? nothing : String(prefix)
    datasets = get(raw, "datasets", nothing)
    if datasets !== nothing
        datasets = String[String(d) for d in datasets]
    end
    output = get(raw, "output", nothing)
    if output === nothing
        name = get(raw, "name", nothing)
        output = name === nothing ? "compare_methods.json" :
            "results/$(replace(String(name), r"[^A-Za-z0-9._-]+" => "_")).json"
    end
    name = get(raw, "name", nothing)

    return (;
        name = name === nothing ? nothing : String(name),
        output=String(output),
        k, θ, seed, reduction, skip_completed, prefix, datasets,
        baseline, challenger, methods,
        inject_cfg=raw,
        config_path=path,
    )
end

# ── Suite runner ──────────────────────────────────────────────────────────────

function run_compare_suite(; graphs,
    baseline_method::SolveMethod, challenger_method::SolveMethod,
    baseline_spec::Dict{String,Any}, challenger_spec::Dict{String,Any},
    k::Int, θ::Int, reduction::ReductionMode.T, seed::UInt64, inject,
    save_path::Union{Nothing,AbstractString},
    skip_completed::Bool=true,
    cfg_meta::Dict{String,Any}=Dict{String,Any}())

    isempty(graphs) && error("No graphs to compare (check data/ index, --prefix= / datasets / dataset arg)")

    out_path = save_path === nothing ? nothing : resolve_benchmark_save_path(save_path)
    results = Dict{String,Any}[]
    if out_path !== nothing && skip_completed
        results, _prev = load_existing_results(out_path)
    end
    done = Set(String(r["dataset"]) for r in results if haskey(r, "dataset") &&
        get(r, "status", "") == "ok")

    cfg_meta = merge(Dict{String,Any}(
        "k" => k,
        "theta" => θ,
        "seed" => seed,
        "reduction" => reduction_name(reduction),
    ), cfg_meta)

    planned = length(graphs)
    println("Compare suite: $planned graph(s) ascending by |E|" *
            (out_path === nothing ? "" : "  → $out_path"))
    if !isempty(done)
        println("Resuming: $(length(done)) already completed")
    end

    for (i, entry) in enumerate(graphs)
        key = entry.key
        if key in done
            println("[$i/$planned] skip (completed): $key  (|E|=$(entry.edges))")
            continue
        end
        println("[$i/$planned] $key  (|E|=$(entry.edges))")
        Random.seed!(seed)
        try
            row = compare_one_graph!(key, entry.edges,
                baseline_method, challenger_method,
                baseline_spec, challenger_spec,
                k, θ, reduction, seed, inject)
            # Replace prior failed/partial row for this dataset if any.
            filter!(r -> get(r, "dataset", nothing) != key, results)
            push!(results, row)
        catch e
            @error "Compare failed on $key" exception=(e, catch_backtrace())
            filter!(r -> get(r, "dataset", nothing) != key, results)
            push!(results, Dict{String,Any}(
                "dataset" => key,
                "edge_count" => entry.edges,
                "status" => "error",
                "error" => sprint(showerror, e),
            ))
        end

        if out_path !== nothing
            # Keep results ordered by edge count for readability.
            sort!(results; by=r -> (get(r, "edge_count", typemax(Int)),
                String(get(r, "dataset", ""))))
            payload = build_suite_payload(cfg_meta, results, baseline_spec, challenger_spec;
                planned=planned)
            checkpoint_suite!(out_path, payload)
        end
    end

    # Final summary
    ok = [r for r in results if get(r, "status", "") == "ok"]
    summary = summarize_results(results)
    println()
    println("==================== SUITE SUMMARY ====================")
    println("  planned / completed  : $planned / $(summary["completed"])")
    println("  challenger wins      : $(summary["challenger_wins"])")
    println("  baseline wins        : $(summary["baseline_wins"])")
    println("  ties                 : $(summary["ties"])")
    println("  paper beats (strict) : $(summary["paper_beats"])")
    if out_path !== nothing
        println("  checkpoint           : $out_path")
    end
    println("=======================================================")

    if out_path !== nothing
        sort!(results; by=r -> (get(r, "edge_count", typemax(Int)),
            String(get(r, "dataset", ""))))
        payload = build_suite_payload(cfg_meta, results, baseline_spec, challenger_spec;
            planned=planned)
        checkpoint_suite!(out_path, payload)
    elseif length(ok) == 1
        # Single-graph CLI without --save: nothing to write (legacy stdout-only).
    end

    return results
end

function summarize_multi(results::Vector, ref::AbstractString)
    counts = Dict{String,Any}()
    for r in results
        get(r, "status", "") == "ok" || continue
        versus = get(r, "versus", Dict{String,Any}())
        versus isa AbstractDict || continue
        for (id, verdict) in versus
            bucket = get!(counts, String(id), Dict{String,Int}(
                "challenger" => 0, "baseline" => 0, "tie" => 0))
            key = string(verdict)
            key in ("challenger", "baseline", "tie") || continue
            bucket[key] += 1
        end
    end
    return Dict{String,Any}(
        "completed" => count(r -> get(r, "status", "") == "ok", results),
        "reference" => String(ref),
        "versus" => counts,
    )
end

function run_multi_suite(; graphs, specs::Vector{Dict{String,Any}},
    k::Int, θ::Int, reduction::ReductionMode.T, seed::UInt64, inject,
    save_path::Union{Nothing,AbstractString},
    skip_completed::Bool=true,
    cfg_meta::Dict{String,Any}=Dict{String,Any}())

    isempty(graphs) && error("No graphs to compare (check data/ index, --prefix= / datasets / dataset arg)")
    isempty(specs) && error("Multi-method compare needs at least one method")

    out_path = save_path === nothing ? nothing : resolve_benchmark_save_path(save_path)
    results = Dict{String,Any}[]
    if out_path !== nothing && skip_completed
        results, _prev = load_existing_results(out_path)
    end
    done = Set(String(r["dataset"]) for r in results if haskey(r, "dataset") &&
        get(r, "status", "") == "ok")

    ref = String(get(specs[1], "id", get(specs[1], "label", "method")))
    cfg_meta = merge(Dict{String,Any}(
        "k" => k,
        "theta" => θ,
        "seed" => seed,
        "reduction" => reduction_name(reduction),
    ), cfg_meta)

    planned = length(graphs)
    println("Multi compare: $planned graph(s), $(length(specs)) methods, reference=$ref" *
            (out_path === nothing ? "" : "  → $out_path"))
    if !isempty(done)
        println("Resuming: $(length(done)) already completed")
    end

    for (i, entry) in enumerate(graphs)
        key = entry.key
        if key in done
            println("[$i/$planned] skip (completed): $key  (|E|=$(entry.edges))")
            continue
        end
        println("[$i/$planned] $key  (|E|=$(entry.edges))")
        Random.seed!(seed)
        try
            row = compare_multi_one_graph!(key, entry.edges, specs,
                k, θ, reduction, seed, inject)
            filter!(r -> get(r, "dataset", nothing) != key, results)
            push!(results, row)
        catch e
            @error "Compare failed on $key" exception=(e, catch_backtrace())
            filter!(r -> get(r, "dataset", nothing) != key, results)
            push!(results, Dict{String,Any}(
                "dataset" => key,
                "edge_count" => entry.edges,
                "status" => "error",
                "error" => sprint(showerror, e),
            ))
        end

        if out_path !== nothing
            sort!(results; by=r -> (get(r, "edge_count", typemax(Int)),
                String(get(r, "dataset", ""))))
            payload = Dict{String,Any}(
                "name" => get(cfg_meta, "name", nothing),
                "config_path" => get(cfg_meta, "config_path", nothing),
                "k" => k,
                "theta" => θ,
                "seed" => string(seed),
                "reduction" => reduction_name(reduction),
                "prefix" => get(cfg_meta, "prefix", nothing),
                "reference" => ref,
                "methods" => [side_spec_to_dict(s) for s in specs],
                "planned" => planned,
                "results" => results,
                "summary" => summarize_multi(results, ref),
            )
            checkpoint_suite!(out_path, payload)
        end
    end

    summary = summarize_multi(results, ref)
    println()
    println("==================== SUITE SUMMARY ====================")
    println("  planned / completed  : $planned / $(summary["completed"])")
    println("  reference            : $ref")
    versus = summary["versus"]
    if versus isa AbstractDict
        for (id, bucket) in versus
            println("  $id vs $ref : challenger=$(bucket["challenger"]) " *
                    "baseline=$(bucket["baseline"]) tie=$(bucket["tie"])")
        end
    end
    if out_path !== nothing
        println("  checkpoint           : $out_path")
    end
    println("=======================================================")
    return results
end

# ── Entry points ──────────────────────────────────────────────────────────────

function main_from_config(config_path::AbstractString)
    cfg = load_compare_config(resolve_config_path(config_path))
    θ = cfg.θ
    k = cfg.k
    θ > k || throw(ArgumentError("θ must be > k (got θ=$θ k=$k)"))

    # CLI can override a few knobs from the config.
    cli_k, cli_θ = parse_k_theta_defaults()
    # Only apply if user passed flags (detect via ARGS); keep config defaults otherwise.
    for arg in ARGS
        startswith(arg, "--k=") && (k = cli_k)
        startswith(arg, "--theta=") && (θ = cli_θ)
    end
    seed = parse_string_flag("seed") === nothing ? cfg.seed : parse_seed_flag()
    reduction = any(a -> startswith(a, "--reduce="), ARGS) ?
        parse_reduction_flag(cfg.reduction) : cfg.reduction
    prefix = let p = parse_string_flag("prefix")
        p === nothing ? cfg.prefix : p
    end
    save_path = let s = parse_benchmark_save()
        s === nothing ? cfg.output : s
    end

    inject = if any(a -> a in ("--inject", "--no-inject") || startswith(a, "--u=") ||
            startswith(a, "--v=") || startswith(a, "--inject-attempts="), ARGS)
        parse_inject(; default_nU=θ, default_nV=θ, default_enabled=true)
    else
        inject_from_config(cfg.inject_cfg, θ)
    end
    if inject.enabled && k >= inject.nU * inject.nV
        throw(ArgumentError("--k must be < u*v for injection; got k=$k, u=$(inject.nU), v=$(inject.nV)"))
    end

    graphs = resolve_graph_queue(; datasets=cfg.datasets, prefix=prefix,
        single=resolve_dataset_arg())
    if isempty(graphs)
        throw(ArgumentError(
            "No graphs to compare (check data/ index, \"prefix\" / \"datasets\", or dataset arg)"))
    end

    if cfg.methods !== nothing
        Random.seed!(seed)
        run_multi_suite(;
            graphs=graphs,
            specs=cfg.methods,
            k=k, θ=θ, reduction=reduction, seed=seed, inject=inject,
            save_path=save_path,
            skip_completed=cfg.skip_completed,
            cfg_meta=Dict{String,Any}(
                "name" => cfg.name,
                "config_path" => cfg.config_path,
                "prefix" => prefix,
            ),
        )
        return
    end

    cfg.baseline === nothing && throw(ArgumentError("Config needs \"baseline\""))
    cfg.challenger === nothing && throw(ArgumentError("Config needs \"challenger\""))
    baseline_method = build_side_from_spec(cfg.baseline)
    challenger_method = build_side_from_spec(cfg.challenger)

    Random.seed!(seed)
    run_compare_suite(;
        graphs=graphs,
        baseline_method=baseline_method,
        challenger_method=challenger_method,
        baseline_spec=cfg.baseline,
        challenger_spec=cfg.challenger,
        k=k, θ=θ, reduction=reduction, seed=seed, inject=inject,
        save_path=save_path,
        skip_completed=cfg.skip_completed,
        cfg_meta=Dict{String,Any}(
            "name" => cfg.name,
            "config_path" => cfg.config_path,
            "prefix" => prefix,
        ),
    )
end

function main_from_cli()
    baseline_name = parse_named_method("baseline")
    challenger_name = parse_named_method("challenger")
    (baseline_name === nothing || challenger_name === nothing) && usage_and_exit()

    k, θ = parse_k_theta_defaults()
    θ > k || throw(ArgumentError("θ must be > k (got θ=$θ k=$k)"))
    inject = parse_inject(; default_nU=θ, default_nV=θ, default_enabled=true)
    if inject.enabled && k >= inject.nU * inject.nV
        throw(ArgumentError("--k must be < u*v for injection; got k=$k, u=$(inject.nU), v=$(inject.nV)"))
    end
    reduction = parse_reduction_flag()
    seed = parse_seed_flag()
    prefix = parse_string_flag("prefix")
    dataset = resolve_dataset_arg()
    save_path = parse_benchmark_save()

    graphs = resolve_graph_queue(; prefix=prefix, single=dataset)
    isempty(graphs) && usage_and_exit()

    # Multi-graph CLI requires --save= so checkpoints have a destination.
    if length(graphs) > 1 && save_path === nothing
        throw(ArgumentError("Multi-graph compare needs --save=PATH.json (checkpoint file)"))
    end

    baseline_method = build_side_cli(baseline_name, "baseline")
    challenger_method = build_side_cli(challenger_name, "challenger")
    baseline_spec = Dict{String,Any}(
        "method" => baseline_name,
        "label" => baseline_name,
        "params" => Dict{String,Any}(),  # filled below from constructed method meta when possible
    )
    challenger_spec = Dict{String,Any}(
        "method" => challenger_name,
        "label" => challenger_name,
        "params" => Dict{String,Any}(),
    )
    # Record CLI overrides into the saved specs.
    for (spec, side) in ((baseline_spec, "baseline"), (challenger_spec, "challenger"))
        ants = parse_int_override("$side-ants")
        gens = parse_int_override("$side-gens")
        kmax = parse_int_override("$side-kmax")
        beta = parse_float_override("$side-beta")
        retry_p = parse_float_override("$side-p")
        max_steps = parse_int_override("$side-max-steps")
        n_seeds = parse_int_override("$side-n-seeds")
        local_reduction = parse_bool_override("$side-local-reduction")
        prune_by_bound = parse_bool_override("$side-prune-by-bound")
        one_colony = parse_bool_override("$side-one-colony")
        greedy_fill = parse_string_flag("$side-greedy-fill")
        eta_exp = parse_float_override("$side-eta-exp")
        best_only_pheromone = parse_bool_override("$side-best-only-pheromone")
        pheromone = parse_int_override("$side-pheromone")
        evaporation = parse_float_override("$side-evaporation")
        ants !== nothing && (spec["params"]["ants"] = ants)
        gens !== nothing && (spec["params"]["generations"] = gens)
        kmax !== nothing && (spec["params"]["kmax"] = kmax)
        beta !== nothing && (spec["params"]["beta"] = beta)
        retry_p !== nothing && (spec["params"]["p"] = retry_p)
        max_steps !== nothing && (spec["params"]["max_steps"] = max_steps)
        n_seeds !== nothing && (spec["params"]["n_seeds"] = n_seeds)
        local_reduction !== nothing && (spec["params"]["local_reduction"] = local_reduction)
        prune_by_bound !== nothing && (spec["params"]["prune_by_bound"] = prune_by_bound)
        one_colony !== nothing && (spec["params"]["one_colony"] = one_colony)
        greedy_fill !== nothing && (spec["params"]["greedy_fill"] = greedy_fill)
        eta_exp !== nothing && (spec["params"]["eta_exp"] = eta_exp)
        best_only_pheromone !== nothing && (spec["params"]["best_only_pheromone"] = best_only_pheromone)
        pheromone !== nothing && (spec["params"]["pheromone"] = pheromone)
        evaporation !== nothing && (spec["params"]["evaporation"] = evaporation)
    end

    Random.seed!(seed)
    results = run_compare_suite(;
        graphs=graphs,
        baseline_method=baseline_method,
        challenger_method=challenger_method,
        baseline_spec=baseline_spec,
        challenger_spec=challenger_spec,
        k=k, θ=θ, reduction=reduction, seed=seed, inject=inject,
        save_path=save_path,
        skip_completed=true,
        cfg_meta=Dict{String,Any}("prefix" => prefix),
    )

    # Legacy single-graph + --save=: also write the flat one-graph shape if only
    # one result, matching the previous schema (top-level dataset / verdict).
    if save_path !== nothing && length(results) == 1 && get(results[1], "status", "") == "ok"
        row = results[1]
        flat = Dict{String,Any}(
            "dataset" => row["dataset"],
            "k" => k,
            "theta" => θ,
            "seed" => string(seed),
            "reduction" => string(reduction),
            "inject" => row["inject"],
            "edge_count" => row["edge_count"],
            "nU" => get(row, "nU", nothing),
            "nV" => get(row, "nV", nothing),
            "reduced_nU" => row["reduced_nU"],
            "reduced_nV" => row["reduced_nV"],
            "reduced_edges" => get(row, "reduced_edges", nothing),
            "reduced_density" => get(row, "reduced_density", nothing),
            "heat" => get(row, "heat", nothing),
            "baseline" => row["baseline"],
            "challenger" => row["challenger"],
            "verdict" => row["verdict"],
            "challenger_beats_baseline" => row["challenger_beats_baseline"],
            "edge_delta" => row["edge_delta"],
            "edge_pct_increase" => row["edge_pct_increase"],
            "baseline_wall_time_s" => row["baseline_wall_time_s"],
            "challenger_wall_time_s" => row["challenger_wall_time_s"],
            # Suite envelope also kept for resume compatibility.
            "results" => results,
            "summary" => summarize_results(results),
            "baseline_spec" => side_spec_to_dict(baseline_spec),
            "challenger_spec" => side_spec_to_dict(challenger_spec),
        )
        save_benchmark_json(resolve_benchmark_save_path(save_path), flat)
    end
end

function main()
    if "--list" in ARGS || "-h" in ARGS || "--help" in ARGS
        println("Registered SolveMethods:")
        for name in list_methods()
            println("  - $name")
        end
        println()
        println("JSON config: julia bin/compare-methods.jl --config=configs/….json")
        return
    end

    config = parse_string_flag("config")
    if config !== nothing
        main_from_config(config)
    else
        main_from_cli()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
