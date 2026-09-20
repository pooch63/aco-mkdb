#=
=================================================================================
Head-to-head method comparison on one graph or a suite ordered by |E|.

Runs two registered SolveMethods (same or different; arbitrary kwargs) on each
reduced graph and reports whether the challenger beats the baseline under paper
semantics (θ-feasible + more edges).

JSON config (recommended for A/B / ablations):
  julia bin/compare-methods.jl --config=configs/aco_pheromone_ablation.json

Single graph (CLI overrides for ants / gens / kmax / beta / p / max-steps still work):
  julia bin/compare-methods.jl amazon/boxes --baseline=heuristic --challenger=aco
  julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=aco \\
      --baseline-ants=10 --challenger-ants=50 --save=aco_ants_ab.json
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
    return Dict{String,Any}("method" => method, "label" => label, "params" => params)
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
    params = Dict{String,Any}()
    for (k, v) in kwargs
        params[string(k)] = v
    end
    if name == "aco"
        params["ants"] = ants === nothing ? get(params, "ants", 100) : ants
        get!(params, "iterations", 5)
        get!(params, "prefer_smaller_side", true)
        get!(params, "neighbor_scope_limit", true)
        get!(params, "parallelize", false)
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
    haskey(raw, "baseline") || throw(ArgumentError("Config needs \"baseline\""))
    haskey(raw, "challenger") || throw(ArgumentError("Config needs \"challenger\""))

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
        if !haskey(d, "params")
            d["params"] = Dict{String,Any}(string(k) => v for (k, v) in d
                if !(string(k) in ("method", "label", "params")))
        elseif !(d["params"] isa AbstractDict)
            throw(ArgumentError("\"$side_name.params\" must be an object"))
        else
            d["params"] = Dict{String,Any}(string(k) => v for (k, v) in d["params"])
        end
        return d
    end

    baseline = normalize_side(raw["baseline"], "baseline")
    challenger = normalize_side(raw["challenger"], "challenger")

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
        baseline, challenger,
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

    baseline_method = build_side_from_spec(cfg.baseline)
    challenger_method = build_side_from_spec(cfg.challenger)

    graphs = resolve_graph_queue(; datasets=cfg.datasets, prefix=prefix,
        single=resolve_dataset_arg())
    if isempty(graphs)
        throw(ArgumentError(
            "No graphs to compare (check data/ index, \"prefix\" / \"datasets\", or dataset arg)"))
    end

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
        ants !== nothing && (spec["params"]["ants"] = ants)
        gens !== nothing && (spec["params"]["generations"] = gens)
        kmax !== nothing && (spec["params"]["kmax"] = kmax)
        beta !== nothing && (spec["params"]["beta"] = beta)
        retry_p !== nothing && (spec["params"]["p"] = retry_p)
        max_steps !== nothing && (spec["params"]["max_steps"] = max_steps)
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
