#=
=================================================================================
Timed multi-dataset benchmark suite
=================================================================================
Reads a JSON config listing datasets and options, runs the same benchmarks as
`load.jl --benchmark=...` on each, with a per-function process timeout
(default 10^3 seconds), and writes aggregate results to a JSON file.

Usage:
  julia test.jl configs/suite.json
  julia test.jl path/to/suite.json

Config schema (all fields except `datasets` are optional):
  {
    "datasets": ["amazon/boxes", "amazon/appliances"],
    "benchmark": "aco,pivot",
    "k": 6,
    "theta": 7,
    "timeout_seconds": 1000,
    "reduction": "hi",
    "seed": 1,
    "output": "results/suite_run.json",
    "aco": {
      "ants": 10,
      "iterations": 100,
      "pheremone": 1,
      "evaporation": 0.9
    }
  }

Dataset entries may be strings or objects that override top-level defaults:
  { "name": "amazon/grocery", "timeout_seconds": 500, "k": 4 }
=================================================================================
=#

using Distributed
using Dates
using Random
using JSON3

const ROOT = @__DIR__
const SRC = joinpath(ROOT, "src")
const DEFAULT_TIMEOUT_SECONDS = 10^3

# ---- local includes (driver) ------------------------------------------------

isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))
isdefined(@__MODULE__, :__OPPONENT_JL__) || include(joinpath(SRC, "opponent.jl"))
isdefined(@__MODULE__, :__GA_JL__) || include(joinpath(SRC, "ga.jl"))
isdefined(@__MODULE__, :__PARALLEL_TABU_JL__) || include(joinpath(SRC, "parallel_tabu.jl"))
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath(SRC, "aco.jl"))
isdefined(@__MODULE__, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))
isdefined(@__MODULE__, :__BENCHMARK_JL__) || include(joinpath(ROOT, "benchmark.jl"))

# ---- worker bootstrap -------------------------------------------------------

"""
Load solver/benchmark code and named entry points onto process `p`.
"""
function setup_worker!(p::Integer)
    root = ROOT
    src = SRC
    @everywhere [p] begin
        ROOT = $root
        SRC = $src
        isdefined(Main, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
        isdefined(Main, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
        isdefined(Main, :__IO_JL__) || include(joinpath(SRC, "io.jl"))
        isdefined(Main, :__OPPONENT_JL__) || include(joinpath(SRC, "opponent.jl"))
        isdefined(Main, :__GA_JL__) || include(joinpath(SRC, "ga.jl"))
        isdefined(Main, :__PARALLEL_TABU_JL__) || include(joinpath(SRC, "parallel_tabu.jl"))
        isdefined(Main, :__ACO_JL__) || include(joinpath(SRC, "aco.jl"))
        isdefined(Main, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))
        isdefined(Main, :__BENCHMARK_JL__) || include(joinpath(ROOT, "benchmark.jl"))

        function worker_warmup(k, θ, reduction, aco_options, target_syms)
            targets = Set{Symbol}(Symbol(s) for s in target_syms)
            warmup_benchmarks!(k, θ, reduction, aco_options; targets=targets)
            return nothing
        end

        function worker_graph_memory(graph_path, k, θ, reduction)
            g, edges = load_bipartite_graph(graph_path)
            stats = benchmark_graph_memory(g, edges, k, θ, reduction)
            # Return only JSON-safe / lightweight fields (avoid shipping FrozenBipartite).
            return (
                mutable_bytes = stats.mutable_bytes,
                frozen_bytes = stats.frozen_bytes,
                reduced_nU = length(stats.fg.u_ids),
                reduced_nV = length(stats.fg.v_ids),
            )
        end

        function worker_pivot(graph_path, k, θ, reduction)
            g, _ = load_bipartite_graph(graph_path)
            ps = benchmark_pivot!(g, k, θ, reduction)
            return (
                time = ps.time,
                allocated = ps.allocated,
                rss_delta = ps.rss_delta,
                opt_edges = ps.opt_edges,
                nU = length(ps.sol.U),
                nV = length(ps.sol.V),
                edges = Subgraph.edge_count(ps.fg_eval, ps.sol),
                missing = Subgraph.missing_edges(ps.fg_eval, ps.sol),
            )
        end

        function worker_aco(graph_path, k, θ, aco_options, opt_edges)
            g, _ = load_bipartite_graph(graph_path)
            as = benchmark_aco!(g, k, θ, aco_options; opt_edges=opt_edges, early_stop=true)
            return (
                time = as.time,
                allocated = as.allocated,
                rss_delta = as.rss_delta,
                ants = as.ants,
                iterations_budget = as.iterations_budget,
                first_hit_iteration = as.first_hit_iteration,
                final_edges = as.final_edges,
                opt_edges = as.opt_edges,
                nU = length(as.sol.U),
                nV = length(as.sol.V),
            )
        end
    end
    return p
end

const _WORKER_READY = Set{Int}()

function ensure_worker!()
    if nprocs() == 1 || workers() == [1]
        empty!(_WORKER_READY)
        addprocs(1)
    end
    p = workers()[end]
    if p ∉ _WORKER_READY
        setup_worker!(p)
        push!(_WORKER_READY, p)
    end
    return p
end

"""
Run a named worker function (`fn_name`) with `args` under a hard process timeout.

Dispatches via a small remote thunk that resolves `fn_name` on the worker, so
Julia 1.12 does not need to deserialize driver-side method bindings.
"""
function with_process_timeout(timeout_seconds::Real, fn_name::Symbol, args...)
    p = ensure_worker!()
    argv = Any[args...]
    future = remotecall(p, fn_name, argv) do name, argv
        getfield(Main, name)(argv...)
    end

    start_time = time()
    while !isready(future)
        if time() - start_time > timeout_seconds
            @warn "Process timed out after $(timeout_seconds)s. Terminating worker $p…"
            rmprocs(p; waitfor=0)
            delete!(_WORKER_READY, p)
            try
                addprocs(1)
                np = workers()[end]
                setup_worker!(np)
                push!(_WORKER_READY, np)
            catch e
                @warn "Failed to respawn worker after timeout" exception = e
            end
            throw(ErrorException("Execution timed out after $(timeout_seconds)s."))
        end
        sleep(0.05)
    end

    return fetch(future)
end

# ---- config parsing ---------------------------------------------------------

function parse_reduction_name(value::AbstractString)
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

function normalize_dataset_entry(entry, defaults::Dict{String,Any})
    if entry isa AbstractString
        cfg = copy(defaults)
        cfg["name"] = String(entry)
        return cfg
    end
    d = json_to_dict(entry)
    cfg = merge(copy(defaults), d)
    haskey(cfg, "name") || throw(ArgumentError("Dataset entry missing \"name\": $entry"))
    cfg["name"] = String(cfg["name"])
    return cfg
end

function load_suite_config(path::AbstractString)
    raw = JSON3.read(read(path, String))
    root = json_to_dict(raw)

    haskey(root, "datasets") || throw(ArgumentError("Config must contain a \"datasets\" array"))

    aco_base = Dict{String,Any}(
        "ants" => 10,
        "iterations" => 100,
        "pheremone" => 1,
        "evaporation" => 0.9,
        "subspecies" => 1,
    )
    if haskey(root, "aco")
        aco_base = merge(aco_base, json_to_dict(root["aco"]))
    end

    defaults = Dict{String,Any}(
        "benchmark" => get(root, "benchmark", "aco,pivot"),
        "k" => get(root, "k", 6),
        "theta" => get(root, "theta", 7),
        "timeout_seconds" => get(root, "timeout_seconds", DEFAULT_TIMEOUT_SECONDS),
        "reduction" => get(root, "reduction", "hi"),
        "seed" => get(root, "seed", nothing),
        "aco" => aco_base,
    )

    datasets = [normalize_dataset_entry(e, defaults) for e in root["datasets"]]
    output = get(root, "output", nothing)
    if output === nothing
        stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
        output = joinpath("results", "suite_$stamp.json")
    end

    return (; datasets, output=String(output), defaults)
end

function aco_options_from(cfg::Dict{String,Any})
    aco = cfg["aco"]
    aco isa AbstractDict || throw(ArgumentError("\"aco\" must be an object"))
    pheremone = Int(get(aco, "pheremone", 1))
    num_ants = Int(get(aco, "ants", 10))
    num_iterations = Int(get(aco, "iterations", 100))
    evaporation = Float64(get(aco, "evaporation", 0.9))
    num_subspecies = Int(get(aco, "subspecies", 1))
    prefer_smaller_side = Bool(get(aco, "prefer_smaller_side", true))
    elite_seed = Bool(get(aco, "elite_seed", true))
    elite_seed_ants = Int(get(aco, "elite_seed_ants", 3))
    elite_seed_remove = Int(get(aco, "elite_seed_remove", 2))
    return (; pheremone, num_ants, num_iterations, evaporation, num_subspecies,
        prefer_smaller_side, elite_seed, elite_seed_ants, elite_seed_remove)
end

# ---- serialization ----------------------------------------------------------

function serialize_pivot_stats(ps, k::Int)
    ps === nothing && return nothing
    return Dict{String,Any}(
        "wall_time_s" => ps.time,
        "allocated_bytes" => ps.allocated,
        "rss_delta_bytes" => ps.rss_delta,
        "optimal_edges" => ps.opt_edges,
        "solution" => Dict(
            "nU" => ps.nU,
            "nV" => ps.nV,
            "edges" => ps.edges,
            "missing" => ps.missing,
            "k" => k,
        ),
    )
end

function serialize_aco_stats(as)
    as === nothing && return nothing
    return Dict{String,Any}(
        "wall_time_s" => as.time,
        "allocated_bytes" => as.allocated,
        "rss_delta_bytes" => as.rss_delta,
        "ants" => as.ants,
        "iterations_budget" => as.iterations_budget,
        "iterations_to_optimal" => as.first_hit_iteration,
        "final_edges" => as.final_edges,
        "optimal_edges" => as.opt_edges,
        "nU" => as.nU,
        "nV" => as.nV,
    )
end

is_timeout(e) = e isa ErrorException && occursin("timed out", e.msg)

# ---- per-dataset timed run --------------------------------------------------

function run_dataset_suite(cfg::Dict{String,Any})
    name = cfg["name"]
    k = Int(cfg["k"])
    θ = Int(cfg["theta"])
    timeout = Float64(cfg["timeout_seconds"])
    reduction = parse_reduction_name(String(cfg["reduction"]))
    targets = parse_benchmark_targets(String(cfg["benchmark"]))
    aco_options = aco_options_from(cfg)
    seed = cfg["seed"]
    target_syms = String[String(t) for t in targets]

    graph_path = abspath(resolve_graph_path(name))
    result = Dict{String,Any}(
        "dataset" => name,
        "graph_path" => graph_path,
        "k" => k,
        "theta" => θ,
        "timeout_seconds" => timeout,
        "benchmark" => String(cfg["benchmark"]),
        "reduction" => String(cfg["reduction"]),
        "status" => "ok",
        "error" => nothing,
        "graph" => nothing,
        "pivot" => nothing,
        "aco" => nothing,
    )

    if !isfile(graph_path)
        result["status"] = "missing_graph"
        result["error"] = "Graph not found at $graph_path"
        @warn result["error"]
        return result
    end

    if seed !== nothing && (:aco in targets)
        Random.seed!(UInt64(seed))
    end

    println()
    println("══════════════════════════════════════════════════")
    println("Dataset: $name")
    println("  path=$graph_path  k=$k  θ=$θ  timeout=$(timeout)s")
    println("  targets=$(join(sort(target_syms), ","))")
    println("══════════════════════════════════════════════════")

    local edge_count
    try
        g_local, edge_count = load_bipartite_graph(graph_path)
        result["nU"] = length(g_local.adjU)
        result["nV"] = length(g_local.adjV)
        result["edges"] = edge_count
        g_local = nothing
        GC.gc()
    catch e
        result["status"] = "error"
        result["error"] = "Failed to load graph: " * sprint(showerror, e)
        return result
    end

    try
        with_process_timeout(min(timeout, 120.0), :worker_warmup, k, θ, reduction, aco_options, target_syms)
    catch e
        @warn "Warmup failed or timed out; continuing" exception = e
    end

    try
        gs = with_process_timeout(timeout, :worker_graph_memory, graph_path, k, θ, reduction)
        result["graph"] = Dict{String,Any}(
            "edges" => edge_count,
            "mutable_bytes" => gs.mutable_bytes,
            "frozen_bytes" => gs.frozen_bytes,
            "reduced_nU" => gs.reduced_nU,
            "reduced_nV" => gs.reduced_nV,
        )
    catch e
        result["graph"] = is_timeout(e) ?
            Dict{String,Any}("status" => "timeout") :
            Dict{String,Any}("status" => "error", "error" => sprint(showerror, e))
        is_timeout(e) && (result["status"] = "partial_timeout")
    end

    pivot_opt_edges = nothing

    if :pivot in targets
        try
            ps = with_process_timeout(timeout, :worker_pivot, graph_path, k, θ, reduction)
            result["pivot"] = serialize_pivot_stats(ps, k)
            pivot_opt_edges = ps.opt_edges
        catch e
            if is_timeout(e)
                result["pivot"] = Dict{String,Any}("status" => "timeout")
                result["status"] = "partial_timeout"
            else
                result["pivot"] = Dict{String,Any}("status" => "error", "error" => sprint(showerror, e))
                result["status"] = "partial_error"
            end
        end
    end

    if :aco in targets
        opt = pivot_opt_edges
        try
            as = with_process_timeout(timeout, :worker_aco, graph_path, k, θ, aco_options, opt)
            result["aco"] = serialize_aco_stats(as)
        catch e
            if is_timeout(e)
                result["aco"] = Dict{String,Any}("status" => "timeout")
                result["status"] = result["status"] == "ok" ? "partial_timeout" : result["status"]
            else
                result["aco"] = Dict{String,Any}("status" => "error", "error" => sprint(showerror, e))
                result["status"] = result["status"] == "ok" ? "partial_error" : result["status"]
            end
        end
    end

    return result
end

# ---- main -------------------------------------------------------------------

function save_results(path::AbstractString, payload)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
    println("Wrote results → $path")
end

function main()
    if length(ARGS) < 1
        println(stderr, "Usage: julia test.jl <path/to/suite.json>")
        exit(1)
    end

    config_path = ARGS[1]
    if !isfile(config_path)
        println(stderr, "Error: config not found: $config_path")
        exit(1)
    end

    suite = load_suite_config(config_path)
    println("Suite config: $config_path")
    println("Datasets: $(length(suite.datasets))")
    println("Output:   $(suite.output)")
    println("Default timeout: $(DEFAULT_TIMEOUT_SECONDS)s (overridable per dataset)")

    ensure_worker!()

    started = Dates.now()
    dataset_results = Dict{String,Any}[]

    for cfg in suite.datasets
        push!(dataset_results, run_dataset_suite(cfg))
    end

    payload = Dict{String,Any}(
        "config_path" => abspath(config_path),
        "started_at" => string(started),
        "finished_at" => string(Dates.now()),
        "default_timeout_seconds" => DEFAULT_TIMEOUT_SECONDS,
        "datasets" => dataset_results,
    )

    save_results(suite.output, payload)
    println("Done. $(length(dataset_results)) dataset(s) benchmarked.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
