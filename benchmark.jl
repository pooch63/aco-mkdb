#=
=================================================================================
Paper / algorithm benchmarks for a single loaded graph.

Invoked from load.jl via:
  julia load.jl <dataset> --benchmark=aco,pivot
  julia load.jl <dataset> --benchmark=pivot
  julia load.jl <dataset> --benchmark=aco
  julia load.jl <dataset> --benchmark=heuristic
  julia load.jl <dataset> --benchmark=ga
  julia load.jl <dataset> --benchmark=aco,pivot,heuristic,ga
  julia load.jl <dataset> --benchmark=aco,ga --save=run.json

Reports:
  - shared graph-reduction wall time (reductions run once; solvers get
    ReductionMode.none on the already-reduced graph)
  - memory of the loaded (and frozen) graph structure
  - wall time + allocated bytes + peak-RSS delta for pivot / ACO / heuristic / GA
  - ACO: total wall time and time-to-best (elapsed until the reported best
    subspecies solution was last improved)
  - when pivot is also run: iterations until ACO first matches the pivot
    optimum (by edge count + θ-feasibility), then early-stops
  - optional JSON dump via --save=<name> → results/<name> (or an explicit path)
=================================================================================
=#

const __BENCHMARK_JL__ = true

using Printf

# Default directory for `--save=` filenames with no directory component.
const BENCHMARK_RESULTS_DIR = joinpath(@__DIR__, "results")

"""
Human-readable byte count (binary units).
"""
function format_bytes(n::Integer)
    n = Int64(n)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    x = Float64(abs(n))
    i = 1
    while x >= 1024 && i < length(units)
        x /= 1024
        i += 1
    end
    return string(round(x; digits=2), " ", units[i])
end

"""Wall / elapsed seconds always shown to 4 decimal places (e.g. `0.0000`)."""
format_seconds(t::Real) = @sprintf("%.4f", Float64(t))

"""
Process high-water RSS in bytes (Julia's Sys.maxrss is already in bytes).
"""
rss_bytes() = Int64(Sys.maxrss())

"""
Run `f()` once, returning `(value, wall_s, allocated_bytes, rss_delta_bytes)`.

RSS delta uses the process high-water mark, so it only grows within a process —
callers should measure heavier solvers first when comparing peak RSS, or treat
allocated bytes as the primary allocator metric.
"""
function measure_call(f)
    GC.gc()
    GC.gc()
    rss_before = rss_bytes()
    timed = @timed f()
    rss_after = rss_bytes()
    return (
        value = timed.value,
        time = timed.time,
        allocated = Int64(timed.bytes),
        rss_delta = max(Int64(0), rss_after - rss_before),
        rss_peak = rss_after,
    )
end

function print_metric_block(title::AbstractString; kwargs...)
    println()
    println("── $title ──")
    for (k, v) in kwargs
        if v isa Integer && (endswith(String(k), "bytes") || endswith(String(k), "rss") ||
                             endswith(String(k), "allocated") || endswith(String(k), "memory"))
            println("  $(rpad(String(k), 28)) $(format_bytes(v))  ($v bytes)")
        elseif v isa AbstractFloat && endswith(String(k), "s")
            println("  $(rpad(String(k), 28)) $(format_seconds(v))s")
        elseif v === nothing
            println("  $(rpad(String(k), 28)) —")
        else
            println("  $(rpad(String(k), 28)) $v")
        end
    end
end

"""
θ-feasible and at least as many edges as the pivot optimum.
"""
function is_optimal_solution(fg::FrozenBipartite, sol::SubGraph, k::Int, θ::Int, opt_edges::Int)
    Subgraph.missing_edges(fg, sol) > k && return false
    (length(sol.U) ≥ θ && length(sol.V) ≥ θ) || (opt_edges == 0 && Subgraph.edge_count(fg, sol) == 0) || return false
    return Subgraph.edge_count(fg, sol) >= opt_edges
end

function describe_solution(fg::FrozenBipartite, sol::SubGraph, k::Int)
    edges = Subgraph.edge_count(fg, sol)
    missing = Subgraph.missing_edges(fg, sol)
    return "|U|=$(length(sol.U)) |V|=$(length(sol.V)) edges=$edges missing=$missing (k=$k)"
end

"""
Parse `--benchmark=aco,pivot,heuristic,ga` into a Set of symbols.
"""
function parse_benchmark_targets(raw::AbstractString)
    targets = Set{Symbol}()
    for part in split(raw, ',')
        name = lowercase(strip(part))
        isempty(name) && continue
        if name == "aco"
            push!(targets, :aco)
        elseif name == "pivot"
            push!(targets, :pivot)
        elseif name == "heuristic"
            push!(targets, :heuristic)
        elseif name == "ga"
            push!(targets, :ga)
        else
            throw(ArgumentError("Unknown benchmark target '$name' (expected aco, pivot, heuristic, and/or ga)"))
        end
    end
    isempty(targets) && throw(ArgumentError("--benchmark= needs at least one of: aco, pivot, heuristic, ga"))
    return targets
end

"""
Return `--save=<path>` if present, otherwise `nothing`.
"""
function parse_benchmark_save()
    for arg in ARGS
        if startswith(arg, "--save=")
            path = strip(split(arg, "=", limit=2)[2])
            isempty(path) && throw(ArgumentError("--save= requires a file name or path"))
            return path
        elseif arg == "--save"
            throw(ArgumentError("--save requires a value, e.g. --save=run.json"))
        end
    end
    return nothing
end

"""
Resolve a `--save=` argument to an absolute path.

Bare filenames go under `results/`; paths with a directory component are kept
(relative to the process cwd, or absolute as given). A missing `.json` suffix
is appended.
"""
function resolve_benchmark_save_path(raw::AbstractString)
    path = String(raw)
    if !endswith(lowercase(path), ".json")
        path *= ".json"
    end
    if dirname(path) == "" || dirname(path) == "."
        path = joinpath(BENCHMARK_RESULTS_DIR, basename(path))
    end
    return abspath(path)
end

"""
Warm up compilation on a tiny synthetic graph so measured runs exclude JIT cost
without paying for a full solve on the real instance.
"""
function warmup_benchmarks!(k::Int, θ::Int, reduction::ReductionMode.T, aco_options;
    targets::Set{Symbol}, ga_options=(; N=10, O=2, k_mutate=0.02, generations=500))
    pheremone, num_ants, _, evaporation, num_subspecies = aco_options

    g = BipartiteGraph{Int}()
    # Small dense biclique plus a few dangling edges — enough to touch reductions,
    # branching, and ACO without dominating wall time.
    for u in 1:8, v in 1:8
        add_edge!(g, u, v, u * 100 + v)
    end
    add_edge!(g, 9, 1, 901)
    add_edge!(g, 1, 9, 109)

    k_w = min(k, 2)
    θ_w = min(θ, 3)
    @assert θ_w > k_w

    if :pivot in targets
        find_kmdb!(deepcopy(g), true, BranchMode.pivot, k_w, θ_w, reduction)
    end
    if :aco in targets
        aco(deepcopy(g), pheremone, min(num_ants, 2), 1, evaporation, k_w, θ_w, num_subspecies;
            parallelize=false, force_gc=false, reduction=reduction)
    end
    if :heuristic in targets
        g_h = deepcopy(g)
        fg_h = if reduction == ReductionMode.none
            freeze(g_h)
        else
            apply_graph_reductions!(g_h, k_w, θ_w, nothing, nothing, true, reduction)
        end
        if length(fg_h.u_ids) >= θ_w && length(fg_h.v_ids) >= θ_w
            theta_based_heuristic(fg_h, k_w, θ_w; return_invalid=true)
        end
    end
    if :ga in targets
        global U = Set{Int}()
        global V = Set{Int}()
        ga(deepcopy(g), k_w, θ_w, min(ga_options.N, 4), ga_options.O, ga_options.k_mutate, 2;
            reduction=reduction, repair=RepairMode.mixed)
    end
    GC.gc()
    return nothing
end

function benchmark_graph_memory(g::BipartiteGraph, edge_count::Int, k::Int, θ::Int,
    reduction::ReductionMode.T)
    mutable_bytes = Base.summarysize(g)
    g_frozen_src = deepcopy(g)
    fg = apply_graph_reductions!(g_frozen_src, k, θ, nothing, nothing, true, reduction)
    frozen_bytes = Base.summarysize(fg)
    print_metric_block("Graph structure";
        nU = length(g.adjU),
        nV = length(g.adjV),
        edges = edge_count,
        mutable_graph_bytes = mutable_bytes,
        frozen_after_reduction_bytes = frozen_bytes,
        reduced_nU = length(fg.u_ids),
        reduced_nV = length(fg.v_ids),
    )
    return (; mutable_bytes, frozen_bytes, fg)
end

function benchmark_pivot!(g::BipartiteGraph, k::Int, θ::Int, reduction::ReductionMode.T)
    println()
    println("Running pivot (branch-and-bound)…")
    g_run = deepcopy(g)
    # Peak RSS: leave force allocations visible (no mid-run GC beyond Julia's own).
    m = measure_call() do
        find_kmdb!(g_run, true, BranchMode.pivot, k, θ, reduction)
    end
    sols = m.value
    sol = isempty(sols) ? SubGraph() : first(sols)
    # Edge counts against the post-reduction graph used by search / ACO.
    g_eval = deepcopy(g)
    fg_eval = apply_graph_reductions!(g_eval, k, θ, nothing, nothing, true, reduction)
    opt_edges = Subgraph.edge_count(fg_eval, sol)
    print_metric_block("Pivot";
        wall_time_s = m.time,
        allocated_bytes = m.allocated,
        rss_delta_bytes = m.rss_delta,
        rss_peak_bytes = m.rss_peak,
        solution = describe_solution(fg_eval, sol, k),
        optimal_edges = opt_edges,
    )
    return (; sol, opt_edges, fg_eval, time = m.time, allocated = m.allocated,
        rss_delta = m.rss_delta)
end

"""
Run the θ-based construction heuristic (`theta_based_heuristic`).
"""
function benchmark_heuristic!(g::BipartiteGraph, k::Int, θ::Int, reduction::ReductionMode.T;
    opt_edges::Union{Nothing,Int}=nothing)
    println()
    println("Running θ-heuristic…")
    g_run = deepcopy(g)
    m = measure_call() do
        fg = if reduction == ReductionMode.none
            freeze(g_run)
        else
            apply_graph_reductions!(g_run, k, θ, nothing, nothing, true, reduction)
        end
        if length(fg.u_ids) < θ || length(fg.v_ids) < θ
            return (fg, SubGraph(Set(), Set()))
        end
        return (fg, theta_based_heuristic(fg, k, θ; return_invalid=true))
    end
    fg_eval, sol = m.value
    final_edges = Subgraph.edge_count(fg_eval, sol)
    matched = opt_edges !== nothing && is_optimal_solution(fg_eval, sol, k, θ, opt_edges)

    print_metric_block("θ-heuristic";
        wall_time_s = m.time,
        allocated_bytes = m.allocated,
        rss_delta_bytes = m.rss_delta,
        rss_peak_bytes = m.rss_peak,
        solution = describe_solution(fg_eval, sol, k),
        final_edges = final_edges,
        optimal_edges = opt_edges,
        matched_optimal = matched,
    )

    return (; sol, fg_eval, time = m.time, allocated = m.allocated, rss_delta = m.rss_delta,
        final_edges, opt_edges, matched_optimal = matched)
end

"""
Run the genetic algorithm (`ga`), matching load.jl defaults unless `ga_options` overrides.
"""
function benchmark_ga!(g::BipartiteGraph, k::Int, θ::Int, reduction::ReductionMode.T,
    ga_options; opt_edges::Union{Nothing,Int}=nothing)
    N = ga_options.N
    O = ga_options.O
    k_mutate = ga_options.k_mutate
    generations = ga_options.generations

    println()
    println("Running GA (N=$N O=$O generations=$generations k_mutate=$k_mutate repair=mixed)…")

    # Reset GA globals that accumulate across generations/runs.
    global U = Set{Int}()
    global V = Set{Int}()

    g_run = deepcopy(g)
    m = measure_call() do
        ga(g_run, k, θ, N, O, k_mutate, generations; reduction=reduction, repair=RepairMode.mixed)
    end
    sol = m.value

    g_eval = deepcopy(g)
    fg_eval = if reduction == ReductionMode.none
        freeze(g_eval)
    else
        apply_graph_reductions!(g_eval, k, θ, nothing, nothing, true, reduction)
    end
    final_edges = Subgraph.edge_count(fg_eval, sol)
    matched = opt_edges !== nothing && is_optimal_solution(fg_eval, sol, k, θ, opt_edges)

    print_metric_block("GA";
        wall_time_s = m.time,
        allocated_bytes = m.allocated,
        rss_delta_bytes = m.rss_delta,
        rss_peak_bytes = m.rss_peak,
        population_N = N,
        generations = generations,
        solution = describe_solution(fg_eval, sol, k),
        final_edges = final_edges,
        optimal_edges = opt_edges,
        matched_optimal = matched,
    )

    return (; sol, fg_eval, time = m.time, allocated = m.allocated, rss_delta = m.rss_delta,
        N, generations, final_edges, opt_edges, matched_optimal = matched)
end

"""
Run ACO. When `opt_edges` is provided (from pivot), track the first iteration
that matches the optimum and early-stop.

Pass `reduction=ReductionMode.none` when `g` is already reduced so ACO does not
re-run graph reductions inside the timed call.
"""
function benchmark_aco!(g::BipartiteGraph, k::Int, θ::Int, aco_options;
    opt_edges::Union{Nothing,Int}=nothing, early_stop::Bool=true,
    reduction::ReductionMode.T=ReductionMode.all_reductions)
    pheremone, num_ants, num_iterations, evaporation, num_subspecies = aco_options
    prefer_smaller_side = get(aco_options, :prefer_smaller_side, true)
    neighbor_scope_limit = get(aco_options, :neighbor_scope_limit, true)
    elite_seed = get(aco_options, :elite_seed, true)
    elite_seed_ants = get(aco_options, :elite_seed_ants, 3)
    elite_seed_remove = get(aco_options, :elite_seed_remove, 2)

    first_hit = Ref{Union{Nothing,Int}}(nothing)

    println()
    println("Running ACO (ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies prefer_smaller_side=$prefer_smaller_side neighbor_scope_limit=$neighbor_scope_limit elite_seed=$elite_seed)…")

    g_run = deepcopy(g)
    m = measure_call() do
        aco(g_run, pheremone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
            parallelize=false,
            force_gc=false,  # keep peak memory meaningful for the paper
            prefer_smaller_side=prefer_smaller_side,
            neighbor_scope_limit=neighbor_scope_limit,
            elite_seed=elite_seed,
            elite_seed_ants=elite_seed_ants,
            elite_seed_remove=elite_seed_remove,
            reduction=reduction,
            iteration_callback = (iter, best_compact, compact_fg, _remapping, _elapsed_s) -> begin
                if opt_edges === nothing || first_hit[] !== nothing
                    return true
                end
                # compact_fg uses compact ids, matching best_compact.
                edges = Subgraph.edge_count(compact_fg, best_compact)
                missing = Subgraph.missing_edges(compact_fg, best_compact)
                θ_ok = (length(best_compact.U) ≥ θ && length(best_compact.V) ≥ θ) ||
                    (opt_edges == 0 && edges == 0)
                if missing <= k && θ_ok && edges >= opt_edges
                    first_hit[] = iter
                    return !early_stop  # false => stop
                end
                return true
            end)
    end

    sols, best_iterations, best_times, _pheromones, _remapping = m.value
    g_eval = deepcopy(g)
    fg_eval = apply_graph_reductions!(g_eval, k, θ, nothing, nothing, true, reduction)
    best_idx = argmax(i -> Subgraph.edge_count(fg_eval, sols[i]), eachindex(sols))
    sol = sols[best_idx]
    iterations_to_best = best_iterations[best_idx]
    time_to_best = best_times[best_idx]
    found = first_hit[] !== nothing

    # Report solution quality on a reduced graph matching ACO's reductions.
    final_edges = Subgraph.edge_count(fg_eval, sol)

    print_metric_block("ACO";
        wall_time_s = m.time,
        time_to_best_s = time_to_best,
        allocated_bytes = m.allocated,
        rss_delta_bytes = m.rss_delta,
        rss_peak_bytes = m.rss_peak,
        ants = num_ants,
        iterations_budget = num_iterations,
        iterations_to_best = iterations_to_best,
        iterations_to_optimal = found ? first_hit[] : nothing,
        best_source = iterations_to_best == 0 ?
            (Subgraph.vertex_count(sol) == 0 ?
                "no ACO solution" :
                "forced-seed incumbent (ants never improved)") :
            "ACO search (iteration $iterations_to_best)",
        matched_optimal_note = found ?
            "matched optimum at iteration $(first_hit[])" :
            (opt_edges === nothing ? "no pivot optimum provided" : "optimum not reached within budget"),
        solution = describe_solution(fg_eval, sol, k),
        final_edges = final_edges,
        optimal_edges = opt_edges,
        matched_optimal = found || (opt_edges !== nothing && final_edges >= opt_edges &&
            length(sol.U) ≥ θ && length(sol.V) ≥ θ),
    )

    if found
        println()
        println("  ACO first matched pivot optimum at iteration $(first_hit[]) " *
                "with $num_ants ants ($(format_bytes(m.allocated)) allocated, " *
                "$(format_seconds(time_to_best))s to best, " *
                "$(format_seconds(m.time))s total wall).")
    elseif opt_edges !== nothing
        println()
        println("  ACO did not match pivot optimum ($opt_edges edges) within " *
                "$num_iterations iterations × $num_ants ants.")
    end

    println()
    if iterations_to_best == 0 && Subgraph.vertex_count(sol) == 0
        println("  ACO found no solution within budget $num_iterations " *
                "(total wall $(format_seconds(m.time))s).")
    elseif iterations_to_best == 0
        println("  ACO never improved on the forced-seed incumbent (still best at iter 0 / " *
                "$(format_seconds(time_to_best))s; budget $num_iterations, " *
                "total wall $(format_seconds(m.time))s).")
    else
        println("  ACO final solution found at iteration $iterations_to_best " *
                "($(format_seconds(time_to_best))s; budget $num_iterations, " *
                "total wall $(format_seconds(m.time))s).")
    end

    return (; sol, time = m.time, time_to_best = time_to_best,
        allocated = m.allocated, rss_delta = m.rss_delta,
        first_hit_iteration = first_hit[], iterations_to_best = iterations_to_best,
        ants = num_ants, iterations_budget = num_iterations, final_edges, opt_edges)
end

"""
Entry point used by load.jl when `--benchmark=...` is set.

Graph reductions run once up front (timed separately). Solvers then receive the
already-reduced mutable graph with `ReductionMode.none` so their wall times
exclude reduction cost.
"""
function run_benchmarks!(g::BipartiteGraph, edge_count::Int, targets::Set{Symbol},
    k::Int, θ::Int, reduction::ReductionMode.T, aco_options;
    ga_options=(; N=10, O=2, k_mutate=0.02, generations=500))
    println()
    println("==================== BENCHMARK ====================")
    println("targets=$(join(sort!(collect(String(t) for t in targets)), ","))  k=$k  θ=$θ")
    println("reduction=$reduction  (applied once; solvers use none)")

    println()
    println("Warming up (excluded from timings)…")
    warmup_benchmarks!(k, θ, reduction, aco_options; targets=targets, ga_options=ga_options)

    mutable_bytes = Base.summarysize(g)
    g_reduced = deepcopy(g)
    println()
    println("Reducing graph once (shared across solvers)…")
    m_red = measure_call() do
        apply_graph_reductions!(g_reduced, k, θ, nothing, nothing, true, reduction)
    end
    fg = m_red.value
    frozen_bytes = Base.summarysize(fg)
    print_metric_block("Graph reduction";
        wall_time_s = m_red.time,
        allocated_bytes = m_red.allocated,
        rss_delta_bytes = m_red.rss_delta,
        reduced_nU = length(fg.u_ids),
        reduced_nV = length(fg.v_ids),
    )
    print_metric_block("Graph structure";
        nU = length(g.adjU),
        nV = length(g.adjV),
        edges = edge_count,
        mutable_graph_bytes = mutable_bytes,
        frozen_after_reduction_bytes = frozen_bytes,
        reduced_nU = length(fg.u_ids),
        reduced_nV = length(fg.v_ids),
    )
    graph_stats = (; mutable_bytes, frozen_bytes, fg,
        reduction_time = m_red.time, reduction_allocated = m_red.allocated,
        reduction_rss_delta = m_red.rss_delta)

    # Solvers search the already-reduced graph; skip re-reduction in each call.
    solver_reduction = ReductionMode.none

    pivot_stats = nothing
    if :pivot in targets
        pivot_stats = benchmark_pivot!(g_reduced, k, θ, solver_reduction)
    end

    opt_edges = pivot_stats === nothing ? nothing : pivot_stats.opt_edges

    heuristic_stats = nothing
    if :heuristic in targets
        heuristic_stats = benchmark_heuristic!(g_reduced, k, θ, solver_reduction; opt_edges=opt_edges)
    end

    ga_stats = nothing
    if :ga in targets
        ga_stats = benchmark_ga!(g_reduced, k, θ, solver_reduction, ga_options; opt_edges=opt_edges)
    end

    aco_stats = nothing
    if :aco in targets
        aco_stats = benchmark_aco!(g_reduced, k, θ, aco_options;
            opt_edges=opt_edges, early_stop=true, reduction=solver_reduction)
    end

    println()
    println("==================== SUMMARY ======================")
    println("  reduction time       : $(format_seconds(graph_stats.reduction_time))s")
    println("  graph mutable memory : $(format_bytes(graph_stats.mutable_bytes))")
    println("  graph frozen memory  : $(format_bytes(graph_stats.frozen_bytes))")
    if pivot_stats !== nothing
        println("  pivot time           : $(format_seconds(pivot_stats.time))s")
        println("  pivot allocated      : $(format_bytes(pivot_stats.allocated))")
        println("  pivot RSS Δ          : $(format_bytes(pivot_stats.rss_delta))")
        println("  pivot optimum edges  : $(pivot_stats.opt_edges)")
    end
    if heuristic_stats !== nothing
        println("  heuristic time       : $(format_seconds(heuristic_stats.time))s")
        println("  heuristic allocated  : $(format_bytes(heuristic_stats.allocated))")
        println("  heuristic RSS Δ      : $(format_bytes(heuristic_stats.rss_delta))")
        println("  heuristic edges      : $(heuristic_stats.final_edges)" *
                (heuristic_stats.opt_edges === nothing ? "" : " / $(heuristic_stats.opt_edges) optimal"))
    end
    if ga_stats !== nothing
        println("  GA time              : $(format_seconds(ga_stats.time))s")
        println("  GA allocated         : $(format_bytes(ga_stats.allocated))")
        println("  GA RSS Δ             : $(format_bytes(ga_stats.rss_delta))")
        println("  GA edges             : $(ga_stats.final_edges)" *
                (ga_stats.opt_edges === nothing ? "" : " / $(ga_stats.opt_edges) optimal"))
    end
    if aco_stats !== nothing
        println("  ACO time             : $(format_seconds(aco_stats.time))s")
        println("  ACO time → best      : $(format_seconds(aco_stats.time_to_best))s" *
                (aco_stats.iterations_to_best == 0 ?
                    (Subgraph.vertex_count(aco_stats.sol) == 0 ?
                        "  (no ACO solution)" :
                        "  (forced-seed incumbent; ants never improved)") :
                    ""))
        println("  ACO allocated        : $(format_bytes(aco_stats.allocated))")
        println("  ACO RSS Δ           : $(format_bytes(aco_stats.rss_delta))")
        println("  ACO ants             : $(aco_stats.ants)")
        println("  ACO iters → best     : $(aco_stats.iterations_to_best)" *
                (aco_stats.iterations_to_best == 0 ?
                    (Subgraph.vertex_count(aco_stats.sol) == 0 ?
                        "  (no ACO solution)" :
                        "  (forced-seed incumbent)") :
                    ""))
        println("  ACO iters → optimal  : $(aco_stats.first_hit_iteration === nothing ? "not found" : aco_stats.first_hit_iteration)")
        println("  ACO final edges      : $(aco_stats.final_edges)" *
                (aco_stats.opt_edges === nothing ? "" : " / $(aco_stats.opt_edges) optimal"))
    end
    println("===================================================")

    return (; graph_stats, pivot_stats, heuristic_stats, ga_stats, aco_stats)
end

# Minimal JSON writer (avoids requiring JSON3 in load.jl).
function _bench_json_escape(s::AbstractString)
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

_bench_json_val(x::Bool) = x ? "true" : "false"
_bench_json_val(x::Integer) = string(x)
_bench_json_val(x::AbstractFloat) = isfinite(x) ? format_seconds(x) : "null"
_bench_json_val(::Nothing) = "null"
_bench_json_val(x::AbstractString) = "\"" * _bench_json_escape(x) * "\""
_bench_json_val(x::Symbol) = _bench_json_val(string(x))
function _bench_json_val(xs::AbstractVector)
    return "[" * join((_bench_json_val(x) for x in xs), ",") * "]"
end
function _bench_json_val(d::AbstractDict)
    return "{" * join(("\"$(_bench_json_escape(string(k)))\":$(_bench_json_val(v))" for (k, v) in d), ",") * "}"
end

"""
Convert benchmark named-tuples into plain Dict/Vector/Number/String/nothing
trees suitable for JSON (drops live SubGraph / FrozenBipartite objects).
"""
function benchmark_results_to_dict(results; k::Int=0, θ::Int=0,
    dataset::AbstractString="", targets=nothing, seed=nothing,
    reduction=nothing, edge_count::Union{Nothing,Int}=nothing)
    function sol_summary(fg, sol)
        sol === nothing && return nothing
        return Dict(
            "nU" => length(sol.U),
            "nV" => length(sol.V),
            "edges" => Subgraph.edge_count(fg, sol),
            "missing" => Subgraph.missing_edges(fg, sol),
            "k" => k,
        )
    end

    out = Dict{String,Any}(
        "dataset" => String(dataset),
        "k" => k,
        "theta" => θ,
        "targets" => targets === nothing ? String[] : sort!(collect(String(t) for t in targets)),
        "seed" => seed === nothing ? nothing : string(seed),
        "reduction" => reduction === nothing ? nothing : string(reduction),
        "edge_count" => edge_count,
    )

    gs = results.graph_stats
    out["graph"] = Dict(
        "mutable_bytes" => gs.mutable_bytes,
        "frozen_bytes" => gs.frozen_bytes,
        "reduced_nU" => length(gs.fg.u_ids),
        "reduced_nV" => length(gs.fg.v_ids),
        "reduction_time_s" => get(gs, :reduction_time, nothing),
        "reduction_allocated_bytes" => get(gs, :reduction_allocated, nothing),
        "reduction_rss_delta_bytes" => get(gs, :reduction_rss_delta, nothing),
    )

    if results.pivot_stats !== nothing
        ps = results.pivot_stats
        out["pivot"] = Dict(
            "wall_time_s" => ps.time,
            "allocated_bytes" => ps.allocated,
            "rss_delta_bytes" => ps.rss_delta,
            "optimal_edges" => ps.opt_edges,
            "solution" => sol_summary(ps.fg_eval, ps.sol),
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
            "solution" => sol_summary(hs.fg_eval, hs.sol),
        )
    end

    if results.ga_stats !== nothing
        gs_ga = results.ga_stats
        out["ga"] = Dict(
            "wall_time_s" => gs_ga.time,
            "allocated_bytes" => gs_ga.allocated,
            "rss_delta_bytes" => gs_ga.rss_delta,
            "population_N" => gs_ga.N,
            "generations" => gs_ga.generations,
            "final_edges" => gs_ga.final_edges,
            "optimal_edges" => gs_ga.opt_edges,
            "matched_optimal" => gs_ga.matched_optimal,
            "solution" => sol_summary(gs_ga.fg_eval, gs_ga.sol),
        )
    end

    if results.aco_stats !== nothing
        as = results.aco_stats
        out["aco"] = Dict(
            "wall_time_s" => as.time,
            "time_to_best_s" => as.time_to_best,
            "allocated_bytes" => as.allocated,
            "rss_delta_bytes" => as.rss_delta,
            "ants" => as.ants,
            "iterations_budget" => as.iterations_budget,
            "iterations_to_best" => as.iterations_to_best,
            "iterations_to_optimal" => as.first_hit_iteration,
            "final_edges" => as.final_edges,
            "optimal_edges" => as.opt_edges,
            "nU" => length(as.sol.U),
            "nV" => length(as.sol.V),
        )
    end

    return out
end

"""
Write benchmark results JSON to `path` (creates parent directories as needed).
"""
function save_benchmark_json(path::AbstractString, payload::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        print(io, _bench_json_val(payload))
        println(io)
    end
    println("Wrote benchmark results → $path")
    return path
end
