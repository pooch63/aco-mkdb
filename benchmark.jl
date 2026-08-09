#=
=================================================================================
Paper / algorithm benchmarks for a single loaded graph.

Invoked from load.jl via:
  julia load.jl <dataset> --benchmark=aco,pivot
  julia load.jl <dataset> --benchmark=pivot
  julia load.jl <dataset> --benchmark=aco

Reports:
  - memory of the loaded (and frozen) graph structure
  - wall time + allocated bytes + peak-RSS delta for pivot and/or ACO
  - when pivot is also run: iterations / ants / time until ACO first matches
    the pivot optimum (by edge count + θ-feasibility), then early-stops
=================================================================================
=#

const __BENCHMARK_JL__ = true

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
            println("  $(rpad(String(k), 28)) $(round(v; digits=4))s")
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
Parse `--benchmark=aco,pivot` into a Set of symbols (`:aco`, `:pivot`).
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
        else
            throw(ArgumentError("Unknown benchmark target '$name' (expected aco and/or pivot)"))
        end
    end
    isempty(targets) && throw(ArgumentError("--benchmark= needs at least one of: aco, pivot"))
    return targets
end

"""
Warm up compilation on a tiny synthetic graph so measured runs exclude JIT cost
without paying for a full solve on the real instance.
"""
function warmup_benchmarks!(k::Int, θ::Int, reduction::ReductionMode.T, aco_options;
    targets::Set{Symbol})
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
            parallelize=false, force_gc=false)
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
Run ACO. When `opt_edges` is provided (from pivot), track the first iteration
that matches the optimum and early-stop.
"""
function benchmark_aco!(g::BipartiteGraph, k::Int, θ::Int, aco_options;
    opt_edges::Union{Nothing,Int}=nothing, early_stop::Bool=true)
    pheremone, num_ants, num_iterations, evaporation, num_subspecies = aco_options
    prefer_smaller_side = get(aco_options, :prefer_smaller_side, true)
    elite_seed = get(aco_options, :elite_seed, true)
    elite_seed_ants = get(aco_options, :elite_seed_ants, 3)
    elite_seed_remove = get(aco_options, :elite_seed_remove, 2)

    first_hit = Ref{Union{Nothing,Int}}(nothing)

    println()
    println("Running ACO (ants=$num_ants iterations=$num_iterations pheromone=$pheremone evaporation=$evaporation subspecies=$num_subspecies prefer_smaller_side=$prefer_smaller_side elite_seed=$elite_seed)…")

    g_run = deepcopy(g)
    m = measure_call() do
        aco(g_run, pheremone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
            parallelize=false,
            force_gc=false,  # keep peak memory meaningful for the paper
            prefer_smaller_side=prefer_smaller_side,
            elite_seed=elite_seed,
            elite_seed_ants=elite_seed_ants,
            elite_seed_remove=elite_seed_remove,
            iteration_callback = (iter, best_compact, compact_fg, _remapping) -> begin
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

    sols = m.value
    g_eval = deepcopy(g)
    fg_eval = apply_graph_reductions!(g_eval, k, θ, nothing, nothing, true, ReductionMode.all_reductions)
    sol = argmax(s -> Subgraph.edge_count(fg_eval, s), sols)
    found = first_hit[] !== nothing

    # Report solution quality on a reduced graph matching ACO's reductions.
    final_edges = Subgraph.edge_count(fg_eval, sol)

    print_metric_block("ACO";
        wall_time_s = m.time,
        allocated_bytes = m.allocated,
        rss_delta_bytes = m.rss_delta,
        rss_peak_bytes = m.rss_peak,
        ants = num_ants,
        iterations_budget = num_iterations,
        iterations_to_optimal = found ? first_hit[] : nothing,
        time_to_optimal_note = found ?
            "early-stopped at iteration $(first_hit[]) (wall time above includes time-to-hit)" :
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
                "$(round(m.time; digits=4))s wall).")
    elseif opt_edges !== nothing
        println()
        println("  ACO did not match pivot optimum ($opt_edges edges) within " *
                "$num_iterations iterations × $num_ants ants.")
    end

    return (; sol, time = m.time, allocated = m.allocated, rss_delta = m.rss_delta,
        first_hit_iteration = first_hit[], ants = num_ants,
        iterations_budget = num_iterations, final_edges, opt_edges)
end

"""
Entry point used by load.jl when `--benchmark=...` is set.
"""
function run_benchmarks!(g::BipartiteGraph, edge_count::Int, targets::Set{Symbol},
    k::Int, θ::Int, reduction::ReductionMode.T, aco_options)
    println()
    println("==================== BENCHMARK ====================")
    println("targets=$(join(sort!(collect(String(t) for t in targets)), ","))  k=$k  θ=$θ")
    println("reduction=$reduction")

    println()
    println("Warming up (excluded from timings)…")
    warmup_benchmarks!(k, θ, reduction, aco_options; targets=targets)

    graph_stats = benchmark_graph_memory(g, edge_count, k, θ, reduction)

    pivot_stats = nothing
    if :pivot in targets
        pivot_stats = benchmark_pivot!(g, k, θ, reduction)
    end

    aco_stats = nothing
    if :aco in targets
        opt_edges = pivot_stats === nothing ? nothing : pivot_stats.opt_edges
        aco_stats = benchmark_aco!(g, k, θ, aco_options;
            opt_edges=opt_edges, early_stop=true)
    end

    println()
    println("==================== SUMMARY ======================")
    println("  graph mutable memory : $(format_bytes(graph_stats.mutable_bytes))")
    println("  graph frozen memory  : $(format_bytes(graph_stats.frozen_bytes))")
    if pivot_stats !== nothing
        println("  pivot time           : $(round(pivot_stats.time; digits=4))s")
        println("  pivot allocated      : $(format_bytes(pivot_stats.allocated))")
        println("  pivot RSS Δ          : $(format_bytes(pivot_stats.rss_delta))")
        println("  pivot optimum edges  : $(pivot_stats.opt_edges)")
    end
    if aco_stats !== nothing
        println("  ACO time             : $(round(aco_stats.time; digits=4))s")
        println("  ACO allocated        : $(format_bytes(aco_stats.allocated))")
        println("  ACO RSS Δ           : $(format_bytes(aco_stats.rss_delta))")
        println("  ACO ants             : $(aco_stats.ants)")
        println("  ACO iters → optimal  : $(aco_stats.first_hit_iteration === nothing ? "not found" : aco_stats.first_hit_iteration)")
        println("  ACO final edges      : $(aco_stats.final_edges)" *
                (aco_stats.opt_edges === nothing ? "" : " / $(aco_stats.opt_edges) optimal"))
    end
    println("===================================================")

    return (; graph_stats, pivot_stats, aco_stats)
end

"""
Convert benchmark named-tuples into plain Dict/Vector/Number/String/nothing
trees suitable for JSON3.write (drops live SubGraph / FrozenBipartite objects).
"""
function benchmark_results_to_dict(results; k::Int=0)
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

    out = Dict{String,Any}()

    gs = results.graph_stats
    out["graph"] = Dict(
        "mutable_bytes" => gs.mutable_bytes,
        "frozen_bytes" => gs.frozen_bytes,
        "reduced_nU" => length(gs.fg.u_ids),
        "reduced_nV" => length(gs.fg.v_ids),
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

    if results.aco_stats !== nothing
        as = results.aco_stats
        out["aco"] = Dict(
            "wall_time_s" => as.time,
            "allocated_bytes" => as.allocated,
            "rss_delta_bytes" => as.rss_delta,
            "ants" => as.ants,
            "iterations_budget" => as.iterations_budget,
            "iterations_to_optimal" => as.first_hit_iteration,
            "final_edges" => as.final_edges,
            "optimal_edges" => as.opt_edges,
            "nU" => length(as.sol.U),
            "nV" => length(as.sol.V),
        )
    end

    return out
end
