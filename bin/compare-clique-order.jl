#=
Compare clique-order with the θ-heuristic on real graphs.

Picks indexed graphs under data/ in ascending |E| and scores both solvers on
the same reduced instance. Each solver runs on both orientations: the
θ-heuristic grows U and grows V, and clique-order searches side U and side V.
The better θ-feasible solution (more edges) is kept from each pair. A graph
whose reduced sides are each smaller than θ cannot hold a θ-feasible
biclique, so it is skipped and does not count toward --N. Reported times are
the sum of the two orientations. A win is the paper rule: θ-feasible, k-valid,
and strictly more edges.

    julia bin/compare-clique-order.jl
    julia bin/compare-clique-order.jl --N=20
    julia bin/compare-clique-order.jl --N=10 --offset=10 --prefix=konect-small
    julia bin/compare-clique-order.jl --N=5 --k=2 --theta=5 --reduce=lo \
        --save=results/clique_order_vs_theta.json

--N=          how many graphs to compare (default 10); reduced-away graphs are skipped
--offset=     skip this many smallest graphs before scanning (default 0)
--prefix=     restrict to a dataset prefix (konect-small, amazon, …)
--k=          defect budget (default 2)
--theta=      minimum side size (default 5)
--reduce=     shared reduction before both solvers: lo | hi | none (default lo)
--save=       write the per-graph rows as JSON
=#

const ROOT = dirname(@__DIR__)
const SRC = joinpath(ROOT, "src")

isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))
isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))

using JSON3
using Printf

function _flag(name::AbstractString)
    prefix = "--$name="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return strip(split(arg, "=", limit=2)[2])
    end
    return nothing
end

function _int_flag(name::AbstractString, default::Int)
    raw = _flag(name)
    raw === nothing && return default
    return parse(Int, raw)
end

function _reduction_flag()
    raw = _flag("reduce")
    raw === nothing && return ReductionMode.simple
    raw in ("lo", "simple") && return ReductionMode.simple
    raw in ("hi", "all", "all_reductions") && return ReductionMode.all_reductions
    raw == "none" && return ReductionMode.none
    error("Unsupported --reduce=$raw (lo|hi|none)")
end

_reduction_name(r::ReductionMode.T) =
    r == ReductionMode.simple ? "lo" :
    r == ReductionMode.all_reductions ? "hi" : "none"

function _usage()
    println("""
    julia bin/compare-clique-order.jl [--N=10] [--offset=0] [--prefix=] \\
        [--k=2] [--theta=5] [--reduce=lo|hi|none] [--save=PATH]
    """)
end

"""Indexed graphs after `offset`, smallest |E| first. The caller stops after `n` comparisons."""
function candidate_graphs(n::Int, offset::Int, prefix::Union{Nothing,String})
    n > 0 || error("--N must be positive, got $n")
    offset >= 0 || error("--offset must be ≥ 0, got $offset")
    data_root = joinpath(ROOT, "data")
    entries = order_graphs_by_edges(; data_root=data_root, ascending=true, prefix=prefix)
    start = min(offset + 1, length(entries) + 1)
    return entries, entries[start:end]
end

function _swap_frozen(fg::FrozenBipartite{T}) where {T}
    return FrozenBipartite{T}(
        fg.v_ids, fg.u_ids,
        fg.v_index, fg.u_index,
        fg.v_offsets, fg.u_adj, fg.v_edge_data,
        fg.u_offsets, fg.v_adj, fg.edge_data,
    )
end

"""Keep `sol_v` only when it beats `sol_u` on edge count among θ-feasible packs."""
function _pick_side(fg::FrozenBipartite, sol_u::SubGraph, sol_v::SubGraph, θ::Int)
    better_than_best(fg, sol_v, sol_u, θ) && return sol_v, "V"
    return sol_u, "U"
end

function _warmup!()
    g = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(g, u, v, nothing)
    end
    fg = freeze(g)
    theta_based_heuristic(fg, 2, 5; return_invalid=true)
    theta_based_heuristic(_swap_frozen(fg), 2, 5; return_invalid=true)
    clique_order(deepcopy(g), 2, 5; side=:U)
    clique_order(deepcopy(g), 2, 5; side=:V)
    return nothing
end

function _load_quiet(path::AbstractString)
    return redirect_stdout(devnull) do
        load_bipartite_graph(String(path))[1]
    end
end

function _pct(delta::Int, base::Int)
    base == 0 && return nothing
    return 100.0 * delta / base
end

"""Load `entry` and apply the shared reduction. The returned graph is that reduced instance."""
function load_reduced(entry, k::Int, θ::Int, reduction::ReductionMode.T)
    g = _load_quiet(entry.path)
    work = deepcopy(g)
    fg = apply_graph_reductions!(work, k, θ, nothing, nothing, true, reduction)
    return work, fg
end

"""
Score both orientations of each solver on an already reduced graph.

The θ-heuristic grows U, then grows V on a swapped copy. Clique-order searches
`:U` and `:V` from separate copies, each with its own `super_reduce!`. Times
are those two calls; the kept solution is the one with more edges once it is
θ-feasible.
"""
function compare_one(entry, work, fg, k::Int, θ::Int)
    t_hu = @elapsed sol_hu = theta_based_heuristic(fg, k, θ; return_invalid=true)
    t_hv = @elapsed sol_hv_s = theta_based_heuristic(_swap_frozen(fg), k, θ; return_invalid=true)
    sol_hv = SubGraph(sol_hv_s.V, sol_hv_s.U)
    sol_h, side_h = _pick_side(fg, sol_hu, sol_hv, θ)

    t_ou = @elapsed found_u = clique_order(deepcopy(work), k, θ; side=:U)
    t_ov = @elapsed found_v = clique_order(deepcopy(work), k, θ; side=:V)
    sol_o, side_o = _pick_side(fg, found_u.solution, found_v.solution, θ)
    found = side_o == "U" ? found_u : found_v

    sh = score_solution(fg, sol_h, k, θ)
    so = score_solution(fg, sol_o, k, θ)
    shu = score_solution(fg, sol_hu, k, θ)
    shv = score_solution(fg, sol_hv, k, θ)
    sou = score_solution(fg, found_u.solution, k, θ)
    sov = score_solution(fg, found_v.solution, k, θ)
    verdict = compare_results(
        MethodResult("heuristic", sol_h),
        MethodResult("clique-order", sol_o),
        fg, k, θ)
    delta = so.edges - sh.edges
    return Dict{String,Any}(
        "dataset" => entry.key,
        "edges" => entry.edges,
        "reduced_nU" => length(fg.u_ids),
        "reduced_nV" => length(fg.v_ids),
        "reduced_edges" => length(fg.v_adj),
        "heuristic_edges" => sh.edges,
        "heuristic_missing" => sh.missing,
        "heuristic_nU" => sh.nU,
        "heuristic_nV" => sh.nV,
        "heuristic_theta_feasible" => sh.theta_feasible,
        "heuristic_k_valid" => sh.k_valid,
        "heuristic_side" => side_h,
        "heuristic_u_edges" => shu.edges,
        "heuristic_u_nU" => shu.nU,
        "heuristic_u_nV" => shu.nV,
        "heuristic_u_theta_feasible" => shu.theta_feasible,
        "heuristic_u_s" => t_hu,
        "heuristic_v_edges" => shv.edges,
        "heuristic_v_nU" => shv.nU,
        "heuristic_v_nV" => shv.nV,
        "heuristic_v_theta_feasible" => shv.theta_feasible,
        "heuristic_v_s" => t_hv,
        "heuristic_s" => t_hu + t_hv,
        "order_edges" => so.edges,
        "order_missing" => so.missing,
        "order_nU" => so.nU,
        "order_nV" => so.nV,
        "order_theta_feasible" => so.theta_feasible,
        "order_k_valid" => so.k_valid,
        "order_side" => side_o,
        "order_u_edges" => sou.edges,
        "order_u_nU" => sou.nU,
        "order_u_nV" => sou.nV,
        "order_u_theta_feasible" => sou.theta_feasible,
        "order_u_s" => t_ou,
        "order_u_clique" => found_u.clique_size,
        "order_v_edges" => sov.edges,
        "order_v_nU" => sov.nU,
        "order_v_nV" => sov.nV,
        "order_v_theta_feasible" => sov.theta_feasible,
        "order_v_s" => t_ov,
        "order_v_clique" => found_v.clique_size,
        "order_s" => t_ou + t_ov,
        "order_clique_size" => found.clique_size,
        "order_kept" => found.kept,
        "order_search_side" => side_o,
        "edge_delta" => delta,
        "edge_pct" => _pct(delta, sh.edges),
        "verdict" => string(verdict),
        "order_wins" => verdict == :challenger && so.valid,
        "status" => "ok",
    )
end

_yn(x) = x ? "yes" : "no"

function _winner(row)
    row["order_wins"] && return "clique-order"
    if row["verdict"] == "baseline" && row["heuristic_theta_feasible"] && row["heuristic_k_valid"]
        return "theta"
    end
    row["verdict"] == "tie" && return "tie"
    return "—"
end

function _secs(t)
    ms = t * 1000
    ms < 0.1 && return "<0.1ms"
    ms < 100 && return string(round(ms; digits=1), "ms")
    return string(round(Int, ms), "ms")
end

function _print_orient(label, side, edges, nU, nV, feas, seconds, note="")
    println("  $(rpad(label, 13)) $side  $(lpad(edges, 6)) edges  " *
            "$(nU)×$(nV)  θ=$(_yn(feas))  $(_secs(seconds))$note")
end

function _print_row(row)
    pct = row["edge_pct"]
    pct_s = pct === nothing ? "" : " ($(round(pct; digits=1))%)"
    println("  $(row["dataset"])  |E_R|=$(row["reduced_edges"])  " *
            "|U|=$(row["reduced_nU"]) |V|=$(row["reduced_nV"])")
    _print_orient("theta", "U", row["heuristic_u_edges"],
        row["heuristic_u_nU"], row["heuristic_u_nV"],
        row["heuristic_u_theta_feasible"], row["heuristic_u_s"])
    _print_orient("theta", "V", row["heuristic_v_edges"],
        row["heuristic_v_nU"], row["heuristic_v_nV"],
        row["heuristic_v_theta_feasible"], row["heuristic_v_s"])
    println("  theta best $(row["heuristic_side"])   " *
            "$(row["heuristic_edges"]) edges  $(row["heuristic_nU"])×$(row["heuristic_nV"])  " *
            "total $(_secs(row["heuristic_s"]))")
    _print_orient("clique-order", "U", row["order_u_edges"],
        row["order_u_nU"], row["order_u_nV"],
        row["order_u_theta_feasible"], row["order_u_s"],
        "  clique=$(row["order_u_clique"])")
    _print_orient("clique-order", "V", row["order_v_edges"],
        row["order_v_nU"], row["order_v_nV"],
        row["order_v_theta_feasible"], row["order_v_s"],
        "  clique=$(row["order_v_clique"])")
    println("  order best $(row["order_side"])   " *
            "$(row["order_edges"]) edges  $(row["order_nU"])×$(row["order_nV"])  " *
            "clique=$(row["order_clique_size"]) kept=$(row["order_kept"])  " *
            "total $(_secs(row["order_s"]))")
    println("  $(_winner(row))  Δ $(row["edge_delta"]) edges$pct_s")
end

function _print_detail(row)
    pct = row["edge_pct"]
    pct_s = pct === nothing ? "" : " ($(round(pct; digits=1))%)"
    println("  $(row["dataset"])  " *
            "theta $(row["heuristic_edges"]) ($(row["heuristic_side"]), $(_secs(row["heuristic_s"])))  " *
            "order $(row["order_edges"]) ($(row["order_side"]), $(_secs(row["order_s"])))  " *
            "Δ $(row["edge_delta"])$pct_s")
end

function print_report(rows)
    ok = [row for row in rows if row["status"] == "ok"]
    wins = [row for row in ok if row["order_wins"]]
    losses = [row for row in ok if row["verdict"] == "baseline" &&
              row["heuristic_theta_feasible"] && row["heuristic_k_valid"]]
    ties = [row for row in ok if row["verdict"] == "tie"]
    accounted = Set(objectid(row) for row in Iterators.flatten((wins, losses, ties)))
    other = [row for row in ok if objectid(row) ∉ accounted]
    sort!(wins; by=r -> something(r["edge_pct"], 0.0), rev=true)

    println()
    println("clique-order outperforms θ-heuristic on $(length(wins)) / $(length(ok)) graphs")
    for row in wins
        _print_detail(row)
    end
    println()
    println("θ-heuristic is ahead on $(length(losses)) / $(length(ok))")
    for row in losses
        _print_detail(row)
    end
    if !isempty(ties)
        println()
        println("ties ($(length(ties))): " * join((row["dataset"] for row in ties), ", "))
    end
    if !isempty(other)
        println()
        println("no θ-feasible winner ($(length(other)))")
        for row in other
            _print_detail(row)
        end
    end
    failed = [row for row in rows if row["status"] != "ok"]
    if !isempty(failed)
        println()
        println("failed ($(length(failed))): " * join((row["dataset"] for row in failed), ", "))
    end
end

function save_rows(path::AbstractString, rows, meta)
    mkpath(dirname(abspath(path)))
    payload = Dict{String,Any}("meta" => meta, "rows" => rows)
    open(path, "w") do io
        JSON3.pretty(io, payload)
        println(io)
    end
    println()
    println("wrote $path")
end

function main()
    any(a -> a in ("-h", "--help"), ARGS) && (_usage(); return)
    n = _int_flag("N", 10)
    offset = _int_flag("offset", 0)
    k = _int_flag("k", 2)
    θ = _int_flag("theta", 5)
    k >= 0 || error("--k must be ≥ 0")
    θ > k || error("--theta must be > --k (got theta=$θ k=$k)")
    prefix = _flag("prefix")
    reduction = _reduction_flag()
    save_path = _flag("save")

    available, chosen = candidate_graphs(n, offset, prefix)
    println("clique-order vs θ-heuristic")
    println("k=$k  θ=$θ  reduce=$(_reduction_name(reduction))  " *
            "N=$n  pool=$(length(chosen)) of $(length(available))  offset=$offset" *
            (prefix === nothing ? "" : "  prefix=$prefix"))
    println("graphs with a reduced side below θ are skipped and do not count toward N")
    isempty(chosen) && error("No graphs selected. Check data/ and --N / --offset / --prefix.")

    println("warmup (compile both solvers on K6,6)…")
    _warmup!()

    rows = Dict{String,Any}[]
    skipped = String[]
    println()
    for entry in chosen
        length(rows) >= n && break
        println("[$(length(rows) + 1)/$n] $(entry.key)")
        flush(stdout)
        row = try
            work, fg = load_reduced(entry, k, θ, reduction)
            if length(fg.u_ids) < θ || length(fg.v_ids) < θ
                push!(skipped, entry.key)
                println("  skip  reduced |U|=$(length(fg.u_ids)) |V|=$(length(fg.v_ids)) |E|=$(length(fg.v_adj))")
                continue
            end
            compare_one(entry, work, fg, k, θ)
        catch err
            Dict{String,Any}(
                "dataset" => entry.key,
                "edges" => entry.edges,
                "status" => "error",
                "error" => sprint(showerror, err),
            )
        end
        push!(rows, row)
        if row["status"] == "ok"
            _print_row(row)
        else
            println("$(row["dataset"])  ERROR  $(row["error"])")
        end
        flush(stdout)
    end
    if !isempty(skipped)
        println()
        println("skipped $(length(skipped)) (reduced side < θ): " * join(skipped, ", "))
    end
    if length(rows) < n
        println("stopped after $(length(rows)) comparable graphs; the pool ran out")
    end

    print_report(rows)
    if save_path !== nothing
        save_rows(save_path, rows, Dict{String,Any}(
            "k" => k,
            "theta" => θ,
            "reduction" => _reduction_name(reduction),
            "N" => n,
            "offset" => offset,
            "prefix" => prefix,
            "sides" => "both",
        ))
    end
    return rows
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end
