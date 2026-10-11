#=
Compare clique-order-edge with the θ-heuristic on real graphs.

Picks indexed graphs under data/ in ascending |E| and scores both solvers on
the same reduced instance. Clique-order-edge walks one side (the smaller
side of the reduced graph). The θ-heuristic grows U and grows V; the better
θ-feasible solution is the baseline. A graph whose reduced sides are each
smaller than θ is skipped and does not count toward --N. A win is the paper
rule: θ-feasible, k-valid, and strictly more edges.

    julia bin/compare-clique-order-edge.jl
    julia bin/compare-clique-order-edge.jl --N=10 --prefix=konect-small
    julia bin/compare-clique-order-edge.jl --N=5 --k=2 --theta=5 --reduce=lo \
        --save=results/clique_order_edge_vs_theta.json
=#

const ROOT = dirname(@__DIR__)
const SRC = joinpath(ROOT, "src")

isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))
isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))

using JSON3

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
    julia bin/compare-clique-order-edge.jl [--N=10] [--offset=0] [--prefix=] \\
        [--k=2] [--theta=5] [--reduce=lo|hi|none] [--save=PATH]
    """)
end

"""Indexed graphs after `offset`, smallest |E| first."""
function candidate_graphs(n::Int, offset::Int, prefix::Union{Nothing,AbstractString})
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
    clique_order_edge(deepcopy(g), 2, 5)
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

"""Load `entry` and apply the shared reduction."""
function load_reduced(entry, k::Int, θ::Int, reduction::ReductionMode.T)
    g = _load_quiet(entry.path)
    work = deepcopy(g)
    fg = apply_graph_reductions!(work, k, θ, nothing, nothing, true, reduction)
    return work, fg
end

"""
Score the θ-heuristic on both orientations and clique-order-edge on one side.

The kept θ solution is the orientation with more edges once it is θ-feasible.
Clique-order-edge uses its default side, the smaller part of the graph it
receives after `super_reduce!`.
"""
function compare_one(entry, work, fg, k::Int, θ::Int)
    t_hu = @elapsed sol_hu = theta_based_heuristic(fg, k, θ; return_invalid=true)
    t_hv = @elapsed sol_hv_s = theta_based_heuristic(_swap_frozen(fg), k, θ; return_invalid=true)
    sol_hv = SubGraph(sol_hv_s.V, sol_hv_s.U)
    sol_h, side_h = _pick_side(fg, sol_hu, sol_hv, θ)

    t_e = @elapsed found = clique_order_edge(deepcopy(work), k, θ)
    sol_e = found.solution

    sh = score_solution(fg, sol_h, k, θ)
    se = score_solution(fg, sol_e, k, θ)
    verdict = compare_results(
        MethodResult("heuristic", sol_h),
        MethodResult("clique-order-edge", sol_e),
        fg, k, θ)
    delta = se.edges - sh.edges
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
        "heuristic_s" => t_hu + t_hv,
        "edge_edges" => se.edges,
        "edge_missing" => se.missing,
        "edge_nU" => se.nU,
        "edge_nV" => se.nV,
        "edge_theta_feasible" => se.theta_feasible,
        "edge_k_valid" => se.k_valid,
        "edge_side" => found.search_is_u ? "U" : "V",
        "edge_kept" => found.kept,
        "edge_components" => found.components,
        "edge_projection_n" => found.projection_n,
        "edge_projection_m" => found.projection_m,
        "edge_s" => t_e,
        "edge_delta" => delta,
        "edge_pct" => _pct(delta, sh.edges),
        "verdict" => string(verdict),
        "edge_wins" => verdict == :challenger && se.valid,
        "status" => "ok",
    )
end

_yn(x) = x ? "yes" : "no"

function _winner(row)
    row["edge_wins"] && return "clique-order-edge"
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

function _print_row(row)
    pct = row["edge_pct"]
    pct_s = pct === nothing ? "" : " ($(round(pct; digits=1))%)"
    println("  $(row["dataset"])  |E_R|=$(row["reduced_edges"])  " *
            "|U|=$(row["reduced_nU"]) |V|=$(row["reduced_nV"])")
    println("  theta $(row["heuristic_side"])  $(row["heuristic_edges"]) edges  " *
            "$(row["heuristic_nU"])×$(row["heuristic_nV"])  " *
            "θ=$(_yn(row["heuristic_theta_feasible"]))  $(_secs(row["heuristic_s"]))")
    println("  edge  $(row["edge_side"])  $(row["edge_edges"]) edges  " *
            "$(row["edge_nU"])×$(row["edge_nV"])  " *
            "θ=$(_yn(row["edge_theta_feasible"]))  kept=$(row["edge_kept"])  " *
            "components=$(row["edge_components"])  " *
            "projection $(row["edge_projection_n"])/$(row["edge_projection_m"])  " *
            "$(_secs(row["edge_s"]))")
    println("  $(_winner(row))  Δ $(row["edge_delta"]) edges$pct_s")
end

function _print_detail(row)
    pct = row["edge_pct"]
    pct_s = pct === nothing ? "" : " ($(round(pct; digits=1))%)"
    println("  $(row["dataset"])  " *
            "theta $(row["heuristic_edges"]) ($(row["heuristic_side"]), $(_secs(row["heuristic_s"])))  " *
            "edge $(row["edge_edges"]) ($(row["edge_side"]), $(_secs(row["edge_s"])))  " *
            "Δ $(row["edge_delta"])$pct_s")
end

function print_report(rows)
    ok = [row for row in rows if row["status"] == "ok"]
    wins = [row for row in ok if row["edge_wins"]]
    losses = [row for row in ok if row["verdict"] == "baseline" &&
              row["heuristic_theta_feasible"] && row["heuristic_k_valid"]]
    ties = [row for row in ok if row["verdict"] == "tie"]
    accounted = Set(objectid(row) for row in Iterators.flatten((wins, losses, ties)))
    other = [row for row in ok if objectid(row) ∉ accounted]
    sort!(wins; by=r -> something(r["edge_pct"], 0.0), rev=true)

    println()
    println("clique-order-edge outperforms θ-heuristic on $(length(wins)) / $(length(ok)) graphs")
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
    println("clique-order-edge vs θ-heuristic")
    println("k=$k  θ=$θ  reduce=$(_reduction_name(reduction))  " *
            "N=$n  pool=$(length(chosen)) of $(length(available))  offset=$offset" *
            (prefix === nothing ? "" : "  prefix=$prefix"))
    println("edge-order searches one side; θ-heuristic keeps the better of U and V")
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
            "sides" => "edge-order one side; theta both",
        ))
    end
    return rows
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    main()
end
