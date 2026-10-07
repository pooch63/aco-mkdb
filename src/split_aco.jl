#= Split ACO: force top seeds from the smaller side into vertex-trail ACO.

Seeds start as the maximal vertices under neighborhood inclusion (u ≼ v when
N(u) ⊆ N(v); identical neighborhoods keep the earliest vertex in degree
order). A k-defective biclique that uses a dominated u has an equal-or-better
counterpart that swaps u for a maximal v, and the seed at v covers it.
On N(a)={1,2}, N(b)={1,2,3}, N(c)={4}, a ≼ b, so the seeds are b and c.

Those candidates are then ranked by node S_C — the maximum neighbourhood-Jaccard
score S_C of any incident edge (`neighborhood_scores`, the same edge score
`neighborhood_edge_reduction!` in pulse.jl keeps) — and only the top `n_seeds`
(default 50) are kept.

`one_colony` (default) runs a single ACO on this graph. Ant i on iteration t
is planted with seed `roots[mod1(i + (t-1)*ants, n_seeds)]`, so the top seeds
are forced in without a separate colony per seed. Pass `one_colony=false` to
restore the per-seed search:

  1. Take the closed 3-hop neighborhood of r: same side = {r} ∪ N²(r),
     opposite side = N(same side). Every k-defective biclique containing r
     (with θ > k) lives inside this induced subgraph.
  2. Renumber that induced subgraph to ids 1..n (common-neighbor reduction
     indexes by vertex id) and apply common-neighbor reduction. If r is
     pruned, no θ-feasible solution contains r and the root is skipped.
  3. Run `aco` on that reduced local graph with r forced into every ant
     (`seed_nodes`), so each local best contains r. Results are mapped back
     to the original vertex ids.

The global best is the best result (θ-feasible by edges, else fitness).

With `prune_by_bound`, an ant whose `upper_bound` cannot beat the incumbent
stops and its pheromone deposits are subtracted, as if the ant had not run.
=#

using Base.Threads

const __SPLIT_ACO_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath("aco", "algorithm.jl"))

"""
Closed 3-hop neighborhood of `root` (original id on side `root_is_u`).

Returns `(same, opposite)` original-id vectors: `same` = {root} ∪ N²(root),
`opposite` = N(same). With `rank` (position in the full same-side degree
order, including vertices that are not seeds), only same-side vertices ranked
after `root` are kept — the N²₊ / N³₊ scheme from `branch_from_all_u!`, which
finds each solution from its earliest root only.
"""
function three_hop_neighborhood(fg::FrozenBipartite, root::Int, root_is_u::Bool;
    rank::Union{Nothing,Dict{Int,Int}}=nothing)
    nbrs_same = root_is_u ? (x -> neighbors_u(fg, x)) : (x -> neighbors_v(fg, x))
    nbrs_opp = root_is_u ? (x -> neighbors_v(fg, x)) : (x -> neighbors_u(fg, x))
    root_rank = rank === nothing ? 0 : rank[root]

    one_hop = nbrs_same(root)
    same = Set{Int}(root)
    for w in one_hop
        for x in nbrs_opp(w)
            if rank === nothing || get(rank, x, 0) > root_rank
                push!(same, x)
            end
        end
    end
    opposite = Set{Int}(one_hop)
    for x in same
        x == root && continue
        union!(opposite, nbrs_same(x))
    end
    return collect(same), collect(opposite)
end

"""
    compact_induced_bipartite(fg, us, vs) -> (BipartiteGraph, GraphRemapping)

Induced subgraph of `fg` on original ids `us` ⊆ U and `vs` ⊆ V, with vertices
renumbered to `1:|us|` and `1:|vs|`.

`common_neighbor_reduction!` indexes adjacency by vertex id and requires those
ids to be `1..n`. The remapping sends the compact ids back to the originals
after ACO.
"""
function compact_induced_bipartite(fg::FrozenBipartite, us::AbstractVector{Int}, vs::AbstractVector{Int})
    u_ids = collect(Int, us)
    v_ids = collect(Int, vs)
    v_of = Dict{Int,Int}(v => i for (i, v) in enumerate(v_ids))
    g = BipartiteGraph{Nothing}()
    for i in eachindex(u_ids)
        add_u!(g, i)
    end
    for i in eachindex(v_ids)
        add_v!(g, i)
    end
    for (cu, u) in enumerate(u_ids)
        for v in neighbors_u(fg, u)
            cv = get(v_of, v, 0)
            cv == 0 && continue
            add_edge!(g, cu, cv, nothing)
        end
    end
    return g, GraphRemapping(u_ids, v_ids)
end

"""
    maximal_neighborhood_seeds(fg, is_u, order) -> Vector{Int}

Seeds among `order` (one side, highest degree first) that are maximal under
neighborhood inclusion.

`u ≼ v` when `N(u) ⊆ N(v)`. `u` is dropped when some `v` in `order` has
`N(u) ⊆ N(v)` and either the inclusion is proper or `v` is earlier in `order`
(the representative for an identical neighborhood). A solution that uses a
dropped `u` has an equal-or-better counterpart covered by the seed at that `v`.

Example: `N(a) = {1,2}`, `N(b) = {1,2,3}`, `N(c) = {4}`. Then `a ≼ b`, and
the seeds are `b` and `c`.
"""
function maximal_neighborhood_seeds(fg::FrozenBipartite, is_u::Bool, order::AbstractVector{Int})
    isempty(order) && return Int[]
    n_same = is_u ? length(fg.u_ids) : length(fg.v_ids)
    index = is_u ? fg.u_index : fg.v_index

    pos = zeros(Int, n_same)
    deg = zeros(Int, n_same)
    any_edge = false
    for (i, id) in enumerate(order)
        si = index[id]
        pos[si] = i
        d = is_u ? length(neighbor_range_u(fg, si)) : length(neighbor_range_v(fg, si))
        deg[si] = d
        any_edge |= d > 0
    end
    # ∅ ⊆ N(v) for every v. Every empty vertex is dominated unless the whole
    # side is empty, in which case one representative is enough.
    any_edge || return Int[order[1]]

    stamp = zeros(Int, n_same)
    count = zeros(Int, n_same)
    cur = 0
    keep = trues(length(order))

    # True when x shares every neighbor of si and is a strictly larger
    # neighborhood, or the earlier representative of an identical one.
    function dominated_by(x::Int, si::Int, d::Int, i::Int)
        if stamp[x] != cur
            stamp[x] = cur
            count[x] = 1
        else
            count[x] += 1
        end
        count[x] == d || return false
        x == si && return false
        pos[x] == 0 && return false
        dx = deg[x]
        return dx > d || (dx == d && pos[x] < i)
    end

    for (i, id) in enumerate(order)
        si = index[id]
        d = deg[si]
        if d == 0
            keep[i] = false
            continue
        end
        cur += 1
        dominated = false
        if is_u
            for k in neighbor_range_u(fg, si)
                dominated && break
                vi = fg.v_adj[k]
                for t in neighbor_range_v(fg, vi)
                    dominated_by(fg.u_adj[t], si, d, i) && (dominated = true)
                end
            end
        else
            for t in neighbor_range_v(fg, si)
                dominated && break
                ui = fg.u_adj[t]
                for k in neighbor_range_u(fg, ui)
                    dominated_by(fg.v_adj[k], si, d, i) && (dominated = true)
                end
            end
        end
        keep[i] = !dominated
    end
    return collect(Int, order[keep])
end

"""
    incident_sc(fg, is_u; θ=5) -> Vector{Float64}

Node S_C for one side, indexed by the dense side index (parallel to
`fg.u_ids` / `fg.v_ids`). A vertex's score is the maximum S_C of its incident
edges. S_C is the neighbourhood-Jaccard edge score from `neighborhood_scores`.
"""
function incident_sc(fg::FrozenBipartite, is_u::Bool; θ::Int=5)
    edge_scores = neighborhood_scores(fg; θ=max(θ, 1))
    if is_u
        scores = zeros(length(fg.u_ids))
        for ui in eachindex(scores)
            best = 0.0
            for k in neighbor_range_u(fg, ui)
                best = max(best, edge_scores[k])
            end
            scores[ui] = best
        end
        return scores
    end
    scores = zeros(length(fg.v_ids))
    for ui in eachindex(fg.u_ids)
        for k in neighbor_range_u(fg, ui)
            scores[fg.v_adj[k]] = max(scores[fg.v_adj[k]], edge_scores[k])
        end
    end
    return scores
end

"""
    top_sc_seeds(fg, is_u, candidates, n_seeds; θ) -> Vector{Int}

The `n_seeds` vertices in `candidates` with the highest node S_C (max incident
edge score). `n_seeds <= 0` or a candidate list no longer than `n_seeds`
returns every candidate in its original order. Otherwise the result is sorted
by score descending; ties keep the candidate order.
"""
function top_sc_seeds(fg::FrozenBipartite, is_u::Bool, candidates::AbstractVector{Int},
    n_seeds::Int; θ::Int=5)
    (n_seeds <= 0 || length(candidates) <= n_seeds) && return collect(Int, candidates)
    scores = incident_sc(fg, is_u; θ=θ)
    index = is_u ? fg.u_index : fg.v_index
    ranked = sort(collect(Int, candidates); by=id -> scores[index[id]], rev=true)
    return ranked[1:n_seeds]
end

"""Mutable graph with the same original vertex ids as `fg`."""
function mutable_from_frozen(fg::FrozenBipartite)
    g = BipartiteGraph{Nothing}()
    for u in fg.u_ids
        add_u!(g, u)
    end
    for v in fg.v_ids
        add_v!(g, v)
    end
    for ui in eachindex(fg.u_ids)
        u = fg.u_ids[ui]
        for k in neighbor_range_u(fg, ui)
            add_edge!(g, u, fg.v_ids[fg.v_adj[k]], nothing)
        end
    end
    return g
end

function seed_subgraph(root::Int, root_is_u::Bool)
    return root_is_u ? SubGraph(Set([root]), Set{Int}()) : SubGraph(Set{Int}(), Set([root]))
end

root_in_solution(root::Int, root_is_u::Bool, sg::SubGraph) =
    root_is_u ? (root in sg.U) : (root in sg.V)

"""
One ACO on `fg`. Each ant is planted with a different seed from `roots`;
later iterations walk further down the list. See `round_robin_seed`.
"""
function run_split_one_colony(fg::FrozenBipartite, k::Int, θ::Int,
    pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64,
    num_subspecies::Int, roots::Vector{Int}, order::AbstractVector{Int},
    candidates::AbstractVector{Int}, root_is_u::Bool, n_seeds::Int;
    parallelize::Bool=false, verbose::Bool=false, prune_by_bound::Bool=true,
    aco_kwargs...)
    n_roots = length(roots)
    n_side = length(order)
    n_candidates = length(candidates)
    slots = min(n_roots, max(num_ants, 0) * max(num_iterations, 0))
    println("Split ACO: one colony on $(root_is_u ? "U" : "V") side=$n_roots of $n_side " *
            "(top $(n_seeds <= 0 ? n_candidates : min(n_seeds, n_candidates)) by S_C " *
            "from $n_candidates) " *
            "ants=$num_ants iterations=$num_iterations assigned_seeds=$slots")

    if n_roots == 0 || length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return (; best=SubGraph(), best_root=nothing, root_is_u, n_roots, n_side, n_ran=0,
            iterations_to_best=nothing, time_to_best_s=nothing, one_colony=true)
    end

    seeds = SubGraph[seed_subgraph(root, root_is_u) for root in roots]
    t0 = time_ns()
    g = mutable_from_frozen(fg)
    kw = Dict{Symbol,Any}(pairs(aco_kwargs))
    delete!(kw, :ant_seeds)
    delete!(kw, :seed_nodes)
    kw[:parallelize] = parallelize
    kw[:prune_by_bound] = prune_by_bound
    kw[:reduction] = ReductionMode.none
    kw[:ant_seeds] = seeds

    t_before_aco = (time_ns() - t0) / 1e9
    function run_colony()
        return aco(g, pheromone, num_ants, num_iterations, evaporation, k, θ, num_subspecies; kw...)
    end
    sols, best_iters, best_times, _, _ =
        verbose ? run_colony() : redirect_stdout(run_colony, devnull)

    b = 1
    for s in 2:length(sols)
        better_than_best(fg, sols[s], sols[b], θ) && (b = s)
    end
    best = sols[b]
    best_root = nothing
    for root in roots
        if root_in_solution(root, root_is_u, best)
            best_root = root
            break
        end
    end
    itb = best_iters[b]
    ttb = t_before_aco + best_times[b]
    println("Split ACO best: root=$(best_root) |U|=$(length(best.U)) |V|=$(length(best.V)) " *
            "edges=$(Subgraph.edge_count(fg, best))")
    return (; best, best_root, root_is_u, n_roots, n_side, n_ran=1,
        iterations_to_best=itb, time_to_best_s=ttb, one_colony=true)
end

"""
Run split ACO on frozen graph `fg`. Returns a NamedTuple with the best
subgraph (original ids) plus per-root bookkeeping.

`aco_kwargs` are forwarded to every `aco` call.
`reduce_seeds` restricts roots to `maximal_neighborhood_seeds`.
`n_seeds` then keeps the top vertices among those by node S_C (default 50).
`one_colony` (default true) runs one ACO on `fg` and round-robins those seeds
across ants and iterations. `one_colony=false` runs one colony per seed on
that seed's 3-hop subgraph.
`prune_by_bound` stops an ant whose `upper_bound` cannot beat the incumbent
and subtracts that ant's pheromone deposits (default on).
`local_reduction` (default true) runs common-neighbor reduction on each
renumbered 3-hop graph before `aco`. Ignored when `one_colony` is set.
"""
function split_aco(fg::FrozenBipartite, k::Int, θ::Int,
    pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64,
    num_subspecies::Int;
    forward_only::Bool=false,
    local_reduction::Bool=true,
    reduce_seeds::Bool=true,
    n_seeds::Int=50,
    one_colony::Bool=true,
    parallelize::Bool=false,
    verbose::Bool=false,
    prune_by_bound::Bool=true,
    aco_kwargs...)
    root_is_u = length(fg.u_ids) <= length(fg.v_ids)
    order = get_degree_order(fg, root_is_u, true)
    candidates = reduce_seeds ? maximal_neighborhood_seeds(fg, root_is_u, order) : order
    roots = top_sc_seeds(fg, root_is_u, candidates, n_seeds; θ=θ)
    if one_colony
        return run_split_one_colony(fg, k, θ, pheromone, num_ants, num_iterations,
            evaporation, num_subspecies, roots, order, candidates, root_is_u, n_seeds;
            parallelize=parallelize, verbose=verbose, prune_by_bound=prune_by_bound,
            aco_kwargs...)
    end
    # Rank the full side, not just the seeds, so a dominated vertex that sorts
    # after its dominator can still be added inside that dominator's forward cut.
    rank = forward_only ? Dict{Int,Int}(r => i for (i, r) in enumerate(order)) : nothing

    n_roots = length(roots)
    local_best = Vector{SubGraph}(undef, n_roots)
    local_itb = zeros(Int, n_roots)
    local_ttb = fill(Inf, n_roots)
    local_sizes = zeros(Int, n_roots)
    ran = falses(n_roots)

    t0 = time_ns()

    function solve_root(i::Int)
        root = roots[i]
        local_best[i] = SubGraph()
        same, opp = three_hop_neighborhood(fg, root, root_is_u; rank=rank)
        us, vs = root_is_u ? (same, opp) : (opp, same)
        (length(us) < θ || length(vs) < θ) && return

        # Ids 1..n so common-neighbor reduction indexes the neighborhood,
        # not the original id span of the parent graph.
        lg, remap = compact_induced_bipartite(fg, us, vs)
        root_c = root_is_u ? findfirst(==(root), remap.u_original) :
                             findfirst(==(root), remap.v_original)
        root_c === nothing && return
        if local_reduction
            lfg = apply_graph_reductions!(lg, k, θ, nothing, nothing, true, ReductionMode.simple)
            root_kept = root_is_u ? haskey(lfg.u_index, root_c) : haskey(lfg.v_index, root_c)
            root_kept || return
            (length(lfg.u_ids) < θ || length(lfg.v_ids) < θ) && return
        end
        local_sizes[i] = length(lg.adjU) + length(lg.adjV)

        seed = root_is_u ? SubGraph(Set([root_c]), Set{Int}()) : SubGraph(Set{Int}(), Set([root_c]))
        t_start = (time_ns() - t0) / 1e9
        sols, best_iters, best_times, _, _ = aco(lg, pheromone, num_ants, num_iterations,
            evaporation, k, θ, num_subspecies;
            parallelize=false, seed_nodes=seed, reduction=ReductionMode.none,
            prune_by_bound=prune_by_bound,
            aco_kwargs...)
        isempty(sols) && return
        sols = [remap_subgraph(remap, s) for s in sols]
        b = 1
        for s in 2:length(sols)
            better_than_best(fg, sols[s], sols[b], θ) && (b = s)
        end
        local_best[i] = sols[b]
        local_itb[i] = best_iters[b]
        local_ttb[i] = t_start + best_times[b]
        ran[i] = true
        return
    end

    function run_all()
        if parallelize
            @threads :dynamic for i in 1:n_roots
                solve_root(i)
            end
        else
            for i in 1:n_roots
                solve_root(i)
            end
        end
    end

    verbose ? run_all() : redirect_stdout(run_all, devnull)

    best_idx = 0
    for i in 1:n_roots
        ran[i] || continue
        if best_idx == 0 || better_than_best(fg, local_best[i], local_best[best_idx], θ)
            best_idx = i
        end
    end

    n_ran = count(ran)
    n_side = length(order)
    n_candidates = length(candidates)
    sizes = local_sizes[ran]
    println("Split ACO: seeds on $(root_is_u ? "U" : "V") side=$n_roots of $n_side " *
            "(top $(n_seeds <= 0 ? n_candidates : min(n_seeds, n_candidates)) by S_C " *
            "from $n_candidates) " *
            "ran=$n_ran skipped=$(n_roots - n_ran) " *
            "local |U|+|V| mean=$(isempty(sizes) ? 0 : round(sum(sizes) / length(sizes); digits=1)) " *
            "max=$(isempty(sizes) ? 0 : maximum(sizes))")

    if best_idx == 0
        return (; best=SubGraph(), best_root=nothing, root_is_u, n_roots, n_side, n_ran,
            iterations_to_best=nothing, time_to_best_s=nothing, one_colony=false)
    end
    best = local_best[best_idx]
    println("Split ACO best: root=$(roots[best_idx]) |U|=$(length(best.U)) |V|=$(length(best.V)) " *
            "edges=$(Subgraph.edge_count(fg, best))")
    return (; best, best_root=roots[best_idx], root_is_u, n_roots, n_side, n_ran,
        iterations_to_best=local_itb[best_idx], time_to_best_s=local_ttb[best_idx],
        one_colony=false)
end
