#=
Clique-order by common-neighbor edge weight.

Same opening as `clique-order`: `super_reduce!`, then the compatibility
projection of one side. A θ-feasible k-defective biclique's searched side is
a clique of that projection. This file does not search the other side.

Edge weights are the common-neighbor counts from `_weighted_projection` in
`super-reduce.jl` (the codegree `C`, not the defect weight `θ − C`). The
walk is:

  1. Start at the heaviest edge. Ties go to the edge touching the
     highest bipartite degree, then the other endpoint's degree, then the
     smaller vertex id.
  2. Add both endpoints. The endpoint with higher bipartite degree is the
     tip (smaller id when the degrees match).
  3. From the tip, step along the heaviest edge to a neighbor that is not
     yet in any component. Ties break toward that neighbor's bipartite
     degree. The neighbor has to be adjacent, in the projection, to every
     vertex already chosen, and the opposite-side packing has to keep at
     least θ vertices. A failure undoes the add and the next edge is tried.
     Adding a searched-side vertex only raises opposite costs, so a packing
     under θ cannot recover.
  4. When the tip has no such neighbor — every neighbor is already in a
     component, or each remaining one failed the check — start over from
     the heaviest edge whose endpoints have not been added.
  5. Every feasible prefix is scored by edge count. The walk returns the
     best one. The opposite side is that prefix's greedy cost packing.

`within_clique=true` runs `max_clique_heuristic` on this projection first.
The seed and every later step then use only edges of that clique.

    julia src/clique-order-edge.jl
=#

const __CLIQUE_ORDER_EDGE_JL__ = true

isdefined(@__MODULE__, :__CLIQUE_ORDER_JL__) || include("clique-order.jl")

"""
Weighted compatibility graph on one side.

`side_id[i]` is the compact id on the searched side. `adj[i]` and `weight[i]`
are parallel: neighbors sorted by descending common-neighbor count, then
descending bipartite degree, then smaller index. `seeds` lists each undirected
edge once as `(weight, a, b)` in the order the walk tries them. `bi[i]` is
the bipartite degree.
"""
struct EdgeOrderGraph
    search_is_u::Bool
    side_id::Vector{Int}
    adj::Vector{Vector{Int}}
    weight::Vector{Vector{Int}}
    bi::Vector{Int}
    seeds::Vector{NTuple{3,Int}}
    edges::Int
end

function _empty_edge_graph(search_is_u::Bool)
    return EdgeOrderGraph(search_is_u, Int[], Vector{Int}[], Vector{Int}[],
        Int[], NTuple{3,Int}[], 0)
end

"""
Codegree dicts for `ids`. `τ > 0` reuses `_weighted_projection`. `τ ≤ 0`
keeps every pair; a pair that shares no neighbor has weight 0.
"""
function _projection_codegree(ids::Vector{Int}, small_adj::Dict{Int,Set{Int}}, θ::Int, τ::Int)
    if τ > 0
        _, codeg, _, _ = _weighted_projection(ids, small_adj, θ, τ)
        return codeg
    end
    n = length(ids)
    lists, buckets = _candidate_buckets(ids, small_adj)
    stamp = zeros(Int, n)
    cnt = zeros(Int, n)
    touched = Int[]
    cur = Ref(0)
    codeg = [Dict{Int,Int}() for _ in 1:n]
    @inbounds for i in 1:n
        cur[] += 1
        stamp_id = cur[]
        empty!(touched)
        for j in lists[i]
            for w in buckets[j]
                if stamp[w] != stamp_id
                    stamp[w] = stamp_id
                    cnt[w] = 1
                    push!(touched, w)
                else
                    cnt[w] += 1
                end
            end
        end
        for w in touched
            w <= i && continue
            codeg[i][w] = cnt[w]
            codeg[w][i] = cnt[w]
        end
        for w in (i + 1):n
            haskey(codeg[i], w) && continue
            codeg[i][w] = 0
            codeg[w][i] = 0
        end
    end
    return codeg
end

"""
    edge_order_graph(fg, k, θ; search_is_u=nothing) -> EdgeOrderGraph

Compatibility projection of one side. With `search_is_u === nothing` the
smaller side is used, U when the two sides have equal size. An edge exists
when the two vertices share at least `θ − k` opposite neighbors, and its
weight is that count.
"""
function edge_order_graph(fg::FrozenBipartite, k::Int, θ::Int;
        search_is_u::Union{Nothing,Bool}=nothing)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    if search_is_u === nothing
        search_is_u = nU <= nV
    end
    nside = search_is_u ? nU : nV
    nside == 0 && return _empty_edge_graph(search_is_u)
    τ = θ - k
    neighborhoods = side_neighborhoods(fg, search_is_u)
    ids = Int[]
    for i in 1:nside
        (τ <= 0 || length(neighborhoods[i]) >= τ) && push!(ids, i)
    end
    isempty(ids) && return _empty_edge_graph(search_is_u)
    codeg = _projection_codegree(ids, neighborhoods, θ, τ)
    keep = Int[]
    for i in eachindex(codeg)
        isempty(codeg[i]) || push!(keep, i)
    end
    isempty(keep) && return _empty_edge_graph(search_is_u)

    old_to_new = zeros(Int, length(ids))
    for (neu, old) in enumerate(keep)
        old_to_new[old] = neu
    end
    n = length(keep)
    side_id = [ids[old] for old in keep]
    adj = [Int[] for _ in 1:n]
    weight = [Int[] for _ in 1:n]
    bi = [_side_degree(fg, search_is_u, side_id[i]) for i in 1:n]
    seeds = NTuple{3,Int}[]
    for (neu, old) in enumerate(keep)
        for (j, c) in codeg[old]
            nj = old_to_new[j]
            nj == 0 && continue
            push!(adj[neu], nj)
            push!(weight[neu], c)
            nj > neu && push!(seeds, (c, neu, nj))
        end
    end
    for i in 1:n
        order = sortperm(eachindex(adj[i]); by=t -> (-weight[i][t], -bi[adj[i][t]], adj[i][t]))
        adj[i] = adj[i][order]
        weight[i] = weight[i][order]
    end
    sort!(seeds; by=s -> (
        -s[1],
        -max(bi[s[2]], bi[s[3]]),
        -min(bi[s[2]], bi[s[3]]),
        min(side_id[s[2]], side_id[s[3]]),
        max(side_id[s[2]], side_id[s[3]]),
    ))
    return EdgeOrderGraph(search_is_u, side_id, adj, weight, bi, seeds, length(seeds))
end

"""Higher bipartite degree, then smaller compact id, is the tip."""
function _edge_tip(eg::EdgeOrderGraph, a::Int, b::Int)
    da, db = eg.bi[a], eg.bi[b]
    ida, idb = eg.side_id[a], eg.side_id[b]
    if da > db || (da == db && ida <= idb)
        return a, b
    end
    return b, a
end

"""True when `nbr` is a projection neighbor of every vertex in `path`."""
function _joins_path(adj::Vector{Vector{Int}}, mark::Vector{Int}, stamp::Base.RefValue{Int},
        nbr::Int, path::Vector{Int})
    stamp[] += 1
    s = stamp[]
    for j in adj[nbr]
        mark[j] = s
    end
    for p in path
        mark[p] == s || return false
    end
    return true
end

function _note_edge_best!(st::OracleState, k::Int, θ::Int,
        best_e::Base.RefValue{Int}, best::Vector{Int})
    e, nsel = oracle_score(st, k)
    if st.nL >= θ && nsel >= θ && e > best_e[]
        best_e[] = e
        empty!(best)
        append!(best, st.members)
    end
    return e, nsel
end

"""
Grow one component from projection edge `(a, b)`.

Returns false when the two endpoints already pack fewer than θ opposite
vertices. Those vertices stay unused. A successful add is marked in `used`
and is not tried in a later component.
"""
function _edge_walk!(st::OracleState, fg::FrozenBipartite, eg::EdgeOrderGraph,
        a::Int, b::Int, k::Int, θ::Int, used::BitVector, mark::Vector{Int},
        stamp::Base.RefValue{Int}, best_e::Base.RefValue{Int}, best::Vector{Int};
        in_clique::Union{Nothing,BitVector}=nothing)
    tip, other = _edge_tip(eg, a, b)
    reset_oracle!(st)
    add_left!(st, fg, eg.side_id[other])
    add_left!(st, fg, eg.side_id[tip])
    _, nsel = oracle_score(st, k)
    if nsel < θ
        reset_oracle!(st)
        return false
    end
    used[a] = true
    used[b] = true
    path = [other, tip]
    _note_edge_best!(st, k, θ, best_e, best)
    while true
        placed = false
        nbrs = eg.adj[tip]
        for nbr in nbrs
            used[nbr] && continue
            in_clique !== nothing && !in_clique[nbr] && continue
            _joins_path(eg.adj, mark, stamp, nbr, path) || continue
            add_left!(st, fg, eg.side_id[nbr])
            _, nsel = oracle_score(st, k)
            if nsel < θ
                remove_left!(st, fg)
                continue
            end
            used[nbr] = true
            push!(path, nbr)
            _note_edge_best!(st, k, θ, best_e, best)
            tip = nbr
            placed = true
            break
        end
        placed || break
    end
    return true
end

"""
Mask of one clique from `max_clique_heuristic` on the projection.

Indices match `eg.side_id`. The second value is the clique size.
"""
function _edge_clique_mask(eg::EdgeOrderGraph)
    members, _ = max_clique_heuristic(eg.adj)
    _is_clique(eg.adj, members) ||
        error("edge-order clique heuristic returned a set that is not a clique")
    mask = falses(length(eg.side_id))
    for i in members
        mask[i] = true
    end
    return mask, length(members)
end

"""
    edge_order_search!(st, fg, eg, k, θ; in_clique=nothing) -> (members, components)

Park `st` on the feasible prefix with the most edges. `members` are compact
ids on the searched side. `components` is how many edges grew a pack with at
least θ opposite vertices. `in_clique` limits seeds and growth steps to edges
inside that vertex set.
"""
function edge_order_search!(st::OracleState, fg::FrozenBipartite, eg::EdgeOrderGraph,
        k::Int, θ::Int; in_clique::Union{Nothing,BitVector}=nothing)
    st.search_is_u == eg.search_is_u ||
        error("edge-order oracle side does not match the projection")
    n = length(eg.side_id)
    best = Int[]
    best_e = Ref(-1)
    used = falses(n)
    mark = zeros(Int, max(n, 1))
    stamp = Ref(0)
    components = 0
    for (_, a, b) in eg.seeds
        (used[a] || used[b]) && continue
        if in_clique !== nothing && !(in_clique[a] && in_clique[b])
            continue
        end
        _edge_walk!(st, fg, eg, a, b, k, θ, used, mark, stamp, best_e, best;
            in_clique=in_clique) || continue
        components += 1
    end
    _replay!(st, fg, best)
    return best, components
end

function _edge_order_empty(; search_is_u::Bool=true)
    return (;
        solution=SubGraph(),
        kept=0,
        search_is_u,
        kept_ids=Int[],
        components=0,
        projection_n=0,
        projection_m=0,
        reduced_nU=0,
        reduced_nV=0,
        reduced_m=0,
        edges=0,
        clique_size=0,
        reduce_s=0.0,
        project_s=0.0,
        clique_s=0.0,
        order_s=0.0,
    )
end

"""
    clique_order_edge(g, k, θ; verbose=false, side=:smaller, within_clique=false) -> NamedTuple

Mutates `g`. Applies `super_reduce!`, weights one side's compatibility edges
by common-neighbor count, and walks those edges. Vertex ids in the result
are original. `side` is forwarded to `super_reduce!` and selects the
projection: `:smaller` (the default) and `:larger` follow the reduced graph,
and `:U` / `:V` force that part.

`within_clique` runs `max_clique_heuristic` on the projection and keeps the
walk on edges of that clique.
"""
function clique_order_edge(g::BipartiteGraph, k::Int, θ::Int;
        verbose::Bool=false, side::Symbol=:smaller, within_clique::Bool=false)
    k >= 0 || throw(ArgumentError("k must be ≥ 0, got $k"))
    side in (:smaller, :larger, :U, :V) ||
        throw(ArgumentError("side must be :smaller, :larger, :U, or :V, got :$side"))
    forced = side == :U ? true : side == :V ? false : nothing

    t0 = time_ns()
    super_reduce!(g, k, θ; side=side)
    reduce_s = (time_ns() - t0) / 1e9
    reduced_nU = length(g.adjU)
    reduced_nV = length(g.adjV)
    reduced_m = _edge_count(g)

    fg0 = freeze(g)
    compact_fg, remap = compact_frozen(fg0)
    t1 = time_ns()
    eg = edge_order_graph(compact_fg, k, θ; search_is_u=forced)
    project_s = (time_ns() - t1) / 1e9
    n = length(eg.side_id)
    if n == 0
        verbose && println("Clique-order-edge: compatibility graph is empty " *
                           "(|U|=$reduced_nU |V|=$reduced_nV)")
        return (; _edge_order_empty(; search_is_u=eg.search_is_u)...,
            reduce_s, project_s, reduced_nU, reduced_nV, reduced_m)
    end

    in_clique = nothing
    clique_size = 0
    clique_s = 0.0
    if within_clique
        t_c = time_ns()
        in_clique, clique_size = _edge_clique_mask(eg)
        clique_s = (time_ns() - t_c) / 1e9
    end

    t2 = time_ns()
    st = OracleState(compact_fg, eg.search_is_u)
    _, components = edge_order_search!(st, compact_fg, eg, k, θ; in_clique=in_clique)
    order_s = (time_ns() - t2) / 1e9

    kept_ids = [_side_original(remap, eg.search_is_u, u) for u in st.members]
    bare = st.nL == 0 ? SubGraph() : oracle_solution(compact_fg, st, k)
    solution = st.nL == 0 ? SubGraph() : remap_subgraph(remap, bare)
    edges = st.nL == 0 ? 0 : Subgraph.edge_count(compact_fg, bare)
    verbose && println("Clique-order-edge: search $(eg.search_is_u ? "U" : "V") " *
                       "projection_n=$n projection_m=$(eg.edges) " *
                       (within_clique ? "clique=$clique_size " : "") *
                       "components=$components kept=$(length(kept_ids)) edges=$edges")
    return (;
        solution,
        kept=length(kept_ids),
        search_is_u=eg.search_is_u,
        kept_ids,
        components,
        projection_n=n,
        projection_m=eg.edges,
        reduced_nU,
        reduced_nV,
        reduced_m,
        edges,
        clique_size,
        reduce_s,
        project_s,
        clique_s,
        order_s,
    )
end

"""Run the walk on `g` with no `super_reduce!`. Ids in `kept` are original."""
function _edge_order_direct(g::BipartiteGraph, k::Int, θ::Int;
        search_is_u::Bool=true, within_clique::Bool=false)
    fg, remap = compact_frozen(freeze(g))
    eg = edge_order_graph(fg, k, θ; search_is_u=search_is_u)
    in_clique = nothing
    clique_size = 0
    if within_clique
        in_clique, clique_size = _edge_clique_mask(eg)
    end
    st = OracleState(fg, eg.search_is_u)
    members, components = edge_order_search!(st, fg, eg, k, θ; in_clique=in_clique)
    oracle_state_consistent(st) || error("oracle state diverged")
    if !isempty(members)
        inv = Dict(eg.side_id[i] => i for i in eachindex(eg.side_id))
        _is_clique(eg.adj, [inv[u] for u in members]) ||
            error("walk kept a set that is not a projection clique")
    end
    kept = [_side_original(remap, eg.search_is_u, u) for u in members]
    bare = st.nL == 0 ? SubGraph() : oracle_solution(fg, st, k)
    edges = st.nL == 0 ? 0 : Subgraph.edge_count(fg, bare)
    missing = st.nL == 0 ? 0 : Subgraph.missing_edges(fg, bare)
    seed_ids = Int[]
    if !isempty(eg.seeds)
        _, a, b = eg.seeds[1]
        seed_ids = [
            _side_original(remap, eg.search_is_u, eg.side_id[a]),
            _side_original(remap, eg.search_is_u, eg.side_id[b]),
        ]
    end
    return (; eg, kept, components, edges, missing, bare, seed_ids, clique_size)
end

"""Checks the weighted walk: codegree, tie-break, rejection, restart, biclique."""
function clique_order_edge_selftest()
    # Weight is the common-neighbor count. Defect weight max(0, θ − C) would be 0.
    pair = BipartiteGraph{Nothing}()
    for v in 1:4
        add_edge!(pair, 1, v, nothing)
        add_edge!(pair, 2, v, nothing)
    end
    weighed = _edge_order_direct(pair, 0, 2)
    length(weighed.eg.seeds) == 1 || error("pair edges $(length(weighed.eg.seeds))")
    weighed.eg.seeds[1][1] == 4 || error("pair weight $(weighed.eg.seeds[1][1]), expected 4")
    Set(weighed.kept) == Set([1, 2]) || error("pair kept $(weighed.kept)")
    weighed.edges == 8 || error("pair edges $(weighed.edges)")
    weighed.missing == 0 || error("pair defects")

    # Equal codegree. The edge touching the higher bipartite degree starts,
    # and an equal later pack does not replace it.
    tied = BipartiteGraph{Nothing}()
    for v in 1:4
        add_edge!(tied, 1, v, nothing)
        add_edge!(tied, 2, v, nothing)
    end
    for v in 5:9
        add_edge!(tied, 1, v, nothing)
    end
    for v in 10:13
        add_edge!(tied, 3, v, nothing)
        add_edge!(tied, 4, v, nothing)
    end
    add_edge!(tied, 3, 14, nothing)
    tie = _edge_order_direct(tied, 0, 2)
    tie.eg.seeds[1][1] == 4 || error("tie weight $(tie.eg.seeds[1][1])")
    Set(tie.seed_ids) == Set([1, 2]) || error("tie seed $(tie.seed_ids), expected {1,2}")
    Set(tie.kept) == Set([1, 2]) || error("tie kept $(tie.kept)")
    tie.edges == 8 || error("tie edges $(tie.edges)")
    tie.components == 2 || error("tie components $(tie.components)")

    # U3 shares six neighbors with U1 and none with U2, so it is adjacent to
    # the tip and still cannot join {U1, U2}. U4 shares five with both. The
    # pair has 12 edges; the triple has 15, and U3 stays out.
    trap = BipartiteGraph{Nothing}()
    for v in 1:6
        add_edge!(trap, 1, v, nothing)
        add_edge!(trap, 2, v, nothing)
    end
    add_edge!(trap, 2, 20, nothing)
    for v in 7:12
        add_edge!(trap, 1, v, nothing)
        add_edge!(trap, 3, v, nothing)
    end
    for v in 1:5
        add_edge!(trap, 4, v, nothing)
    end
    trapped = _edge_order_direct(trap, 0, 2)
    Set(trapped.kept) == Set([1, 2, 4]) || error("trap kept $(trapped.kept)")
    trapped.edges == 15 || error("trap edges $(trapped.edges)")
    trapped.missing == 0 || error("trap defects")
    3 in trapped.kept && error("U3 joined a set it does not coexist with")

    # The heaviest edge is a 2×12. The later component is a 3×9 and has more edges.
    parts = BipartiteGraph{Nothing}()
    for v in 1:12
        add_edge!(parts, 1, v, nothing)
        add_edge!(parts, 2, v, nothing)
    end
    for u in 3:5, v in 30:38
        add_edge!(parts, u, v, nothing)
    end
    parted = _edge_order_direct(parts, 0, 2)
    parted.components == 2 || error("restart components $(parted.components)")
    Set(parted.kept) == Set([3, 4, 5]) || error("restart kept $(parted.kept)")
    parted.edges == 27 || error("restart edges $(parted.edges)")
    parted.missing == 0 || error("restart defects")

    full = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(full, u, v, nothing)
    end
    base = freeze(deepcopy(full))
    found = clique_order_edge(full, 2, 5; verbose=false)
    found.kept == 6 || error("biclique kept $(found.kept)")
    found.components == 1 || error("biclique components $(found.components)")
    length(found.solution.U) == 6 && length(found.solution.V) == 6 ||
        error("biclique sides $(length(found.solution.U)), $(length(found.solution.V))")
    Subgraph.edge_count(base, found.solution) == 36 || error("biclique edges")
    Subgraph.missing_edges(base, found.solution) == 0 || error("biclique defects")
    full_v = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(full_v, u, v, nothing)
    end
    forced = clique_order_edge(full_v, 2, 5; side=:V)
    forced.search_is_u && error("side=:V searched U")
    forced.edges == 36 || error("side V edges $(forced.edges)")

    # U3 meets both endpoints pairwise and still drives the opposite pack under θ.
    overlap = BipartiteGraph{Nothing}()
    for v in 1:6
        add_edge!(overlap, 1, v, nothing)
    end
    for v in (1, 2, 3, 4, 7, 8)
        add_edge!(overlap, 2, v, nothing)
    end
    for v in (5, 6, 7, 8)
        add_edge!(overlap, 3, v, nothing)
    end
    before = freeze(deepcopy(overlap))
    overlapped = clique_order_edge(overlap, 0, 2; verbose=false)
    Set(overlapped.kept_ids) == Set([1, 2]) || error("overlap kept $(overlapped.kept_ids)")
    Set(overlapped.solution.U) == Set([1, 2]) || error("overlap U $(overlapped.solution.U)")
    Set(overlapped.solution.V) == Set([1, 2, 3, 4]) || error("overlap V $(overlapped.solution.V)")
    Subgraph.missing_edges(before, overlapped.solution) == 0 || error("overlap defects")
    Subgraph.edge_count(before, overlapped.solution) == 8 || error("overlap edges")

    # U1–U2 and the edges of {U1,U3..U6} all have codegree 20. U2's private
    # neighbors make that edge the first seed, so the open walk spends U1 and
    # the rest of the clique is an 80-edge K4. The clique-restricted walk
    # keeps U1 and returns the 100-edge K5.
    steal = BipartiteGraph{Nothing}()
    for v in 1:20
        add_edge!(steal, 1, v, nothing)
        add_edge!(steal, 2, v, nothing)
    end
    for v in 41:50
        add_edge!(steal, 2, v, nothing)
    end
    for u in (1, 3, 4, 5, 6), v in 21:40
        add_edge!(steal, u, v, nothing)
    end
    stolen = _edge_order_direct(steal, 0, 2)
    stolen.edges == 80 || error("steal edges $(stolen.edges), expected 80")
    Set(stolen.kept) == Set([3, 4, 5, 6]) || error("steal kept $(stolen.kept)")
    held = _edge_order_direct(steal, 0, 2; within_clique=true)
    held.clique_size == 5 || error("steal clique $(held.clique_size)")
    held.edges == 100 || error("steal within-clique edges $(held.edges), expected 100")
    Set(held.kept) == Set([1, 3, 4, 5, 6]) || error("steal within-clique kept $(held.kept)")
    held.missing == 0 || error("steal within-clique defects")
    return true
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    clique_order_edge_selftest()
    println("clique-order-edge selftest passed")
end
