#=
Clique-order heuristic.

Same opening as `clique-find`: `super_reduce!`, then the compatibility
projection of one side. A θ-feasible k-defective biclique's searched side is
always a clique of that projection, so the search only builds cliques.

The degree-ordered prefix of the single maximum clique is not that side.
On graphs where that prefix loses to the θ-heuristic, the heuristic's
vertices are usually inside the clique and the losses are the vertices the
prefix is forced to keep. The rest of the time one of them lies in a
different clique of the same projection. The pack that is returned is the
best of three constructions:

  * the degree prefix of the maximum clique
  * the best subset of its highest-degree members
  * cliques grown from the highest-degree seeds by marginal edge count

Each committed vertex uses `add_left!`. The opposite side is the greedy
cost packing of that left set. A feasible pack is kept when it has strictly
more edges.

`clique-order-edge.jl` is the same one-sided setup with a different order:
projection edges are weighted by common-neighbor count, and the walk follows
those weights. This file keeps the degree prefix.

    julia src/clique-order.jl
=#

const __CLIQUE_ORDER_JL__ = true

isdefined(@__MODULE__, :__CLIQUE_FIND_JL__) || include("clique-find.jl")

"""
Try `ids` in order. `ids` are compact ids on `st`'s searched side.

Each committed vertex updates the opposite-side costs. The best pack is kept.
The scan stops when that pack has fewer than `θ` opposite vertices, because
further adds cannot bring it back. `st` is rewound to the vertices of the
best pack.
"""
function clique_order_keep!(st::OracleState, fg::FrozenBipartite, ids, k::Int, θ::Int)
    best = Int[]
    best_sg = SubGraph()
    for u in ids
        add_left!(st, fg, u)
        _, nsel = oracle_score(st, k)
        sg = oracle_solution(fg, st, k)
        if Subgraph.vertex_count(sg) > 0 &&
           (Subgraph.vertex_count(best_sg) == 0 || better_than_best(fg, sg, best_sg, θ))
            best = copy(st.members)
            best_sg = sg
        end
        nsel < θ && break
    end
    while length(st.members) > length(best)
        remove_left!(st, fg)
    end
    return st
end

const _EXACT_CAP = 20
const _EXACT_NODES = 400_000
const _GROW_SEEDS = 12
const _GROW_CANDS = 64
const _GROW_LIMIT = 80

function _replay!(st::OracleState, fg::FrozenBipartite, members::Vector{Int})
    reset_oracle!(st)
    for u in members
        add_left!(st, fg, u)
    end
    return st
end

function _pack_view(fg::FrozenBipartite, st::OracleState, k::Int)
    members = copy(st.members)
    sg = st.nL == 0 ? SubGraph() : oracle_solution(fg, st, k)
    return members, sg
end

"""Keep `cand` when it beats `curr` on the same rule as the degree prefix."""
_prefer_pack(fg::FrozenBipartite, cand::SubGraph, curr::SubGraph, θ::Int) =
    better_than_best(fg, cand, curr, θ)

"""
Best subset of `ids`. `ids` is degree-ordered and a clique, so every subset
is a legal left side. The scan stops adding once the opposite side is below
`θ`, because further vertices only raise costs.
"""
function _exact_subset!(st::OracleState, fg::FrozenBipartite, ids::Vector{Int},
        k::Int, θ::Int, best_e::Base.RefValue{Int}, best::Vector{Int},
        nodes::Base.RefValue{Int}, i::Int)
    nodes[] += 1
    if i > length(ids)
        e, nsel = oracle_score(st, k)
        if st.nL >= θ && nsel >= θ && e > best_e[]
            best_e[] = e
            empty!(best)
            append!(best, st.members)
        end
        return best
    end
    nodes[] > _EXACT_NODES && return best
    # Add the next vertex before skipping it. `ids` is highest-degree first,
    # so a strong pack turns up early and the bound can drop the rest.
    add_left!(st, fg, ids[i])
    e, nsel = oracle_score(st, k)
    if nsel >= θ
        if st.nL >= θ && e > best_e[]
            best_e[] = e
            empty!(best)
            append!(best, st.members)
        end
        if !(best_e[] >= 0 && oracle_upper_bound(st, k) <= best_e[])
            _exact_subset!(st, fg, ids, k, θ, best_e, best, nodes, i + 1)
        end
    end
    remove_left!(st, fg)
    nodes[] > _EXACT_NODES && return best
    _exact_subset!(st, fg, ids, k, θ, best_e, best, nodes, i + 1)
    return best
end

"""Edge-greedy clique grown from projection index `seed`. Returns compact ids."""
function _grow_from!(st::OracleState, fg::FrozenBipartite, cg::CliqueGraph,
        seed::Int, k::Int, θ::Int, bi::Vector{Int})
    reset_oracle!(st)
    add_left!(st, fg, cg.side_id[seed])
    best = Int[]
    best_e = -1
    e, nsel = oracle_score(st, k)
    if st.nL >= θ && nsel >= θ
        best_e = e
        best = copy(st.members)
    end
    cand = copy(cg.adj[seed])
    mark = zeros(Int, length(cg.adj))
    stamp = 0
    selected = 1
    while !isempty(cand) && selected < _GROW_LIMIT
        if length(cand) > _GROW_CANDS
            sort!(cand; by=j -> (-bi[j], j))
            resize!(cand, _GROW_CANDS)
        end
        pick = 0
        pick_e = -1
        for j in cand
            add_left!(st, fg, cg.side_id[j])
            e, nsel = oracle_score(st, k)
            remove_left!(st, fg)
            nsel < θ && continue
            if e > pick_e
                pick_e = e
                pick = j
            end
        end
        pick == 0 && break
        add_left!(st, fg, cg.side_id[pick])
        e, nsel = oracle_score(st, k)
        if st.nL >= θ && nsel >= θ && e > best_e
            best_e = e
            best = copy(st.members)
        end
        stamp += 1
        for j in cg.adj[pick]
            mark[j] = stamp
        end
        nxt = Int[]
        for j in cand
            j == pick && continue
            mark[j] == stamp && push!(nxt, j)
        end
        cand = nxt
        selected += 1
    end
    return best
end

"""
    clique_order_search!(st, fg, cg, proj, k, θ) -> OracleState

Park `st` on the best left set found from the maximum clique `proj`
(indices into `cg`) and from edge-greedy growth on the rest of `cg`.
"""
function clique_order_search!(st::OracleState, fg::FrozenBipartite, cg::CliqueGraph,
        proj::Vector{Int}, k::Int, θ::Int)
    ordered = _clique_degree_order(fg, cg, proj)
    ids = [cg.side_id[i] for i in ordered]
    reset_oracle!(st)
    clique_order_keep!(st, fg, ids, k, θ)
    best_members, best_sg = _pack_view(fg, st, k)

    if !isempty(ids)
        use = length(ids) > _EXACT_CAP ? ids[1:_EXACT_CAP] : ids
        reset_oracle!(st)
        found = Int[]
        _exact_subset!(st, fg, use, k, θ, Ref(-1), found, Ref(0), 1)
        if !isempty(found)
            _replay!(st, fg, found)
            members, sg = _pack_view(fg, st, k)
            if _prefer_pack(fg, sg, best_sg, θ)
                best_members = members
                best_sg = sg
            end
        end
    end

    n = length(cg.side_id)
    if n > 0
        bi = [_side_degree(fg, cg.search_is_u, cg.side_id[i]) for i in 1:n]
        seeds = sortperm(bi; rev=true)
        nseeds = min(length(seeds), _GROW_SEEDS)
        for s in 1:nseeds
            grown = _grow_from!(st, fg, cg, seeds[s], k, θ, bi)
            isempty(grown) && continue
            _replay!(st, fg, grown)
            members, sg = _pack_view(fg, st, k)
            if _prefer_pack(fg, sg, best_sg, θ)
                best_members = members
                best_sg = sg
            end
        end
    end

    _replay!(st, fg, best_members)
    return st
end

"""Clique indices sorted by descending bipartite degree, then smaller compact id."""
function _clique_degree_order(fg::FrozenBipartite, cg::CliqueGraph, members::Vector{Int})
    return sort(members; by=i -> (
        -_side_degree(fg, cg.search_is_u, cg.side_id[i]),
        cg.side_id[i],
    ))
end

function _side_original(remap::GraphRemapping, search_is_u::Bool, compact::Int)
    return search_is_u ? remap.u_original[compact] : remap.v_original[compact]
end

function _clique_order_empty(; search_is_u::Bool=true)
    return (;
        solution=SubGraph(),
        clique_size=0,
        kept=0,
        degeneracy=0,
        search_is_u,
        visit=Int[],
        kept_ids=Int[],
        projection_n=0,
        projection_m=0,
        reduced_nU=0,
        reduced_nV=0,
        reduced_m=0,
        edges=0,
        reduce_s=0.0,
        project_s=0.0,
        clique_s=0.0,
        order_s=0.0,
    )
end

"""
    clique_order(g, k, θ; verbose=false, side=:smaller) -> NamedTuple

Mutates `g`. Applies `super_reduce!`, finds a large compatibility clique, and
keeps the best pack among the degree prefix, the best subset of that clique,
and cliques grown by marginal edges. Vertex ids are original.

`side` is forwarded to `super_reduce!`. `:smaller` (the default) and `:larger`
still let the compatibility graph pick the smaller side of the reduced graph.
`:U` and `:V` project that part and search it, including when the two parts
have equal size.
"""
function clique_order(g::BipartiteGraph, k::Int, θ::Int;
        verbose::Bool=false, side::Symbol=:smaller)
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
    cg = compatibility_graph(compact_fg, k, θ; search_is_u=forced)
    project_s = (time_ns() - t1) / 1e9
    n = length(cg.side_id)
    if n == 0
        verbose && println("Clique-order: compatibility graph is empty " *
                           "(|U|=$reduced_nU |V|=$reduced_nV)")
        return (; _clique_order_empty(; search_is_u=cg.search_is_u)...,
            reduce_s, project_s, reduced_nU, reduced_nV, reduced_m)
    end

    t2 = time_ns()
    members, degeneracy = max_clique_heuristic(cg.adj)
    clique_s = (time_ns() - t2) / 1e9
    _is_clique(cg.adj, members) ||
        error("clique-order found a vertex set that is not a clique")

    t3 = time_ns()
    ordered = _clique_degree_order(compact_fg, cg, members)
    ids = [cg.side_id[i] for i in ordered]
    st = OracleState(compact_fg, cg.search_is_u)
    clique_order_search!(st, compact_fg, cg, members, k, θ)
    order_s = (time_ns() - t3) / 1e9

    visit = [_side_original(remap, cg.search_is_u, u) for u in ids]
    kept_ids = [_side_original(remap, cg.search_is_u, u) for u in st.members]
    bare = st.nL == 0 ? SubGraph() : oracle_solution(compact_fg, st, k)
    solution = st.nL == 0 ? SubGraph() : remap_subgraph(remap, bare)
    edges = st.nL == 0 ? 0 : Subgraph.edge_count(compact_fg, bare)
    verbose && println("Clique-order: search $(cg.search_is_u ? "U" : "V") " *
                       "projection_n=$n degeneracy=$degeneracy " *
                       "clique=$(length(members)) kept=$(length(kept_ids)) edges=$edges")
    return (;
        solution,
        clique_size=length(members),
        kept=length(kept_ids),
        degeneracy,
        search_is_u=cg.search_is_u,
        visit,
        kept_ids,
        projection_n=n,
        projection_m=cg.edges,
        reduced_nU,
        reduced_nV,
        reduced_m,
        edges,
        reduce_s,
        project_s,
        clique_s,
        order_s,
    )
end

"""Checks the degree-ordered keep on a hand-built overlap and a full biclique."""
function clique_order_selftest()
    # U1 and U2 share {V1..V4}. U3 misses that intersection, so with k = 0 the
    # opposite pack drops below θ and the scan returns {U1, U2}. U4 would have
    # fit after undoing U3, and is not tried.
    g = BipartiteGraph{Nothing}()
    for v in 1:6
        add_edge!(g, 1, v, nothing)
    end
    for v in (1, 2, 3, 4, 7, 8)
        add_edge!(g, 2, v, nothing)
    end
    for v in (5, 6, 7, 8, 9)
        add_edge!(g, 3, v, nothing)
    end
    for v in (1, 2, 9)
        add_edge!(g, 4, v, nothing)
    end
    fg, remap = compact_frozen(freeze(g))
    length(fg.u_ids) == 4 || error("keep fixture |U|")
    order = sort(collect(1:4); by=u -> (-length(neighbor_range_u(fg, u)), u))
    st = OracleState(fg, true)
    clique_order_keep!(st, fg, order, 0, 2)
    oracle_state_consistent(st) || error("oracle state diverged after keep")
    kept = [remap.u_original[u] for u in st.members]
    Set(kept) == Set([1, 2]) || error("kept $kept, expected {1,2}")
    4 in kept && error("U4 was tried after the opposite side fell below θ")
    sg = oracle_solution(fg, st, 0)
    Set(remap.v_original[v] for v in sg.V) == Set([1, 2, 3, 4]) ||
        error("opposite pack $(collect(remap.v_original[v] for v in sg.V))")
    Subgraph.missing_edges(fg, sg) == 0 || error("pack exceeds k")

    full = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(full, u, v, nothing)
    end
    base = freeze(deepcopy(full))
    found = clique_order(full, 2, 5; verbose=false)
    found.clique_size == 6 || error("biclique clique $(found.clique_size)")
    found.kept == 6 || error("biclique kept $(found.kept)")
    length(found.solution.U) == 6 && length(found.solution.V) == 6 ||
        error("biclique sides $(length(found.solution.U)), $(length(found.solution.V))")
    Subgraph.edge_count(base, found.solution) == 36 || error("biclique edges")
    Subgraph.missing_edges(base, found.solution) == 0 || error("biclique defects")

    # Pairwise common-neighbor clique. U3 misses N(U1) ∩ N(U2), so k = 0 drops it.
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
    overlapped = clique_order(overlap, 0, 2; verbose=false)
    overlapped.clique_size == 3 || error("overlap clique $(overlapped.clique_size)")
    overlapped.kept == 2 || error("overlap kept $(overlapped.kept)")
    Set(overlapped.kept_ids) == Set([1, 2]) || error("overlap kept ids $(overlapped.kept_ids)")
    overlapped.visit[end] == 3 || error("U3 should be tried last, visit $(overlapped.visit)")
    Set(overlapped.solution.U) == Set([1, 2]) || error("overlap U $(overlapped.solution.U)")
    Set(overlapped.solution.V) == Set([1, 2, 3, 4]) || error("overlap V $(overlapped.solution.V)")
    Subgraph.missing_edges(before, overlapped.solution) == 0 || error("overlap defects")
    Subgraph.edge_count(before, overlapped.solution) == 8 || error("overlap edges")

    # A later θ-feasible prefix with fewer edges must not replace the best.
    # U1 and U2 cover V1..V8 (16 edges). U3 keeps the pack θ-feasible at 3×3
    # (9 edges) and is tried last because its degree is smaller.
    keep_best = BipartiteGraph{Nothing}()
    for u in 1:2, v in 1:8
        add_edge!(keep_best, u, v, nothing)
    end
    for v in 1:3
        add_edge!(keep_best, 3, v, nothing)
    end
    kept_fg, kept_remap = compact_frozen(freeze(keep_best))
    order = sort(collect(1:3); by=u -> (-length(neighbor_range_u(kept_fg, u)), u))
    st = OracleState(kept_fg, true)
    clique_order_keep!(st, kept_fg, order, 0, 2)
    kept = [kept_remap.u_original[u] for u in st.members]
    Set(kept) == Set([1, 2]) || error("best prefix kept $kept, expected {1,2}")
    sg = oracle_solution(kept_fg, st, 0)
    length(sg.V) == 8 || error("best prefix opposite $(length(sg.V)), expected 8")
    Subgraph.edge_count(kept_fg, sg) == 16 || error("best prefix edges")

    # U6 has the highest degree, so the degree prefix has to keep it, but it
    # shares only V1..V3 with the K_{5,12} on U1..U5. That prefix is feasible
    # and far below 60 edges. The subset search has to drop U6.
    poison = BipartiteGraph{Nothing}()
    for u in 1:5, v in 1:12
        add_edge!(poison, u, v, nothing)
    end
    for v in 1:3
        add_edge!(poison, 6, v, nothing)
    end
    for v in 13:40
        add_edge!(poison, 6, v, nothing)
    end
    pfg, premap = compact_frozen(freeze(poison))
    pcg = compatibility_graph(pfg, 2, 5; search_is_u=true)
    pmembers, _ = max_clique_heuristic(pcg.adj)
    length(pmembers) == 6 || error("poison clique $(length(pmembers)), expected 6")
    pids = [pcg.side_id[i] for i in _clique_degree_order(pfg, pcg, pmembers)]
    pst = OracleState(pfg, true)
    clique_order_keep!(pst, pfg, pids, 2, 5)
    psg = oracle_solution(pfg, pst, 2)
    pdeg = Subgraph.edge_count(pfg, psg)
    pdeg < 60 || error("degree prefix already has $pdeg edges")
    clique_order_search!(pst, pfg, pcg, pmembers, 2, 5)
    psg = oracle_solution(pfg, pst, 2)
    Subgraph.edge_count(pfg, psg) >= 60 ||
        error("search edges $(Subgraph.edge_count(pfg, psg)), degree prefix $pdeg")
    Set(premap.u_original[u] for u in pst.members) == Set(1:5) ||
        error("search kept $(pst.members)")
    poisoned = clique_order(poison, 2, 5; side=:U)
    poisoned.edges >= 60 || error("end-to-end poison edges $(poisoned.edges)")
    return true
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    clique_order_selftest()
    println("clique-order selftest passed")
end
