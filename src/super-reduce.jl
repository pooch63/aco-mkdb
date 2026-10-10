const __SUPER_REDUCE_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__IO_JL__) || include("io.jl")
isdefined(@__MODULE__, :__PATHS_JL__) || include("paths.jl")
isdefined(@__MODULE__, :__THETA_HEURISTIC_JL__) || include("theta_heuristic.jl")

using Random

# After a shell, finish the core by materializing edges when at most this many
# vertices still meet the degree floor. Larger sets stay on the shell loop.
const SUPER_REDUCE_EDGE_CAP = 20_000

# Stop the outer core+CNN loop and each side's degree peel after this many
# rounds, even if another round could still delete vertices.
const SUPER_REDUCE_MAX_ROUNDS = 10

"""
    super_reduce!(g, k, θ; edge_cap=SUPER_REDUCE_EDGE_CAP,
                  max_rounds=SUPER_REDUCE_MAX_ROUNDS,
                  prune_edges=true, prune_vertices=true, stages=2,
                  efficient=true, side=:smaller, incumbent=nothing,
                  stats=nothing) -> g

With `efficient` (the default), run each requested stage once: the defect-weight
projection, then one `common_neighbor_reduction!`, then the common-neighbor-weight
projection. `efficient=false` repeats that pair of deletions until neither
removes a vertex or an edge, or `max_rounds` of the loop have run. In that
mode both weightings share one projection and run before each common-neighbor pass.

`side` selects the projected part: `:smaller` (the default) or `:larger`.
Equal part sizes project `U`. Two vertices on that part are joined when they
share `C ≥ θ − k` opposite neighbors. A part with fewer than `θ` vertices is
deleted whichever side is projected, since it cannot sit in a θ-feasible
biclique. The first stage stores weight `max(0, θ − C)`,
a nonnegative defect lower bound of at most `k`, and deletes vertices and
edges that cannot sit in a clique of order ≥ θ. The second stage writes the
common-neighbor count `C` onto those edges and deletes vertices that cannot
match the incumbent solution's neighborhood weight. `incumbent`, when
given, is that solution; otherwise stage 2 calls the θ-heuristic. Replace
that side with the surviving vertices.

Any two vertices that sit together in a θ-feasible k-defective biclique share
at least `θ − k` neighbors, and each of them has at least `θ − 1` partners on
its own side. The defect pass and the unweighted clique cuts delete only
vertices that cannot belong to such a biclique. The common-neighbor pass
additionally drops a vertex `v` when the sum of every edge weight in `E(N(v))`
is below that same sum for every vertex of the θ-heuristic solution.

Each pass repeats, in order, until the projection stops shrinking: delete
vertices of degree less than `θ − 1`; then the edge cuts; then, on the
common-neighbor weighting, the neighborhood-weight cut.

`prune_edges` drops a projected edge `{u,v}` when `S = N(u) ∩ N(v)` has fewer
than `θ − 2` vertices, when `E(S)` has fewer than `(θ − 2)(θ − 3) / 2` edges,
or, on the defect weighting, when every triangle through the edge has total
defect weight above `2k`. One missing edge is charged to two pairs, so the
three edge weights may sum to `2k`.

`prune_vertices` drops `v` when fewer than `θ − 1` projected neighbors remain,
or, after the reweight, when the edges of `E(N(v))` sum to less than the
incumbent neighborhood weight.

The weighted rules run only when the surviving side still has at most
`edge_cap` vertices (`edge_cap < 0` uses `SUPER_REDUCE_EDGE_CAP`). `stats`,
when given, receives `:rounds`, `:core_removed`, `:weight_edges_cut`,
`:weight_vertices_cut`, `:mwc_vertices_cut`, and `:weight_skipped`.
"""
function super_reduce!(g::BipartiteGraph, k::Int, θ::Int;
        edge_cap::Int=SUPER_REDUCE_EDGE_CAP, max_rounds::Int=SUPER_REDUCE_MAX_ROUNDS,
        prune_edges::Bool=true, prune_vertices::Bool=true, stages::Int=2,
        efficient::Bool=true, side::Symbol=:smaller,
        incumbent::Union{Nothing,SubGraph}=nothing,
        stats::Union{Nothing,Dict{Symbol,Int}}=nothing)
    num_U = isempty(g.adjU) ? 0 : maximum(keys(g.adjU))
    num_V = isempty(g.adjV) ? 0 : maximum(keys(g.adjV))
    rounds = 0
    core_removed = 0
    if stats !== nothing
        get!(stats, :weight_edges_cut, 0)
        get!(stats, :weight_vertices_cut, 0)
        get!(stats, :mwc_vertices_cut, 0)
        get!(stats, :weight_skipped, 0)
    end
    if efficient
        core_removed += _prune_side!(g, k, θ, edge_cap, max_rounds,
            prune_edges, prune_vertices, stages, stats;
            only_stage=1, side=side, incumbent=incumbent)
        rounds = 1
        if stages >= 2
            common_neighbor_reduction!(g, k, θ, num_U, num_V)
            core_removed += _prune_side!(g, k, θ, edge_cap, max_rounds,
                prune_edges, prune_vertices, stages, stats;
                only_stage=2, side=side, incumbent=incumbent)
            # Stage 2 deletes vertices after the common-neighbor pass, so the
            # opposite side can fall under the degree floor without another scan.
            core_removed += _drop_low_degree!(g, θ - k)
            rounds = 2
        end
    else
        while rounds < max_rounds
            u0 = length(g.adjU)
            v0 = length(g.adjV)
            e0 = _edge_count(g)
            core_removed += _prune_side!(g, k, θ, edge_cap, max_rounds,
                prune_edges, prune_vertices, stages, stats; side=side, incumbent=incumbent)
            common_neighbor_reduction!(g, k, θ, num_U, num_V)
            rounds += 1
            length(g.adjU) == u0 && length(g.adjV) == v0 && _edge_count(g) == e0 && break
        end
    end
    if stats !== nothing
        stats[:rounds] = rounds
        stats[:core_removed] = core_removed
    end
    return g
end

"""
Stage 2 only: common-neighbor weights and the incumbent neighborhood-weight cut,
then a degree-floor cascade. Used when a better edge lower bound should prune
again without repeating the defect-weight stage.
"""
function super_reduce_stage2!(g::BipartiteGraph, k::Int, θ::Int;
        edge_cap::Int=SUPER_REDUCE_EDGE_CAP, max_rounds::Int=SUPER_REDUCE_MAX_ROUNDS,
        prune_edges::Bool=true, prune_vertices::Bool=true, side::Symbol=:smaller,
        incumbent::Union{Nothing,SubGraph}=nothing,
        stats::Union{Nothing,Dict{Symbol,Int}}=nothing)
    _prune_side!(g, k, θ, edge_cap, max_rounds,
        prune_edges, prune_vertices, 2, stats;
        only_stage=2, side=side, incumbent=incumbent)
    _drop_low_degree!(g, θ - k)
    return g
end

_edge_count(g::BipartiteGraph) = sum(length, values(g.adjU); init=0)

"""Delete vertices with fewer than `τ` neighbors, cascading as degrees fall."""
function _drop_low_degree!(g::BipartiteGraph, τ::Int)
    τ <= 0 && return 0
    removed = 0
    changed = true
    while changed
        changed = false
        for is_u in (true, false)
            adj = is_u ? g.adjU : g.adjV
            victims = Int[]
            for (id, nbrs) in adj
                length(nbrs) < τ && push!(victims, id)
            end
            for id in victims
                rem_node_structural!(g, is_u, id)
                removed += 1
                changed = true
            end
        end
    end
    return removed
end

"""True when the projection runs on `U` for this `side` (`:smaller` or `:larger`)."""
function _project_u(nU::Int, nV::Int, side::Symbol)
    side == :smaller && return nU <= nV
    side == :larger && return nU >= nV
    throw(ArgumentError("side must be :smaller or :larger, got :$side"))
end

"""Delete vertices on the chosen side that lie outside its (θ − 1)-core, then apply the weighted rules. Returns how many were deleted."""
function _prune_side!(g::BipartiteGraph, k::Int, θ::Int, edge_cap::Int, max_rounds::Int,
        prune_edges::Bool, prune_vertices::Bool, stages::Int, stats::Union{Nothing,Dict{Symbol,Int}};
        only_stage::Union{Nothing,Int}=nothing, side::Symbol=:smaller,
        incumbent::Union{Nothing,SubGraph}=nothing)
    nU = length(g.adjU)
    nV = length(g.adjV)
    (nU == 0 || nV == 0) && return 0
    min_edges = θ - 1
    min_edges <= 0 && return 0

    small_is_u = _project_u(nU, nV, side)
    small = small_is_u ? g.adjU : g.adjV
    n = length(small)
    ids = collect(keys(small))
    opp_n = small_is_u ? nV : nU
    if n < θ
        for id in ids
            rem_node_structural!(g, small_is_u, id)
        end
        return n
    end
    # The other part is too small to host a θ-feasible biclique. This fires
    # when `side=:larger` and the smaller part is below θ.
    if opp_n < θ
        opp_is_u = !small_is_u
        opp = opp_is_u ? g.adjU : g.adjV
        for id in collect(keys(opp))
            rem_node_structural!(g, opp_is_u, id)
        end
        return opp_n
    end

    τ = θ - k
    # Every pair is compatible, and the side is large enough for a clique of minimum degree θ − 1.
    τ <= 0 && return 0

    cand = Int[]
    sizehint!(cand, n)
    for id in ids
        length(small[id]) >= τ && push!(cand, id)
    end
    keep = _compatibility_core(cand, small, τ, min_edges, edge_cap, max_rounds)
    if (prune_edges || prune_vertices) && !isempty(keep)
        cap = edge_cap < 0 ? SUPER_REDUCE_EDGE_CAP : edge_cap
        if length(keep) <= cap
            keep, ecut, vcut = _weight_prune(keep, small, k, θ, prune_edges, prune_vertices;
                g=g, small_is_u=small_is_u, stages=stages, only_stage=only_stage,
                stats=stats, incumbent=incumbent)
            if stats !== nothing
                stats[:weight_edges_cut] = get(stats, :weight_edges_cut, 0) + ecut
                stats[:weight_vertices_cut] = get(stats, :weight_vertices_cut, 0) + vcut
            end
        elseif stats !== nothing
            stats[:weight_skipped] = get(stats, :weight_skipped, 0) + 1
        end
    end
    keep_set = Set(keep)
    removed = 0
    for id in ids
        id in keep_set && continue
        rem_node_structural!(g, small_is_u, id)
        removed += 1
    end
    return removed
end

"""
(θ − 1)-core of the graph on `cand` where an edge exists iff the two vertices
have ≥ τ common neighbors in `small_adj`'s opposite side.
"""
function _compatibility_core(cand::Vector{Int}, small_adj::Dict{Int,Set{Int}},
        τ::Int, min_edges::Int, edge_cap::Int, max_rounds::Int)
    n = length(cand)
    (n == 0 || n < min_edges + 1) && return Int[]

    cand_to_large, buckets = _candidate_buckets(cand, small_adj)
    alive = trues(n)
    stamp = zeros(Int, n)
    cnt = zeros(Int, n)
    touched = Int[]
    cur = Ref(0)

    shells = 0
    while shells < max_rounds
        shells += 1
        victims = Int[]
        n_keep = 0
        for i in 1:n
            alive[i] || continue
            if _meets_degree(i, alive, min_edges, cand_to_large, buckets, τ, stamp, cnt, cur)
                n_keep += 1
            else
                push!(victims, i)
            end
        end
        if isempty(victims)
            return [cand[i] for i in 1:n if alive[i]]
        end
        for i in victims
            alive[i] = false
        end
        n_keep == 0 && return Int[]
        # One edge build finishes every later shell when the survivor set is modest.
        if edge_cap >= 0 && n_keep <= edge_cap
            deg = fill(min_edges, n)
            return _core_from_edges(cand, alive, deg, min_edges, cand_to_large, buckets, τ, stamp, cnt, touched, cur)
        end
        _compact_buckets!(buckets, alive)
    end
    return [cand[i] for i in 1:n if alive[i]]
end

function _compact_buckets!(buckets::Vector{Vector{Int}}, alive::BitVector)
    for b in buckets
        w = 0
        @inbounds for x in b
            if alive[x]
                w += 1
                b[w] = x
            end
        end
        resize!(b, w)
    end
    return buckets
end

function _candidate_buckets(cand::Vector{Int}, small_adj::Dict{Int,Set{Int}})
    n = length(cand)
    large_index = Dict{Int,Int}()
    buckets = Vector{Int}[]
    cand_to_large = Vector{Vector{Int}}(undef, n)
    for i in 1:n
        nbrs = small_adj[cand[i]]
        mapped = Vector{Int}(undef, length(nbrs))
        t = 0
        for v in nbrs
            j = get(large_index, v, 0)
            if j == 0
                j = length(buckets) + 1
                large_index[v] = j
                push!(buckets, Int[])
            end
            t += 1
            @inbounds mapped[t] = j
            push!(buckets[j], i)
        end
        cand_to_large[i] = mapped
    end
    return cand_to_large, buckets
end

"""True when `i` has at least `min_edges` selected partners with ≥ τ common neighbors."""
function _meets_degree(i::Int, selected::BitVector, min_edges::Int,
        cand_to_large::Vector{Vector{Int}}, buckets::Vector{Vector{Int}},
        τ::Int, stamp::Vector{Int}, cnt::Vector{Int}, cur::Ref{Int})
    cur[] += 1
    stamp_id = cur[]
    found = 0
    @inbounds for j in cand_to_large[i]
        for w in buckets[j]
            selected[w] || continue
            c = stamp[w] == stamp_id ? cnt[w] + 1 : 1
            stamp[w] = stamp_id
            cnt[w] = c
            if w != i && c == τ
                found += 1
                found >= min_edges && return true
            end
        end
    end
    return false
end

function _core_from_edges(ids::Vector{Int}, alive::BitVector, deg::Vector{Int}, min_edges::Int,
        cand_to_large::Vector{Vector{Int}}, buckets::Vector{Vector{Int}},
        τ::Int, stamp::Vector{Int}, cnt::Vector{Int}, touched::Vector{Int}, cur::Ref{Int})
    n = length(ids)
    in_core = BitVector(undef, n)
    @inbounds for i in 1:n
        in_core[i] = alive[i] && deg[i] >= min_edges
    end
    comp = [Int[] for _ in 1:n]
    @inbounds for i in 1:n
        in_core[i] || continue
        cur[] += 1
        stamp_id = cur[]
        empty!(touched)
        for j in cand_to_large[i]
            for w in buckets[j]
                in_core[w] || continue
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
            cnt[w] >= τ || continue
            push!(comp[i], w)
            push!(comp[w], i)
        end
    end

    rdeg = [length(comp[i]) for i in 1:n]
    removed = falses(n)
    q = Int[]
    for i in 1:n
        if !in_core[i] || rdeg[i] < min_edges
            removed[i] = true
            in_core[i] && push!(q, i)
        end
    end
    head = 1
    while head <= length(q)
        i = q[head]
        head += 1
        for j in comp[i]
            removed[j] && continue
            rdeg[j] -= 1
            if rdeg[j] < min_edges
                removed[j] = true
                push!(q, j)
            end
        end
    end
    return [ids[i] for i in 1:n if !removed[i]]
end

@inline _choose2(n::Int) = n < 2 ? 0 : n * (n - 1) ÷ 2

"""
Weighted projection, reduced twice on the same adjacency.

`ids` are original vertex ids on one side. `small_adj` is that side's
bipartite adjacency. Edges exist for pairs with `C ≥ θ − k`. The first pass
uses defect weights `max(0, θ − C)`. The second pass, when `prune_vertices`
is set, overwrites those weights with `C` and applies the neighborhood-weight
cut. `incumbent`, or a θ-heuristic solution when `incumbent` is omitted,
sets that cut's threshold; without `g` and without `incumbent` the threshold is 0.

Returns surviving ids, how many projected edges were deleted, and how many
vertices were deleted.
"""
function _weight_prune(ids::Vector{Int}, small_adj::Dict{Int,Set{Int}},
        k::Int, θ::Int, prune_edges::Bool, prune_vertices::Bool;
        g::Union{Nothing,BipartiteGraph}=nothing, small_is_u::Bool=true, stages::Int=2,
        only_stage::Union{Nothing,Int}=nothing,
        incumbent::Union{Nothing,SubGraph}=nothing,
        stats::Union{Nothing,Dict{Symbol,Int}}=nothing)
    n = length(ids)
    n == 0 && return Int[], 0, 0
    τ = θ - k
    # Pairwise edges still need codegree ≥ θ − k (weight ≤ k). A triangle's
    # three weights sum to at most 2k, so its codegree floor is θ − 2k.
    τ_clique = θ - 2 * k
    wadj, codeg, lists, buckets = _weighted_projection(ids, small_adj, θ, τ)
    alive = trues(n)
    min_edges = θ - 1
    need = θ - 2

    m = length(buckets)
    ostamp = zeros(Int, max(m, 1))
    cstamp = zeros(Int, n)
    sstamp = zeros(Int, n)
    ccnt = zeros(Int, n)
    cur = Ref(0)
    sbuf = Int[]
    run1 = only_stage === nothing || only_stage == 1
    run2 = only_stage === nothing ? stages >= 2 : only_stage == 2
    ecut = 0
    vcut = 0
    if run1
        ecut, vcut = _reduce_projection!(wadj, alive, lists, buckets, τ_clique, need, min_edges,
            Int[], false, prune_edges, prune_vertices,
            ostamp, cstamp, sstamp, ccnt, cur, sbuf, stats)
    end
    if run2 && prune_vertices && any(alive)
        _apply_codegree_weights!(wadj, codeg)
        members = _stage2_members(g, k, θ, ids, alive, small_is_u, incumbent)
        e2, v2 = _reduce_projection!(wadj, alive, lists, buckets, τ_clique, need, min_edges,
            members, true, prune_edges, true,
            ostamp, cstamp, sstamp, ccnt, cur, sbuf, stats)
        ecut += e2
        vcut += v2
    end
    return [ids[i] for i in 1:n if alive[i]], ecut, vcut
end

"""Degree core, edge cuts, then vertex clique cuts, repeated until nothing else drops."""
function _reduce_projection!(wadj::Vector{Dict{Int,Int}}, alive::BitVector,
        lists::Vector{Vector{Int}}, buckets::Vector{Vector{Int}},
        τ::Int, need::Int, min_edges::Int, members::Vector{Int}, use_weight::Bool,
        prune_edges::Bool, prune_vertices::Bool,
        ostamp::Vector{Int}, cstamp::Vector{Int}, sstamp::Vector{Int},
        ccnt::Vector{Int}, cur::Ref{Int}, sbuf::Vector{Int},
        stats::Union{Nothing,Dict{Symbol,Int}})
    ecut = 0
    vcut = 0
    passes = sum(length, wadj; init=0) ÷ 2 + length(wadj) + 1
    while passes > 0
        passes -= 1
        dv = prune_vertices ? _apply_vertex_rule!(alive, wadj, min_edges) : 0
        cut = prune_edges ? _cut_heavy_edges!(wadj, alive, lists, buckets, τ, need,
            ostamp, cstamp, sstamp, ccnt, cur, sbuf) : 0
        # Recompute from whoever is still alive. A fixed threshold falls once
        # deleted vertices take edges out of the incumbent neighborhoods.
        lb = use_weight ? _incumbent_neighborhood_lb(members, wadj, alive) : 0
        mw = (prune_vertices && use_weight) ? _prune_neighborhood_cliques!(alive, wadj, lb) : 0
        ecut += cut
        vcut += dv + mw
        if stats !== nothing && mw > 0
            stats[:mwc_vertices_cut] = get(stats, :mwc_vertices_cut, 0) + mw
        end
        dv == 0 && cut == 0 && mw == 0 && break
    end
    return ecut, vcut
end

"""
Edges for pairs with `C ≥ τ`. `wadj` holds defect weights `max(0, θ − C)` and
`codeg` holds `C`. The opposite-side index lists score triangles.
"""
function _weighted_projection(ids::Vector{Int}, small_adj::Dict{Int,Set{Int}}, θ::Int, τ::Int)
    n = length(ids)
    lists, buckets = _candidate_buckets(ids, small_adj)
    stamp = zeros(Int, n)
    cnt = zeros(Int, n)
    touched = Int[]
    cur = Ref(0)
    wadj = [Dict{Int,Int}() for _ in 1:n]
    codeg = [Dict{Int,Int}() for _ in 1:n]
    τ <= 0 && return wadj, codeg, lists, buckets
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
            c = cnt[w]
            c >= τ || continue
            weight = max(0, θ - c)
            wadj[i][w] = weight
            wadj[w][i] = weight
            codeg[i][w] = c
            codeg[w][i] = c
        end
    end
    return wadj, codeg, lists, buckets
end

"""Replace each surviving defect weight with the common-neighbor count `C`."""
function _apply_codegree_weights!(wadj::Vector{Dict{Int,Int}}, codeg::Vector{Dict{Int,Int}})
    for i in eachindex(wadj)
        for j in keys(wadj[i])
            j <= i && continue
            c = codeg[i][j]
            wadj[i][j] = c
            wadj[j][i] = c
        end
    end
    return wadj
end

"""
Drop projected edges that cannot lie in a θ-clique.

`S = N(u) ∩ N(v)` in the projection. The edge goes when `|S| < θ − 2`, when
`E(S)` has fewer than `(θ − 2)(θ − 3) / 2` edges, or when every `z ∈ S` forms
a triangle whose opposite-side codegree is below `τ = θ − 2k`.
"""
function _cut_heavy_edges!(wadj::Vector{Dict{Int,Int}}, alive::BitVector,
        lists::Vector{Vector{Int}}, buckets::Vector{Vector{Int}},
        τ::Int, need::Int,
        ostamp::Vector{Int}, cstamp::Vector{Int}, sstamp::Vector{Int},
        ccnt::Vector{Int}, cur::Ref{Int}, sbuf::Vector{Int})
    edges = Tuple{Int,Int}[]
    for i in eachindex(wadj)
        alive[i] || continue
        for j in keys(wadj[i])
            j > i && alive[j] && push!(edges, (i, j))
        end
    end
    cut = 0
    for (i, j) in edges
        haskey(wadj[i], j) || continue
        if _projected_edge_ok(i, j, wadj, alive, lists, buckets, τ, need,
                ostamp, cstamp, sstamp, ccnt, cur, sbuf)
            continue
        end
        delete!(wadj[i], j)
        delete!(wadj[j], i)
        cut += 1
    end
    return cut
end

"""True when `sbuf` spans at least `need_e` projected edges."""
function _has_induced_edges(sbuf::Vector{Int}, wadj::Vector{Dict{Int,Int}}, need_e::Int)
    need_e <= 0 && return true
    eS = 0
    for a in eachindex(sbuf)
        wz = wadj[sbuf[a]]
        for b in (a + 1):length(sbuf)
            haskey(wz, sbuf[b]) || continue
            eS += 1
            eS >= need_e && return true
        end
    end
    return false
end

"""True when `{i,j}` still has a large enough, dense enough shared neighborhood and a triangle of codegree `≥ τ` (`τ = θ − 2k`)."""
function _projected_edge_ok(i::Int, j::Int, wadj::Vector{Dict{Int,Int}}, alive::BitVector,
        lists::Vector{Vector{Int}}, buckets::Vector{Vector{Int}},
        τ::Int, need::Int,
        ostamp::Vector{Int}, cstamp::Vector{Int}, sstamp::Vector{Int},
        ccnt::Vector{Int}, cur::Ref{Int}, sbuf::Vector{Int})
    cur[] += 1
    mark_i = cur[]
    for z in keys(wadj[i])
        alive[z] && (cstamp[z] = mark_i)
    end
    cur[] += 1
    mark_s = cur[]
    empty!(sbuf)
    for z in keys(wadj[j])
        if alive[z] && z != i && cstamp[z] == mark_i
            sstamp[z] = mark_s
            push!(sbuf, z)
        end
    end
    length(sbuf) < need && return false
    _has_induced_edges(sbuf, wadj, _choose2(need)) || return false
    (need <= 0 || τ <= 0) && return true

    # Score each z ∈ S by the triangle's total defect weight θ − |N(i)∩N(j)∩N(z)|.
    # That total is ≤ 2k exactly when the triple shares at least τ = θ − 2k
    # opposite neighbors. The minimum over S exceeds 2k when no z reaches τ,
    # so the edge is pruned.
    cur[] += 1
    mark_j = cur[]
    for x in lists[j]
        ostamp[x] = mark_j
    end
    cur[] += 1
    mark_c = cur[]
    for x in lists[i]
        ostamp[x] == mark_j || continue
        for z in buckets[x]
            alive[z] || continue
            if cstamp[z] != mark_c
                cstamp[z] = mark_c
                ccnt[z] = 1
            else
                ccnt[z] += 1
            end
            if ccnt[z] == τ && sstamp[z] == mark_s && z != i && z != j
                return true
            end
        end
    end
    return false
end

"""
Delete every live vertex with fewer than `min_edges` live neighbors, and
cascade. Returns how many vertices were deleted.
"""
function _apply_vertex_rule!(alive::BitVector, wadj::Vector{Dict{Int,Int}}, min_edges::Int)
    min_edges <= 0 && return 0
    n = length(alive)
    light = fill(0, n)
    for i in 1:n
        alive[i] || continue
        for j in keys(wadj[i])
            alive[j] && (light[i] += 1)
        end
    end
    queued = falses(n)
    q = Int[]
    for i in 1:n
        if alive[i] && light[i] < min_edges
            queued[i] = true
            push!(q, i)
        end
    end
    head = 1
    removed = 0
    while head <= length(q)
        i = q[head]
        head += 1
        alive[i] || continue
        alive[i] = false
        removed += 1
        for j in keys(wadj[i])
            alive[j] || continue
            light[j] -= 1
            if light[j] < min_edges && !queued[j]
                queued[j] = true
                push!(q, j)
            end
        end
    end
    return removed
end

"""Sum of the weights of every live edge in `E(N(v))`."""
function _neighborhood_weight(v::Int, wadj::Vector{Dict{Int,Int}}, alive::BitVector)
    nbrs = Int[]
    for j in keys(wadj[v])
        alive[j] && push!(nbrs, j)
    end
    total = 0
    for a in eachindex(nbrs)
        wi = wadj[nbrs[a]]
        for b in (a + 1):length(nbrs)
            haskey(wi, nbrs[b]) && (total += wi[nbrs[b]])
        end
    end
    return total
end

"""
Drop every live `v` whose edges in `E(N(v))` sum to less than `lb`.
Returns how many vertices were deleted.
"""
function _prune_neighborhood_cliques!(alive::BitVector, wadj::Vector{Dict{Int,Int}}, lb::Int)
    lb <= 0 && return 0
    kill = Int[]
    for i in eachindex(alive)
        alive[i] || continue
        _neighborhood_weight(i, wadj, alive) < lb && push!(kill, i)
    end
    for i in kill
        alive[i] = false
    end
    return length(kill)
end

"""Projection indices of `sg` on the side being reduced. Vertices already deleted are skipped."""
function _members_from_subgraph(sg::SubGraph, ids::Vector{Int}, alive::BitVector, small_is_u::Bool)
    side = small_is_u ? sg.U : sg.V
    id_of = Dict{Int,Int}()
    for i in eachindex(ids)
        alive[i] && (id_of[ids[i]] = i)
    end
    members = Int[]
    for id in side
        i = get(id_of, id, 0)
        i == 0 && continue
        push!(members, i)
    end
    return members
end

"""
Vertices that set the stage-2 neighborhood-weight threshold.

A supplied incumbent is used when any of its vertices are still alive on the
projected side. Otherwise the θ-heuristic is solved on `g`.
"""
function _stage2_members(g::Union{Nothing,BipartiteGraph}, k::Int, θ::Int,
        ids::Vector{Int}, alive::BitVector, small_is_u::Bool,
        incumbent::Union{Nothing,SubGraph})
    if incumbent !== nothing
        members = _members_from_subgraph(incumbent, ids, alive, small_is_u)
        if !isempty(members) || g === nothing
            return members
        end
    end
    return g === nothing ? Int[] : _heuristic_member_indices(g, k, θ, ids, alive, small_is_u)
end

"""Projection indices of the θ-heuristic solution on the side being reduced."""
function _heuristic_member_indices(g::BipartiteGraph, k::Int, θ::Int,
        ids::Vector{Int}, alive::BitVector, small_is_u::Bool)
    fg = freeze(g)
    sg = theta_based_heuristic(fg, k, θ)
    side = small_is_u ? sg.U : sg.V
    id_of = Dict{Int,Int}()
    for i in eachindex(ids)
        alive[i] && (id_of[ids[i]] = i)
    end
    members = Int[]
    for id in side
        i = get(id_of, id, 0)
        i == 0 && continue
        push!(members, i)
    end
    return members
end

"""
Smallest sum of `E(N(v))` over the incumbent vertices. Each of those vertices
meets a cut at this value.
"""
function _incumbent_neighborhood_lb(members::Vector{Int}, wadj::Vector{Dict{Int,Int}},
        alive::BitVector)
    isempty(members) && return 0
    lb = typemax(Int)
    for v in members
        alive[v] || continue
        s = _neighborhood_weight(v, wadj, alive)
        s < lb && (lb = s)
    end
    return lb == typemax(Int) ? 0 : lb
end

function _copy_bipartite(g::BipartiteGraph{T}) where {T}
    h = BipartiteGraph{T}()
    for (u, nbrs) in g.adjU
        h.adjU[u] = copy(nbrs)
    end
    for (v, nbrs) in g.adjV
        h.adjV[v] = copy(nbrs)
    end
    for (e, data) in g.edge_data
        h.edge_data[e] = data
    end
    return h
end

"""
Plant an `nU × nV` biclique with exactly `k` missing edges on existing vertices.

Same rule as `inject_biclique!` in `bin/load.jl`: the CSV is not modified.
Returns the number of edges inserted.
"""
function _plant_biclique!(g::BipartiteGraph, nU::Int, nV::Int, k::Int, rng::AbstractRNG;
        attempts::Int=20)
    users = collect(keys(g.adjU))
    items = collect(keys(g.adjV))
    if length(users) < nU || length(items) < nV
        throw(ArgumentError("graph has $(length(users))×$(length(items)) vertices; cannot plant $nU×$nV"))
    end
    if k < 0 || k >= nU * nV
        throw(ArgumentError("k must satisfy 0 ≤ k < u*v; got k=$k, u=$nU, v=$nV"))
    end
    for _ in 1:attempts
        shuffle!(rng, users)
        shuffle!(rng, items)
        chosen_U = users[1:nU]
        chosen_V = items[1:nV]
        absent = Tuple{Int,Int}[]
        for u in chosen_U, v in chosen_V
            has_edge(g, u, v) || push!(absent, (u, v))
        end
        length(absent) < k && continue
        shuffle!(rng, absent)
        inserted = 0
        for (u, v) in absent[(k + 1):end]
            add_edge!(g, u, v, nothing)
            inserted += 1
        end
        return inserted
    end
    throw(ArgumentError("could not plant a $nU×$nV biclique with k=$k missing edges"))
end

function _apply_cnn!(g::BipartiteGraph, k::Int, θ::Int)
    num_U = isempty(g.adjU) ? 0 : maximum(keys(g.adjU))
    num_V = isempty(g.adjV) ? 0 : maximum(keys(g.adjV))
    common_neighbor_reduction!(g, k, θ, num_U, num_V)
    return g
end

function _graph_counts(g::BipartiteGraph)
    return (
        nU = length(g.adjU),
        nV = length(g.adjV),
        edges = _edge_count(g),
    )
end

function _read_dataset_list(path::AbstractString)
    names = String[]
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, '#') && continue
        push!(names, s)
    end
    return names
end

function _json_escape(s::AbstractString)
    return replace(replace(s, '\\' => "\\\\"), '"' => "\\\"")
end

"""
Time three reductions of one graph, each from a fresh copy.

`once` is `super_reduce!` with `efficient=true`: the defect-weight stage,
one common-neighbor pass, and the neighborhood-weight stage, each a single
time. `full` is `efficient=false`: both weighted stages and a common-neighbor
pass, repeated until the graph stops shrinking. Copies are freed between
phases so only the original and one working graph are live.
"""
function _reduce_phases(g::BipartiteGraph, k::Int, θ::Int;
        prune_edges::Bool=true, prune_vertices::Bool=true,
        progress::Union{Nothing,AbstractString}=nothing)
    function mark(label::AbstractString)
        progress === nothing && return
        println("phase\t", progress, "\t", label)
        flush(stdout)
        return nothing
    end
    mark("cnn")
    work = _copy_bipartite(g)
    t_cnn = @elapsed _apply_cnn!(work, k, θ)
    cnn = _graph_counts(work)
    work = nothing

    mark("once")
    work = _copy_bipartite(g)
    st_once = Dict{Symbol,Int}()
    t_once = @elapsed super_reduce!(work, k, θ; stats=st_once,
        prune_edges=prune_edges, prune_vertices=prune_vertices, efficient=true)
    once = _graph_counts(work)
    work = nothing

    mark("full")
    work = _copy_bipartite(g)
    st_full = Dict{Symbol,Int}()
    t_full = @elapsed super_reduce!(work, k, θ; stats=st_full,
        prune_edges=prune_edges, prune_vertices=prune_vertices, efficient=false)
    full = _graph_counts(work)
    work = nothing
    mark("done")
    return (; cnn, t_cnn, once, t_once, st_once, full, t_full, st_full)
end

"""
Compare common-neighbor reduction, one `super_reduce!` pass, and the repeated
reduction on every dataset in `data/datasets.txt` (paper defaults k = 2, θ = 5).
"""
function report_super_reduce(; k::Int=2, θ::Int=5, datasets::Union{Nothing,Vector{String}}=nothing,
        list_path::AbstractString="data/datasets.txt",
        save_path::AbstractString="results/super_reduce_phases.json",
        prune_edges::Bool=true, prune_vertices::Bool=true,
        inject::Bool=true, seed::Int=1)
    names = datasets === nothing ? _read_dataset_list(list_path) : datasets
    rng = MersenneTwister(seed)
    println("reduction phases   k=$k  θ=$θ  graphs=$(length(names))  edge_prune=$(prune_edges)  vertex_prune=$(prune_vertices)  inject=$(inject)  seed=$seed")
    if inject
        println("planted biclique   |U|=$θ  |V|=$θ  missing edges=$k")
    end
    hdr = ("dataset", "nU", "nV", "edges",
        "cnn_nU", "cnn_nV", "cnn_edges", "cnn_s",
        "once_nU", "once_nV", "once_edges", "once_s",
        "full_nU", "full_nV", "full_edges", "full_s", "full_rounds")
    println(join(hdr, "\t"))
    flush(stdout)

    rows = Dict{String,Any}[]
    mkpath(dirname(save_path))
    warmed = false
    for name in names
        path = resolve_graph_path(name)
        if !isfile(path)
            println("skip\t$name\tmissing")
            flush(stdout)
            continue
        end
        g = redirect_stdout(devnull) do
            load_bipartite_graph(path)[1]
        end
        planted = 0
        if inject
            try
                planted = _plant_biclique!(g, θ, θ, k, rng)
            catch err
                println("skip\t$name\tplant failed: $err")
                flush(stdout)
                continue
            end
        end
        base = _graph_counts(g)
        if !warmed
            _reduce_phases(g, k, θ; prune_edges=prune_edges, prune_vertices=prune_vertices)
            warmed = true
            GC.gc()
        end
        ph = _reduce_phases(g, k, θ; prune_edges=prune_edges, prune_vertices=prune_vertices,
            progress=name)
        g = nothing
        cnn, once, full = ph.cnn, ph.once, ph.full
        st = ph.st_once
        extra_nodes = (cnn.nU + cnn.nV) - (once.nU + once.nV)
        extra_edges = cnn.edges - once.edges
        vals = (name, base.nU, base.nV, base.edges,
            cnn.nU, cnn.nV, cnn.edges, round(ph.t_cnn, digits=3),
            once.nU, once.nV, once.edges, round(ph.t_once, digits=3),
            full.nU, full.nV, full.edges, round(ph.t_full, digits=3),
            get(ph.st_full, :rounds, -1))
        println(join(vals, "\t"))
        flush(stdout)
        push!(rows, Dict{String,Any}(
            "dataset" => name,
            "nU" => base.nU, "nV" => base.nV, "edges" => base.edges,
            "cnn_nU" => cnn.nU, "cnn_nV" => cnn.nV, "cnn_edges" => cnn.edges,
            "cnn_s" => ph.t_cnn,
            "once_nU" => once.nU, "once_nV" => once.nV, "once_edges" => once.edges,
            "once_s" => ph.t_once, "once_rounds" => get(st, :rounds, -1),
            "full_nU" => full.nU, "full_nV" => full.nV, "full_edges" => full.edges,
            "full_s" => ph.t_full, "full_rounds" => get(ph.st_full, :rounds, -1),
            # Default `super_reduce!` is the single pass. Kept so older
            # comparisons of that call against CNR still read `sup_*`.
            "sup_nU" => once.nU, "sup_nV" => once.nV, "sup_edges" => once.edges,
            "sup_s" => ph.t_once,
            "extra_nodes" => extra_nodes, "extra_edges" => extra_edges,
            "rounds" => get(st, :rounds, -1),
            "core_removed" => get(st, :core_removed, -1),
            "weight_edges_cut" => get(st, :weight_edges_cut, 0),
            "weight_vertices_cut" => get(st, :weight_vertices_cut, 0),
            "mwc_vertices_cut" => get(st, :mwc_vertices_cut, 0),
            "weight_skipped" => get(st, :weight_skipped, 0),
            "prune_edges" => prune_edges, "prune_vertices" => prune_vertices,
            "k" => k, "theta" => θ,
            "injected" => inject, "plant_edges" => planted, "seed" => seed,
        ))
        _write_report(save_path, rows)
        GC.gc()
    end

    return rows
end

function _write_report(save_path::AbstractString, rows::Vector{Dict{String,Any}})
    open(save_path, "w") do io
        println(io, "[")
        for (i, row) in enumerate(rows)
            print(io, "  {")
            first = true
            for (key, val) in row
                first || print(io, ", ")
                first = false
                print(io, '"', _json_escape(key), '"', ": ")
                if val isa AbstractString
                    print(io, '"', _json_escape(val), '"')
                elseif val isa AbstractFloat
                    print(io, JSON_fmt(val))
                else
                    print(io, val)
                end
            end
            println(io, i == length(rows) ? "}" : "},")
        end
        println(io, "]")
    end
    return rows
end

JSON_fmt(x::AbstractFloat) = string(round(x; digits=6))

"""
Run `super_reduce!` with `side=:smaller` and `side=:larger` on every dataset.

A row records the surviving order and edge count of each choice, plus whether
a θ-feasible θ-heuristic solution is still present. Fewer surviving edges is
the stronger reduction.
"""
function report_side_choice(; k::Int=2, θ::Int=5, datasets::Union{Nothing,Vector{String}}=nothing,
        list_path::AbstractString="data/datasets.txt",
        save_path::AbstractString="results/super_reduce_side.json")
    names = datasets === nothing ? _read_dataset_list(list_path) : datasets
    println("super-reduce side choice   k=$k  θ=$θ  graphs=$(length(names))")
    hdr = ("dataset", "nU", "nV", "edges",
        "small_n", "small_e", "small_s", "large_n", "large_e", "large_s",
        "winner", "heur_ok", "small_keeps_heur", "large_keeps_heur")
    println(join(hdr, "\t"))
    flush(stdout)

    rows = Dict{String,Any}[]
    mkpath(dirname(save_path))
    for name in names
        path = resolve_graph_path(name)
        if !isfile(path)
            println("skip\t$name\tmissing")
            flush(stdout)
            continue
        end
        g = redirect_stdout(devnull) do
            load_bipartite_graph(path)[1]
        end
        base = _graph_counts(g)
        fg = freeze(g)
        heur = theta_based_heuristic(fg, k, θ)
        heur_miss = Subgraph.missing_edges(fg, heur)
        heur_ok = length(heur.U) >= θ && length(heur.V) >= θ && heur_miss <= k
        g_small = _copy_bipartite(g)
        g_large = _copy_bipartite(g)
        g = nothing
        fg = nothing
        t_small = @elapsed super_reduce!(g_small, k, θ; side=:smaller)
        t_large = @elapsed super_reduce!(g_large, k, θ; side=:larger)
        sm = _graph_counts(g_small)
        lg = _graph_counts(g_large)
        small_keeps = !heur_ok || _keeps_solution(g_small, heur)
        large_keeps = !heur_ok || _keeps_solution(g_large, heur)
        small_n = sm.nU + sm.nV
        large_n = lg.nU + lg.nV
        winner = if sm.edges < lg.edges || (sm.edges == lg.edges && small_n < large_n)
            "smaller"
        elseif lg.edges < sm.edges || (lg.edges == sm.edges && large_n < small_n)
            "larger"
        else
            "tie"
        end
        vals = (name, base.nU, base.nV, base.edges,
            small_n, sm.edges, round(t_small, digits=3),
            large_n, lg.edges, round(t_large, digits=3),
            winner, heur_ok, small_keeps, large_keeps)
        println(join(vals, "\t"))
        flush(stdout)
        push!(rows, Dict{String,Any}(
            "dataset" => name,
            "nU" => base.nU, "nV" => base.nV, "edges" => base.edges,
            "small_nU" => sm.nU, "small_nV" => sm.nV, "small_edges" => sm.edges,
            "large_nU" => lg.nU, "large_nV" => lg.nV, "large_edges" => lg.edges,
            "small_s" => t_small, "large_s" => t_large,
            "winner" => winner,
            "heur_ok" => heur_ok,
            "small_keeps_heur" => small_keeps,
            "large_keeps_heur" => large_keeps,
            "k" => k, "theta" => θ,
        ))
        _write_report(save_path, rows)
        g_small = nothing
        g_large = nothing
        GC.gc()
    end
    return rows
end

function _keeps_solution(g::BipartiteGraph, sg)
    for u in sg.U
        haskey(g.adjU, u) || return false
    end
    for v in sg.V
        haskey(g.adjV, v) || return false
    end
    return true
end

"""
Read `"k:theta"` entries from `scripts/kt-sweep.bash`.

Duplicate pairs are kept once, in first-seen order. Comment lines are ignored.
"""
function _read_kt_pairs(path::AbstractString)
    pairs = Tuple{Int,Int}[]
    seen = Set{Tuple{Int,Int}}()
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, '#') && continue
        for m in eachmatch(r"(\d+)\s*:\s*(\d+)", s)
            pair = (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
            pair in seen && continue
            push!(seen, pair)
            push!(pairs, pair)
        end
    end
    return pairs
end

"""
Run `report_super_reduce` on every pair in `scripts/kt-sweep.bash`.

Each pair is written to `out_dir/k{k}t{θ}.json`.
"""
function report_kt_sweep(; pairs_path::AbstractString="scripts/kt-sweep.bash",
        out_dir::AbstractString="results/reduce_sweep",
        datasets::Union{Nothing,Vector{String}}=nothing,
        prune_edges::Bool=true, prune_vertices::Bool=true,
        inject::Bool=true, seed::Int=1)
    pairs = _read_kt_pairs(pairs_path)
    isempty(pairs) && error("no (k, θ) pairs in $pairs_path")
    mkpath(out_dir)
    println("reduce sweep  pairs=$(length(pairs))  out=$out_dir  file=$pairs_path")
    flush(stdout)
    for (k, θ) in pairs
        save_path = joinpath(out_dir, "k$(k)t$(θ).json")
        println()
        println("========== k=$k  θ=$θ  →  $save_path ==========")
        flush(stdout)
        report_super_reduce(; k=k, θ=θ, datasets=datasets, save_path=save_path,
            prune_edges=prune_edges, prune_vertices=prune_vertices,
            inject=inject, seed=seed)
    end
    return nothing
end

function _super_reduce_main()
    k = 2
    θ = 5
    prune_edges = true
    prune_vertices = true
    sides = false
    sweep = false
    inject = true
    seed = 1
    save_path = nothing
    pairs_path = "scripts/kt-sweep.bash"
    out_dir = "results/reduce_sweep"
    only = String[]
    for a in ARGS
        if startswith(a, "--k=")
            k = parse(Int, split(a, '=', limit=2)[2])
        elseif startswith(a, "--theta=")
            θ = parse(Int, split(a, '=', limit=2)[2])
        elseif startswith(a, "--save=")
            save_path = split(a, '=', limit=2)[2]
        elseif startswith(a, "--seed=")
            seed = parse(Int, split(a, '=', limit=2)[2])
        elseif startswith(a, "--sweep-file=")
            pairs_path = split(a, '=', limit=2)[2]
        elseif startswith(a, "--out-dir=")
            out_dir = split(a, '=', limit=2)[2]
        elseif a == "--no-inject"
            inject = false
        elseif a == "--no-edge-prune"
            prune_edges = false
        elseif a == "--no-vertex-prune"
            prune_vertices = false
        elseif a == "--sides"
            sides = true
        elseif a == "--sweep"
            sweep = true
        else
            push!(only, a)
        end
    end
    datasets = isempty(only) ? nothing : only
    if sweep
        report_kt_sweep(; pairs_path=pairs_path, out_dir=out_dir,
            datasets=datasets, prune_edges=prune_edges,
            prune_vertices=prune_vertices, inject=inject, seed=seed)
    elseif sides
        report_side_choice(; k=k, θ=θ, datasets=datasets)
    elseif save_path === nothing
        report_super_reduce(; k=k, θ=θ, datasets=datasets,
            prune_edges=prune_edges, prune_vertices=prune_vertices,
            inject=inject, seed=seed)
    else
        report_super_reduce(; k=k, θ=θ, datasets=datasets,
            save_path=save_path,
            prune_edges=prune_edges, prune_vertices=prune_vertices,
            inject=inject, seed=seed)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    _super_reduce_main()
end
