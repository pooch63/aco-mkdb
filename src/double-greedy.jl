#=
Double-greedy heuristic for a k-defective biclique.

1. Seed with the highest-degree vertex and every neighbor. That star is a
   biclique with no missing edges.
2. Until both sides have at least θ vertices, add a vertex on the smaller
   side. Adding it may create more than k missing edges. Those are removed by
   dropping vertices on the opposite side, highest nondegree first, stopping
   at k and never dropping that side below θ. The vertex kept is the one whose
   subgraph has the most edges after that trim. If the opposite side is still
   below θ, nothing is removed, and the add is legal only when it already
   fits in k.
3. Once both sides have at least θ vertices, add the vertex on either side
   that contributes the most edges without exceeding k. No further vertices
   are removed.

Ties break toward fewer defects, then the smaller vertex id, then U.
=#

const __DOUBLE_GREEDY_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")

struct DoubleGreedyResult
    sol::SubGraph
    seed_is_u::Bool
    seed_id::Int
    additions::Int
end

function _activate_u!(fg::FrozenBipartite, inU::BitVector, degV::Vector{Int}, ui::Int)
    inU[ui] = true
    @inbounds for t in neighbor_range_u(fg, ui)
        degV[fg.v_adj[t]] += 1
    end
    return nothing
end

function _activate_v!(fg::FrozenBipartite, inV::BitVector, degU::Vector{Int}, vi::Int)
    inV[vi] = true
    @inbounds for t in neighbor_range_v(fg, vi)
        degU[fg.u_adj[t]] += 1
    end
    return nothing
end

function _deactivate_u!(fg::FrozenBipartite, inU::BitVector, degV::Vector{Int}, ui::Int)
    inU[ui] = false
    @inbounds for t in neighbor_range_u(fg, ui)
        degV[fg.v_adj[t]] -= 1
    end
    return nothing
end

function _deactivate_v!(fg::FrozenBipartite, inV::BitVector, degU::Vector{Int}, vi::Int)
    inV[vi] = false
    @inbounds for t in neighbor_range_v(fg, vi)
        degU[fg.u_adj[t]] -= 1
    end
    return nothing
end

"""
Highest-degree vertex as `(is_u, dense_index, id)`.

Equal degrees take the smaller id. The same id on both sides keeps U.
"""
function _max_degree_seed(fg::FrozenBipartite)
    best_deg = -1
    best_is_u = true
    best_idx = 0
    best_id = 0
    found = false
    for i in eachindex(fg.u_ids)
        d = length(neighbor_range_u(fg, i))
        id = fg.u_ids[i]
        if !found || d > best_deg || (d == best_deg && id < best_id)
            found = true
            best_deg = d
            best_is_u = true
            best_idx = i
            best_id = id
        end
    end
    for i in eachindex(fg.v_ids)
        d = length(neighbor_range_v(fg, i))
        id = fg.v_ids[i]
        if !found || d > best_deg || (d == best_deg && id < best_id)
            found = true
            best_deg = d
            best_is_u = false
            best_idx = i
            best_id = id
        end
    end
    return found, best_is_u, best_idx, best_id
end

"""Vertices on the side that is still shorter. Both sides when the sizes match."""
function _shorter_sides(n_u::Int, n_v::Int)
    if n_u < n_v
        return (true,)
    elseif n_v < n_u
        return (false,)
    else
        return (true, false)
    end
end

"""
Order in which to drop opposite-side vertices: highest nondegree, then largest id.

`deg[i]` is that vertex's degree into the side that just grew.
"""
function _trim_order(ids::Vector{Int}, inn::BitVector, deg::Vector{Int}, n_same::Int)
    recs = Tuple{Int,Int,Int,Int}[]
    for i in eachindex(ids)
        inn[i] || continue
        d = deg[i]
        push!(recs, (n_same - d, ids[i], i, d))
    end
    sort!(recs, by = r -> (-r[1], -r[2], r[3]))
    return recs
end

"""
Edges and missing edges after adding dense index `idx`, trimming the opposite
side down to at most k defects and at least θ vertices.

Returns `nothing` when the add cannot meet both constraints. `stamp` marks the
candidate's neighbors for this `clock` value.
"""
function _score_growth(fg::FrozenBipartite, is_u::Bool, idx::Int,
        inU::BitVector, inV::BitVector, degU::Vector{Int}, degV::Vector{Int},
        n_u::Int, n_v::Int, edges::Int, k::Int, θ::Int,
        stamp::Vector{Int}, clock::Int)
    add_e = is_u ? degU[idx] : degV[idx]
    n_same = (is_u ? n_u : n_v) + 1
    n_opp = is_u ? n_v : n_u
    new_edges = edges + add_e
    new_missing = n_same * n_opp - new_edges
    if new_missing <= k
        return (edges=new_edges, missing=new_missing)
    end
    n_opp >= θ || return nothing
    # Every removed vertex clears at most n_same defects.
    if new_missing - (n_opp - θ) * n_same > k
        return nothing
    end

    if is_u
        @inbounds for t in neighbor_range_u(fg, idx)
            stamp[fg.v_adj[t]] = clock
        end
        inn, deg, ids = inV, degV, fg.v_ids
    else
        @inbounds for t in neighbor_range_v(fg, idx)
            stamp[fg.u_adj[t]] = clock
        end
        inn, deg, ids = inU, degU, fg.u_ids
    end

    recs = Tuple{Int,Int,Int,Int}[]
    for i in eachindex(ids)
        inn[i] || continue
        d = deg[i] + (stamp[i] == clock ? 1 : 0)
        push!(recs, (n_same - d, ids[i], i, d))
    end
    sort!(recs, by = r -> (-r[1], -r[2], r[3]))

    miss = new_missing
    kept_edges = new_edges
    left = n_opp
    for (nondeg, _, _, d) in recs
        left == θ && break
        miss -= nondeg
        kept_edges -= d
        left -= 1
        miss <= k && break
    end
    miss <= k || return nothing
    return (edges=kept_edges, missing=miss)
end

"""
Best addition on `sides` that can be made to fit in k without taking the
opposite side below θ. The winner is the one with the most edges afterward.
"""
function _best_growth(fg::FrozenBipartite, sides, inU::BitVector, inV::BitVector,
        degU::Vector{Int}, degV::Vector{Int}, n_u::Int, n_v::Int, edges::Int,
        k::Int, θ::Int, stamp::Vector{Int}, clock::Base.RefValue{Int})
    best_edges = -1
    best_miss = typemax(Int)
    best_id = typemax(Int)
    best_is_u = true
    best_idx = 0
    found = false
    for is_u in sides
        inn = is_u ? inU : inV
        ids = is_u ? fg.u_ids : fg.v_ids
        for i in eachindex(ids)
            inn[i] && continue
            clock[] += 1
            scored = _score_growth(fg, is_u, i, inU, inV, degU, degV,
                n_u, n_v, edges, k, θ, stamp, clock[])
            scored === nothing && continue
            id = ids[i]
            better = !found || scored.edges > best_edges ||
                (scored.edges == best_edges && scored.missing < best_miss) ||
                (scored.edges == best_edges && scored.missing == best_miss && id < best_id) ||
                (scored.edges == best_edges && scored.missing == best_miss &&
                    id == best_id && is_u && !best_is_u)
            if better
                found = true
                best_edges = scored.edges
                best_miss = scored.missing
                best_id = id
                best_is_u = is_u
                best_idx = i
            end
        end
    end
    return found ? (best_is_u, best_idx) : nothing
end

"""Drop opposite vertices until missing edges are at most k, keeping at least θ."""
function _trim_to_k!(fg::FrozenBipartite, added_is_u::Bool,
        inU::BitVector, inV::BitVector, degU::Vector{Int}, degV::Vector{Int},
        n_same::Int, n_opp::Int, edges::Int, missing::Int, k::Int, θ::Int)
    missing <= k && return n_opp, edges, missing
    if added_is_u
        recs = _trim_order(fg.v_ids, inV, degV, n_same)
    else
        recs = _trim_order(fg.u_ids, inU, degU, n_same)
    end
    for (nondeg, _, i, d) in recs
        n_opp == θ && break
        if added_is_u
            _deactivate_v!(fg, inV, degU, i)
        else
            _deactivate_u!(fg, inU, degV, i)
        end
        n_opp -= 1
        edges -= d
        missing -= nondeg
        missing <= k && break
    end
    return n_opp, edges, missing
end

"""
Best feasible addition on `sides` with no removals.

Among vertices whose new defects keep the total at most `k`, the one with the
most edges into the current opposite side wins.
"""
function _best_addition(fg::FrozenBipartite, sides, inU::BitVector, inV::BitVector,
        degU::Vector{Int}, degV::Vector{Int}, n_u::Int, n_v::Int, missing::Int, k::Int)
    best_edges = -1
    best_miss = typemax(Int)
    best_id = typemax(Int)
    best_is_u = true
    best_idx = 0
    found = false
    for is_u in sides
        inn = is_u ? inU : inV
        deg = is_u ? degU : degV
        ids = is_u ? fg.u_ids : fg.v_ids
        opp = is_u ? n_v : n_u
        for i in eachindex(ids)
            inn[i] && continue
            add_e = deg[i]
            add_m = opp - add_e
            missing + add_m > k && continue
            id = ids[i]
            better = !found || add_e > best_edges ||
                (add_e == best_edges && add_m < best_miss) ||
                (add_e == best_edges && add_m == best_miss && id < best_id) ||
                (add_e == best_edges && add_m == best_miss && id == best_id && is_u && !best_is_u)
            if better
                found = true
                best_edges = add_e
                best_miss = add_m
                best_id = id
                best_is_u = is_u
                best_idx = i
            end
        end
    end
    return found ? (best_is_u, best_idx) : nothing
end

"""
    double_greedy(fg, k, θ) -> DoubleGreedyResult

See the file header. A subgraph that cannot reach θ on both sides is returned
as built.
"""
function double_greedy(fg::FrozenBipartite, k::Int, θ::Int)
    k >= 0 || throw(ArgumentError("k must be ≥ 0, got $k"))
    θ >= 1 || throw(ArgumentError("θ must be ≥ 1, got $θ"))

    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    found, seed_is_u, seed_idx, seed_id = _max_degree_seed(fg)
    if !found
        return DoubleGreedyResult(SubGraph(), true, 0, 0)
    end

    inU = falses(nU)
    inV = falses(nV)
    degU = zeros(Int, nU)
    degV = zeros(Int, nV)
    n_u = 0
    n_v = 0
    edges = 0

    if seed_is_u
        _activate_u!(fg, inU, degV, seed_idx)
        n_u = 1
        for t in neighbor_range_u(fg, seed_idx)
            _activate_v!(fg, inV, degU, fg.v_adj[t])
            n_v += 1
            edges += 1
        end
    else
        _activate_v!(fg, inV, degU, seed_idx)
        n_v = 1
        for t in neighbor_range_v(fg, seed_idx)
            _activate_u!(fg, inU, degV, fg.u_adj[t])
            n_u += 1
            edges += 1
        end
    end
    missing = n_u * n_v - edges

    stamp = zeros(Int, max(nU, nV))
    clock = Ref(0)
    additions = 0
    limit = nU + nV

    while (n_u < θ || n_v < θ) && additions < limit
        pick = _best_growth(fg, _shorter_sides(n_u, n_v), inU, inV, degU, degV,
            n_u, n_v, edges, k, θ, stamp, clock)
        pick === nothing && break
        is_u, idx = pick
        if is_u
            add_e = degU[idx]
            _activate_u!(fg, inU, degV, idx)
            n_u += 1
            edges += add_e
            missing = n_u * n_v - edges
            n_v, edges, missing = _trim_to_k!(fg, true, inU, inV, degU, degV,
                n_u, n_v, edges, missing, k, θ)
        else
            add_e = degV[idx]
            _activate_v!(fg, inV, degU, idx)
            n_v += 1
            edges += add_e
            missing = n_u * n_v - edges
            n_u, edges, missing = _trim_to_k!(fg, false, inU, inV, degU, degV,
                n_v, n_u, edges, missing, k, θ)
        end
        additions += 1
    end

    if n_u >= θ && n_v >= θ
        while additions < limit
            pick = _best_addition(fg, (true, false), inU, inV, degU, degV,
                n_u, n_v, missing, k)
            pick === nothing && break
            is_u, idx = pick
            if is_u
                edges += degU[idx]
                _activate_u!(fg, inU, degV, idx)
                n_u += 1
            else
                edges += degV[idx]
                _activate_v!(fg, inV, degU, idx)
                n_v += 1
            end
            missing = n_u * n_v - edges
            additions += 1
        end
    end

    U = Set{Int}()
    V = Set{Int}()
    sizehint!(U, n_u)
    sizehint!(V, n_v)
    for i in 1:nU
        inU[i] && push!(U, fg.u_ids[i])
    end
    for i in 1:nV
        inV[i] && push!(V, fg.v_ids[i])
    end
    return DoubleGreedyResult(SubGraph(U, V), seed_is_u, seed_id, additions)
end
