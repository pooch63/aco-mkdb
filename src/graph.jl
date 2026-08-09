const __GRAPH_JL__ = true

# =============================================================================
# graph.jl
#
# Three-phase design:
#   Phase 1 (mutation)  -> BipartiteGraph   : add/remove vertices & edges
#   freeze()             -> FrozenBipartite  : one-time O(V+E) conversion
#   Phase 2 (split)      -> partition into SubGraphs, one O(V+E) pass
#   Phase 3 (read-only)  -> accumulate/query against SubGraphs
#
# Each phase runs exactly once, so we pay a one-time conversion cost between
# phases rather than keeping a single mutation-friendly structure (hash sets)
# alive for read-heavy phases where it's the wrong tool.
#
# The public API for phases 2 & 3 is entirely in terms of SubGraph (you
# specify which U/V vertex ids belong to each subgraph); the dense internal
# lookup arrays used for fast iteration are built and discarded internally.
#
# Phase 3 has two flavors of query:
#   - BATCH  (split, subgraph_edge_counts, accumulate_edges!): one O(V+E)
#     sweep that answers for ALL subgraphs in `subgraphs` at once. Use this
#     when you need results for every subgraph.
#   - POINT  (degree_u, degree_v, neighbors_u, neighbors_v, and their
#     `_in_subgraph` variants): O(deg(vertex)) queries about a single vertex,
#     optionally restricted to a single SubGraph. Prefer bind_membership!
#     so restriction is a dense bit test on CSR indices; otherwise falls back
#     to Set membership. For whole-graph candidate scans, prefer the inverted
#     degrees_into_subgraph_*! helpers (walk sg's cut, not all of E).
# =============================================================================

# --------------------------------------------------------------------------
# PHASE 1: mutable graph, arbitrary Int vertex ids, Dict{Int,Set{Int}} adjacency
# --------------------------------------------------------------------------

struct BipartiteGraph{T}
    adjU::Dict{Int, Set{Int}}            # u => neighbors in V
    adjV::Dict{Int, Set{Int}}            # v => neighbors in U
    edge_data::Dict{Tuple{Int,Int}, T}   # (u, v) => metadata
end

BipartiteGraph{T}() where {T} =
    BipartiteGraph{T}(Dict{Int,Set{Int}}(), Dict{Int,Set{Int}}(), Dict{Tuple{Int,Int},T}())

# --- vertices ---
function add_u!(g::BipartiteGraph, u::Int)
    get!(g.adjU, u, Set{Int}())
    return g
end

function add_v!(g::BipartiteGraph, v::Int)
    get!(g.adjV, v, Set{Int}())
    return g
end

function rem_u!(g::BipartiteGraph, u::Int)
    haskey(g.adjU, u) || return false
    for v in g.adjU[u]
        delete!(g.adjV[v], u)
        delete!(g.edge_data, (u, v))
    end
    delete!(g.adjU, u)
    return true
end

function rem_v!(g::BipartiteGraph, v::Int)
    haskey(g.adjV, v) || return false
    for u in g.adjV[v]
        delete!(g.adjU[u], v)
        delete!(g.edge_data, (u, v))
    end
    delete!(g.adjV, v)
    return true
end

function rem_node!(g::BipartiteGraph, is_u::Bool, node_id::Int)
    if is_u
        return rem_u!(g, node_id)
    else
        return rem_v!(g, node_id)
    end
end

# --- edges ---
function add_edge!(g::BipartiteGraph{T}, u::Int, v::Int, data::T) where {T}
    add_u!(g, u)
    add_v!(g, v)
    push!(g.adjU[u], v)
    push!(g.adjV[v], u)
    g.edge_data[(u, v)] = data
    return g
end

function rem_edge!(g::BipartiteGraph, u::Int, v::Int)
    haskey(g.edge_data, (u, v)) || return false
    delete!(g.adjU[u], v)
    delete!(g.adjV[v], u)
    delete!(g.edge_data, (u, v))
    return true
end

# Adjacency-only removal for structural passes (e.g. graph reduction) that do
# not need edge metadata. Skips edge_data tuple hashing; stale edge_data entries
# are harmless because freeze() walks adjacency lists, not edge_data keys.
function rem_edge_structural!(g::BipartiteGraph, u::Int, v::Int)
    haskey(g.adjU, u) || return false
    v in g.adjU[u] || return false
    delete!(g.adjU[u], v)
    delete!(g.adjV[v], u)
    return true
end

function rem_u_structural!(g::BipartiteGraph, u::Int)
    haskey(g.adjU, u) || return false
    for v in g.adjU[u]
        delete!(g.adjV[v], u)
    end
    delete!(g.adjU, u)
    return true
end

function rem_v_structural!(g::BipartiteGraph, v::Int)
    haskey(g.adjV, v) || return false
    for u in g.adjV[v]
        delete!(g.adjU[u], v)
    end
    delete!(g.adjV, v)
    return true
end

function rem_node_structural!(g::BipartiteGraph, is_u::Bool, node_id::Int)
    if is_u
        return rem_u_structural!(g, node_id)
    else
        return rem_v_structural!(g, node_id)
    end
end

function add_edge!(g::BipartiteGraph{Nothing}, u::Int, v::Int, ::Nothing)
    add_u!(g, u)
    add_v!(g, v)
    push!(g.adjU[u], v)
    push!(g.adjV[v], u)
    return g
end

function rem_edge!(g::BipartiteGraph{Nothing}, u::Int, v::Int)
    return rem_edge_structural!(g, u, v)
end

function rem_u!(g::BipartiteGraph{Nothing}, u::Int)
    return rem_u_structural!(g, u)
end

function rem_v!(g::BipartiteGraph{Nothing}, v::Int)
    return rem_v_structural!(g, v)
end

has_edge(g::BipartiteGraph{Nothing}, u::Int, v::Int) =
    haskey(g.adjU, u) && v in g.adjU[u]

# --- queries (still available on the live graph if needed) ---
node_exists(g::BipartiteGraph, is_u::Bool, node_id::Int) = is_u ? haskey(g.adjU, node_id) : haskey(g.adjV, node_id)
neighbors_u(g::BipartiteGraph, u::Int) = g.adjU[u]
neighbors_v(g::BipartiteGraph, v::Int) = g.adjV[v]
get_neighbors(g::BipartiteGraph, is_u::Bool, node_id::Int) = is_u ? neighbors_u(g, node_id) : neighbors_v(g, node_id)
degree_u(g::BipartiteGraph, u::Int) = length(g.adjU[u])
degree_v(g::BipartiteGraph, v::Int) = length(g.adjV[v])
get_degree(g::BipartiteGraph, is_u::Bool, node_id::Int) = is_u ? degree_u(g, node_id) : degree_v(g, node_id)
edge_data(g::BipartiteGraph, u::Int, v::Int) = g.edge_data[(u, v)]
has_edge(g::BipartiteGraph, u::Int, v::Int) = haskey(g.edge_data, (u, v))


# --------------------------------------------------------------------------
# FREEZE: one-time O(V+E) conversion to CSR-style dense arrays.
# Call this exactly once, after all removal is finished forever.
#
# Builds BOTH directions:
#   forward (U -> V): u_offsets / v_adj / edge_data
#   reverse (V -> U): v_offsets / u_adj / v_edge_data
# so that neighbor/degree queries are O(deg(vertex)) from either side,
# instead of forcing a V-side query to scan every U vertex's edge list.
# --------------------------------------------------------------------------

struct FrozenBipartite{T}
    u_ids::Vector{Int}              # dense u-index -> original u id
    v_ids::Vector{Int}              # dense v-index -> original v id
    u_index::Dict{Int,Int}          # original u id -> dense u-index
    v_index::Dict{Int,Int}          # original v id -> dense v-index

    u_offsets::Vector{Int}          # CSR offsets into v_adj / edge_data (length nU+1)
    v_adj::Vector{Int}              # flattened neighbor dense v-indices, sorted per u
    edge_data::Vector{T}            # parallel to v_adj

    v_offsets::Vector{Int}          # CSR offsets into u_adj / v_edge_data (length nV+1)
    u_adj::Vector{Int}              # flattened neighbor dense u-indices, sorted per v
    v_edge_data::Vector{T}          # parallel to u_adj
end

function freeze(g::BipartiteGraph{T}) where {T}
    u_ids = collect(keys(g.adjU))
    v_ids = collect(keys(g.adjV))
    u_index = Dict(u => i for (i, u) in enumerate(u_ids))
    v_index = Dict(v => i for (i, v) in enumerate(v_ids))

    nU = length(u_ids)
    nV = length(v_ids)

    # ---- forward CSR (U -> V) ----
    u_offsets = Vector{Int}(undef, nU + 1)
    v_adj = Int[]
    edata = T[]
    nE = sum(length(nbrs) for nbrs in values(g.adjU); init=0)
    sizehint!(v_adj, nE)
    sizehint!(edata, nE)

    u_offsets[1] = 1
    for (i, u) in enumerate(u_ids)
        nbrs = sort!(collect(g.adjU[u]))   # sorted -> cache-friendly, mergeable later
        for v in nbrs
            push!(v_adj, v_index[v])
            push!(edata, g.edge_data[(u, v)])
        end
        u_offsets[i + 1] = u_offsets[i] + length(nbrs)
    end

    # ---- reverse CSR (V -> U), built from the forward arrays via counting sort ----
    nE = length(v_adj)
    v_deg = zeros(Int, nV)
    for vi in v_adj
        v_deg[vi] += 1
    end
    v_offsets = Vector{Int}(undef, nV + 1)
    v_offsets[1] = 1
    for i in 1:nV
        v_offsets[i + 1] = v_offsets[i] + v_deg[i]
    end

    u_adj = Vector{Int}(undef, nE)
    v_edata = Vector{T}(undef, nE)
    fill_pos = collect(v_offsets[1:nV])   # next write position per v, mutated below
    for ui in 1:nU
        for k in u_offsets[ui]:(u_offsets[ui + 1] - 1)
            vi = v_adj[k]
            pos = fill_pos[vi]
            u_adj[pos] = ui
            v_edata[pos] = edata[k]
            fill_pos[vi] = pos + 1
        end
    end
    # u_adj ends up sorted per v because ui is iterated in increasing order.

    return FrozenBipartite{T}(
        u_ids, v_ids, u_index, v_index,
        u_offsets, v_adj, edata,
        v_offsets, u_adj, v_edata,
    )
end

function freeze(g::BipartiteGraph{Nothing})
    u_ids = collect(keys(g.adjU))
    v_ids = collect(keys(g.adjV))
    u_index = Dict(u => i for (i, u) in enumerate(u_ids))
    v_index = Dict(v => i for (i, v) in enumerate(v_ids))

    nU = length(u_ids)
    nV = length(v_ids)

    u_offsets = Vector{Int}(undef, nU + 1)
    v_adj = Int[]
    edata = Nothing[]
    nE = sum(length(nbrs) for nbrs in values(g.adjU); init=0)
    sizehint!(v_adj, nE)
    sizehint!(edata, nE)

    u_offsets[1] = 1
    for (i, u) in enumerate(u_ids)
        nbrs = sort!(collect(g.adjU[u]))
        for v in nbrs
            push!(v_adj, v_index[v])
            push!(edata, nothing)
        end
        u_offsets[i + 1] = u_offsets[i] + length(nbrs)
    end

    nE = length(v_adj)
    v_deg = zeros(Int, nV)
    for vi in v_adj
        v_deg[vi] += 1
    end
    v_offsets = Vector{Int}(undef, nV + 1)
    v_offsets[1] = 1
    for i in 1:nV
        v_offsets[i + 1] = v_offsets[i] + v_deg[i]
    end

    u_adj = Vector{Int}(undef, nE)
    v_edata = Vector{Nothing}(undef, nE)
    fill_pos = collect(v_offsets[1:nV])
    for ui in 1:nU
        for k in u_offsets[ui]:(u_offsets[ui + 1] - 1)
            vi = v_adj[k]
            pos = fill_pos[vi]
            u_adj[pos] = ui
            v_edata[pos] = edata[k]
            fill_pos[vi] = pos + 1
        end
    end

    return FrozenBipartite{Nothing}(
        u_ids, v_ids, u_index, v_index,
        u_offsets, v_adj, edata,
        v_offsets, u_adj, v_edata,
    )
end

function build_frozen(edges, nU, nV)
    g = BipartiteGraph{Nothing}()
    for u in 1:nU
        add_u!(g, u)
    end
    for v in 1:nV
        add_v!(g, v)
    end
    for (u, v) in edges
        add_edge!(g, u, v, nothing)
    end
    return freeze(g)
end

# Range into v_adj/edge_data for the neighbors of dense u-index ui
@inline neighbor_range_u(fg::FrozenBipartite, ui::Int) = fg.u_offsets[ui]:(fg.u_offsets[ui+1] - 1)
# Range into u_adj/v_edge_data for the neighbors of dense v-index vi
@inline neighbor_range_v(fg::FrozenBipartite, vi::Int) = fg.v_offsets[vi]:(fg.v_offsets[vi+1] - 1)

# kept for backward compatibility with existing callers of the old name
@inline neighbor_range(fg::FrozenBipartite, ui::Int) = neighbor_range_u(fg, ui)


# --------------------------------------------------------------------------
# SubGraph: U/V vertex ids, plus optional dense membership BitVectors.
#
# Masks are indexed by FrozenBipartite dense indices (same as CSR adj).
# Call bind_membership!(sg, fg) once, then mutate via the fg-aware
# Subgraph.add_node! / remove_node! / add! / minus! overloads so bits stay
# in sync. Mask-less mutations invalidate the masks; the next
# ensure_membership! rebuilds them from the Sets.
# --------------------------------------------------------------------------

mutable struct SubGraph
    U::Set{Int}                          # original u ids in this subgraph
    V::Set{Int}                          # original v ids in this subgraph
    u_in::Union{Nothing, BitVector}      # dense membership, or nothing
    v_in::Union{Nothing, BitVector}
end

function SubGraph(U::AbstractSet, V::AbstractSet)
    return SubGraph(Set{Int}(u for u in U), Set{Int}(v for v in V), nothing, nothing)
end
SubGraph() = SubGraph(Set{Int}(), Set{Int}(), nothing, nothing)

subgraph_length(sg::SubGraph) = length(sg.U) + length(sg.V)

"""
    bind_membership!(sg, fg)

Allocate / refresh dense membership masks for `sg` against `fg`. Subsequent
degree/nondegree queries use contiguous bit tests on CSR neighbor indices.
"""
function bind_membership!(sg::SubGraph, fg::FrozenBipartite)
    u_in = falses(length(fg.u_ids))
    v_in = falses(length(fg.v_ids))
    @inbounds for u in sg.U
        ui = get(fg.u_index, u, nothing)
        ui !== nothing && (u_in[ui] = true)
    end
    @inbounds for v in sg.V
        vi = get(fg.v_index, v, nothing)
        vi !== nothing && (v_in[vi] = true)
    end
    sg.u_in = u_in
    sg.v_in = v_in
    return sg
end

@inline function invalidate_membership!(sg::SubGraph)
    sg.u_in = nothing
    sg.v_in = nothing
    return sg
end

"""Ensure masks exist and match `fg`'s dimensions; no-op if already bound."""
function ensure_membership!(sg::SubGraph, fg::FrozenBipartite)
    if sg.u_in === nothing || sg.v_in === nothing ||
       length(sg.u_in) != length(fg.u_ids) || length(sg.v_in) != length(fg.v_ids)
        bind_membership!(sg, fg)
    end
    return sg
end

@inline function _membership_bound(sg::SubGraph, fg::FrozenBipartite)
    return sg.u_in !== nothing && sg.v_in !== nothing &&
           length(sg.u_in) == length(fg.u_ids) && length(sg.v_in) == length(fg.v_ids)
end

"""
    degrees_into_subgraph_u!(hits, fg, sg)

`hits[ui] = |N(u) ∩ sg.V|` for every dense u-index. Walks adjacency of nodes
in `sg.V` (inverted), so cost tracks the cut of `sg` rather than full `|E|`.
"""
function degrees_into_subgraph_u!(hits::Vector{Int}, fg::FrozenBipartite, sg::SubGraph)
    length(hits) == length(fg.u_ids) || throw(DimensionMismatch("hits length must equal nU"))
    fill!(hits, 0)
    @inbounds for v in sg.V
        vi = get(fg.v_index, v, nothing)
        vi === nothing && continue
        for k in neighbor_range_v(fg, vi)
            hits[fg.u_adj[k]] += 1
        end
    end
    return hits
end

"""
    degrees_into_subgraph_v!(hits, fg, sg)

`hits[vi] = |N(v) ∩ sg.U|` for every dense v-index.
"""
function degrees_into_subgraph_v!(hits::Vector{Int}, fg::FrozenBipartite, sg::SubGraph)
    length(hits) == length(fg.v_ids) || throw(DimensionMismatch("hits length must equal nV"))
    fill!(hits, 0)
    @inbounds for u in sg.U
        ui = get(fg.u_index, u, nothing)
        ui === nothing && continue
        for k in neighbor_range_u(fg, ui)
            hits[fg.v_adj[k]] += 1
        end
    end
    return hits
end

# --------------------------------------------------------------------------
# Compact a reduced FrozenBipartite so vertex ids become 1..nU / 1..nV.
# CSR arrays are reused unchanged (they already use dense indices internally).
# --------------------------------------------------------------------------

struct GraphRemapping
    u_original::Vector{Int}   # compact u id -> original u id
    v_original::Vector{Int}   # compact v id -> original v id
end

"""
    compact_frozen(fg) -> (compact_fg, remapping)

Reorder a (possibly reduced) frozen graph so U ids are `1:nU` and V ids are
`1:nV`. The returned `remapping` maps compact subgraph ids back to the
original ids via `remap_subgraph`.
"""
function compact_frozen(fg::FrozenBipartite{T}) where {T}
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    remapping = GraphRemapping(copy(fg.u_ids), copy(fg.v_ids))
    compact = FrozenBipartite{T}(
        collect(1:nU), collect(1:nV),
        Dict(i => i for i in 1:nU), Dict(i => i for i in 1:nV),
        fg.u_offsets, fg.v_adj, fg.edge_data,
        fg.v_offsets, fg.u_adj, fg.v_edge_data,
    )
    return compact, remapping
end

"""Map a subgraph on compact ids back to original vertex ids."""
function remap_subgraph(r::GraphRemapping, sg::SubGraph)
    return SubGraph(
        Set(r.u_original[u] for u in sg.U),
        Set(r.v_original[v] for v in sg.V),
    )
end

"""
    induce_frozen(fg, keep_U, keep_V) -> FrozenBipartite

Build a new frozen bipartite graph containing only the vertices whose *dense*
indices appear in `keep_U` / `keep_V` (in that order). New vertex ids are
`1:length(keep_U)` and `1:length(keep_V)`.

An edge `(u, v)` is kept iff both endpoints are kept; neighbor indices in the
CSR are remapped to the new dense numbering. Prefer passing `keep_U` /
`keep_V` in increasing old-index order (as `eachindex` filters do).
"""
function induce_frozen(fg::FrozenBipartite{T}, keep_U::AbstractVector{Int},
    keep_V::AbstractVector{Int}) where {T}
    nU = length(keep_U)
    nV = length(keep_V)
    nU == 0 && throw(ArgumentError("keep_U must be non-empty"))
    nV == 0 && throw(ArgumentError("keep_V must be non-empty"))

    # old dense index → new dense index (0 = dropped)
    u_new = zeros(Int, length(fg.u_ids))
    v_new = zeros(Int, length(fg.v_ids))
    @inbounds for (ni, oi) in enumerate(keep_U)
        u_new[oi] = ni
    end
    @inbounds for (ni, oi) in enumerate(keep_V)
        v_new[oi] = ni
    end

    u_ids = collect(1:nU)
    v_ids = collect(1:nV)
    u_index = Dict{Int,Int}(i => i for i in 1:nU)
    v_index = Dict{Int,Int}(i => i for i in 1:nV)

    u_offsets = Vector{Int}(undef, nU + 1)
    v_adj = Int[]
    edata = T[]
    sizehint!(v_adj, length(fg.v_adj))
    sizehint!(edata, length(fg.edge_data))

    u_offsets[1] = 1
    @inbounds for (ni, oi) in enumerate(keep_U)
        for k in neighbor_range_u(fg, oi)
            new_vi = v_new[fg.v_adj[k]]
            if new_vi != 0
                push!(v_adj, new_vi)
                push!(edata, fg.edge_data[k])
            end
        end
        u_offsets[ni + 1] = length(v_adj) + 1
    end

    nE = length(v_adj)
    v_deg = zeros(Int, nV)
    @inbounds for vi in v_adj
        v_deg[vi] += 1
    end
    v_offsets = Vector{Int}(undef, nV + 1)
    v_offsets[1] = 1
    @inbounds for i in 1:nV
        v_offsets[i + 1] = v_offsets[i] + v_deg[i]
    end

    u_adj = Vector{Int}(undef, nE)
    v_edata = Vector{T}(undef, nE)
    fill_pos = copy(v_offsets[1:nV])
    @inbounds for ui in 1:nU
        for k in u_offsets[ui]:(u_offsets[ui + 1] - 1)
            vi = v_adj[k]
            pos = fill_pos[vi]
            u_adj[pos] = ui
            v_edata[pos] = edata[k]
            fill_pos[vi] = pos + 1
        end
    end

    return FrozenBipartite{T}(
        u_ids, v_ids, u_index, v_index,
        u_offsets, v_adj, edata,
        v_offsets, u_adj, v_edata,
    )
end

# --------------------------------------------------------------------------
# internal helper: build dense assignment arrays from a Vector{SubGraph}.
# 0 means "not assigned to any of these subgraphs". Not exported/public --
# every public BATCH function below builds and discards this internally so
# you never have to think about dense indices. POINT queries (further down)
# deliberately skip this -- they test Set membership directly instead, since
# building a full-graph dense array for a single-vertex query is overkill.
# --------------------------------------------------------------------------

function _dense_assignment(fg::FrozenBipartite, subgraphs::Vector{SubGraph})
    u_subgraph = zeros(Int, length(fg.u_ids))
    v_subgraph = zeros(Int, length(fg.v_ids))
    for (s, sg) in enumerate(subgraphs)
        for u in sg.U
            ui = get(fg.u_index, u, nothing)
            ui === nothing || (u_subgraph[ui] = s)
        end
        for v in sg.V
            vi = get(fg.v_index, v, nothing)
            vi === nothing || (v_subgraph[vi] = s)
        end
    end
    return u_subgraph, v_subgraph
end


# --------------------------------------------------------------------------
# PHASE 2: split into subgraphs, single O(V + E) pass.
# --------------------------------------------------------------------------

"""
    split_fg(fg, subgraphs::Vector{SubGraph}) -> Vector{Vector{Tuple{Int,Int}}}

Given the subgraphs you want (each specifying its own U/V vertex ids), find
all edges that fall inside each one, in a single O(V+E) pass over the frozen
graph. An edge is kept only if both endpoints belong to the same subgraph.
Returns `sub_edges[s]` = list of (u_id, v_id) pairs for `subgraphs[s]`.

Vertex ids in `subgraphs` that don't exist in `fg` are silently ignored.
"""
function split_fg(fg::FrozenBipartite, subgraphs::Vector{SubGraph})
    u_subgraph, v_subgraph = _dense_assignment(fg, subgraphs)
    n_sub = length(subgraphs)

    sub_edges = [Tuple{Int,Int}[] for _ in 1:n_sub]
    for ui in eachindex(fg.u_ids)
        s = u_subgraph[ui]
        s == 0 && continue
        for k in neighbor_range_u(fg, ui)
            vi = fg.v_adj[k]
            if v_subgraph[vi] == s
                push!(sub_edges[s], (fg.u_ids[ui], fg.v_ids[vi]))
            end
        end
    end
    return sub_edges
end


# --------------------------------------------------------------------------
# PHASE 3a: BATCH read-only accumulation over ALL subgraphs at once.
# --------------------------------------------------------------------------

"""
    subgraph_edge_counts(fg, subgraphs::Vector{SubGraph}) -> Vector{Int}

Single O(V+E) sweep producing the edge count for every subgraph at once.
`counts[s]` corresponds to `subgraphs[s]`.
"""
function subgraph_edge_counts(fg::FrozenBipartite, subgraphs::Vector{SubGraph})
    u_subgraph, v_subgraph = _dense_assignment(fg, subgraphs)
    n_sub = length(subgraphs)

    counts = zeros(Int, n_sub)
    for ui in eachindex(fg.u_ids)
        s = u_subgraph[ui]
        s == 0 && continue
        for k in neighbor_range_u(fg, ui)
            if v_subgraph[fg.v_adj[k]] == s
                counts[s] += 1
            end
        end
    end
    return counts
end

"""
    accumulate_edges!(dest, fg, subgraphs::Vector{SubGraph}, f)

Generic read-only accumulation for phase 3: walks every edge belonging to any
of `subgraphs` exactly once and calls `f(dest, s, u_id, v_id, data)` — where
`s` is the index into `subgraphs` — to fold it in. Use this instead of writing
a bespoke O(V+E) loop for every new whole-graph query.
"""
function accumulate_edges!(dest, fg::FrozenBipartite, subgraphs::Vector{SubGraph}, f::F) where {F}
    u_subgraph, v_subgraph = _dense_assignment(fg, subgraphs)
    for ui in eachindex(fg.u_ids)
        s = u_subgraph[ui]
        s == 0 && continue
        for k in neighbor_range_u(fg, ui)
            vi = fg.v_adj[k]
            if v_subgraph[vi] == s
                f(dest, s, fg.u_ids[ui], fg.v_ids[vi], fg.edge_data[k])
            end
        end
    end
    return dest
end


# --------------------------------------------------------------------------
# PHASE 3b: POINT queries -- one vertex, whole graph or one SubGraph.
#
# These are O(deg(vertex)) and never touch `_dense_assignment`. Restriction
# to a SubGraph is done with a direct `in` test against sg.U / sg.V (Set
# membership, O(1) each), which is the right tool when you only need one
# vertex's answer rather than a sweep over every subgraph.
# --------------------------------------------------------------------------

# ---- whole-graph neighbors, by original id ----

"""
    is_neighbor(fg, is_u, node_id, other_id) -> Bool

Return whether `other_id` is adjacent to `node_id` in the frozen graph, where
`is_u` tells which side `node_id` lives on. Scans the CSR neighbor range for
the dense index of `other_id` (allocation-free).

Note: forward CSR lists are sorted by original neighbor id; reverse lists are
sorted by dense u-index. A single binary-search key does not work for both, so
we use a linear dense-index scan.
"""
function is_neighbor(fg::FrozenBipartite, is_u::Bool, node_id::Int, other_id::Int)
    if is_u
        ui = get(fg.u_index, node_id, nothing)
        ui === nothing && return false
        vi = get(fg.v_index, other_id, nothing)
        vi === nothing && return false
        @inbounds for k in neighbor_range_u(fg, ui)
            fg.v_adj[k] == vi && return true
        end
        return false
    else
        vi = get(fg.v_index, node_id, nothing)
        vi === nothing && return false
        ui = get(fg.u_index, other_id, nothing)
        ui === nothing && return false
        @inbounds for k in neighbor_range_v(fg, vi)
            fg.u_adj[k] == ui && return true
        end
        return false
    end
end

"""
    neighbors_u(fg, u_id) -> Vector{Int}

All V-side neighbor ids of `u_id` in the whole frozen graph. Returns an empty
vector if `u_id` isn't in `fg`.
"""
function neighbors_u(fg::FrozenBipartite, u_id::Int)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return Int[]
    return [fg.v_ids[fg.v_adj[k]] for k in neighbor_range_u(fg, ui)]
end

"""
    neighbors_v(fg, v_id) -> Vector{Int}

All U-side neighbor ids of `v_id` in the whole frozen graph. Returns an empty
vector if `v_id` isn't in `fg`.
"""
function neighbors_v(fg::FrozenBipartite, v_id::Int)
    vi = get(fg.v_index, v_id, nothing)
    vi === nothing && return Int[]
    return [fg.u_ids[fg.u_adj[k]] for k in neighbor_range_v(fg, vi)]
end

# ---- whole-graph degree, by original id ----

"""
    degree_u(fg, u_id) -> Int

Total number of V-side neighbors of `u_id`. 0 if `u_id` isn't in `fg`.
"""
function degree_u(fg::FrozenBipartite, u_id::Int)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return 0
    return length(neighbor_range_u(fg, ui))
end

"""
    degree_v(fg, v_id) -> Int

Total number of U-side neighbors of `v_id`. 0 if `v_id` isn't in `fg`.
"""
function degree_v(fg::FrozenBipartite, v_id::Int)
    vi = get(fg.v_index, v_id, nothing)
    vi === nothing && return 0
    return length(neighbor_range_v(fg, vi))
end

# ---- neighbors restricted to a SubGraph ----

"""
    neighbors_u_in_subgraph(fg, u_id, sg::SubGraph) -> Vector{Int}

V-side neighbors of `u_id` that also belong to `sg.V`. `u_id` itself need not
be a member of `sg.U` -- this just answers "which of u_id's neighbors fall
inside sg", regardless of whether u_id is considered part of sg.
"""
function neighbors_u_in_subgraph(fg::FrozenBipartite, u_id::Int, sg::SubGraph)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return Int[]
    out = Int[]
    for k in neighbor_range_u(fg, ui)
        vid = fg.v_ids[fg.v_adj[k]]
        vid in sg.V && push!(out, vid)
    end
    return out
end

"""
    neighbors_v_in_subgraph(fg, v_id, sg::SubGraph) -> Vector{Int}

U-side neighbors of `v_id` that also belong to `sg.U`.
"""
function neighbors_v_in_subgraph(fg::FrozenBipartite, v_id::Int, sg::SubGraph)
    vi = get(fg.v_index, v_id, nothing)
    vi === nothing && return Int[]
    out = Int[]
    for k in neighbor_range_v(fg, vi)
        uid = fg.u_ids[fg.u_adj[k]]
        uid in sg.U && push!(out, uid)
    end
    return out
end

# ---- degree / nondegree restricted to a SubGraph ----

@inline function _degree_u_against_mask(fg::FrozenBipartite, ui::Int, v_in::BitVector)
    count = 0
    @inbounds for k in neighbor_range_u(fg, ui)
        v_in[fg.v_adj[k]] && (count += 1)
    end
    return count
end

@inline function _degree_v_against_mask(fg::FrozenBipartite, vi::Int, u_in::BitVector)
    count = 0
    @inbounds for k in neighbor_range_v(fg, vi)
        u_in[fg.u_adj[k]] && (count += 1)
    end
    return count
end

"""
    degree_in_subgraph_u(fg, u_id, sg::SubGraph) -> Int

Number of `u_id`'s neighbors that fall inside `sg.V`. `u_id` may or may not
itself be a member of `sg.U`. Uses dense bit masks when bound.
"""
function degree_in_subgraph_u(fg::FrozenBipartite, u_id::Int, sg::SubGraph)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return 0
    if sg.v_in !== nothing && length(sg.v_in) == length(fg.v_ids)
        return _degree_u_against_mask(fg, ui, sg.v_in)
    end
    count = 0
    @inbounds for k in neighbor_range_u(fg, ui)
        fg.v_ids[fg.v_adj[k]] in sg.V && (count += 1)
    end
    return count
end

"""
    degree_in_subgraph_v(fg, v_id, sg::SubGraph) -> Int

Number of `v_id`'s neighbors that fall inside `sg.U`.
"""
function degree_in_subgraph_v(fg::FrozenBipartite, v_id::Int, sg::SubGraph)
    vi = get(fg.v_index, v_id, nothing)
    vi === nothing && return 0
    if sg.u_in !== nothing && length(sg.u_in) == length(fg.u_ids)
        return _degree_v_against_mask(fg, vi, sg.u_in)
    end
    count = 0
    @inbounds for k in neighbor_range_v(fg, vi)
        fg.u_ids[fg.u_adj[k]] in sg.U && (count += 1)
    end
    return count
end

degree_in_subgraph(fg::FrozenBipartite, is_u::Bool, node_id::Int, sg::SubGraph) =
    is_u ? degree_in_subgraph_u(fg, node_id, sg) : degree_in_subgraph_v(fg, node_id, sg)

"""
    nondegree_in_subgraph_u(fg, u_id, sg::SubGraph) -> Int

Number of nodes in `sg.V` that `u_id` does NOT share an edge with
(i.e. `length(sg.V) - degree_in_subgraph_u`). `u_id` need not itself be a
member of `sg.U`; if `u_id` isn't in `fg` at all, this is `length(sg.V)`
since it shares no edges with anything.
"""
function nondegree_in_subgraph_u(fg::FrozenBipartite, u_id::Int, sg::SubGraph)
    return length(sg.V) - degree_in_subgraph_u(fg, u_id, sg)
end

"""
    nondegree_in_subgraph_v(fg, v_id, sg::SubGraph) -> Int

Number of nodes in `sg.U` that `v_id` does NOT share an edge with
(i.e. `length(sg.U) - degree_in_subgraph_v`).
"""
function nondegree_in_subgraph_v(fg::FrozenBipartite, v_id::Int, sg::SubGraph)
    return length(sg.U) - degree_in_subgraph_v(fg, v_id, sg)
end

function nondegree_in_subgraph(fg::FrozenBipartite, is_u::Bool, node_id::Int, sg::SubGraph)
    return is_u ? nondegree_in_subgraph_u(fg, node_id, sg) : nondegree_in_subgraph_v(fg, node_id, sg)
end


module Subgraph
    import ..SubGraph, ..BipartiteGraph, ..FrozenBipartite, .._dense_assignment, ..neighbor_range_u, ..neighbor_range_v
    import ..bind_membership!, ..ensure_membership!, ..invalidate_membership!, .._membership_bound
    export add_node!, remove_node!, add!, minus, minus!, missing_edges, edge_count, nonneighbors_in_subgraph, vertex_count, clone
    export bind_membership!, ensure_membership!, invalidate_membership!

    @inline function _mark!(sg::SubGraph, fg::FrozenBipartite, is_u::Bool, node::Int, val::Bool)
        if is_u
            ui = get(fg.u_index, node, nothing)
            ui !== nothing && (sg.u_in[ui] = val)
        else
            vi = get(fg.v_index, node, nothing)
            vi !== nothing && (sg.v_in[vi] = val)
        end
        return nothing
    end

    function add_node!(sg::SubGraph, is_u::Bool, node::Int)
        if is_u
            push!(sg.U, node)
        else
            push!(sg.V, node)
        end
        # Cannot update dense masks without fg — drop them so the next
        # ensure_membership! rebuilds from the Sets.
        invalidate_membership!(sg)
        return sg
    end

    function add_node!(sg::SubGraph, fg::FrozenBipartite, is_u::Bool, node::Int)
        if is_u
            push!(sg.U, node)
        else
            push!(sg.V, node)
        end
        if _membership_bound(sg, fg)
            _mark!(sg, fg, is_u, node, true)
        end
        return sg
    end

    function remove_node!(sg::SubGraph, is_u::Bool, node::Int)
        if is_u
            delete!(sg.U, node)
        else
            delete!(sg.V, node)
        end
        invalidate_membership!(sg)
        return sg
    end

    function remove_node!(sg::SubGraph, fg::FrozenBipartite, is_u::Bool, node::Int)
        if is_u
            delete!(sg.U, node)
        else
            delete!(sg.V, node)
        end
        if _membership_bound(sg, fg)
            _mark!(sg, fg, is_u, node, false)
        end
        return sg
    end

    function has_node(sg::SubGraph, is_u::Bool, node::Int)
        return is_u ? node in sg.U : node in sg.V
    end

    function add(S::SubGraph, to_add::SubGraph)
        return SubGraph(union(S.U, to_add.U), union(S.V, to_add.V))
    end

    function add!(S::SubGraph, to_add::SubGraph)
        union!(S.U, to_add.U)
        union!(S.V, to_add.V)
        invalidate_membership!(S)
        return S
    end

    function add!(S::SubGraph, fg::FrozenBipartite, to_add::SubGraph)
        union!(S.U, to_add.U)
        union!(S.V, to_add.V)
        if _membership_bound(S, fg)
            @inbounds for u in to_add.U
                ui = get(fg.u_index, u, nothing)
                ui !== nothing && (S.u_in[ui] = true)
            end
            @inbounds for v in to_add.V
                vi = get(fg.v_index, v, nothing)
                vi !== nothing && (S.v_in[vi] = true)
            end
        end
        return S
    end

    function minus(sg1::SubGraph, sg2::SubGraph)
        return SubGraph(setdiff(sg1.U, sg2.U), setdiff(sg1.V, sg2.V))
    end

    function minus!(sg1::SubGraph, sg2::SubGraph)
        setdiff!(sg1.U, sg2.U)
        setdiff!(sg1.V, sg2.V)
        invalidate_membership!(sg1)
        return sg1
    end

    function minus!(sg1::SubGraph, fg::FrozenBipartite, sg2::SubGraph)
        if _membership_bound(sg1, fg)
            @inbounds for u in sg2.U
                if u in sg1.U
                    ui = get(fg.u_index, u, nothing)
                    ui !== nothing && (sg1.u_in[ui] = false)
                end
            end
            @inbounds for v in sg2.V
                if v in sg1.V
                    vi = get(fg.v_index, v, nothing)
                    vi !== nothing && (sg1.v_in[vi] = false)
                end
            end
        end
        setdiff!(sg1.U, sg2.U)
        setdiff!(sg1.V, sg2.V)
        return sg1
    end

    function missing_edges(fg::FrozenBipartite, sg::SubGraph)
        return length(sg.U) * length(sg.V) - edge_count(fg, sg)
    end

    """
        edge_count(g, sg::SubGraph) -> Int

    Count edges between `sg.U` and `sg.V`. Works on a mutable `BipartiteGraph`
    (Dict adjacency) or a frozen graph; use the mutable overload during
    reduction passes to avoid repeated `freeze` calls.
    """
    function edge_count(g::BipartiteGraph, sg::SubGraph)
        count = 0
        for u in sg.U
            neighbors = get(g.adjU, u, nothing)
            neighbors === nothing && continue
            for v in neighbors
                v in sg.V && (count += 1)
            end
        end
        return count
    end

    function edge_count(fg::FrozenBipartite, sg::SubGraph)
        count = 0
        if sg.v_in !== nothing && length(sg.v_in) == length(fg.v_ids)
            @inbounds for u in sg.U
                ui = get(fg.u_index, u, nothing)
                ui === nothing && continue
                for k in neighbor_range_u(fg, ui)
                    sg.v_in[fg.v_adj[k]] && (count += 1)
                end
            end
            return count
        end
        for u in sg.U
            ui = get(fg.u_index, u, nothing)
            ui === nothing && continue
            for k in neighbor_range_u(fg, ui)
                fg.v_ids[fg.v_adj[k]] in sg.V && (count += 1)
            end
        end
        return count
    end

    """
        nonneighbors_in_subgraph(fg, is_u, node_id, sg::SubGraph) -> Set{Int}

    Nodes on the *other* side of `sg` that `node_id` does NOT share an edge
    with. If `is_u`, this returns the subset of `sg.V` not adjacent to
    `node_id` (a U-side id); otherwise it returns the subset of `sg.U` not
    adjacent to `node_id` (a V-side id). `node_id` need not itself be a
    member of `sg`. If `node_id` isn't in `fg` at all, every node on the
    opposite side of `sg` is returned, since it shares no edges with
    anything.
    """
    function nonneighbors_in_subgraph(fg::FrozenBipartite, is_u::Bool, node_id::Int, sg::SubGraph)
        if is_u
            ui = get(fg.u_index, node_id, nothing)
            neighbor_set = Set{Int}()
            if ui !== nothing
                for k in neighbor_range_u(fg, ui)
                    push!(neighbor_set, fg.v_ids[fg.v_adj[k]])
                end
            end
            return setdiff(sg.V, neighbor_set)
        else
            vi = get(fg.v_index, node_id, nothing)
            neighbor_set = Set{Int}()
            if vi !== nothing
                for k in neighbor_range_v(fg, vi)
                    push!(neighbor_set, fg.u_ids[fg.u_adj[k]])
                end
            end
            return setdiff(sg.U, neighbor_set)
        end
    end

    function vertex_count(sg::SubGraph)
        return length(sg.U) + length(sg.V)
    end

    function clone(sg::SubGraph)
        return SubGraph(
            copy(sg.U),
            copy(sg.V),
            sg.u_in === nothing ? nothing : copy(sg.u_in),
            sg.v_in === nothing ? nothing : copy(sg.v_in),
        )
    end
end
