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
#     optionally restricted to a single SubGraph via direct Set membership
#     tests. Use this when you only need one vertex's answer and building the
#     whole-graph dense assignment arrays would be overkill.
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

# --- queries (still available on the live graph if needed) ---
neighbors_u(g::BipartiteGraph, u::Int) = g.adjU[u]
neighbors_v(g::BipartiteGraph, v::Int) = g.adjV[v]
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
    sizehint!(v_adj, length(g.edge_data))
    sizehint!(edata, length(g.edge_data))

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

# Range into v_adj/edge_data for the neighbors of dense u-index ui
@inline neighbor_range_u(fg::FrozenBipartite, ui::Int) = fg.u_offsets[ui]:(fg.u_offsets[ui+1] - 1)
# Range into u_adj/v_edge_data for the neighbors of dense v-index vi
@inline neighbor_range_v(fg::FrozenBipartite, vi::Int) = fg.v_offsets[vi]:(fg.v_offsets[vi+1] - 1)

# kept for backward compatibility with existing callers of the old name
@inline neighbor_range(fg::FrozenBipartite, ui::Int) = neighbor_range_u(fg, ui)


# --------------------------------------------------------------------------
# SubGraph: what you actually specify. Just the U/V vertex ids you want
# grouped together. No dense-index bookkeeping required from you.
# --------------------------------------------------------------------------

struct SubGraph
    U::Set{Int}   # original u ids in this subgraph
    V::Set{Int}   # original v ids in this subgraph
end

subgraph_length(sg::SubGraph) = length(sg.U) + length(sg.V)

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
    split(fg, subgraphs::Vector{SubGraph}) -> Vector{Vector{Tuple{Int,Int}}}

Given the subgraphs you want (each specifying its own U/V vertex ids), find
all edges that fall inside each one, in a single O(V+E) pass over the frozen
graph. An edge is kept only if both endpoints belong to the same subgraph.
Returns `sub_edges[s]` = list of (u_id, v_id) pairs for `subgraphs[s]`.

Vertex ids in `subgraphs` that don't exist in `fg` are silently ignored.
"""
function split(fg::FrozenBipartite, subgraphs::Vector{SubGraph})
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
    subgraph_edge_count(fg, sg::SubGraph) -> Int

Single-subgraph version, when you only need one count.
"""
function subgraph_edge_count(fg::FrozenBipartite, sg::SubGraph)
    u_subgraph, v_subgraph = _dense_assignment(fg, [sg])
    count = 0
    for ui in eachindex(fg.u_ids)
        u_subgraph[ui] == 1 || continue
        for k in neighbor_range_u(fg, ui)
            v_subgraph[fg.v_adj[k]] == 1 && (count += 1)
        end
    end
    return count
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

"""
    degree_in_subgraph_u(fg, u_id, sg::SubGraph) -> Int

Number of `u_id`'s neighbors that fall inside `sg.V`. `u_id` may or may not
itself be a member of `sg.U`.
"""
function degree_in_subgraph_u(fg::FrozenBipartite, u_id::Int, sg::SubGraph)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return 0
    count = 0
    for k in neighbor_range_u(fg, ui)
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
    count = 0
    for k in neighbor_range_v(fg, vi)
        fg.u_ids[fg.u_adj[k]] in sg.U && (count += 1)
    end
    return count
end

"""
    nondegree_in_subgraph_u(fg, u_id, sg::SubGraph) -> Int

Number of `u_id`'s neighbors that fall OUTSIDE `sg.V`
(i.e. total degree minus `degree_in_subgraph_u`).
"""
function nondegree_in_subgraph_u(fg::FrozenBipartite, u_id::Int, sg::SubGraph)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return 0
    total = length(neighbor_range_u(fg, ui))
    return total - degree_in_subgraph_u(fg, u_id, sg)
end

"""
    nondegree_in_subgraph_v(fg, v_id, sg::SubGraph) -> Int

Number of `v_id`'s neighbors that fall OUTSIDE `sg.U`
(i.e. total degree minus `degree_in_subgraph_v`).
"""
function nondegree_in_subgraph_v(fg::FrozenBipartite, v_id::Int, sg::SubGraph)
    vi = get(fg.v_index, v_id, nothing)
    vi === nothing && return 0
    total = length(neighbor_range_v(fg, vi))
    return total - degree_in_subgraph_v(fg, v_id, sg)
end