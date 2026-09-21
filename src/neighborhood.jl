#=
Neighborhood-Jaccard edge scoring (S_C).

For each edge (u, v) in a bipartite graph G = (U, V, E) we define:

    C_U(u, v) = mean of top-θ J(N(u), N(u')) over u' ∈ N(v) \ {u}
    C_V(u, v) = mean of top-θ J(N(v), N(v')) over v' ∈ N(u) \ {v}

where J(A, B) = |A ∩ B| / |A ∪ B| is the Jaccard similarity between two
neighbor sets (both subsets of V for the U-side, both subsets of U for the
V-side).

The combined score is:

    S_C(u, v) = combine(C_U(u, v), C_V(u, v))

By default `combine = (*)` reproduces the product C_U · C_V.
Pass `combine = min` to use min(C_U, C_V) instead.

Edges with high S_C scores are incident to vertices whose common-neighbor
structure is dense and symmetric — a strong signal that the edge participates
in a dense (near-complete) bipartite subgraph.

Complexity: O(Σ_v deg(v)² · d_maxU  +  Σ_u deg(u)² · d_maxV).
In practice this is fast because Jaccard is computed via the CSR
dirty-accumulation trick (no allocations per pair).
=#

const __NEIGHBORHOOD_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
    _jaccard_accumulate!(cnt, dirty, fg, ref_ui, query_ui) -> Float64

Compute J(N(ref_u), N(query_u)) using a dirty accumulation array `cnt`
(length nV, caller-allocated). Both neighbor sets are subsets of V.

After the call `cnt` and `dirty` are reset to their pre-call state (all zeros /
empty), so the same buffers can be reused across many calls.

Returns 0.0 if either vertex has no neighbors (undefined Jaccard).
"""
@inline function _jaccard_u(fg::FrozenBipartite,
                            cnt::Vector{Int8}, dirty::Vector{Int},
                            ref_ui::Int, query_ui::Int)
    r_ref   = neighbor_range_u(fg, ref_ui)
    r_query = neighbor_range_u(fg, query_ui)
    (isempty(r_ref) || isempty(r_query)) && return 0.0

    # Mark ref neighbors with flag 1
    @inbounds for k in r_ref
        vi = fg.v_adj[k]
        if cnt[vi] == 0
            push!(dirty, vi)
        end
        cnt[vi] = 1
    end

    intersection = 0
    union_size   = length(r_ref)

    # Walk query neighbors: count intersection, extend union
    @inbounds for k in r_query
        vi = fg.v_adj[k]
        if cnt[vi] == 1
            intersection += 1
        elseif cnt[vi] == 0
            union_size += 1
            push!(dirty, vi)
            cnt[vi] = 2   # seen only in query — still reset correctly below
        end
        # cnt[vi] == 2 means already counted in union extension; skip
    end

    # Reset
    @inbounds for vi in dirty
        cnt[vi] = 0
    end
    empty!(dirty)

    return union_size == 0 ? 0.0 : intersection / union_size
end

"""
    _jaccard_v(fg, cnt, dirty, ref_vi, query_vi) -> Float64

Compute J(N(ref_v), N(query_v)) where both neighbor sets are subsets of U.
`cnt` has length nU.
"""
@inline function _jaccard_v(fg::FrozenBipartite,
                            cnt::Vector{Int8}, dirty::Vector{Int},
                            ref_vi::Int, query_vi::Int)
    r_ref   = neighbor_range_v(fg, ref_vi)
    r_query = neighbor_range_v(fg, query_vi)
    (isempty(r_ref) || isempty(r_query)) && return 0.0

    @inbounds for k in r_ref
        ui = fg.u_adj[k]
        if cnt[ui] == 0
            push!(dirty, ui)
        end
        cnt[ui] = 1
    end

    intersection = 0
    union_size   = length(r_ref)

    @inbounds for k in r_query
        ui = fg.u_adj[k]
        if cnt[ui] == 1
            intersection += 1
        elseif cnt[ui] == 0
            union_size += 1
            push!(dirty, ui)
            cnt[ui] = 2
        end
    end

    @inbounds for ui in dirty
        cnt[ui] = 0
    end
    empty!(dirty)

    return union_size == 0 ? 0.0 : intersection / union_size
end

# ─────────────────────────────────────────────────────────────────────────────
# Public scoring function
# ─────────────────────────────────────────────────────────────────────────────

"""
    neighborhood_scores(fg, pos_θ=nothing; θ=nothing, theta=nothing, pad_zeros=true, combine=(*)) -> Vector{Float64}

Compute the neighbourhood-Jaccard score S_C(u, v) for every edge in `fg`,
returned in CSR order (parallel to fg.v_adj).

    C_U(u, v) = mean of top-θ Jaccard similarities J(N(u), N(u')) over u' ∈ N(v) \\ {u}
    C_V(u, v) = mean of top-θ Jaccard similarities J(N(v), N(v')) over v' ∈ N(u) \\ {v}
    S_C(u, v) = combine(C_U(u, v), C_V(u, v))

The default `θ = 5`.
When `pad_zeros = true` (default), if there are fewer than θ co-neighbours,
the sum of available top similarities is divided by θ (treating missing partners
as similarity 0.0). If `pad_zeros = false`, the sum is divided by min(n_partners, θ).
The default `combine = (*)` gives the product C_U · C_V (original formula).
Pass `combine = min` to use min(C_U, C_V) instead.

Edges where either side of the biclique neighbourhood is degenerate (degree 1
on both the u- and v-side, so no u'/v' co-neighbours exist) receive score 0.
"""
function neighborhood_scores(fg::FrozenBipartite, pos_θ::Union{Int,Nothing}=nothing;
                             θ::Union{Int,Nothing}=nothing,
                             theta::Union{Int,Nothing}=nothing,
                             pad_zeros::Bool=true,
                             combine = (*))::Vector{Float64}
    effective_θ = something(theta, θ, pos_θ, 5)
    effective_θ > 0 || throw(ArgumentError("θ must be positive, got $effective_θ"))

    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    nE = length(fg.v_adj)

    scores = zeros(Float64, nE)

    # ── C_U pass: for each u, compute mean top-θ Jaccard to co-neighbours u' ──
    # C_U(u, v) accumulated into cu_sum[k] (slot k = edge (u,v)).
    cu_sum = zeros(Float64, nE)

    cnt_v  = zeros(Int8, nV)   # scratch for Jaccard over V-neighbors
    dirty_v = Int[]
    sizehint!(dirty_v, 64)
    sim_buf = Float64[]
    sizehint!(sim_buf, 64)

    for vi in 1:nV
        r_v = neighbor_range_v(fg, vi)
        deg_v = length(r_v)
        deg_v <= 1 && continue   # no u' candidate → C_U = 0 for all edges into vi

        # For each u (slot l in r_v), compute mean top-θ J with other u' (slot m)
        for l in r_v
            ui = fg.u_adj[l]
            # Find the forward CSR slot for (ui → vi)
            # (we need to accumulate into cu_sum[k] where fg.v_adj[k] == vi)
            fwd_slot = _find_fwd_slot(fg, ui, vi)
            fwd_slot == 0 && continue

            empty!(sim_buf)
            for m in r_v
                u2i = fg.u_adj[m]
                u2i == ui && continue
                push!(sim_buf, _jaccard_u(fg, cnt_v, dirty_v, ui, u2i))
            end

            n = length(sim_buf)
            if n == 0
                cu_sum[fwd_slot] = 0.0
            else
                n_top = min(n, effective_θ)
                if n > effective_θ
                    partialsort!(sim_buf, 1:effective_θ, rev=true)
                end
                s = 0.0
                @inbounds for idx in 1:n_top
                    s += sim_buf[idx]
                end
                denom = pad_zeros ? effective_θ : n_top
                cu_sum[fwd_slot] = s / denom
            end
        end
    end

    # ── C_V pass: for each v, compute mean top-θ Jaccard to co-neighbours v' ──
    cv_sum = zeros(Float64, nE)

    cnt_u  = zeros(Int8, nU)
    dirty_u = Int[]
    sizehint!(dirty_u, 64)

    for ui in 1:nU
        r_u = neighbor_range_u(fg, ui)
        deg_u = length(r_u)
        deg_u <= 1 && continue

        for k in r_u
            vi = fg.v_adj[k]

            empty!(sim_buf)
            for m in r_u
                v2i = fg.v_adj[m]
                v2i == vi && continue
                push!(sim_buf, _jaccard_v(fg, cnt_u, dirty_u, vi, v2i))
            end

            n = length(sim_buf)
            if n == 0
                cv_sum[k] = 0.0
            else
                n_top = min(n, effective_θ)
                if n > effective_θ
                    partialsort!(sim_buf, 1:effective_θ, rev=true)
                end
                s = 0.0
                @inbounds for idx in 1:n_top
                    s += sim_buf[idx]
                end
                denom = pad_zeros ? effective_θ : n_top
                cv_sum[k] = s / denom
            end
        end
    end

    # ── Combine ───────────────────────────────────────────────────────────────
    @inbounds for k in 1:nE
        scores[k] = combine(cu_sum[k], cv_sum[k])
    end

    return scores
end

"""
    _find_fwd_slot(fg, ui, vi) -> Int

Return the CSR slot index k such that fg.v_adj[k] == vi within the neighbor
range of ui, or 0 if not found. O(deg(u)).
"""
@inline function _find_fwd_slot(fg::FrozenBipartite, ui::Int, vi::Int)::Int
    @inbounds for k in neighbor_range_u(fg, ui)
        fg.v_adj[k] == vi && return k
    end
    return 0
end
