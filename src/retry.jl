#=
=================================================================================
Retry search for maximum k-defective edge biclique.

Hyperparameters:
  β ∈ [0, 1) — probability of backtracking even when expandable candidates exist
  p > 0 — rate of the truncated geometric that picks how many nodes to drop
  max_steps — hard budget on add/remove operations (search never terminates
              without a budget because resampling can loop)

1. S ← ∅; keep B ← best seen.
2. Sample a seed vertex ∝ deg_G; add it to S (path stack).
3. L ← last vertex on the path.
   C ← {u ∈ N(L) \ S | missing(S ∪ {u}) ≤ k}.
4. If (S nonempty and rand() < β) or C = ∅: backtrack (step 6).
5. Sample u ∈ C with P ∝ η = deg_S(u) + deg_G(u)·σ(-|S|), where
   σ(-|S|) = 1 / (1 + exp(|S|)) (same η as ACO, without pheromone).
   Add u; goto 3.
6. Sample r ∈ {1,…,|S|} with
     P(r=x) = ((1−e^{−p}) / (1−e^{−p n})) · e^{−p(x−1)},  n = |S|.
   With probability 1/2 remove the first r nodes on the path; otherwise remove
   the last r (updating L). If S = ∅, goto 2 (reseed); else goto 3.

Stop when `max_steps` add/remove operations have been performed; return B.
=================================================================================
=#

const __RETRY_JL__ = true

isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")

using Random

@inline function _retry_deg_G(fg::FrozenBipartite, node::Node)
    return node.is_u ? degree_u(fg, node.id) : degree_v(fg, node.id)
end

@inline function _retry_deg_S(fg::FrozenBipartite, node::Node, sg::SubGraph)
    return degree_in_subgraph(fg, node.is_u, node.id, sg)
end

"""
ACO-style η (no pheromone): deg_S + deg_G · σ(-|S|), with
σ(-|S|) = 1 / (1 + exp(|S|)).
"""
@inline function retry_node_desirability(fg::FrozenBipartite, node::Node,
    sg::SubGraph, sigmoid_neg_size::Float64)
    return Float64(_retry_deg_S(fg, node, sg) +
        _retry_deg_G(fg, node) * sigmoid_neg_size)
end

"""
All vertices on both sides as `Node`s, for degree-proportional seeding.
"""
function _retry_all_nodes(fg::FrozenBipartite)
    nodes = Node[]
    sizehint!(nodes, length(fg.u_ids) + length(fg.v_ids))
    for u in fg.u_ids
        push!(nodes, Node(true, u))
    end
    for v in fg.v_ids
        push!(nodes, Node(false, v))
    end
    return nodes
end

"""
Neighbors of `L` that are not already in `S` and keep the defect ≤ `defect_k`.
"""
function retry_neighbor_candidates(fg::FrozenBipartite, S::SubGraph, L::Node,
    defect_k::Int)
    budget = defect_k - Subgraph.missing_edges(fg, S)
    budget < 0 && return Node[]

    nbr_ids = L.is_u ? neighbors_u(fg, L.id) : neighbors_v(fg, L.id)
    # Neighbors of a U-vertex are V-ids (is_u=false), and vice versa.
    cand_is_u = !L.is_u
    C = Node[]
    sizehint!(C, length(nbr_ids))
    for nid in nbr_ids
        Subgraph.has_node(S, cand_is_u, nid) && continue
        nondeg = nondegree_in_subgraph(fg, cand_is_u, nid, S)
        nondeg <= budget && push!(C, Node(cand_is_u, nid))
    end
    return C
end

"""
Sample how many nodes to remove under the truncated geometric
P(X=x) = ((1−e^{−p})/(1−e^{−p n})) · e^{−p(x−1)} for x = 1,…,n.
"""
function retry_backtrack_count(n::Int, p::Float64)
    n <= 1 && return n
    # CDF(x) = (1 − e^{−p x}) / (1 − e^{−p n})
    denom = 1 - exp(-p * n)
    if denom <= 0.0
        # p extremely small → nearly uniform on 1..n
        return rand(1:n)
    end
    r = rand()
    @inbounds for x in 1:n
        (1 - exp(-p * x)) / denom >= r && return x
    end
    return n
end

"""
Remove `r` nodes from the front or end of `path` (and from `S`).
Each node removal counts as one step toward `max_steps`.
Returns the updated step count.
"""
function _retry_remove_nodes!(fg::FrozenBipartite, S::SubGraph, path::Vector{Node},
    r::Int, from_front::Bool, steps::Int, max_steps::Int)
    removed = 0
    while removed < r && !isempty(path) && steps < max_steps
        node = from_front ? popfirst!(path) : pop!(path)
        Subgraph.remove_node!(S, fg, node.is_u, node.id)
        steps += 1
        removed += 1
    end
    return steps
end

"""
If `S` improves on `(best_edges, best_fit)`, return a clone of `S` and true;
otherwise return `B` and false.
"""
function _retry_maybe_improve(fg::FrozenBipartite, S::SubGraph, B::SubGraph,
    best_edges::Int, best_fit::Float64, θ::Int)
    edges = Subgraph.edge_count(fg, S)
    fit = Float64(instance_fitness(fg, S, θ))
    if edges > best_edges || (edges == best_edges && fit > best_fit)
        return Subgraph.clone(S), true, edges, fit
    end
    return B, false, best_edges, best_fit
end

"""
Run retry search on a frozen bipartite graph.

Returns `(best, steps_to_best)` where `steps_to_best` is the operation index
at which the returned best first appeared (0 if empty).
"""
function retry_search(fg::FrozenBipartite, defect_k::Int, θ::Int, β::Float64,
    p::Float64, max_steps::Int)
    if length(fg.u_ids) < 1 || length(fg.v_ids) < 1 || max_steps < 1
        return SubGraph(), 0
    end
    (0.0 <= β < 1.0) || throw(ArgumentError("β must be in [0, 1), got $β"))
    p > 0.0 || throw(ArgumentError("p must be > 0, got $p"))

    all_nodes = _retry_all_nodes(fg)
    isempty(all_nodes) && return SubGraph(), 0

    S = SubGraph()
    ensure_membership!(S, fg)
    path = Node[]
    B = SubGraph()
    best_edges = 0
    best_fit = -Inf
    steps_to_best = 0
    steps = 0

    while steps < max_steps
        # Seed when empty (initial start or after full backtrack).
        if isempty(path)
            seed = linear_sample_one(n -> Float64(_retry_deg_G(fg, n)), all_nodes)
            Subgraph.add_node!(S, fg, seed.is_u, seed.id)
            push!(path, seed)
            steps += 1
            B, improved, best_edges, best_fit =
                _retry_maybe_improve(fg, S, B, best_edges, best_fit, θ)
            improved && (steps_to_best = steps)
            steps >= max_steps && break
        end

        L = path[end]
        C = retry_neighbor_candidates(fg, S, L, defect_k)

        should_backtrack = (!isempty(path) && rand() < β) || isempty(C)
        if should_backtrack
            r = retry_backtrack_count(length(path), p)
            from_front = rand() < 0.5
            steps = _retry_remove_nodes!(fg, S, path, r, from_front, steps, max_steps)
            continue
        end

        # Same η damping as ACO advance.jl (sigmoid of −|S|).
        sigmoid_neg_size = 1 / (1 + exp(Subgraph.vertex_count(S)))
        chosen = linear_sample_one(
            n -> retry_node_desirability(fg, n, S, sigmoid_neg_size),
            C)
        Subgraph.add_node!(S, fg, chosen.is_u, chosen.id)
        push!(path, chosen)
        steps += 1
        B, improved, best_edges, best_fit =
            _retry_maybe_improve(fg, S, B, best_edges, best_fit, θ)
        improved && (steps_to_best = steps)
    end

    return B, steps_to_best
end

"""
Public entry: reduce (optional), run retry search, return best SubGraph.
"""
function retry(g::BipartiteGraph, defect_k::Int, θ::Int;
    β::Float64=0.04,
    p::Float64=1.0,
    max_steps::Int=10_000,
    reduction::ReductionMode.T=ReductionMode.all_reductions)
    @assert θ > defect_k "θ must be greater than k"
    (0.0 <= β < 1.0) || throw(ArgumentError("β must be in [0, 1), got $β"))
    p > 0.0 || throw(ArgumentError("p must be > 0, got $p"))
    max_steps >= 1 || throw(ArgumentError("max_steps must be ≥ 1, got $max_steps"))

    fg = apply_graph_reductions!(g, defect_k, θ, nothing, nothing, true, reduction)
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph()
    end

    best, _ = retry_search(fg, defect_k, θ, β, p, max_steps)
    return best
end
