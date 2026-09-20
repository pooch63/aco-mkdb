#=
=================================================================================
Variable Neighborhood Search (VNS) for maximum k-defective edge biclique.

Hyperparameter: `kmax` — maximum perturbation intensity.

1. Seed S with the highest-degree vertex and its highest-degree neighbor; also
   keep B ← S (best).
2. Greedily grow S (prefer the side that still needs θ when the other side
   already has ≥ θ).
3. intensity ← 1
4. N ← copy(S); apply `intensity` random perturbations. Each perturbation
   picks a side (U/V, 50/50), then remove vs. add on that side (50/50).
5. Greedily expand N with the same θ-biased growth as step 2.
6. If N has more edges than S: S ← N (and B ← S if improved); intensity ← 1;
   goto 4.
7. If intensity ≥ kmax: stop and return B.
8. intensity ← intensity + 1; goto 4.

Removals are sampled ∝ (Δ_N + 1 − deg_N(n)) (prefer low subgraph-degree).
Additions are among k-feasible candidates on the chosen side, sampled
∝ deg_N(n)·Δ_C + deg_G(n).
=================================================================================
=#

const __VNS_JL__ = true

isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")

using Random

"""
Highest-degree vertex in `fg` across both sides, then its highest-degree neighbor.
Returns a 2-vertex SubGraph, or empty if the graph has no edges.
"""
function vns_seed_edge(fg::FrozenBipartite)
    best_is_u = true
    best_id = -1
    best_deg = -1

    for u in fg.u_ids
        d = degree_u(fg, u)
        if d > best_deg
            best_deg = d
            best_is_u = true
            best_id = u
        end
    end
    for v in fg.v_ids
        d = degree_v(fg, v)
        if d > best_deg
            best_deg = d
            best_is_u = false
            best_id = v
        end
    end

    best_id < 0 && return SubGraph()
    best_deg == 0 && return SubGraph()

    nbrs = best_is_u ? neighbors_u(fg, best_id) : neighbors_v(fg, best_id)
    isempty(nbrs) && return SubGraph()

    nbr_id = argmax(n -> best_is_u ? degree_v(fg, n) : degree_u(fg, n), nbrs)

    if best_is_u
        return SubGraph(Set([best_id]), Set([nbr_id]))
    else
        return SubGraph(Set([nbr_id]), Set([best_id]))
    end
end

"""
Candidate pool for θ-biased greedy growth: if one side already has ≥ θ and the
other does not, restrict to the deficient side.
"""
function vns_growth_candidates(fg::FrozenBipartite, S::SubGraph, defect_k::Int, θ::Int)
    C = candidate_set(fg, S, defect_k)
    nU, nV = length(S.U), length(S.V)
    if nU >= θ && nV < θ
        return SubGraph(Set{Int}(), C.V)
    elseif nV >= θ && nU < θ
        return SubGraph(C.U, Set{Int}())
    else
        return C
    end
end

"""
Greedily add highest subgraph-degree feasible vertices until no candidates remain.
When one side has ≥ θ and the other does not, only grow the deficient side.
"""
function vns_greedily_grow!(fg::FrozenBipartite, S::SubGraph, defect_k::Int, θ::Int)
    ensure_membership!(S, fg)
    while true
        C = vns_growth_candidates(fg, S, defect_k, θ)
        Subgraph.vertex_count(C) == 0 && return S
        is_u, node = argmax_nodes((u, n) -> degree_in_subgraph(fg, u, n, S), C)
        Subgraph.add_node!(S, fg, is_u, node)
    end
end

@inline function _vns_deg_in(fg::FrozenBipartite, node::Node, sg::SubGraph)
    return degree_in_subgraph(fg, node.is_u, node.id, sg)
end

@inline function _vns_deg_G(fg::FrozenBipartite, node::Node)
    return node.is_u ? degree_u(fg, node.id) : degree_v(fg, node.id)
end

"""
One perturbation: pick side U or V with equal probability, then with equal
probability remove a vertex on that side (biased toward low subgraph degree) or
add a k-feasible candidate on that side (biased toward high deg_N·Δ_C + deg_G).
If the chosen action's pool is empty, tries the other action on the same side;
no-ops if both pools are empty.
"""
function vns_perturb_once!(fg::FrozenBipartite, N::SubGraph, defect_k::Int)
    is_u = rand() < 0.5
    want_remove = rand() < 0.5

    verts = is_u ?
        Node[Node(true, u) for u in N.U] :
        Node[Node(false, v) for v in N.V]
    C = candidate_set(fg, N, defect_k)
    cands = is_u ?
        Node[Node(true, u) for u in C.U] :
        Node[Node(false, v) for v in C.V]

    do_remove = want_remove ? !isempty(verts) : isempty(cands) && !isempty(verts)
    if do_remove
        ΔN = maximum(_vns_deg_in(fg, n, N) for n in verts)
        chosen = linear_sample_one(n -> Float64(ΔN + 1 - _vns_deg_in(fg, n, N)), verts)
        Subgraph.remove_node!(N, fg, chosen.is_u, chosen.id)
        return
    end

    isempty(cands) && return
    ΔC = maximum(_vns_deg_G(fg, n) for n in cands)
    chosen = linear_sample_one(
        n -> Float64(_vns_deg_in(fg, n, N) * ΔC + _vns_deg_G(fg, n)),
        cands)
    Subgraph.add_node!(N, fg, chosen.is_u, chosen.id)
end

"""
Apply `intensity` sequential perturbations to a clone of `S`.
"""
function vns_shake(fg::FrozenBipartite, S::SubGraph, defect_k::Int, intensity::Int)
    N = Subgraph.clone(S)
    ensure_membership!(N, fg)
    for _ in 1:intensity
        vns_perturb_once!(fg, N, defect_k)
    end
    return N
end

"""
Run VNS on a frozen bipartite graph. Returns `(best, iterations_to_best)`.
`iterations_to_best` counts accepted improving shakes (0 if only the seed).
"""
function vns_search(fg::FrozenBipartite, defect_k::Int, θ::Int, kmax::Int)
    if length(fg.u_ids) < 1 || length(fg.v_ids) < 1
        return SubGraph(), 0
    end

    S = vns_seed_edge(fg)
    if Subgraph.vertex_count(S) == 0
        return SubGraph(), 0
    end
    ensure_membership!(S, fg)
    vns_greedily_grow!(fg, S, defect_k, θ)

    B = Subgraph.clone(S)
    best_edges = Subgraph.edge_count(fg, B)
    accepted = 0

    intensity = 1
    while true
        N = vns_shake(fg, S, defect_k, intensity)
        vns_greedily_grow!(fg, N, defect_k, θ)
        n_edges = Subgraph.edge_count(fg, N)
        s_edges = Subgraph.edge_count(fg, S)

        if n_edges > s_edges
            S = N
            if n_edges > best_edges
                B = Subgraph.clone(S)
                best_edges = n_edges
                accepted += 1
            end
            intensity = 1
            continue
        end

        intensity >= kmax && break
        intensity += 1
    end

    return B, accepted
end

"""
Public entry: reduce (optional), run VNS, return best SubGraph.
"""
function vns(g::BipartiteGraph, defect_k::Int, θ::Int;
    kmax::Int=10,
    reduction::ReductionMode.T=ReductionMode.all_reductions)
    @assert θ > defect_k "θ must be greater than k"
    kmax >= 1 || throw(ArgumentError("kmax must be ≥ 1, got $kmax"))

    fg = apply_graph_reductions!(g, defect_k, θ, nothing, nothing, true, reduction)
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph()
    end

    best, _ = vns_search(fg, defect_k, θ, kmax)
    return best
end
