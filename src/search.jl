const __SEARCH_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__THETA_HEURISTIC_JL__) || include("theta_heuristic.jl")

using EnumX

function arg_nodes(f, is_max::Bool, sg::SubGraph)
    best_score = is_max ? -Inf : Inf
    best_is_u = false
    best_node = -1

    for u in sg.U
        score = f(true, u)
        if (is_max && score > best_score) || (!is_max && score < best_score)
            best_score = score
            best_is_u = true
            best_node = u
        end
    end
    
    for v in sg.V
       score = f(false, v)
        if (is_max && score > best_score) || (!is_max && score < best_score)
            best_score = score
            best_is_u = false
            best_node = v
        end
    end

    best_node != -1 || error("arg_nodes called on empty candidate set")

    return best_is_u, best_node
end
argmax_nodes(f, sg::SubGraph) = arg_nodes(f, true, sg)
argmin_nodes(f, sg::SubGraph) = arg_nodes(f, false, sg)

@enumx ReductionMode all_reductions simple progressive none

function apply_graph_reductions!(g::BipartiteGraph, k::Int, θ::Int,
    num_U::Union{Int, Nothing}, num_V::Union{Int, Nothing},
    use_heuristic::Bool, reduction::ReductionMode.T)

    num_U = num_U === nothing ? length(g.adjU) : num_U
    num_V = num_V === nothing ? length(g.adjV) : num_V

    if reduction == ReductionMode.none
        return freeze(g)
    end
    
    reduce_graph!(g, k, θ, num_U, num_V)
    last_θ_eff = θ

    if reduction == ReductionMode.progressive || reduction == ReductionMode.all_reductions
        fg = freeze(g)
        u_degrees = Int[]
        for u in fg.u_ids
            push!(u_degrees, degree_u(fg, u))
        end

        isempty(u_degrees) && return fg

        θ_U = maximum(u_degrees) + k
        θ_V = 0

        D = use_heuristic ? theta_based_heuristic(fg, k, θ; return_invalid=false) : SubGraph(Set(), Set())
        D_edges = Subgraph.edge_count(g, D)

        # θ_eff = min(θ_U, θ_V) is non-monotonic; peel each new θ_eff from the
        # post-CNN base, then copy the result back into `g` for the return value.
        g_base = deepcopy(g)

        iterations = 0
        reductions = 0

        while θ_U > θ
            iterations += 1
            θ_V = max(θ, floor(Int, D_edges / θ_U))
            θ_U = max(θ, floor(Int, θ_U / 2))

            θ_eff = min(θ_U, θ_V)
            # Skip passes whose effective threshold matches the last reduce —
            # common when D is empty (θ_V ≡ θ) or θ_U halves but still ≥ last θ_eff.
            if θ_eff != last_θ_eff
                g_round = deepcopy(g_base)
                reduce_graph!(g_round, k, θ_eff, num_U, num_V)
                empty!(g.adjU)
                empty!(g.adjV)
                empty!(g.edge_data)
                for (u, nbrs) in g_round.adjU
                    g.adjU[u] = nbrs
                end
                for (v, nbrs) in g_round.adjV
                    g.adjV[v] = nbrs
                end
                for (e, data) in g_round.edge_data
                    g.edge_data[e] = data
                end
                last_θ_eff = θ_eff
                reductions += 1
            end
        end

        println("Progressive reductions took $(iterations) iterations ($(reductions) new reduces)")
    end

    return freeze(g)
end

function upper_bound(S::SubGraph, C::SubGraph, g::FrozenBipartite, k::Int, S_missing::Int)
    S_edges = length(S.U) * length(S.V) - S_missing
    budget = k - S_missing
    # Note: may have to redo C's structure, because a collect call every time might have significant overhead
    u = sort(collect(C.U), by = u -> nondegree_in_subgraph(g, true, u, S))
    v = sort(collect(C.V), by = v -> nondegree_in_subgraph(g, false, v, S))

    nondegree_U, nondegree_V = 0, 0
    nodes_U, nodes_V = 0, 0

    for i in 1:length(u)
        nondegree_in_S = nondegree_in_subgraph(g, true, u[i], S)
        if nondegree_U + nondegree_in_S ≤ budget
            nondegree_U = nondegree_U + nondegree_in_S
            nodes_U += 1
        else
            break
        end
    end

    e = (length(S.U) + nodes_U) * length(S.V) - S_missing - nondegree_U
    i = nodes_U

    for j in 1:length(v)
        cost_V = nondegree_in_subgraph(g, false, v[j], S)

        if nondegree_V + cost_V > budget
            break
        end
        

        while i > 0 && nondegree_U + nondegree_V + cost_V > budget
            nondegree_U -= nondegree_in_subgraph(g, true, u[i], S)
            i -= 1
        end

        nondegree_V += cost_V
        nodes_V += 1
        e = max(
            e,
            (length(S.U) + i) * (length(S.V) + j) - S_missing - nondegree_U - nondegree_V
        )
    end

    return length(S.U) + nodes_U, length(S.V) + nodes_V, e
end

# If we add u to S, what does C become? What "free" nodes can we add to S that don't limit the search space?
function update(S::SubGraph, C::SubGraph, g::FrozenBipartite, is_u::Bool, node::Int, k::Int, S_missing::Int, depth::Int)
    # C′ = {v ∈ C ∖ {u} | nondegree_{ S ∪ {u} }(v) ≤ k - Ē(S)}

    ensure_membership!(S, g)
    ensure_membership!(C, g)

    new_S_missing = S_missing + nondegree_in_subgraph(g, is_u, node, S)

    Subgraph.remove_node!(C, g, is_u, node)
    Subgraph.add_node!(S, g, is_u, node)

    maximum_nondegree = k - new_S_missing

    C′_u = Set{Int}()
    sizehint!(C′_u, length(C.U))
    for node in C.U
        nondegree_in_subgraph(g, true, node::Int, S) ≤ maximum_nondegree && push!(C′_u, node)
    end

    C′_v = Set{Int}()
    sizehint!(C′_v, length(C.V))
    for node in C.V
        nondegree_in_subgraph(g, false, node::Int, S) ≤ maximum_nondegree && push!(C′_v, node)
    end

    C′ = SubGraph(C′_u, C′_v)
    bind_membership!(C′, g)
    Subgraph.remove_node!(S, g, is_u, node::Int)
    Subgraph.add_node!(C, g, is_u, node::Int)

    # T = S ∪ {u} ∪ C′
    nondegree_in_T(node_is_u, v) = nondegree_in_subgraph(g, node_is_u, v, S) + nondegree_in_subgraph(g, node_is_u, v, C′) + (node_is_u == !is_u && is_neighbor(g, node_is_u, v, node) ? 0 : 1)

    # C′_0 = {v ∈ C′ | nondegree_T(v) = 0}
    C′_0_u = Set(
        node for node in C′_u if nondegree_in_T(true, node::Int) == 0
    )
    C′_0_v = Set(
        node for node in C′_v if nondegree_in_T(false, node::Int) == 0
    )

    C′_0 = SubGraph(C′_0_u, C′_0_v)
    bind_membership!(C′_0, g)

    Subgraph.minus!(C′, g, C′_0)

    if BRANCH_TRACE
        println("  "^(depth + 1), "[update] branched-on=($is_u, $node)  maximum_nondegree=$maximum_nondegree",
                "  C′.U=", sorted_str(C′.U), " C′.V=", sorted_str(C′.V),
                "  C′_0.U=", sorted_str(C′_0.U), " C′_0.V=", sorted_str(C′_0.V))
    end

    return C′, C′_0, maximum_nondegree
end

