include("graph.jl")

function _index_to_node(index::Int, S::SubGraph)
    is_u = index <= length(S.U)
    node = is_u ? S.U[i] : S.V[i - length(S.U)]

    return is_u, node
end

# Returns the largest k-MDB of the graph
function branch_binary(S::SubGraph, C::SubGraph, g::FrozenBipartite, D::SubGraph, k::Int, θ::Int)
    if subgraph_vertex_count(C) == 0
        if subgraph_edge_count(g, S) > subgraph_edge_count(g, D) && subgraph_u_count(S) >= θ && subgraph_v_count(S) >= θ
            return S
        end
        
        return D
    end

    S_nondegrees = [[nondegree_in_subgraph_u(g, u, S) for u in S.U]; [nondegree_in_subgraph_v(g, v, S) for v in S.V]]
    idx = argmax(S_nondegrees)
    is_u, node = _index_to_node(idx, S)
    
    nondegree = is_u ? nondegree_in_subgraph_u(g, node, S) : nondegree_in_subgraph_v(g, node, S)
    if nondegree == 0
        C_nondegrees = [[nondegree_in_subgraph_u(g, u, C) for u in C.U]; [nondegree_in_subgraph_v(g, v, C) for v in C.V]]
        idx = argmax(C_nondegrees)
        _is_u, _node = _index_to_node(idx, C)
        _is_u = _is_u
        node = _node
    end

    C′, C′_0 = update(S, C, g, is_u, node)
    
    # FLAG: This is the reversed order from the way the authors
    # did it, because this way we don't have to make a complete copy
    # of the graph or remove a subgraph
    # If the order of branches turns out to matter, will have to flip this back
    
    # BranchB(S, C ∖ {u})
    remove_node!(C, is_u, node)
    D = branch_binary(S, C, g, D, k, θ)

    # S′ = S ∪ C′_0 ∪ {u}
    add_subgraph!(S′, C′_0)
    add_node!(S′, is_u, node)

    # C′ = C′ ∖ C′_0
    C′_minus_C′_0 = subgraph_minus!(C′, C′_0)
    D = branch_binary(S′, C′_minus_C′_0, g, D, k, θ)

    return D
end

function update(S::SubGraph, C::SubGraph, g::FrozenBipartite, is_u::Bool, node::Int)
    # C′ = {v ∈ C ∖ {u} | nondegree_{ S ∪ {u} }(v) ≤ k - E(S)}
    remove_node!(C, is_u, node)
    add_node!(S, is_u, node)
    
    S_edges = subgraph_edge_count(fg, sg)

    C′_u = Set(
        node for node in C.U if nondegree_in_subgraph(g, true, node::Int, S) ≤ k - S_edges
    )
    C′_v = Set(
        node for node in C.V if nondegree_in_subgraph(g, false, node::Int, S) ≤ k - S_edges
    )

    C′ = SubGraph(C′_u, C′_v)
    remove_node!(S, is_u, node)
    add_node!(C, is_u, node)

    # T = S ∪ {u} ∪ C′
    add_subgraph!(C′, S)
    add_node!(C′, is_u, node)

    # C′_0 = {v ∈ C′ | nondegree_{T}(v) = 0}
    C′_0_u = [
        node for node in C′_u if nondegree_in_subgraph(g, true, node::Int, S) == 0
    ]
    C′_0_v = [
        node for node in C′_v if nondegree_in_subgraph(g, true, node::Int, S) == 0
    ]

    C′_0 = SubGraph(C′_0_u, C′_0_v)
    
    return C′, C′_0
end