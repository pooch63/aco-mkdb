include("graph.jl")

# Returns the largest k-MDB of the graph
function branch_binary(S::SubGraph, C::SubGraph, g::BipartiteGraph, D::SubGraph, θ::Int, k::Int)
    if subgraph_vertex_count(C) == 0
        if subgraph_edge_counts(S) > subgraph_edge_counts(D) && subgraph_u_count(S) >= θ && subgraph_v_count(S) >= θ
            return S
        end
        
        return D
    end

    nondegrees = [nondegree_in_subgraph_u(g, u, S)]
end