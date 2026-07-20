include("graph.jl")

struct Node
    is_u::Bool
    id::Int
    deg::Int
end

function get_ascending_degree_order(g::BipartiteGraph)
    nodes = Vector{Node}()
    sizehint!(nodes, length(g.adjU) + length(g.adjV))

    for u in keys(g.adjU)
        deg = degree_u(g, u)
        push!(nodes, Node(true, u, deg))
    end

    for v in keys(g.adjV)
        deg = degree_v(g, v)
        push!(nodes, Node(false, v, deg))
    end

    sort!(nodes, by = node -> node.deg)

    return nodes
end

function reduce_graph!(g::BipartiteGraph{T}, k::Int, θ::Int) where {T}
    common_neighbor_reduction!(g, k, θ)
end

# Note: REQUIRES node id's to start at 1, otherwise it will crash
function common_neighbor_reduction!(g::BipartiteGraph{T}, k::Int, θ::Int) where {T}
    O = get_ascending_degree_order(g)

    common_neighbors_U = zeros(Int, length(g.adjU))
    common_neighbors_V = zeros(Int, length(g.adjV))

    for node in O
        # It may have already been removed, in which case, just continue
        if !node_exists(g, node.is_u, node.id)
            continue
        end

        neighbors = get_neighbors(g, node.is_u, node.id)

        for v in neighbors
            neighbors_2_hop = get_neighbors(g, !node.is_u, v)
            for w in neighbors_2_hop
                if node.is_u
                    common_neighbors_U[w] += 1
                else
                    common_neighbors_V[w] += 1
                end
            end
        end

        for v in neighbors
            neighbors_2_hop = get_neighbors(g, !node.is_u, v)

            valid_connections = 0
            for w in neighbors_2_hop
                cn = node.is_u ? common_neighbors_U[w] : common_neighbors_V[w]
                if cn >= θ - k
                    valid_connections += 1
                end
            end

            if valid_connections < θ - k
                if node.is_u
                    rem_edge!(g, node.id, v)
                else
                    rem_edge!(g, v, node.id)
                end
            end
        end

        degree = get_degree(g, node.is_u, node.id)
        if degree < θ - k
            rem_node!(g, node.is_u, node.id)
            # println("Removed U=$(node.id)")
        end

        for v in neighbors
            if get_degree(g, !node.is_u, v) < θ - k
                rem_node!(g, !node.is_u, v)
                # println("Removed V=$(v)")
            end
        end
    end
end

function one_non_neighbor_reduction!(g::BipartiteGraph, k::Int, θ::Int)
end

function ordering_reduction!(g::BipartiteGraph, k::Int, θ::Int)
end

function progressive_bounding_reduction!(g::BipartiteGraph, k::Int, θ::Int)
end