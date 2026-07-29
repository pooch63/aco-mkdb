const __REDUCTION_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")

struct DegreeNode
    is_u::Bool
    id::Int
    deg::Int
end

function get_ascending_degree_order(g::BipartiteGraph)
    nodes = Vector{DegreeNode}()
    sizehint!(nodes, length(g.adjU) + length(g.adjV))

    for u in keys(g.adjU)
        deg = degree_u(g, u)
        push!(nodes, DegreeNode(true, u, deg))
    end

    for v in keys(g.adjV)
        deg = degree_v(g, v)
        push!(nodes, DegreeNode(false, v, deg))
    end

    sort!(nodes, by = node -> node.deg)

    return nodes
end

function reduce_graph!(g::BipartiteGraph{T}, k::Int, θ::Int, num_U::Int, num_V::Int) where {T}
    common_neighbor_reduction!(g, k, θ, num_U, num_V)
end

# Note: REQUIRES node id's to start at 1, otherwise it will crash
function common_neighbor_reduction!(g::BipartiteGraph{T}, k::Int, θ::Int, num_U::Int, num_V::Int) where {T}
    O = get_ascending_degree_order(g)

    # Stamp arrays: per-node counts without O(n) fill! between nodes.
    # stamp[w] == cur marks that count[w] is live for this node.
    stamp_U = zeros(Int, num_U)
    stamp_V = zeros(Int, num_V)
    count_U = zeros(Int, num_U)
    count_V = zeros(Int, num_V)
    cur_stamp = 0

    # Reusable neighbor snapshots. Needed because rem_edge_structural! mutates
    # adjacency Sets while we still need the original lists; buffers avoid the
    # per-neighbor `collect` that dominated allocations.
    neighbors_buf = Int[]
    neighbors_2hop_bufs = Vector{Vector{Int}}()

    for node in O
        # It may have already been removed, in which case, just continue
        if !node_exists(g, node.is_u, node.id)
            continue
        end

        nbrs_set = get_neighbors(g, node.is_u, node.id)
        empty!(neighbors_buf)
        sizehint!(neighbors_buf, length(nbrs_set))
        for v in nbrs_set
            push!(neighbors_buf, v)
        end

        threshold = θ - k

        cur_stamp += 1
        stamp = node.is_u ? stamp_U : stamp_V
        count = node.is_u ? count_U : count_V

        n_nbrs = length(neighbors_buf)
        while length(neighbors_2hop_bufs) < n_nbrs
            push!(neighbors_2hop_bufs, Int[])
        end
        for i in 1:n_nbrs
            buf = neighbors_2hop_bufs[i]
            empty!(buf)
            for w in get_neighbors(g, !node.is_u, neighbors_buf[i])
                push!(buf, w)
            end
        end

        for i in 1:n_nbrs
            for w in neighbors_2hop_bufs[i]
                if stamp[w] != cur_stamp
                    stamp[w] = cur_stamp
                    count[w] = 1
                else
                    count[w] += 1
                end
            end
        end

        for i in 1:n_nbrs
            v = neighbors_buf[i]
            valid_connections = 0
            for w in neighbors_2hop_bufs[i]
                cn = stamp[w] == cur_stamp ? count[w] : 0
                if cn >= threshold
                    valid_connections += 1
                end
            end

            if valid_connections < threshold
                if node.is_u
                    rem_edge_structural!(g, node.id, v)
                else
                    rem_edge_structural!(g, v, node.id)
                end
            end
        end

        degree = get_degree(g, node.is_u, node.id)
        if degree < threshold
            rem_node_structural!(g, node.is_u, node.id)
            # println("Removed U=$(node.id)")
        end

        for v in neighbors_buf
            if get_degree(g, !node.is_u, v) < threshold
                rem_node_structural!(g, !node.is_u, v)
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
