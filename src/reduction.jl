const __REDUCTION_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
# Node lives in fitness.jl; must exist before we attach Node(::DegreeNode),
# otherwise this creates a free function that struct Node later replaces.
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__UTILS_JL__) || include("utils.jl")

struct DegreeNode
    is_u::Bool
    id::Int
    deg::Int
end

Node(deg_node::DegreeNode) = Node(deg_node.is_u, deg_node.id)

function get_degree_order(g::BipartiteGraph, desc::Bool)
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

    if desc
        sort!(nodes, by = node -> node.deg)
    else
        sort!(nodes, by = node -> -node.deg)
    end

    return nodes
end
function get_degree_order(g::BipartiteGraph, is_u::Bool, desc::Bool)
    return parallel_sort_by_keys(is_u ? g.u_ids : g.v_ids, (n) -> is_u ? degree_u(g, n) : degree_v(g, n), desc)
end

function reduce_graph!(g::BipartiteGraph{T}, k::Int, θ::Int, num_U::Int, num_V::Int) where {T}
    common_neighbor_reduction!(g, k, θ, num_U, num_V)
end

@inline function _swap_delete!(a::Vector{Int}, x::Int)
    i = findfirst(==(x), a)
    i === nothing && return false
    a[i] = a[end]
    pop!(a)
    return true
end

# Note: REQUIRES node id's to start at 1, otherwise it will crash
function common_neighbor_reduction!(g::BipartiteGraph{T}, k::Int, θ::Int, num_U::Int, num_V::Int) where {T}
    threshold = θ - k

    # Local Vector adjacency for the reduction pass. Ids are 1..num_U / 1..num_V.
    # Iteration is contiguous (vs Set hash probes); sync back to g.adj* at the end.
    adjU = Vector{Vector{Int}}(undef, num_U)
    adjV = Vector{Vector{Int}}(undef, num_V)
    aliveU = falses(num_U)
    aliveV = falses(num_V)

    for (u, nbrs) in g.adjU
        aliveU[u] = true
        adjU[u] = collect(nbrs)
    end
    for (v, nbrs) in g.adjV
        aliveV[v] = true
        adjV[v] = collect(nbrs)
    end

    stamp_U = zeros(Int, num_U)
    stamp_V = zeros(Int, num_V)
    count_U = zeros(Int, num_U)
    count_V = zeros(Int, num_V)
    cur_stamp = 0

    # Snapshot of the current node's neighbors (rem_edge mutates that adjacency list).
    neighbors_buf = Int[]

    O = DegreeNode[]
    sizehint!(O, count(aliveU) + count(aliveV))
    for u in 1:num_U
        aliveU[u] && push!(O, DegreeNode(true, u, length(adjU[u])))
    end
    for v in 1:num_V
        aliveV[v] && push!(O, DegreeNode(false, v, length(adjV[v])))
    end
    sort!(O, by = node -> node.deg)

    @inline function rem_edge!(u::Int, v::Int)
        _swap_delete!(adjU[u], v)
        _swap_delete!(adjV[v], u)
        return nothing
    end

    @inline function rem_node!(is_u::Bool, id::Int)
        if is_u
            aliveU[id] || return
            for v in adjU[id]
                _swap_delete!(adjV[v], id)
            end
            aliveU[id] = false
        else
            aliveV[id] || return
            for u in adjV[id]
                _swap_delete!(adjU[u], id)
            end
            aliveV[id] = false
        end
        return nothing
    end

    for node in O
        if node.is_u
            aliveU[node.id] || continue
            nbrs = adjU[node.id]
        else
            aliveV[node.id] || continue
            nbrs = adjV[node.id]
        end

        empty!(neighbors_buf)
        append!(neighbors_buf, nbrs)

        # deg < threshold ⟹ node cannot participate; skip the 2-hop walk
        if length(neighbors_buf) < threshold
            rem_node!(node.is_u, node.id)
            for x in neighbors_buf
                if node.is_u
                    aliveV[x] && length(adjV[x]) < threshold && rem_node!(false, x)
                else
                    aliveU[x] && length(adjU[x]) < threshold && rem_node!(true, x)
                end
            end
            continue
        end

        cur_stamp += 1
        stamp = node.is_u ? stamp_U : stamp_V
        count = node.is_u ? count_U : count_V

        # Pass 1: count 2-hop occurrences (iterate Vectors in place; no mutation yet)
        for v in neighbors_buf
            nbrs2 = node.is_u ? adjV[v] : adjU[v]
            for w in nbrs2
                if stamp[w] != cur_stamp
                    stamp[w] = cur_stamp
                    count[w] = 1
                else
                    count[w] += 1
                end
            end
        end

        # Pass 2: validate edges. rem_edge runs only after each v's scan finishes,
        # so we never mutate the 2-hop list under iteration.
        for v in neighbors_buf
            valid_connections = 0
            nbrs2 = node.is_u ? adjV[v] : adjU[v]
            for w in nbrs2
                cn = stamp[w] == cur_stamp ? count[w] : 0
                if cn >= threshold
                    valid_connections += 1
                end
            end

            if valid_connections < threshold
                if node.is_u
                    rem_edge!(node.id, v)
                else
                    rem_edge!(v, node.id)
                end
            end
        end

        deg = node.is_u ? length(adjU[node.id]) : length(adjV[node.id])
        if deg < threshold
            rem_node!(node.is_u, node.id)
        end

        for v in neighbors_buf
            if node.is_u
                aliveV[v] && length(adjV[v]) < threshold && rem_node!(false, v)
            else
                aliveU[v] && length(adjU[v]) < threshold && rem_node!(true, v)
            end
        end
    end

    # Sync surviving adjacency back into the mutable graph (structural; edge_data untouched).
    empty!(g.adjU)
    empty!(g.adjV)
    for u in 1:num_U
        aliveU[u] || continue
        g.adjU[u] = Set(adjU[u])
    end
    for v in 1:num_V
        aliveV[v] || continue
        g.adjV[v] = Set(adjV[v])
    end
end
