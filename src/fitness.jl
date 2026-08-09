using StatsBase

const __FITNESS_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")

struct Node
    is_u::Bool
    id::Int
end

# Null node
Node() = Node(false, -1)

struct Instance
    subgraph::SubGraph
    k::Int
end

function softmax_sample(objects::Vector{T}, scores::Vector{S}, n::Int) where {T, S<:Real}
    # softmax with max-subtraction for numerical stability
    w = exp.(scores .- maximum(scores))
    w ./= sum(w)

    # If there are a few objects whose scores are orders of magnitude higher than the rest,
    # then everything else might go to 0. In that instance, return out the objects that are over 0,
    # then softmax on the remaining objects for a total of n - sum(obj .> 0)
    if count(>(0), w) < n
        zero_objects = T[]
        zero_scores = S[]
        for (index, score) in enumerate(scores)
            if w[index] == 0.0
                push!(zero_objects, objects[index])
                push!(zero_scores, score)
            end
        end
        positives = [object for (index, object) in enumerate(objects) if w[index] > 0.0]

        @assert length(positives) > 0 "The exp of all scores was 0"

        append!(positives, softmax_sample(zero_objects, zero_scores, n - length(positives)))
        return positives
    end


    return sample(objects, Weights(w), n; replace=false)
end
function softmax_sample(f::Function, objects::Vector{T}, n::Int) where T
    return softmax_sample(objects, [f(object) for object in objects], n)
end

function softmax_sample_nodes(f, sg::SubGraph, n::Int)
    nodes = Vector{Node}()
    sizehint!(nodes, length(sg.U) + length(sg.V))

    for u in sg.U
        push!(nodes, Node(true, u))
    end
    for v in sg.V
        push!(nodes, Node(false, v))
    end

    isempty(nodes) && error("softmax_sample_nodes called on empty candidate set")
    n <= length(nodes) || error("requested $n samples but only $(length(nodes)) candidates available")

    scores = [f(node.is_u, node.id) for node in nodes]

    w = exp.(scores .- maximum(scores))
    w ./= sum(w)

    sampled = sample(nodes, Weights(w), n; replace=false)

    # NOTE FOR LLMS TO TELL ME: Do i need to do it like this, or can I just return sampled
    return [Node(node.is_u, node.id) for node in sampled]
end

function instance_fitness(fg::FrozenBipartite, instance::SubGraph, θ::Union{Int, Missing})
    min_nodes, max_nodes = minmax(length(instance.U), length(instance.V))

    # Makes ACO _terrible_
    # return min_nodes * max_nodes

    return (min_nodes >= coalesce(θ, 1) ? max_nodes : 1) *
        # Give a higher score to subgraphs with more nodes
        min_nodes ^ 2
        # Linearly penalize a disparity in the graph
        # Room for improvement: if both sides have at least θ nodes, then we don't penalize at all?
        # *(1 + 1 / (abs(length(instance.U) - length(instance.V)) + 1))
end
@inline instance_energy(fg::FrozenBipartite, instance::SubGraph, θ::Int) = -instance_fitness(fg, instance, θ)

# Room for efficiency improvement: we can just compute C once and then order it by descending degree in S
# Then just add nodes from C with a moving pointer until we've hit k
function greedily_add!(fg::FrozenBipartite, S::SubGraph, k::Int)
    # Room for algorithm improvement: could we randomly choose a maximum missing
    # edges variable that could be greater or less than k?
    ensure_membership!(S, fg)
    while Subgraph.missing_edges(fg, S) < k
        C = candidate_set(fg, S, k)

        if Subgraph.vertex_count(C) == 0
            return
        end

        is_u, node = argmax_nodes((u, n) -> degree_in_subgraph(fg, u, n, S), C)

        Subgraph.add_node!(S, fg, is_u, node)
    end
end

function candidate_set(fg::FrozenBipartite, sg::SubGraph, k::Int)
    budget = k - Subgraph.missing_edges(fg, sg)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    hits_u = zeros(Int, nU)
    hits_v = zeros(Int, nV)
    degrees_into_subgraph_u!(hits_u, fg, sg)
    degrees_into_subgraph_v!(hits_v, fg, sg)

    nV_sg = length(sg.V)
    nU_sg = length(sg.U)
    U = Set{Int}()
    V = Set{Int}()
    @inbounds for ui in 1:nU
        u = fg.u_ids[ui]
        Subgraph.has_node(sg, true, u) && continue
        (nV_sg - hits_u[ui]) <= budget && push!(U, u)
    end
    @inbounds for vi in 1:nV
        v = fg.v_ids[vi]
        Subgraph.has_node(sg, false, v) && continue
        (nU_sg - hits_v[vi]) <= budget && push!(V, v)
    end
    return SubGraph(U, V)
end

# When you want the nondegrees
function candidate_set_with_nondegrees(fg::FrozenBipartite, sg::SubGraph, k::Int)
    budget = k - Subgraph.missing_edges(fg, sg)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    hits_u = zeros(Int, nU)
    hits_v = zeros(Int, nV)
    degrees_into_subgraph_u!(hits_u, fg, sg)
    degrees_into_subgraph_v!(hits_v, fg, sg)

    nV_sg = length(sg.V)
    nU_sg = length(sg.U)
    nodes = DegreeNode[]
    sizehint!(nodes, (nU + nV) - Subgraph.vertex_count(sg))

    @inbounds for ui in 1:nU
        u = fg.u_ids[ui]
        if Subgraph.has_node(sg, true, u)
            continue
        end
        deg = hits_u[ui]
        nondegree = nV_sg - deg
        if nondegree <= budget
            push!(nodes, DegreeNode(true, u, deg))
        end
    end
    @inbounds for vi in 1:nV
        v = fg.v_ids[vi]
        if Subgraph.has_node(sg, false, v)
            continue
        end
        deg = hits_v[vi]
        nondegree = nU_sg - deg
        if nondegree <= budget
            push!(nodes, DegreeNode(false, v, deg))
        end
    end

    return nodes
end

function candidate_set_as_node_array(fg::FrozenBipartite, sg::SubGraph, k::Int)
    budget = k - Subgraph.missing_edges(fg, sg)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    hits_u = zeros(Int, nU)
    hits_v = zeros(Int, nV)
    degrees_into_subgraph_u!(hits_u, fg, sg)
    degrees_into_subgraph_v!(hits_v, fg, sg)

    nV_sg = length(sg.V)
    nU_sg = length(sg.U)
    nodes = Node[]
    sizehint!(nodes, (nU + nV) - Subgraph.vertex_count(sg))

    @inbounds for ui in 1:nU
        u = fg.u_ids[ui]
        Subgraph.has_node(sg, true, u) && continue
        (nV_sg - hits_u[ui]) <= budget && push!(nodes, Node(true, u))
    end
    @inbounds for vi in 1:nV
        v = fg.v_ids[vi]
        Subgraph.has_node(sg, false, v) && continue
        (nU_sg - hits_v[vi]) <= budget && push!(nodes, Node(false, v))
    end
    return nodes
end

