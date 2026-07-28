const __FITNESS_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")

struct Node
    is_u::Bool
    id::Int
end

struct Instance
    subgraph::SubGraph
    k::Int
end

function softmax_sample(objects::Vector{T}, scores::Vector{<:Real}, n::Int) where T
    # softmax with max-subtraction for numerical stability
    w = exp.(scores .- maximum(scores))
    w ./= sum(w)
    return sample(objects, Weights(w), n; replace=false)
end
function softmax_sample(f, objects::Vector{T}, n::Int) where T
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

    # return min_nodes + max_nodes

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
    while Subgraph.missing_edges(fg, S) < k
        C = candidate_set(fg, S, k)

        if Subgraph.vertex_count(C) == 0
            return
        end

        is_u, node = argmax_nodes((u, n) -> degree_in_subgraph(fg, u, n, S), C)

        Subgraph.add_node!(S, is_u, node)
    end
end

function candidate_set(fg::FrozenBipartite, sg::SubGraph, k::Int)
    budget = k - Subgraph.missing_edges(fg, sg)
    return SubGraph(
        Set(u for u in fg.u_ids if !Subgraph.has_node(sg, true, u) && nondegree_in_subgraph_u(fg, u, sg) <= budget),
        Set(v for v in fg.v_ids if !Subgraph.has_node(sg, false, v) && nondegree_in_subgraph_v(fg, v, sg) <= budget)
    )
end

function candidate_set_as_node_array(fg::FrozenBipartite, sg::SubGraph, k::Int)
    budget = k - Subgraph.missing_edges(fg, sg)
    return [
        [Node(true, u) for u in fg.u_ids if !Subgraph.has_node(sg, true, u) && nondegree_in_subgraph_u(fg, u, sg) <= budget];
        [Node(false, v) for v in fg.v_ids if !Subgraph.has_node(sg, false, v) && nondegree_in_subgraph_v(fg, v, sg) <= budget]
    ]
end

