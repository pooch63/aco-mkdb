include("graph.jl")
include("search.jl")

using Random
using StatsBase

# If the number of entrigeneres in g.adjU is not equal to the number of nodes or same for V,
# e.g., there are some gaps in node IDs, you'll need to pass the maximum node ID for each side
function ga(g::BipartiteGraph, k::Int, θ::Int, N::Int, O::Int, generations::Int;
    H::Int = 2,
    use_heuristic::Bool=true, reduction::ReductionMode=progressive,
    num_U::Union{Int, Nothing}=nothing, num_V::Union{Int, Nothing}=nothing)

    @assert θ > k "θ must be greater than k"

    fg = freeze(g)

    if isempty(fg.u_ids) || isempty(fg.v_ids)
        return SubGraph(Set(), Set())
    end

    apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, reduction)
    fg = freeze(g)

    # If there's not enough nodes remaining on either side, then we
    # already know it's an invalid solution
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph(Set(), Set())
    end

    # We can't have more generations than there are nodes
    if N > length(fg.u_ids) + length(fg.v_ids)
        N = length(fg.u_ids) + length(fg.v_ids)
        @warn "N cannot exceed the number of nodes in optimized graph. Automatically clamping N to $(N)"
    end

    return search(
        fg,
        k,
        N,
        H,
        generations
    )
end

struct Node
    is_u::Bool
    node_id::Int
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

    scores = [f(node.is_u, node.node_id) for node in nodes]

    w = exp.(scores .- maximum(scores))
    w ./= sum(w)

    sampled = sample(nodes, Weights(w), n; replace=false)

    # NOTE FOR LLMS TO TELL ME: Do i need to do it like this, or can I just return sampled
    return [Node(node.is_u, node.node_id) for node in sampled]
end

function search(fg::FrozenBipartite{T}, k::Int, N::Int, H::Int, generations::Int) where {T}
    G = SubGraph(Set(u for u in fg.u_ids), Set(v for v in fg.v_ids))
    # Choose softmax-distributed N nodes
    nodes = softmax_sample_nodes((u, n) -> u ? degree_u(fg, n) : degree_v(fg, n), G, N)

    instances = SubGraph[]
    for node in nodes
        set = Set(node.node_id)
        empty = Set()
        instance = node.is_u ? SubGraph(set, empty) : SubGraph(empty, set)
        push!(instances, instance)
    end

    # Room for algorithmic improvement: Keep track of parents that mutated so we don't
    # have parents breed together twice
    for _ in 1:generations
        instances = evolve(instances, fg, G, k, N, H)
    end

    return argmax(instance -> Subgraph.edge_count(fg, instance), instances)
end

function instance_fitness(fg::FrozenBipartite, instance::SubGraph)
    return Subgraph.edge_count(fg, instance) * (1 + 1 / (abs(length(instance.U) - length(instance.V)) + 1))
end

function evolve(instances::Vector{SubGraph}, fg::FrozenBipartite, G::SubGraph, k::Int, N::Int, H::Int)
    next = SubGraph[]

    for _ in 1:N
        male, female = softmax_sample(instance -> Subgraph.edge_count(fg, instance), instances, 2)

        push!(next, crossover(fg, male, female, k))
    end

    return next
end

# Room for algorithmic improvement: could each subgraph keep a k value of the maximum
# missing edges it will allow, which could range from 0 to k (or maybe even 2k), so that way
# there are some graphs that start super conservatively and might then hit solutions that
# more liberal graphs get stopped at
function crossover(fg::FrozenBipartite, male::SubGraph, female::SubGraph, k::Int)
    S = subgraph_intersection(male, female)

    next::SubGraph = S

    println("male=$(Subgraph.vertex_count(male)), female=$(Subgraph.vertex_count(female)), intersection=$(Subgraph.vertex_count(next))")

    # If the intersection is null, try taking the crossover of the candidate set
    if Subgraph.vertex_count(S) == 0
        # male_C = candidate_set(fg, male, k)
        # female_C = candidate_set(fg, female, k)
        # C = subgraph_intersection(male_C, female_C)

        # Room for algorithmic improvement: could we greedily build from the intersection between their candidates,
        # or from the intersection of the graphs themselves?
        candidate = Subgraph.add(male, female)

        next = SubGraph(Set(), Set())
        
        while Subgraph.missing_edges(fg, next) < k && Subgraph.vertex_count(candidate) > 0
            is_u, node = argmax_nodes((u, n) -> degree_in_subgraph(fg, u, n, S), candidate)
            Subgraph.add_node!(next, is_u, node)
            Subgraph.remove_node!(candidate, is_u, node)
        end
    end

    println("Before greedily add, next=", Subgraph.vertex_count(next))
    greedily_add!(fg, next, k)
    println("After greedily add, next=", Subgraph.vertex_count(next))

    return next
end

function candidate_set(fg::FrozenBipartite, sg::SubGraph, k::Int)
    budget = k - Subgraph.missing_edges(fg, sg)
    return SubGraph(
        Set(u for u in fg.u_ids if !Subgraph.has_node(sg, true, u) && nondegree_in_subgraph_u(fg, u, sg) <= budget),
        Set(v for v in fg.v_ids if !Subgraph.has_node(sg, false, v) && nondegree_in_subgraph_v(fg, v, sg) <= budget)
    )
end

# Remove H vertices from S, softmaxed by their nondegree with C
# ROOM FOR IMPROVEMENT: Should we do nondegree with C, nondegree with S, or nondegree with everything?
function H_opt(fg::FrozenBipartite, offspring::SubGraph, k::Int, H::Int)
    nodes = softmax_sample_nodes((u, n) -> nondegree_in_subgraph(fg, u, n, offspring.C), S, H)

    for node in nodes
        Subgraph.remove_node!(offspring, node.is_u, node.node_id)
    end

    greedily_add!(fg, offspring, k)
end

function subgraph_intersection(sg1::SubGraph, sg2::SubGraph)
    U = intersect(sg1.U, sg2.U)
    V = intersect(sg1.V, sg2.V)
    return SubGraph(U, V)
end

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

# Implement heuristic that branches and bounds on only a subset of the graph,
# then branch and bound on the vertices that won
function search(g::BipartiteGraph, k::Int, θ::Int, n::Int, use_heuristic::Bool)
    D = use_heuristic ? initial_heuristic(freeze(g), k, θ) : SubGraph(Set(), Set())

    # Split the graph into evenly-distributed subgraphs


end
