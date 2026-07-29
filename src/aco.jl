#= Ant colony optimization to find the maximum biclique =#

const __ACO_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")

struct Pheromones
    U::Vector{Float64}
    V::Vector{Float64}
end

Pheromones(nU::Int, nV::Int) = Pheromones(zeros(nU), zeros(nV))
Pheromones(fg::FrozenBipartite) = Pheromones(length(fg.u_ids), length(fg.v_ids))

get_pheromone(pheromones::Pheromones, node::Node) = (node.is_u ? pheromones.U : pheromones.V)[node.id]
add_pheromone!(pheromones::Pheromones, node::Node, pheromone::Real) = ((node.is_u ? pheromones.U : pheromones.V)[node.id] += pheromone)

function evaporate_pheromones!(pheromones::Pheromones, evaporation::Float64)
    pheromones.U .*= evaporation
    pheromones.V .*= evaporation
end

struct Ant
    explored::SubGraph
end

function aco(fg::FrozenBipartite, pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64, k::Int; reduction::ReductionMode.T=ReductionMode.all_reductions)
    println(length(fg.u_ids), " ", length(fg.v_ids))

    compact_fg, remapping = compact_frozen(fg)
    pheromones = Pheromones(compact_fg)

    ants = [Ant(SubGraph()) for _ in 1:num_ants]
    invalid_ants = Set{Int}()

    best_score::Int = 0
    best_subgraph::SubGraph = SubGraph()

    for _ in 1:num_iterations
        while length(invalid_ants) < num_ants
            for (idx, ant) in enumerate(ants)
                if idx ∉ invalid_ants && !advance_ant!(compact_fg, pheromones, pheromone, ant, k)
                    push!(invalid_ants, idx)
                end
            end
        end

        evaporate_pheromones!(pheromones, evaporation)

        for ant in ants
            score = instance_fitness(compact_fg, ant.explored, missing)
            if score > best_score
                best_score = score
                best_subgraph = ant.explored
            end
        end

        invalid_ants = Set{Int}()
        ants = [Ant(SubGraph()) for _ in 1:num_ants]

        GC.gc()
    end

    return remap_subgraph(remapping, best_subgraph)
end

function node_desirability(fg::FrozenBipartite, sg::SubGraph, pheromones::Pheromones, node::Node)
    return max(1, get_pheromone(pheromones, node) ^ 3) * degree_in_subgraph(fg, node.is_u, node.id, sg)
end

# Returns false if the ant has no further moves
function advance_ant!(fg::FrozenBipartite, pheromones::Pheromones, pheromone::Int, ant::Ant, k::Int)
    # Room for algorithmic improvement: have ants instead just start off of where
    # they are instead of looking across the entire graph
    candidates = candidate_set(fg, ant.explored, k)
    
    if Subgraph.vertex_count(candidates) == 0
        return false
    end

    next = softmax_sample_nodes(
        (u, n) -> node_desirability(fg, ant.explored, pheromones, Node(u, n)),
        candidates,
        1,
    )[1]
    Subgraph.add_node!(ant.explored, next.is_u, next.id)

    add_pheromone!(pheromones, next, pheromone)
    return true
end

function aco(g::BipartiteGraph, k::Int, θ::Int, pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64;
    use_heuristic::Bool=true, reduction::ReductionMode.T=ReductionMode.all_reductions,
    num_U::Union{Int,Nothing}=nothing, num_V::Union{Int,Nothing}=nothing)

    @assert θ > k "θ must be greater than k"

    fg = apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, reduction)

    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph(Set(), Set())
    end

    return aco(fg, pheromone, num_ants, num_iterations, evaporation, k)
end
