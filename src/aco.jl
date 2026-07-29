#= Ant colony optimization to find the maximum biclique =#

using Base.Threads

const __ACO_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")

const ELITE_PHEROMONE_FACTOR = 2

struct Pheromones
    U::Vector{Float64}
    V::Vector{Float64}
end

Pheromones(nU::Int, nV::Int) = Pheromones(ones(nU), ones(nV))
Pheromones(fg::FrozenBipartite) = Pheromones(length(fg.u_ids), length(fg.v_ids))

get_pheromone(pheromones::Pheromones, node::Node) = (node.is_u ? pheromones.U : pheromones.V)[node.id]
add_pheromone!(pheromones::Pheromones, node::Node, pheromone::Real) = ((node.is_u ? pheromones.U : pheromones.V)[node.id] += pheromone)

function evaporate_pheromones!(pheromones::Pheromones, evaporation::Float64)
    pheromones.U .*= evaporation
    pheromones.V .*= evaporation
end

mutable struct Ant
    explored::SubGraph
    last_visited::Node
end

function aco(g::BipartiteGraph, pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64, k::Int, θ::Int; parallelize::Bool=true)
    apply_graph_reductions!(g, k, θ, nothing, nothing, true, ReductionMode.all_reductions)
    
    fg = freeze(g)

    println(length(fg.u_ids), " ", length(fg.v_ids))

    compact_fg, remapping = compact_frozen(fg)
    pheromones = Pheromones(compact_fg)

    ants = [Ant(SubGraph(), Node()) for _ in 1:num_ants]
    invalid_ants = Set{Int}()

    best_score::Int = 0
    best_subgraph::SubGraph = SubGraph()

    explored_ants = [ant for ant in ants]

    for _ in 1:num_iterations
        # Track which ants are still exploring
        active_ants = collect(1:num_ants)

        while !isempty(active_ants)
            if parallelize
                # Divide the remaining ants evenly across available threads
                chunk_size = max(1, cld(length(active_ants), nthreads()))
                
                # 1. Spawn tasks for each chunk
                tasks = map(Iterators.partition(active_ants, chunk_size)) do chunk
                    Threads.@spawn advance_ants!(compact_fg, pheromones, pheromone, ants, k, chunk)
                end
                
                # 2. Wait for all threads to finish their step
                results = fetch.(tasks)
            else
                # Process all active ants sequentially on the main thread.
                # Wrapped in an array to match the structure of multithreaded `results`.
                results = [advance_ants!(compact_fg, pheromones, pheromone, ants, k, active_ants)]
            end
            
            # 3. Safely merge the results on the main thread
            for (local_additions, local_invalids) in results
                # Merge the local additions into the global pheromone tracker
                pheromones.U .+= local_additions.U
                pheromones.V .+= local_additions.V
                
                # Remove dead ants from the active pool for the next iteration
                setdiff!(active_ants, local_invalids)
            end
        end

        evaporate_pheromones!(pheromones, evaporation)

        # Softmax-select a few finished ants by instance fitness and reinforce
        # the last node each visited (elitist deposit after evaporation).
        eligible = [ant for ant in ants if ant.last_visited.id != -1]
        n_elite = min(3, length(eligible))
        if n_elite > 0
            elites = softmax_sample(
                ant -> instance_fitness(compact_fg, ant.explored, θ),
                eligible,
                n_elite,
            )
            for ant in elites
                add_pheromone!(pheromones, ant.last_visited, pheromone * ELITE_PHEROMONE_FACTOR)
            end
        end

        for ant in ants
            score = instance_fitness(compact_fg, ant.explored, missing)
            if score > best_score
                best_score = score
                best_subgraph = ant.explored
            end
        end

        invalid_ants = Set{Int}()
        ants = [Ant(SubGraph(), Node()) for _ in 1:num_ants]

        GC.gc()
    end

    return remap_subgraph(remapping, best_subgraph)
end

# Start and end index are inclusive
function advance_ants!(fg::FrozenBipartite, pheromones::Pheromones, pheromone::Int, ants::Vector{Ant}, k::Int, ant_chunk)
    # Thread-local storage to prevent data races
    additions = Pheromones(fg)
    local_invalid_ants = Int[]
    
    for idx in ant_chunk
        # advance_ant! mutates the ant and adds to local `additions`
        if !advance_ant!(fg, pheromones, additions, pheromone, ants[idx], k)
            push!(local_invalid_ants, idx)
        end
    end

    return additions, local_invalid_ants
end

function node_desirability(pheromones::Pheromones, node::DegreeNode)
    pheromone = get_pheromone(pheromones, Node(node.is_u, node.id))
    @assert pheremone > 0 "Pheremone cannot be zero. Otherwise, evaporation won't penalize underexplored nodes"
    return pheromone^3 * node.deg
end

# Returns false if the ant has no further moves
function advance_ant!(fg::FrozenBipartite, pheromones::Pheromones, additions::Pheromones, pheromone::Int, ant::Ant, k::Int)
    # Room for algorithmic improvement: have ants instead just start off of where
    # they are instead of looking across the entire graph

    candidates = candidate_set_with_nondegrees(fg, ant.explored, k)

    if length(candidates) == 0
        return false
    end

    next_with_deg = softmax_sample(
        node -> node_desirability(pheromones, node),
        candidates,
        1,
    )[1]

    next = Node(next_with_deg)

    Subgraph.add_node!(ant.explored, next.is_u, next.id)
    
    ant.last_visited = next

    add_pheromone!(additions, next, pheromone)
    return true
end
