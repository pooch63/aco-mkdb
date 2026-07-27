const __TABU_JL__ = true

isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")

# For each dictionary, keys are the nodes that are added or removed
struct TabuList
    # Moves where we're adding nodes
    add::Dict{Node, Int}
    # Moves where we're removing nodes
    remove::Dict{Node, Int}
end

TabuList() = TabuList(Dict{Node, Int}(), Dict{Node, Int}())

function move_is_tabu(tabu::TabuList, node::Node, is_add::Bool)
    return is_add ? get(tabu.add, node, 0) > 0 : get(tabu.remove, node, 0) > 0
end
function add_tabu_move!(tabu::TabuList, node::Node, is_add::Bool, tt::Int)
    if is_add && tabu.add[node] == 0
        tabu.add[node] = tt
    elseif !is_add && tabu.remove[node] == 0
        tabu.remove[node] = tt
    end
end
function add_tabu_move!(tabu::TabuList, node::Node, is_add::Bool, tt::Int)
    target_dict = is_add ? tabu.add : tabu.remove
    
    if get(target_dict, node, 0) == 0
        target_dict[node] = tt
    end
    
    return tabu
end

# Modifies the instance in place to be the best instance
function tabu_repair!(fg::FrozenBipartite, instance::SubGraph, k::Int, θ::Int, tt::Int, patience::Int)
    tabu = TabuList()

    best_score = instance_fitness(fg, instance, θ)
    best_instance::SubGraph = deepcopy(instance)

    # println("Score of first best: $(best_score)")
    
    patience_counter = 0

    while patience_counter < patience
        candidate = candidate_set_as_node_array(fg, instance, k)

        # First bool is whether we add it, second bool is whether it's u, third is the node id
        best_move::Tuple{Bool, Bool, Int} = (false, false, -1)
        best_new_score::Int = -1

        function walk_node(node::Node, is_add::Bool)
            if is_add
                Subgraph.add_node!(instance, node.is_u, node.id)
            else
                Subgraph.remove_node!(instance, node.is_u, node.id)
            end

            score = instance_fitness(fg, instance, θ)
            is_tabu = move_is_tabu(tabu, node, is_add)

            if (!is_tabu && score > best_new_score) || score > max(best_new_score, best_score)
                best_new_score = score
                best_move = (is_add, node.is_u, node.id)
            end

            # If this move would make the graph worse, make it tabu
            if score < best_score
                add_tabu_move!(tabu, node, is_add, tt)
            end

            if is_add
                Subgraph.remove_node!(instance, node.is_u, node.id)
            else
                Subgraph.add_node!(instance, node.is_u, node.id)
            end
        end

        for node in candidate
            walk_node(node, true)
        end

        instance_nodes = [
            [Node(true, u) for u in instance.U];
            [Node(false, v) for v in instance.V] 
        ]

        for node in instance_nodes
            walk_node(node, false)
        end

        # Now that we have the best move, actually make it
        if best_move[1]
            Subgraph.add_node!(instance, best_move[2], best_move[3])
        else
            Subgraph.remove_node!(instance, best_move[2], best_move[3])
        end

        # If this was a better move than before, clone the instance
        if best_new_score > best_score
            best_score = best_new_score
            best_instance = deepcopy(instance)
        # Otherwise, increase our patience
        else
            patience_counter += 1
        end
    end

    # println("Score of after loop: $(best_score)")

    # Revert instance back to the best graph we found
    instance.U = best_instance.U
    instance.V = best_instance.V
end
