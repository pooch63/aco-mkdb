function sa(fg::FrozenBipartite, S::SubGraph, k::Int,
    initial_T::Float64, cooling_factor::Float64,
    cooling_interval::Int, patience::Int)

    patience_counter::Int = 0
    iterations_since_cooling_update = 0

    state = S
    state_energy = instance_energy(fg, state)

    T = initial_T

    while patience_counter < patience
        neighbor = deepcopy(state)
        neighbor!(fg, neighbor, k)
        neighbor_energy = instance_energy(fg, neighbor)
        
        # Always switch if we got a better solution
        if neighbor_energy < state_energy
            state = neighbor
        # Still switch anyway at random
        else
            prob_select = exp(-(neighbor_energy - state_energy) / T)
            @show prob_select
            @show state_energy
            @show neighbor_energy
            if rand() < prob_select
                state = neighbor
                state_energy = neighbor_energy
            end
            patience_counter += 1
        end

        iterations_since_cooling_update += 1
        if iterations_since_cooling_update >= cooling_interval
            T *= cooling_factor
            iterations_since_cooling_update = 0
        end
    end

    return state
end

function neighbor!(fg::FrozenBipartite, S::SubGraph, k::Int)
    # Randomly switch between adding a node and removing a node
    add = rand(1, 2) == 1
    G = SubGraph(Set(u for u in fg.u_ids), Set(v for v in fg.v_ids))

    if add
        node = softmax_sample_nodes((u, n) -> degree_in_subgraph(fg, u, n, G), S, 1)[1]
        Subgraph.add_node!(S, node.is_u, node.id)
    else
        node = softmax_sample_nodes((u, n) -> nondegree_in_subgraph(fg, u, n, G), S, 1)[1]
        Subgraph.remove_node!(S, node.is_u, node.id)
    end

    greedily_add!(fg, S, k)
end