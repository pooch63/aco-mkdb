const __THETA_HEURISTIC_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")

"""
Greedy θ×· construction: grow U to θ, trimming V to stay within k missing edges.

If the result fails `|U|≥θ` and `|V|≥θ` and `return_invalid` is false, return empty.
"""
function theta_based_heuristic(g::FrozenBipartite, k::Int, θ::Int;
    incremental::Bool=false, return_invalid::Bool=false)
    sg = SubGraph(Set(), Set(g.v_ids))

    search = copy(g.u_ids)
    best_score::Int = 0
    best_instance::SubGraph = sg

    while (incremental || length(sg.U) < θ) && !isempty(search)
        u = argmax(u -> degree_in_subgraph_u(g, u, sg), search)
        new_missing = nondegree_in_subgraph(g, true, u, sg)
        
        Subgraph.add_node!(sg, true, u)
        deleteat!(search, findfirst(==(u), search))

        sg_missing = Subgraph.missing_edges(g, sg)
        
        if sg_missing > k
            Vs = collect(sg.V)
            nondegrees = [nondegree_in_subgraph(g, false, v, sg) for v in Vs]
            idxs = sortperm(nondegrees, rev=true)

            missing_edges_to_remove = sg_missing - k
            num_nodes_removed = 0

            while missing_edges_to_remove > 0
                num_nodes_removed += 1
                idx = idxs[num_nodes_removed]
                missing_edges_to_remove -= nondegrees[idx]
                delete!(sg.V, Vs[idx])
            end
        end

        if incremental
            score = instance_fitness(g, sg, missing)
            if score > best_score
                best_score = score
                best_instance = deepcopy(sg)
            end
        end
    end

    if !return_invalid && (length(sg.U) < θ || length(sg.V) < θ)
        return SubGraph(Set(), Set())
    end

    return incremental ? best_instance : sg
end
