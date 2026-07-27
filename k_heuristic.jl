const __K_HEURISTIC_JL__ = true

isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")

neighbors_in_subgraph(fg::FrozenBipartite, u::Bool, n::Int, S::SubGraph) =
    u ? neighbors_u_in_subgraph(fg, n, S) : neighbors_v_in_subgraph(fg, n, S)
function neighbors_2_hop(fg::FrozenBipartite, u::Bool, n::Int, S::SubGraph)
    nodes::Set{Int} = Set()
    for neighbor in neighbors_in_subgraph(fg, u, n, S)
        for n2 in neighbors_in_subgraph(fg, !u, n, S)
            push!(nodes, n2)
        end
    end

    return length(nodes)
end

"""
Greedy deletion heuristic: start from the full bipartite graph and repeatedly
remove the node that contributes the most missing edges until `missing ≤ k`.

If the result fails `|U|≥θ` and `|V|≥θ` and `return_invalid` is false, return empty.
"""
function k_based_heuristic(fg::FrozenBipartite, k::Int, θ::Int; return_invalid::Bool=false)
    S = SubGraph(Set(fg.u_ids), Set(fg.v_ids))

    S_missing = Subgraph.missing_edges(fg, S)

    nondegrees = [
        [nondegree_in_subgraph(fg, true, u, S) for u in fg.u_ids];
        [nondegree_in_subgraph(fg, false, v, S) for v in fg.v_ids]
    ]

    S_missing = sum(nondegrees)

    t0 = time()
    iters = 0
    while S_missing > k && (!isempty(S.U) || !isempty(S.V))
        iters += 1
        if iters % 1000 == 0
            println("iter=$iters elapsed=$(round(time() - t0; digits=2))s missing=$S_missing |S|=$(length(S.U) + length(S.V))")
        end

        # Get the node with the highest nondegree. Break ties with the node that has the most 2nd degree neighbors.
        # Still trying to figure out _why_ this works
        is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(fg, u, n, S) + 0.01 * neighbors_2_hop(fg, u, n, S), S)
        Subgraph.remove_node!(S, is_u, node)

        println("Removed ($(is_u), $(node)), S_missing =$(S_missing)")

        S_missing = Subgraph.missing_edges(fg, S)
    end

    println("exited loop after $iters iters in $(round(time() - t0; digits=2))s")

    if !return_invalid && (length(S.U) < θ || length(S.V) < θ)
        return SubGraph(Set(), Set())
    end

    return S
end
