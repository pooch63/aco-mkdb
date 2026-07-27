include("graph.jl")
include("reduction.jl")

function arg_nodes(f, is_max::Bool, sg::SubGraph)
    best_score = is_max ? -Inf : Inf
    best_is_u = false
    best_node = -1

    for u in sg.U
        score = f(true, u)
        if (is_max && score > best_score) || (!is_max && score < best_score)
            best_score = score
            best_is_u = true
            best_node = u
        end
    end
    
    for v in sg.V
       score = f(false, v)
        if (is_max && score > best_score) || (!is_max && score < best_score)
            best_score = score
            best_is_u = false
            best_node = v
        end
    end

    best_node != -1 || error("arg_nodes called on empty candidate set")

    return best_is_u, best_node
end
argmax_nodes(f, sg::SubGraph) = arg_nodes(f, true, sg)
argmin_nodes(f, sg::SubGraph) = arg_nodes(f, false, sg)

@enum ReductionMode all_reductions simple progressive none

function apply_graph_reductions!(g::BipartiteGraph, k::Int, θ::Int,
    num_U::Union{Int, Nothing}, num_V::Union{Int, Nothing},
    use_heuristic::Bool, reduction::ReductionMode)

    num_U = num_U === nothing ? length(g.adjU) : num_U
    num_V = num_V === nothing ? length(g.adjV) : num_V
    
    if reduction == simple || reduction == progressive || reduction == all_reductions
        reduce_graph!(g, k, θ, num_U, num_V)
        fg = freeze(g)
    end
    if reduction == progressive || reduction == all_reductions
        reduce_graph!(g, k, θ, num_U, num_V)
        fg = freeze(g)

        u_degrees = Int[]
        for u in fg.u_ids
            push!(u_degrees, degree_u(fg, u))
        end

        isempty(u_degrees) && return fg

        θ_U = maximum(u_degrees) + k
        θ_V = 0

        D = use_heuristic ? initial_heuristic(fg, k, θ) : SubGraph(Set(), Set())

        while θ_U > θ
            θ_V = max(θ, floor(Int, Subgraph.edge_count(fg, D) / θ_U))
            θ_U = max(θ, floor(Int, θ_U / 2))

            reduce_graph!(g, k, min(θ_U, θ_V), num_U, num_V)
            fg = freeze(g)
        end
    end

    return fg
end

function initial_heuristic(g::FrozenBipartite, k::Int, θ::Int)
    sg = SubGraph(Set(), Set(g.v_ids))

    search = copy(g.u_ids)

    while length(sg.U) < θ && !isempty(search)
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
    end

    if length(sg.V) < θ
        return SubGraph(Set(), Set())
    end

    return sg
end

function upper_bound(S::SubGraph, C::SubGraph, g::FrozenBipartite, k::Int, S_missing::Int)
    S_edges = length(S.U) * length(S.V) - S_missing
    budget = k - S_missing
    # Note: may have to redo C's structure, because a collect call every time might have significant overhead
    u = sort(collect(C.U), by = u -> nondegree_in_subgraph(g, true, u, S))
    v = sort(collect(C.V), by = v -> nondegree_in_subgraph(g, false, v, S))

    nondegree_U, nondegree_V = 0, 0
    nodes_U, nodes_V = 0, 0

    for i in 1:length(u)
        nondegree_in_S = nondegree_in_subgraph(g, true, u[i], S)
        if nondegree_U + nondegree_in_S ≤ budget
            nondegree_U = nondegree_U + nondegree_in_S
            nodes_U += 1
        else
            break
        end
    end

    e = (length(S.U) + nodes_U) * length(S.V) - S_missing - nondegree_U
    i = nodes_U

    for j in 1:length(v)
        cost_V = nondegree_in_subgraph(g, false, v[j], S)

        if nondegree_V + cost_V > budget
            break
        end
        

        while i > 0 && nondegree_U + nondegree_V + cost_V > budget
            nondegree_U -= nondegree_in_subgraph(g, true, u[i], S)
            i -= 1
        end

        nondegree_V += cost_V
        nodes_V += 1
        e = max(
            e,
            (length(S.U) + i) * (length(S.V) + j) - S_missing - nondegree_U - nondegree_V
        )
    end

    return length(S.U) + nodes_U, length(S.V) + nodes_V, e
end

# If we add u to S, what does C become? What "free" nodes can we add to S that don't limit the search space?
function update(S::SubGraph, C::SubGraph, g::FrozenBipartite, is_u::Bool, node::Int, k::Int, S_missing::Int)
    # C′ = {v ∈ C ∖ {u} | nondegree_{ S ∪ {u} }(v) ≤ k - Ē(S)}

    new_S_missing = S_missing + nondegree_in_subgraph(g, is_u, node, S)

    Subgraph.remove_node!(C, is_u, node)
    Subgraph.add_node!(S, is_u, node)

    maximum_nondegree = k - new_S_missing

    C′_u = Set{Int}()
    sizehint!(C′_u, length(C.U))
    for node in C.U
        nondegree_in_subgraph(g, true, node::Int, S) ≤ maximum_nondegree && push!(C′_u, node)
    end

    C′_v = Set{Int}()
    sizehint!(C′_v, length(C.V))
    for node in C.V
        nondegree_in_subgraph(g, false, node::Int, S) ≤ maximum_nondegree && push!(C′_v, node)
    end

    C′ = SubGraph(C′_u, C′_v)
    Subgraph.remove_node!(S, is_u, node::Int)
    Subgraph.add_node!(C, is_u, node::Int)

    # T = S ∪ {u} ∪ C′
    nondegree_in_T(node_is_u, v) = nondegree_in_subgraph(g, node_is_u, v, S) + nondegree_in_subgraph(g, node_is_u, v, C′) + (node_is_u == !is_u && is_neighbor(g, node_is_u, v, node) ? 0 : 1)

    # C′_0 = {v ∈ C′ | nondegree_T(v) = 0}
    C′_0_u = Set(
        node for node in C′_u if nondegree_in_T(true, node::Int) == 0
    )
    C′_0_v = Set(
        node for node in C′_v if nondegree_in_T(false, node::Int) == 0
    )

    C′_0 = SubGraph(C′_0_u, C′_0_v)

    Subgraph.minus!(C′, C′_0)

    if TRACE
        println("    [update] branched-on=$node  maximum_nondegree=$maximum_nondegree",
                "  C′.U=", sorted_str(C′.U), " C′.V=", sorted_str(C′.V),
                "  C′_0.U=", sorted_str(C′_0.U), " C′_0.V=", sorted_str(C′_0.V))
    end

    return C′, C′_0, maximum_nondegree
end
