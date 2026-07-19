include("graph.jl")

const DEBUG = true
const OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES = true

function argmax_nodes(f, sg::SubGraph)
    best_score = -Inf
    best_is_u = false
    best_node = -1

    for u in sg.U
        score = f(true, u)
        if score > best_score
            best_score = score
            best_is_u = true
            best_node = u
        end
    end
    
    for v in sg.V
       score = f(false, v)
        if score > best_score
            best_score = score
            best_is_u = false
            best_node = v
        end
    end

    return best_is_u, best_node
end

# Returns the largest k-MDB of the graph
function branch_binary(S::SubGraph, C::SubGraph, g::FrozenBipartite, D::SubGraph, k::Int, θ::Int)
    @static if @isdefined(DEBUG)
        # @show S
        # @show C
        println("S=", length(S.U) + length(S.V), "C=", length(C.U) + length(C.V))
    end

    @static if @isdefined(OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES)
        max_u = max(length(D.U), θ)
        max_v = max(length(D.V), θ)
        if length(C.U) + length(S.U) < max_u || length(C.V) + length(S.V) < max_v
            @static if @isdefined(DEBUG)
                println("Pruning impossibly small branch")
            end

            return D
        end
    end
    if Subgraph.vertex_count(C) == 0
        @static if @isdefined(DEBUG)
            println("Reached end of candidate set")
        end

        if Subgraph.edge_count(g, S) > Subgraph.edge_count(g, D) && length(S.U) >= θ && length(S.V) >= θ
            # @static if @isdefined(DEBUG)
                println("Achieved a k-MDB")
                # @assert Subgraph.missing_edges(g, S) <= k "INVALID SOLUTION: d̄(S)=$(Subgraph.missing_edges(g, S)) > k=$k"
            # end
            return Subgraph.clone(S)
        end

        return D
    end

    is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, S), C)

    nondegree = is_u ? nondegree_in_subgraph_u(g, node, S) : nondegree_in_subgraph_v(g, node, S)
    if nondegree == 0
        _is_u, _node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, C), C)
        is_u = _is_u
        node = _node
    end

    C′, C′_0 = update(S, C, g, is_u, node, k)

    # FLAG: This is the reversed order from the way the authors
    # did it, because this way we don't have to make a complete copy
    # of the graph or remove a subgraph
    # If the order of branches turns out to matter, will have to flip this back

    # BranchB(S, C ∖ {u})
    Subgraph.remove_node!(C, is_u, node)
    D = branch_binary(S, C, g, D, k, θ)

    # S′ = S ∪ C′_0
    Subgraph.add!(S, C′_0)
    # S′ ∪ {u}
    Subgraph.add_node!(S, is_u, node)

    D = branch_binary(S, C′, g, D, k, θ)

    Subgraph.minus!(S, C′_0)
    Subgraph.remove_node!(S, is_u, node)

    return D
end

function branch_pivot(S::SubGraph, C::SubGraph, g::FrozenBipartite, D::SubGraph, k::Int, θ::Int)
    @static if @isdefined(DEBUG)
        @show S
        @show C
    end

    @static if @isdefined(OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES)
        max_u = max(length(D.U), θ)
        max_v = max(length(D.V), θ)
        if length(C.U) + length(S.U) < max_u || length(C.V) + length(S.V) < max_v
            @static if @isdefined(DEBUG)
                println("Pruning impossibly small branch")
            end

            return D
        end
    end

    # println("S=", length(S.U) + length(S.V), "C=", length(C.U)+length(C.V))

    if Subgraph.vertex_count(C) == 0
        @static if @isdefined(DEBUG)
            println("Reached end of candidate set")
        end

        if subgraph_edge_count(g, S) > subgraph_edge_count(g, D) && length(S.U) >= θ && length(S.V) >= θ
            @static if @isdefined(DEBUG)
                println("Achieved a k-MDB")
            end
            return Subgraph.clone(S)
        end

        return D
    end

    # BRAIN EXERCISE: why does changing this to nondegree in subgraph C not work?
    C_0_u = [u for u in C.U if nondegree_in_subgraph(g, true, u::Int, S) == 0]
    C_0_v = [v for v in C.V if nondegree_in_subgraph(g, false, v::Int, S) == 0]

    C_0 = SubGraph(Set(C_0_u), Set(C_0_v))

    if length(C_0_u) + length(C_0_v) == 0 || Subgraph.vertex_count(Subgraph.minus(C, C_0)) > k - Subgraph.missing_edges(g, S)
        C_U = [u for u in C.U]
        C_V = [v for v in C.V]
        nondegrees = [[nondegree_in_subgraph(g, true, u, S) for u in C_U]; [nondegree_in_subgraph(g, false, v, S) for v in C_V]]
        idx = argmax(nondegrees)
        is_u, node = _index_to_node(idx, C_U, C_V)

        C′, C′_0 = update(S, C, g, is_u, node, k)

        # S′ = S ∪ C′_0
        Subgraph.add!(S, C′_0)
        # S′ ∪ {u}
        Subgraph.add_node!(S, is_u, node)

        D = branch_pivot(S, C′, g, D, k, θ)

        # S = S′ ∖ C′_0 ∖ {u}
        Subgraph.minus!(S, C′_0)
        Subgraph.remove_node!(S, is_u, node)

        # C′ C ∖ {u}
        Subgraph.remove_node!(C, is_u, node)

        D = branch_pivot(S, C, g, D, k, θ)

        # C = C′ ∪ {u}
        Subgraph.add_node!(C, is_u, node)
    else
        C_U = [u for u in C.U]
        C_V = [v for v in C.V]
        nondegrees = [[nondegree_in_subgraph(g, true, u, C_0) for u in C_U]; [nondegree_in_subgraph(g, false, v, C_0) for v in C_V]]
        idx = argmin(nondegrees)
        is_u, node = _index_to_node(idx, C_U, C_V)


        if nondegree_in_subgraph(g, is_u, node, C) > k - Subgraph.missing_edges(g, S) > 0
            C′, C′_0 = update(S, C, g, is_u, node, k)

            # S′ = S ∪ C′_0 ∪ {u}
            Subgraph.add!(S, C′_0)
            Subgraph.add_node!(S, is_u, node)

            D = branch_pivot(S, C′, g, D, k, θ)

            # S = S′ ∖ C′_0 ∖ {u}
            Subgraph.minus!(S, C′_0)
            Subgraph.remove_node!(S, is_u, node)

            # C = C ∖ {u}
            Subgraph.remove_node!(C, is_u, node)
            D = branch_pivot(S, C, g, D, k, θ)
        else
            C′, C′_0 = update(S, C, g, is_u, node, k)

            # S′ = S ∪ C′_0 ∪ {u}
            Subgraph.add!(S, C′_0)
            Subgraph.add_node!(S, is_u, node)

            # Let L = search space
            # L = {u} ∪ nonneighbors_C(u)
            # u ∈ L
            D = branch_pivot(S, C′, g, D, k, θ)

            Subgraph.minus!(S, C′_0)
            Subgraph.remove_node!(S, is_u, node)

            # v = {u} ∪ nonneighbors_C(u)
            nonneighbors = Subgraph.nonneighbors_in_subgraph(g, is_u, node, C)
            for v in nonneighbors
                C′, C′_0 = update(S, C, g, !is_u, v, k)

                # S′ = S ∪ C′_0 ∪ {u}

                Subgraph.add!(S, C′_0)
                Subgraph.add_node!(S, !is_u, v)

                D = branch_pivot(S, C′, g, D, k, θ)

                Subgraph.minus!(S, C′_0)
                Subgraph.remove_node!(S, !is_u, v)

                Subgraph.remove_node!(C, !is_u, v)
            end
        end
    end

    return D
end

function update(S::SubGraph, C::SubGraph, g::FrozenBipartite, is_u::Bool, node::Int, k::Int)
    S_edges = Subgraph.edge_count(g, S)
    S_missing = length(S.U) * length(S.V) - S_edges

    # C′ = {v ∈ C ∖ {u} | nondegree_{ S ∪ {u} }(v) ≤ k - Ē(S)}
    Subgraph.remove_node!(C, is_u, node)
    Subgraph.add_node!(S, is_u, node)

    # maximum_nondegree = k - S_edges
    # println("S_e = $S_edges, S_m = $S_missing")
    maximum_nondegree = k - S_missing

    C′_u = Set(
        node for node in C.U if nondegree_in_subgraph(g, true, node::Int, S) ≤ maximum_nondegree
    )
    C′_v = Set(
        node for node in C.V if nondegree_in_subgraph(g, false, node::Int, S) ≤ maximum_nondegree
    )

    C′ = SubGraph(copy(C′_u), copy(C′_v))
    Subgraph.remove_node!(S, is_u, node::Int)
    Subgraph.add_node!(C, is_u, node::Int)

    # T = S ∪ {u} ∪ C′
    Subgraph.add!(C′, S)
    Subgraph.add_node!(C′, is_u, node::Int)

    # C′_0 = {v ∈ C′ | nondegree_{T}(v) = 0}
    C′_0_u = Set(
        node for node in C′_u if nondegree_in_subgraph(g, true, node::Int, C′) == 0
    )
    C′_0_v = Set(
        node for node in C′_v if nondegree_in_subgraph(g, false, node::Int, C′) == 0
    )

    Subgraph.minus!(C′, S)
    Subgraph.remove_node!(C′, is_u, node::Int)

    C′_0 = SubGraph(C′_0_u, C′_0_v)

    Subgraph.minus!(C′, C′_0)

    return C′, C′_0
end

function heuristic(g::FrozenBipartite, k::Int, θ::Int)
    U′ = Set()
    V′ = copy(g.V)

    while length(U′) < θ
        nondegs = []
    end
end