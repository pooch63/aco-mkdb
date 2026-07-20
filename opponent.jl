include("graph.jl")

const TRACE = false
const OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES = true

# When enabled, throw away any k-MDB that will have fewer U
# or fewer V nodes than the current best k-MBD. If this setting
# is disabled, any k-MDB found that has more edges than the current k-MDB
# will be used
# BE CAREFUL, because if the PRIORITIZE_VERTEX_COUNT setting is not the same as is used in
# the test suite, you may get mismatches
const PRIORITIZE_VERTEX_COUNT = true

sorted_str(s::Set{Int}) = "{" * join(sort(collect(s)), ",") * "}"

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

# BranchB assumes that S ALWAYS has k or fewer missing edges. In other words, S is at all
# times a valid k-MDB. C is the set of all nodes that we still have to search, where each node could be added to S
# and still result in a k-MDB, although we don't necessarily know which subset of C could be added to S.
function branch_binary(S::SubGraph, C::SubGraph, g::FrozenBipartite,
    D::SubGraph, k::Int, θ::Int, S_missing::Int=0, depth::Int=0)
    if TRACE
        me = S_missing
        println("  "^depth, "depth=$depth  S.U=", sorted_str(S.U), " S.V=", sorted_str(S.V),
                "  missing(S)=$me/$k  C.U=", sorted_str(C.U), " C.V=", sorted_str(C.V))
        if me > k
            println("  "^depth, "!!! INVARIANT VIOLATED: missing_edges(S)=$me > k=$k  <-- bug is here or in the parent frame's update() call")
            error("stopping so you can inspect the call stack")
        end
        if !isempty(intersect(S.U, C.U)) || !isempty(intersect(S.V, C.V))
            println("  "^depth, "!!! S and C OVERLAP: S.U∩C.U=", intersect(S.U, C.U), " S.V∩C.V=", intersect(S.V, C.V))
            error("S and C should always be disjoint")
        end
    end

    if OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES
        min_u = PRIORITIZE_VERTEX_COUNT ? max(length(D.U), θ) : θ
        min_v = PRIORITIZE_VERTEX_COUNT ? max(length(D.V), θ) : θ
        if length(C.U) + length(S.U) < min_u || length(C.V) + length(S.V) < min_v
            TRACE && println("  "^depth, "-> pruned (too few reachable vertices, θ=$θ)")
            return D
        end
    end

    # TODO: Remove this check. Performance bottleneck and I'm pretty sure I don't need it
    if S_missing > k
        return D
    end

    if Subgraph.vertex_count(C) == 0
        S_edges = length(S.U) * length(S.V) - S_missing
        # S_edges = Subgraph.edge_count(g, S)
        if S_edges > Subgraph.edge_count(g, D) && length(S.U) >= θ && length(S.V) >= θ
            TRACE && println("  "^depth, "-> LEAF: new best D, S_edges=", S_edges, " D_edges=", Subgraph.edge_count(g,D))
            TRACE && println(" "^depth, " -> S.U=", sort(collect(S.U)), " S.V=", sort(collect(S.V)))
            @assert S_missing<= k "INVALID SOLUTION: d̄(S)=$(Subgraph.missing_edges(g, S)) > k=$k"
            return Subgraph.clone(S)
        end
        TRACE && println("  "^depth, "-> LEAF: not better than D, S_e=$S_edges, D_e=$(Subgraph.edge_count(g, D))")
        return D
    end

    is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, S), C)

    nondegree = is_u ? nondegree_in_subgraph_u(g, node, S) : nondegree_in_subgraph_v(g, node, S)
    if nondegree == 0
        _is_u, _node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, C), C)
        is_u = _is_u
        node = _node
    end

    TRACE && println("  "^depth, "branching on ", is_u ? "u=" : "v=", node, "  (d̄_S=$nondegree)")

    C′, C′_0, maximum_nondegree = update(S, C, g, is_u, node, k, S_missing)

    # FLAG: This is the reversed order from the way the authors
    # did it, because this way we don't have to make a complete copy
    # of the graph or remove a subgraph
    # If the order of branches turns out to matter, will have to flip this back

    # BranchB(S, C ∖ {u})
    Subgraph.remove_node!(C, is_u, node)
    D = branch_binary(S, C, g, D, k, θ, S_missing, depth + 1)

    missing_edges_budget = k - S_missing
    # missing_edges_budget = k - Subgraph.missing_edges(g, S)
    nondegree = nondegree_in_subgraph(g, is_u, node, S)

    # if nondegree <= maximum_nondegree
    if nondegree <= missing_edges_budget
        # S′ = S ∪ C′_0
        Subgraph.add!(S, C′_0)
        # S′ ∪ {u}
        Subgraph.add_node!(S, is_u, node)

        D = branch_binary(S, C′, g, D, k, θ, nondegree + S_missing, depth + 1)

        Subgraph.minus!(S, C′_0)
        Subgraph.remove_node!(S, is_u, node)
    end

    return D
end

# If we add u to S, what does C become? What "free" nodes can we add to S that don't limit the search space?
function update(S::SubGraph, C::SubGraph, g::FrozenBipartite, is_u::Bool, node::Int, k::Int, S_missing::Int)
    # C′ = {v ∈ C ∖ {u} | nondegree_{ S ∪ {u} }(v) ≤ k - Ē(S)}

    new_S_missing = S_missing + nondegree_in_subgraph(g, is_u, node, S)

    Subgraph.remove_node!(C, is_u, node)
    Subgraph.add_node!(S, is_u, node)

    maximum_nondegree = k - new_S_missing

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

    # C′_0 = {v ∈ C′ | nondegree_T(v) = 0}
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

    if TRACE
        println("    [update] branched-on=$node  maximum_nondegree=$maximum_nondegree",
                "  C′.U=", sorted_str(C′.U), " C′.V=", sorted_str(C′.V),
                "  C′_0.U=", sorted_str(C′_0.U), " C′_0.V=", sorted_str(C′_0.V))
    end

    return C′, C′_0, maximum_nondegree
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

function heuristic(g::FrozenBipartite, k::Int, θ::Int)
    U′ = Set()
    V′ = copy(g.V)

    while length(U′) < θ
        nondegs = []
    end
end