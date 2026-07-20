include("graph.jl")

const TRACE = false
const OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES = true

sorted_str(s::Set{Int}) = "{" * join(sort(collect(s)), ",") * "}"

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

@enum BranchMode binary pivot
# BranchB assumes that S ALWAYS has k or fewer missing edges. In other words, S is at all
# times a valid k-MDB. C is the set of all nodes that we still have to search, where each node could be added to S
# and still result in a k-MDB, although we don't necessarily know which subset of C could be added to S.
function branch(S::SubGraph, C::SubGraph, g::FrozenBipartite,
    D::SubGraph, k::Int, θ::Int, mode::BranchMode, S_missing::Int=0, depth::Int=0)
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
        max_u = length(C.U) + length(S.U)
        max_v = length(C.V) + length(S.V)

        if (max_u < θ || max_v < θ) || # No matter how many vertices we add, we won't pass the θ threshold
            (max_u * max_v - S_missing < Subgraph.edge_count(g, D)) # No matter how many edges we add, we won't surpass D. Assumes we're optimizing for edges, not vertices 
            TRACE && println("  "^depth, "-> pruned (too few reachable vertices, θ=$θ, |D_u|=$(length(D.U)), |D_v|=$(length(D.V)))")
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
            @assert S_missing<= k "INVALID SOLUTION: d̄(S)=$(Subgraph.missing_edges(g, S)) > k=$k"
            return Subgraph.clone(S)
        end
        TRACE && println("  "^depth, "-> LEAF: not better than D, S_e=$S_edges, D_e=$(Subgraph.edge_count(g, D))")
        return D
    end

    if mode == binary
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
        D = branch(S, C, g, D, k, θ, mode, S_missing, depth + 1)

        missing_edges_budget = k - S_missing
        # missing_edges_budget = k - Subgraph.missing_edges(g, S)
        nondegree = nondegree_in_subgraph(g, is_u, node, S)

        # if nondegree <= maximum_nondegree
        if nondegree <= missing_edges_budget
            # S′ = S ∪ C′_0
            Subgraph.add!(S, C′_0)
            # S′ ∪ {u}
            Subgraph.add_node!(S, is_u, node)

            D = branch(S, C′, g, D, k, θ, mode, S_missing + nondegree, depth + 1)

            Subgraph.minus!(S, C′_0)
            Subgraph.remove_node!(S, is_u, node)
        end
    elseif mode == pivot
        TRACE && println("  "^depth, "[pivot] entering pivot mode, remaining_budget=$(k - S_missing)")

        # BRAIN EXERCISE: why does changing this to nondegree in subgraph C not work?
        C_0_u = [u for u in C.U if nondegree_in_subgraph(g, true, u::Int, S) == 0]
        C_0_v = [v for v in C.V if nondegree_in_subgraph(g, false, v::Int, S) == 0]

        C_0 = SubGraph(Set(C_0_u), Set(C_0_v))
        TRACE && println("  "^depth, "[pivot] C_0 size=$(length(C_0_u) + length(C_0_v))  C_0.U=", sorted_str(C_0.U), " C_0.V=", sorted_str(C_0.V))

        if length(C_0_u) + length(C_0_v) == 0 || Subgraph.vertex_count(Subgraph.minus(C, C_0)) > k - S_missing
            TRACE && println("  "^depth, "[pivot] branch A: no usable zero-nondegree set or remaining vertices exceed budget")
            is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, S), C)
            C′, C′_0 = update(S, C, g, is_u, node, k, S_missing)

            # S′ = S ∪ C′_0
            Subgraph.add!(S, C′_0)
            # S′ ∪ {u}
            Subgraph.add_node!(S, is_u, node)

            nondegree = nondegree_in_subgraph(g, is_u, node, S)

            TRACE && println("  "^depth, "[pivot] branch A -> recurse with selected node $(is_u ? "u" : "v")=$node")
            D = branch(S, C′, g, D, k, θ, mode, S_missing + nondegree, depth + 1)

            # S = S′ ∖ C′_0 ∖ {u}
            Subgraph.minus!(S, C′_0)
            Subgraph.remove_node!(S, is_u, node)

            # C′ C ∖ {u}
            Subgraph.remove_node!(C, is_u, node)

            TRACE && println("  "^depth, "[pivot] branch A -> recurse on reduced candidate set")
            D = branch(S, C, g, D, k, θ, mode, S_missing, depth + 1)

            # C = C′ ∪ {u}
            Subgraph.add_node!(C, is_u, node)
        else
            is_u, node = argmin_nodes((u, n) -> nondegree_in_subgraph(g, u, n, C), C_0)

            C_nondegree = nondegree_in_subgraph(g, is_u, node, C)
            TRACE && println("  "^depth, "[pivot] branch B: selected $(is_u ? "u" : "v")=$node with C_nondegree=$C_nondegree")

            if C_nondegree > k - S_missing > 0
                TRACE && println("  "^depth, "[pivot] branch B1: C_nondegree exceeds remaining budget")
                C′, C′_0 = update(S, C, g, is_u, node, k, S_missing)

                # S′ = S ∪ C′_0 ∪ {u}
                Subgraph.add!(S, C′_0)
                Subgraph.add_node!(S, is_u, node)

                nondegree = nondegree_in_subgraph(g, is_u, node, S)

                TRACE && println("  "^depth, "[pivot] branch B1 -> recurse with added node $(is_u ? "u" : "v")=$node, nondegree=$nondegree")
                D = branch(S, C′, g, D, k, θ, mode, S_missing + nondegree, depth + 1)

                # S = S′ ∖ C′_0 ∖ {u}
                Subgraph.minus!(S, C′_0)
                Subgraph.remove_node!(S, is_u, node)

                # C = C ∖ {u}
                Subgraph.remove_node!(C, is_u, node)
                TRACE && println("  "^depth, "[pivot] branch B1 -> recurse after removing node $(is_u ? "u" : "v")=$node, nondegree=$nondegree")
                D = branch(S, C, g, D, k, θ, mode, S_missing, depth + 1)
            else
                TRACE && println("  "^depth, "[pivot] branch B2: using the zero-nondegree candidate set")
                C′, C′_0 = update(S, C, g, is_u, node, k, S_missing)

                # S′ = S ∪ C′_0 ∪ {u}
                Subgraph.add!(S, C′_0)
                Subgraph.add_node!(S, is_u, node)

                nondegree = nondegree_in_subgraph(g, is_u, node, S)

                # Let L = search space
                # L = {u} ∪ nonneighbors_C(u)
                # u ∈ L
                TRACE && println("  "^depth, "[pivot] branch B2 -> recurse on primary branch with nondegree=$nondegree")
                D = branch(S, C′, g, D, k, θ, mode, S_missing + nondegree, depth + 1)

                Subgraph.minus!(S, C′_0)
                Subgraph.remove_node!(S, is_u, node)

                Subgraph.remove_node!(C, is_u, node)

                # v = {u} ∪ nonneighbors_C(u)
                nonneighbors = Subgraph.nonneighbors_in_subgraph(g, is_u, node, C)
                TRACE && println("  "^depth, "[pivot] branch B2 -> exploring $(length(nonneighbors)) nonneighbors")
                for v in nonneighbors
                    C′, C′_0 = update(S, C, g, !is_u, v, k, S_missing)

                    # S′ = S ∪ C′_0 ∪ {u}

                    Subgraph.add!(S, C′_0)
                    Subgraph.add_node!(S, !is_u, v)

                    nondegree = nondegree_in_subgraph(g, !is_u, v, S)

                    TRACE && println("  "^depth, "[pivot] branch B2 -> recurse on nonneighbor $(is_u ? "v" : "u")=$v, nondegree=$nondegree")
                    D = branch(S, C′, g, D, k, θ, mode, S_missing + nondegree, depth + 1)

                    Subgraph.minus!(S, C′_0)
                    Subgraph.remove_node!(S, !is_u, v)

                    Subgraph.remove_node!(C, !is_u, v)
                end
            end
        end
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

function heuristic(g::FrozenBipartite, k::Int, θ::Int)
    U′ = Set()
    V′ = copy(g.V)

    while length(U′) < θ
        nondegs = []
    end
end