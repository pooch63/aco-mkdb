include("graph.jl")
include("search.jl")

const TRACE = false
const OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES = true
const OPTIMIZATION_USE_TIGHT_UPPER_BOUNDING_FUNCTION = true
const OPTIMIZATION_ONE_NONNEIGHBOR_REDUCTION = true

# Trying to maximize number of vertices or number of edges?
# In large graphs, this almost certainly doesn't matter. If, somehow, a subgraph
# with the most vertices isn't equal to the subgraph with the most edges, we'd
# want to record both. In the small graphs with a lot of variation that we use for testing,
# it's just good to have this setting enabled. I've spent at least two painful debugging sessions
# trying to figure out why my algorithm wasn't performing only to find it WAS and was just optimizing
# for the wrong thing (which again, doesn't actually matter!), so I'll just set this to edges for now
@enum GraphPart Vertices Edges
const MAXIMIZING = Edges

@assert !(!OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES && OPTIMIZATION_USE_TIGHT_UPPER_BOUNDING_FUNCTION) ||
    error("Cannot enable tight upper bounding without enabling branch pruning")

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

# If the number of entries in g.adjU is not equal to the number of nodes or same for V,
# e.g., there are some gaps in node IDs, you'll need to pass the maximum node ID for each side
function find_kmdb!(g::BipartiteGraph, use_heuristic::Bool, mode::BranchMode, k::Int, θ::Int,
    reduction::ReductionMode=progressive; num_U::Union{Int, Nothing}=nothing, num_V::Union{Int, Nothing}=nothing)

    @assert θ > k "θ must be greater than k"

    fg = freeze(g)

    D = use_heuristic ? initial_heuristic(fg, k, θ) : SubGraph(Set(), Set())

    if isempty(fg.u_ids) || isempty(fg.v_ids)
        return branch(fg, use_heuristic, mode, D, k, θ, mode)
    end

    fg = apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, reduction)

    # If there's not enough nodes remaining on either side, then we
    # already know it's an invalid solution
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph(Set(), Set())
    end

    D = use_heuristic ? initial_heuristic(fg, k, θ) : SubGraph(Set(), Set())

    println("Reduction is complete")
    return branch(
        SubGraph(Set(), Set()),
        SubGraph(Set(u for u in fg.u_ids), Set(v for v in fg.v_ids)),
        fg,
        D,
        k,
        θ,
        mode,
    )
end

function find_kmdb(g::BipartiteGraph, use_heuristic::Bool, mode::BranchMode,
    k::Int, θ::Int, reduction::ReductionMode=progressive)
    return find_kmdb!(deepcopy(g), use_heuristic, mode, k, θ, reduction)
end

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

        if OPTIMIZATION_USE_TIGHT_UPPER_BOUNDING_FUNCTION
            upper_u, upper_v, upper_e = upper_bound(S, C, g, k, S_missing)
        else
            upper_u, upper_v, upper_e = θ, θ, Subgraph.edge_count(g, D)
        end
        if (upper_u < θ || upper_v < θ) || # No matter how many vertices we add, we won't pass the θ threshold
            (upper_e < Subgraph.edge_count(g, D)) # No matter how many edges we add, we won't surpass D. Assumes we're optimizing for edges, not vertices 
            TRACE && println("  "^depth, "-> pruned (too few reachable vertices, θ=$θ, |D_u|=$(length(D.U)), |D_v|=$(length(D.V)))")
            return D
        end
    end

    # TODO: Remove this check. Performance bottleneck and I'm pretty sure I don't need it
    if S_missing > k
        return D
    end

    if Subgraph.vertex_count(C) == 0
        @assert length(S.U) >= θ && length(S.V) >= θ "Should not have reached leaf node that contains invalid solution"

        S_edges = length(S.U) * length(S.V) - S_missing
        # S_edges = Subgraph.edge_count(g, S)
        D_edges = Subgraph.edge_count(g, D)

        if (MAXIMIZING == Vertices && length(S.U) + length(S.V) > length(D.U) + length(D.V)) ||
            (MAXIMIZING == Edges && S_edges > D_edges)
            @assert S_missing <= k "INVALID SOLUTION: d̄(S)=$(Subgraph.missing_edges(g, S)) > k=$k"
            TRACE && println("  "^depth, "-> LEAF: new best D, S_edges=", S_edges, " D_edges=$(D_edges)")
            return Subgraph.clone(S)
        end

        TRACE && println("  "^depth, "-> LEAF: not better than D, S_e=$S_edges, D_e=$(Subgraph.edge_count(g, D)), D_u=", collect(D.U), " D_v=", collect(D.V))
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

            total_nondegree = nondegree + nondegree_in_subgraph(g, is_u, node, C)

            # C′ = C ∖ {u}
            Subgraph.remove_node!(C, is_u, node)

            # One non-neighbor reduction
            if total_nondegree <= 1 && OPTIMIZATION_ONE_NONNEIGHBOR_REDUCTION
                removed_nodes = Vector{Int}()

                if is_u
                    for u in C.U
                        if nondegree_in_subgraph(g, true, u, S) ≥ 1
                            Subgraph.remove_node!(C, true, u)
                            push!(removed_nodes, u)
                            println("one-nonneighbor reduction!")
                        end
                    end
                else
                    for v in C.V
                        if nondegree_in_subgraph(g, false, v, S) ≥ 1
                            Subgraph.remove_node!(C, false, v)
                            push!(removed_nodes, v)
                            println("one-nonneighbor reduction!")
                        end
                    end
                end
            end

            TRACE && println("  "^depth, "[pivot] branch A -> recurse on reduced candidate set")
            D = branch(S, C, g, D, k, θ, mode, S_missing, depth + 1)
                        
            if total_nondegree <= 1 && OPTIMIZATION_ONE_NONNEIGHBOR_REDUCTION
                if is_u
                    union!(C.U, removed_nodes)
                else
                    union!(C.V, removed_nodes)
                end
            end

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
            
                Subgraph.add_node!(C, is_u, node)
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
