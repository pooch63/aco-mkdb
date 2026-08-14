const __OPPONENT_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")

using EnumX

const BRANCH_TRACE = false
const OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES = true
const OPTIMIZATION_USE_TIGHT_UPPER_BOUNDING_FUNCTION = true
const OPTIMIZATION_ONE_NONNEIGHBOR_REDUCTION = true

# Trying to maximize number of vertices or number of edges?
# I've spent at least two painful debugging sessions
# trying to figure out why my algorithm wasn't performing only to find it WAS and was just optimizing
# for the wrong thing, so I'll just set this to edges for now
@enumx GraphPart Vertices Edges
const MAXIMIZING = GraphPart.Edges

@assert MAXIMIZING != GraphPart.Vertices ||
    error("Cannot maximize vertices - this is not yet implemented")
@assert !(!OPTIMIZATION_PRUNE_BRANCHES_TOO_FEW_NODES && OPTIMIZATION_USE_TIGHT_UPPER_BOUNDING_FUNCTION) ||
    error("Cannot enable tight upper bounding without enabling branch pruning")

sorted_str(s::Set{Int}) = "{" * join(sort(collect(s)), ",") * "}"

@enumx BranchMode binary pivot

# Single source of truth for "what are we comparing leaves by"
@inline function solution_score(S::SubGraph, S_missing::Int)
    if MAXIMIZING == GraphPart.Vertices
        return length(S.U) + length(S.V)
    else # GraphPart.Edges
        return length(S.U) * length(S.V) - S_missing
    end
end

"""
Thread-safe top-`n` solution pool for parallel branch-and-bound.

`prune_floor` is the Nth-best score once the pool is full (else `typemin(Int)`),
so threads can prune with a lock-free atomic read without discarding candidates
that still belong in the top-N. Progressive reduction uses `top_score` (#1).
"""
struct SharedTopN
    n::Int
    lock::ReentrantLock
    scores::Vector{Int}          # descending
    sols::Vector{SubGraph}       # parallel to scores
    prune_floor::Threads.Atomic{Int}
end

function _set_prune_floor!(sb::SharedTopN)
    sb.prune_floor[] = length(sb.scores) >= sb.n ? sb.scores[end] : typemin(Int)
end

function SharedTopN(D0::SubGraph, D0_missing::Int, n::Int)
    n >= 1 || throw(ArgumentError("num_solutions must be >= 1, got $n"))
    s0 = solution_score(D0, D0_missing)
    sb = SharedTopN(n, ReentrantLock(), Int[s0], SubGraph[Subgraph.clone(D0)], Threads.Atomic{Int}(typemin(Int)))
    _set_prune_floor!(sb)
    return sb
end

# Lock-free prune threshold (Nth score when full).
best_edges(sb::SharedTopN) = sb.prune_floor[]

# Best (#1) score; used for progressive θ scheduling.
top_score(sb::SharedTopN) = isempty(sb.scores) ? 0 : sb.scores[1]

function _same_solution(a::SubGraph, b::SubGraph)
    return a.U == b.U && a.V == b.V
end

function try_update!(sb::SharedTopN, S::SubGraph, S_missing::Int)
    s = solution_score(S, S_missing)
    s <= sb.prune_floor[] && return false        # fast bail, no locking
    lock(sb.lock) do
        for existing in sb.sols
            _same_solution(existing, S) && return false
        end
        if length(sb.scores) >= sb.n && s <= sb.scores[end]
            return false
        end
        idx = searchsortedfirst(sb.scores, s; rev=true)
        insert!(sb.scores, idx, s)
        insert!(sb.sols, idx, Subgraph.clone(S))
        while length(sb.scores) > sb.n
            pop!(sb.scores)
            pop!(sb.sols)
        end
        _set_prune_floor!(sb)
        return true
    end
end

function snapshot_solutions(sb::SharedTopN)
    lock(sb.lock) do
        return [Subgraph.clone(sg) for sg in sb.sols]
    end
end

# BitVectors / reduce_graph! are indexed by original ids (1..max_id). After a
# prior reduction `length(adjU)` is a count, not a max id.
@inline function original_id_span(ids::Vector{Int}, fallback::Int)
    isempty(ids) && return fallback
    return max(fallback, maximum(ids))
end

"""
Exact search: branch from every U vertex (degree order) with candidate sets
N²₊(u) / N³₊(u), pruned by `reduce_candidate_set` at the current (θ_U, θ_V).

`id_U` / `id_V` are BitVector lengths, indexed by original vertex ids.
"""
function branch_from_all_u!(fg::FrozenBipartite, best::SharedTopN, k::Int, θ::Int,
    mode::BranchMode.T, θ_U::Int, θ_V::Int, id_U::Int, id_V::Int)
    us = get_degree_order(fg, true, true)

    # Multithreading reduces the effectiveness of upper bounding techniques
    # We order degrees because finding those nodes first is likelier to speed the search
    # up, but running completely sequentially turns out to be slower than parallelizing,
    # even if it reduces the effectiveness of upper bounding
    # Room for algorithmic improvement: @threads chunks the indices into
    # [1, 2, 3], [4, 5, 6], etc. But u_i that are later in the list have fewer indices,
    # so we're essentially packing all the u's that don't really remove much from the list
    # into one thread. That one thread is therefore going to take longer than the others.
    # If we wanted to make this faster, we could mix up the degrees, so do [1, 5, 6], [2, 3, 4].
    @threads for i in eachindex(us)
        S = SubGraph(Set(us[i]), Set())

        # N²₊(u): U-nodes at distance 2 from us[i].
        # neighbors_u(u) → V-nbrs; neighbors_v(v) → U-nbrs.
        _2_hop_neighbors = falses(id_U)
        for v in neighbors_u(fg, us[i])
            for _2u in neighbors_v(fg, v)
                _2_hop_neighbors[_2u] = true
            end
        end
        _2_hop_neighbors[us[i]] = false  # open neighborhood

        # N³₊(u): V-nodes adjacent to those 2-hop U-nodes (includes N(u)).
        _3_hop_neighbors = falses(id_V)
        for u in eachindex(_2_hop_neighbors)
            if _2_hop_neighbors[u]
                for v in neighbors_u(fg, u)
                    _3_hop_neighbors[v] = true
                end
            end
        end

        C = SubGraph(
            Set(u for u in eachindex(_2_hop_neighbors) if _2_hop_neighbors[u]),
            Set(v for v in eachindex(_3_hop_neighbors) if _3_hop_neighbors[v])
        )
        C = reduce_candidate_set(fg, C, us[i], θ_U, θ_V, k)
        bind_membership!(S, fg)
        bind_membership!(C, fg)

        branch!(S, C, fg, best, k, θ, mode)
    end
    return nothing
end

"""
Keep only vertices from `seed` that still exist on `fg` after reduction.
"""
function filter_seed_to_graph(fg::FrozenBipartite, seed::SubGraph)
    U = Set{Int}(u for u in seed.U if haskey(fg.u_index, u))
    V = Set{Int}(v for v in seed.V if haskey(fg.v_index, v))
    return SubGraph(U, V)
end

"""
Choose the best initial incumbent among the θ-heuristic (if enabled) and an
optional external seed (e.g. an ACO subgraph). Higher `solution_score` wins.
"""
function choose_initial_seed(fg::FrozenBipartite, k::Int, θ::Int, use_heuristic::Bool,
    initial_seed::Union{Nothing,SubGraph})
    candidates = SubGraph[]
    if use_heuristic
        push!(candidates, theta_based_heuristic(fg, k, θ; return_invalid=false))
    end
    if initial_seed !== nothing
        filtered = filter_seed_to_graph(fg, initial_seed)
        if Subgraph.vertex_count(filtered) > 0
            push!(candidates, filtered)
        end
    end
    isempty(candidates) && return SubGraph(Set(), Set())

    best = candidates[1]
    best_missing = Subgraph.missing_edges(fg, best)
    best_score = solution_score(best, best_missing)
    for c in Iterators.drop(candidates, 1)
        m = Subgraph.missing_edges(fg, c)
        s = solution_score(c, m)
        if s > best_score
            best = c
            best_missing = m
            best_score = s
        end
    end
    return best
end

# If the number of entries in g.adjU is not equal to the number of nodes or same for V,
# e.g., there are some gaps in node IDs, you'll need to pass the maximum node ID for each side.
# Returns the top `num_solutions` subgraphs by `solution_score`, best first.
#
# `initial_seed`: optional external incumbent (original vertex ids). When
# `use_heuristic` is also true, the better of θ-heuristic and `initial_seed`
# seeds the search (useful for measuring ACO→branch-and-bound speedups).
function find_kmdb!(g::BipartiteGraph, use_heuristic::Bool, mode::BranchMode.T, k::Int, θ::Int,
    reduction::ReductionMode.T=ReductionMode.all_reductions; num_U::Union{Int, Nothing}=nothing, num_V::Union{Int, Nothing}=nothing,
    num_solutions::Int=1, initial_seed::Union{Nothing,SubGraph}=nothing)

    num_solutions >= 1 || throw(ArgumentError("num_solutions must be >= 1, got $num_solutions"))
    @assert θ > k "θ must be greater than k"

    num_U = num_U === nothing ? length(g.adjU) : num_U
    num_V = num_V === nothing ? length(g.adjV) : num_V

    # Cap apply_graph_reductions! at simple; progressive is done below so we can
    # later interleave it with branching (which needs the mutable graph).
    base_reduction = reduction == ReductionMode.none ? ReductionMode.none : ReductionMode.simple
    fg = apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, base_reduction)

    if length(fg.u_ids) == 0 || length(fg.v_ids) == 0
        return SubGraph[]
    end

    println("Initial reduction is complete")

    D = choose_initial_seed(fg, k, θ, use_heuristic, initial_seed)
    D_missing = Subgraph.missing_edges(fg, D)
    best = SharedTopN(D, D_missing, num_solutions)

    id_U = original_id_span(fg.u_ids, num_U)
    id_V = original_id_span(fg.v_ids, num_V)

    if reduction == ReductionMode.progressive || reduction == ReductionMode.all_reductions
        u_degrees = Int[]
        for u in fg.u_ids
            push!(u_degrees, degree_u(fg, u))
        end

        θ_U = maximum(u_degrees) + k
        last_θ_eff = θ

        iterations = 0
        reductions = 0

        while θ_U > θ
            iterations += 1
            θ_V = max(θ, floor(Int, top_score(best) / θ_U))
            θ_U = max(θ, floor(Int, θ_U / 2))

            θ_eff = min(θ_U, θ_V)
            # Skip passes whose effective threshold matches the last reduce —
            # common when D is empty (θ_V ≡ θ) or θ_U halves but still ≥ last θ_eff.
            if θ_eff != last_θ_eff
                reduce_graph!(g, k, θ_eff, num_U, num_V)
                fg = freeze(g)
                last_θ_eff = θ_eff
                reductions += 1
                id_U = original_id_span(fg.u_ids, num_U)
                id_V = original_id_span(fg.v_ids, num_V)
            end

            branch_from_all_u!(fg, best, k, θ, mode, θ_U, θ_V, id_U, id_V)
        end
        println("Progressive reductions took $(iterations) iterations ($(reductions) new reduces)")
        if iterations == 0
            # max-degree + k ≤ θ: the progressive loop never enters, but search still must run.
            branch_from_all_u!(fg, best, k, θ, mode, θ, θ, id_U, id_V)
        end
    else
        # none / simple: still run exact search on the (already) reduced graph.
        # Use the true (θ, θ) candidate bounds — not a heuristic-tightened θ_V —
        # so a better balanced solution cannot be pruned away.
        println("Searching without further reduction")
        branch_from_all_u!(fg, best, k, θ, mode, θ, θ, id_U, id_V)
    end

    return snapshot_solutions(best)
end

function find_kmdb(g::BipartiteGraph, use_heuristic::Bool, mode::BranchMode.T,
    k::Int, θ::Int, reduction::ReductionMode.T=ReductionMode.all_reductions; num_solutions::Int=1,
    initial_seed::Union{Nothing,SubGraph}=nothing)
    return find_kmdb!(deepcopy(g), use_heuristic, mode, k, θ, reduction;
        num_solutions=num_solutions, initial_seed=initial_seed)
end

function common_neighbors(fg::FrozenBipartite, is_u::Bool, a::Int, b::Int)
    na = is_u ? neighbors_u(fg, a) : neighbors_v(fg, a)
    nb = is_u ? neighbors_u(fg, b) : neighbors_v(fg, b)
    return length(intersect(na, nb))
end

function reduce_candidate_set(fg::FrozenBipartite, C::SubGraph, u::Int, θ_U::Int, θ_V::Int, k::Int)
    C_U = Set(n for n in C.U if common_neighbors(fg, true, u, n) ≥ θ_V - k)
    C_V = Set(n for n in C.V if degree_v(fg, n) ≥ θ_U - k)
    return SubGraph(C_U, C_V)
end

# BranchB assumes that S ALWAYS has k or fewer missing edges. In other words, S is at all
# times a valid k-MDB. C is the set of all nodes that we still have to search, where each node could be added to S
# and still result in a k-MDB, although we don't necessarily know which subset of C could be added to S.
function branch!(S::SubGraph, C::SubGraph, g::FrozenBipartite,
    best::SharedTopN, k::Int, θ::Int, mode::BranchMode.T, S_missing::Int=0, depth::Int=0)
    ensure_membership!(S, g)
    ensure_membership!(C, g)
    if BRANCH_TRACE
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

        if OPTIMIZATION_USE_TIGHT_UPPER_BOUNDING_FUNCTION
            upper_u, upper_v, upper_e = upper_bound(S, C, g, k, S_missing)
        else
            upper_u, upper_v = length(C.U) + length(S.U), length(C.V) + length(S.V)
            upper_e = upper_u * upper_v
        end
        if (upper_u < θ || upper_v < θ) || # No matter how many vertices we add, we won't pass the θ threshold
            (upper_e < best_edges(best)) # No matter how many edges we add, we won't surpass D. Assumes we're optimizing for edges, not vertices 
            BRANCH_TRACE && println("  "^depth, "-> pruned (too few reachable vertices, θ=$θ, top_score=$(top_score(best)), prune_floor=$(best_edges(best)))")
            return
        end
    end

    # TODO: Remove this check. I'm pretty sure I don't need it
    if S_missing > k
        return
    end

    if Subgraph.vertex_count(C) == 0
        # Starting from a singleton (or a pruned candidate set) can reach a leaf
        # that is not θ-feasible; that is a dead end, not a solution.
        if length(S.U) < θ || length(S.V) < θ
            BRANCH_TRACE && println("  "^depth, "-> leaf skipped (not θ-feasible)")
            return
        end

        if BRANCH_TRACE
            println("  "^depth, "-> LEAF: S_score=$(solution_score(S, S_missing)) D_score=$(best_edges(best))")
        end

        @assert S_missing <= k "INVALID SOLUTION: d̄(S)=$(Subgraph.missing_edges(g, S)) > k=$k"
        try_update!(best, S, S_missing)
        return
    end

    if mode == BranchMode.binary
        is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, S), C)

        nondegree = is_u ? nondegree_in_subgraph_u(g, node, S) : nondegree_in_subgraph_v(g, node, S)
        if nondegree == 0
            _is_u, _node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, C), C)
            is_u = _is_u
            node = _node
        end

        BRANCH_TRACE && println("  "^depth, "branching on ", is_u ? "u=" : "v=", node, "  (d̄_S=$nondegree)")

        C′, C′_0, maximum_nondegree = update(S, C, g, is_u, node, k, S_missing)

        # FLAG: This is the reversed order from the way the authors
        # did it, because this way we don't have to make a complete copy
        # of the graph or remove a subgraph
        # If the order of branches turns out to matter, will have to flip this back

        # BranchB(S, C ∖ {u})
        Subgraph.remove_node!(C, g, is_u, node)
        branch!(S, C, g, best, k, θ, mode, S_missing, depth + 1)

        missing_edges_budget = k - S_missing
        # missing_edges_budget = k - Subgraph.missing_edges(g, S)
        nondegree = nondegree_in_subgraph(g, is_u, node, S)

        # if nondegree <= maximum_nondegree
        if nondegree <= missing_edges_budget
            # S′ = S ∪ C′_0
            Subgraph.add!(S, g, C′_0)
            # S′ ∪ {u}
            Subgraph.add_node!(S, g, is_u, node)

            branch!(S, C′, g, best, k, θ, mode, S_missing + nondegree, depth + 1)

            Subgraph.minus!(S, g, C′_0)
            Subgraph.remove_node!(S, g, is_u, node)
        end
    elseif mode == BranchMode.pivot
        BRANCH_TRACE && println("  "^depth, "[pivot] entering pivot mode, remaining_budget=$(k - S_missing)")

        C_0_u = [u for u in C.U if nondegree_in_subgraph(g, true, u::Int, S) == 0]
        C_0_v = [v for v in C.V if nondegree_in_subgraph(g, false, v::Int, S) == 0]

        C_0 = SubGraph(Set(C_0_u), Set(C_0_v))
        bind_membership!(C_0, g)
        BRANCH_TRACE && println("  "^depth, "[pivot] C_0 size=$(length(C_0_u) + length(C_0_v))  C_0.U=", sorted_str(C_0.U), " C_0.V=", sorted_str(C_0.V))

        if length(C_0_u) + length(C_0_v) == 0 || Subgraph.vertex_count(Subgraph.minus(C, C_0)) > k - S_missing
            BRANCH_TRACE && println("  "^depth, "[pivot] branch A: no usable zero-nondegree set or remaining vertices exceed budget")
            is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(g, u, n, S), C)
            C′, C′_0 = update(S, C, g, is_u, node, k, S_missing, depth)

            # S′ = S ∪ C′_0
            Subgraph.add!(S, g, C′_0)
            # S′ ∪ {u}
            Subgraph.add_node!(S, g, is_u, node)

            nondegree = nondegree_in_subgraph(g, is_u, node, S)

            BRANCH_TRACE && println("  "^depth, "[pivot] branch A -> recurse with selected node $(is_u ? "u" : "v")=$node")
            branch!(S, C′, g, best, k, θ, mode, S_missing + nondegree, depth + 1)

            # S = S′ ∖ C′_0 ∖ {u}
            Subgraph.minus!(S, g, C′_0)
            Subgraph.remove_node!(S, g, is_u, node)

            total_nondegree = nondegree + nondegree_in_subgraph(g, is_u, node, C)

            # C′ = C ∖ {u}
            Subgraph.remove_node!(C, g, is_u, node)

            # One non-neighbor reduction
            if total_nondegree <= 1 && OPTIMIZATION_ONE_NONNEIGHBOR_REDUCTION
                removed_nodes = Vector{Int}()

                if is_u
                    for u in C.U
                        if nondegree_in_subgraph(g, true, u, S) ≥ 1
                            Subgraph.remove_node!(C, g, true, u)
                            push!(removed_nodes, u)
                        end
                    end
                else
                    for v in C.V
                        if nondegree_in_subgraph(g, false, v, S) ≥ 1
                            Subgraph.remove_node!(C, g, false, v)
                            push!(removed_nodes, v)
                        end
                    end
                end
            end

            BRANCH_TRACE && println("  "^depth, "[pivot] branch A -> recurse on reduced candidate set")
            branch!(S, C, g, best, k, θ, mode, S_missing, depth + 1)

            if total_nondegree <= 1 && OPTIMIZATION_ONE_NONNEIGHBOR_REDUCTION
                if is_u
                    for u in removed_nodes
                        Subgraph.add_node!(C, g, true, u)
                    end
                else
                    for v in removed_nodes
                        Subgraph.add_node!(C, g, false, v)
                    end
                end
            end

            # C = C′ ∪ {u}
            Subgraph.add_node!(C, g, is_u, node)
        else
            is_u, node = argmin_nodes((u, n) -> nondegree_in_subgraph(g, u, n, C), C_0)

            C_nondegree = nondegree_in_subgraph(g, is_u, node, C)
            BRANCH_TRACE && println("  "^depth, "[pivot] branch B: selected $(is_u ? "u" : "v")=$node with C_nondegree=$C_nondegree")

            if C_nondegree > k - S_missing > 0
                BRANCH_TRACE && println("  "^depth, "[pivot] branch B1: C_nondegree exceeds remaining budget")

                C′, C′_0 = update(S, C, g, is_u, node, k, S_missing, depth)

                # S′ = S ∪ C′_0 ∪ {u}
                Subgraph.add!(S, g, C′_0)
                Subgraph.add_node!(S, g, is_u, node)

                nondegree = nondegree_in_subgraph(g, is_u, node, S)

                BRANCH_TRACE && println("  "^depth, "[pivot] branch B1 -> recurse with added node $(is_u ? "u" : "v")=$node, nondegree=$nondegree")
                branch!(S, C′, g, best, k, θ, mode, S_missing + nondegree, depth + 1)

                # S = S′ ∖ C′_0 ∖ {u}
                Subgraph.minus!(S, g, C′_0)
                Subgraph.remove_node!(S, g, is_u, node)

                # C = C ∖ {u}
                Subgraph.remove_node!(C, g, is_u, node)
                BRANCH_TRACE && println("  "^depth, "[pivot] branch B1 -> recurse after removing node $(is_u ? "u" : "v")=$node, nondegree=$nondegree")
                branch!(S, C, g, best, k, θ, mode, S_missing, depth + 1)
            
                Subgraph.add_node!(C, g, is_u, node)
            else
                BRANCH_TRACE && println("  "^depth, "[pivot] branch B2: using the zero-nondegree candidate set")
                C′, C′_0 = update(S, C, g, is_u, node, k, S_missing, depth)

                # S′ = S ∪ C′_0 ∪ {u}
                Subgraph.add!(S, g, C′_0)
                Subgraph.add_node!(S, g, is_u, node)

                nondegree = nondegree_in_subgraph(g, is_u, node, S)

                # Let L = search space
                # L = {u} ∪ nonneighbors_C(u)
                # u ∈ L
                BRANCH_TRACE && println("  "^depth, "[pivot] branch B2 -> recurse on primary branch ($is_u, $node) with nondegree=$nondegree")
                branch!(S, C′, g, best, k, θ, mode, S_missing + nondegree, depth + 1)

                Subgraph.minus!(S, g, C′_0)
                Subgraph.remove_node!(S, g, is_u, node)

                Subgraph.remove_node!(C, g, is_u, node)

                # v = {u} ∪ nonneighbors_C(u)
                nonneighbors = Subgraph.nonneighbors_in_subgraph(g, is_u, node, C)
                BRANCH_TRACE && println("  "^depth, "[pivot] branch B2 -> exploring $(length(nonneighbors)) nonneighbors")
                for v in nonneighbors
                    C′, C′_0 = update(S, C, g, !is_u, v, k, S_missing, depth)

                    # S′ = S ∪ C′_0 ∪ {u}

                    Subgraph.add!(S, g, C′_0)
                    Subgraph.add_node!(S, g, !is_u, v)

                    nondegree = nondegree_in_subgraph(g, !is_u, v, S)

                    BRANCH_TRACE && println("  "^depth, "[pivot] branch B2 -> recurse on nonneighbor $(is_u ? "v" : "u")=$v, nondegree=$nondegree")
                    branch!(S, C′, g, best, k, θ, mode, S_missing + nondegree, depth + 1)

                    Subgraph.minus!(S, g, C′_0)
                    Subgraph.remove_node!(S, g, !is_u, v)

                    Subgraph.remove_node!(C, g, !is_u, v)
                end
            end
        end
    end
end

