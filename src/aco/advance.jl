mutable struct Ant
    explored::SubGraph
    last_visited::Node
    species::Int
    candidates::Vector{DegreeNode}
    missing::Int
    # Compact-id nodes in the order each ant appended them during construction.
    addition_order::Vector{Tuple{Bool,Int}}
    # First observed missing-edge count when |S| reaches each size (via advance_ant!).
    missing_at_size::Dict{Int,Int}
    # (is_u, compact id, species deposit) written by this construction.
    # Shared deposit is `dep * SHARED_PHEROMONE_FACTOR`. Bound-prune subtracts these.
    deposits::Vector{Tuple{Bool,Int,Float64}}
    # Set when `upper_bound` shows this ant cannot beat the incumbent.
    bound_pruned::Bool
    # Best greedy fill seen during this construction. Empty when fill is off
    # or no fill has been recorded. The walk in `explored` is left unchanged.
    filled::SubGraph
end

"""Candidate set for an empty S: every vertex with deg_S = 0 (nondegree is always 0)."""
function empty_subgraph_candidates(fg::FrozenBipartite)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    nodes = Vector{DegreeNode}(undef, nU + nV)
    @inbounds for i in 1:nU
        nodes[i] = DegreeNode(true, fg.u_ids[i], 0)
    end
    @inbounds for i in 1:nV
        nodes[nU + i] = DegreeNode(false, fg.v_ids[i], 0)
    end
    return nodes
end

function new_ants(fg::FrozenBipartite, num_ants::Int, num_subspecies::Int)
    template = empty_subgraph_candidates(fg)
    return [Ant(SubGraph(), Node(), mod1(i, num_subspecies), copy(template), 0,
                Tuple{Bool,Int}[], Dict{Int,Int}(), Tuple{Bool,Int,Float64}[], false,
                SubGraph())
            for i in 1:num_ants]
end

"""Record a vertex append and the missing-edge count at the new subgraph size."""
function record_ant_addition!(ant::Ant, next::Node)
    push!(ant.addition_order, (next.is_u, next.id))
    depth = Subgraph.vertex_count(ant.explored)
    if !haskey(ant.missing_at_size, depth)
        ant.missing_at_size[depth] = ant.missing
    end
    return ant
end

function reset_ant_construction_trace!(ant::Ant)
    empty!(ant.addition_order)
    empty!(ant.missing_at_size)
    empty!(ant.deposits)
    ant.bound_pruned = false
    ant.filled = SubGraph()
    return ant
end

"""
Plant `seed` into one ant: set `explored`, recompute `missing`, and rebuild
the candidate set. No-op if `seed` is empty.
"""
function seed_ant_with_subgraph!(ant::Ant, fg::FrozenBipartite, k::Int, seed::SubGraph)
    Subgraph.vertex_count(seed) == 0 && return ant
    sg = SubGraph(copy(seed.U), copy(seed.V))
    last = if !isempty(sg.U)
        Node(true, first(sg.U))
    elseif !isempty(sg.V)
        Node(false, first(sg.V))
    else
        Node()
    end
    ant.explored = sg
    ant.last_visited = last
    ant.missing = Subgraph.missing_edges(fg, sg)
    ant.candidates = candidate_set_with_nondegrees(fg, sg, k)
    reset_ant_construction_trace!(ant)
    return ant
end

"""
Seed every ant with the same (compact-id) subgraph. No-op if `seed` is empty.
"""
function seed_ants_with_subgraph!(ants::Vector{Ant}, fg::FrozenBipartite, k::Int,
    seed::SubGraph)
    Subgraph.vertex_count(seed) == 0 && return ants
    for ant in ants
        seed_ant_with_subgraph!(ant, fg, k, seed)
    end
    return ants
end

"""
Seed of ant `ant_index` (1-based) on iteration `iter`.

Ant `i` on iteration `t` receives `seeds[mod1(i + (t-1)*n_ants, length(seeds))]`.
Iteration 1 uses the front of the list. Later iterations walk forward, so a
colony smaller than the seed list still reaches later seeds.
"""
function round_robin_seed(seeds::AbstractVector{SubGraph}, n_ants::Int, ant_index::Int, iter::Int)
    isempty(seeds) && return nothing
    return seeds[mod1(ant_index + (iter - 1) * n_ants, length(seeds))]
end

"""Plant a distinct seed into each ant. See `round_robin_seed`."""
function seed_ants_round_robin!(ants::Vector{Ant}, fg::FrozenBipartite, k::Int,
    seeds::AbstractVector{SubGraph}, iter::Int)
    isempty(seeds) && return ants
    for (i, ant) in enumerate(ants)
        seed_ant_with_subgraph!(ant, fg, k, round_robin_seed(seeds, length(ants), i, iter))
    end
    return ants
end

subgraph_has_seed(sg::SubGraph, seed::SubGraph) =
    issubset(seed.U, sg.U) && issubset(seed.V, sg.V)

"""Union `seed` into `sg` (mutates)."""
function merge_seed!(sg::SubGraph, seed::SubGraph)
    union!(sg.U, seed.U)
    union!(sg.V, seed.V)
    return sg
end

"""
Drop `added` and any vertex whose nondegree into the updated S exceeds `budget`.
Opposite-side `deg` values are incremented when adjacent to `added`.
`nU_sg` / `nV_sg` are |S.U| / |S.V| *after* the add. Mutates `candidates` in place.
"""
function reduce_candidates!(candidates::Vector{DegreeNode}, fg::FrozenBipartite,
    added::DegreeNode, budget::Int, nU_sg::Int, nV_sg::Int)
    # Compact ACO graphs use dense ids 1:n, so CSR neighbor indices index the mask directly.
    if added.is_u
        ui = fg.u_index[added.id]
        is_nbr = falses(length(fg.v_ids))
        @inbounds for k in neighbor_range_u(fg, ui)
            is_nbr[fg.v_adj[k]] = true
        end
    else
        vi = fg.v_index[added.id]
        is_nbr = falses(length(fg.u_ids))
        @inbounds for k in neighbor_range_v(fg, vi)
            is_nbr[fg.u_adj[k]] = true
        end
    end

    write = 1
    @inbounds for i in eachindex(candidates)
        c = candidates[i]
        if c.is_u == added.is_u && c.id == added.id
            continue
        end
        deg = c.deg
        if c.is_u != added.is_u && is_nbr[c.id]
            deg += 1
        end
        n_opp = c.is_u ? nV_sg : nU_sg
        if n_opp - deg <= budget
            candidates[write] = DegreeNode(c.is_u, c.id, deg)
            write += 1
        end
    end
    resize!(candidates, write - 1)
    return candidates
end

"""Incumbent an ant must beat. `edges` is only used when `feasible`."""
mutable struct BoundTarget
    feasible::Bool
    edges::Int
    fit::Float64
end

function bound_target_from(fg::FrozenBipartite, sg::SubGraph, θ::Int, fitness)
    ok = theta_feasible(sg, θ)
    edges = ok ? Subgraph.edge_count(fg, sg) : 0
    fit = Subgraph.vertex_count(sg) == 0 ? 0.0 : Float64(fitness(fg, sg, θ))
    return BoundTarget(ok, edges, fit)
end

"""Raise `target` when `sg` beats it under the same rule as `better_than_best`."""
function raise_bound_target!(target::BoundTarget, fg::FrozenBipartite, sg::SubGraph, θ::Int, fitness)
    Subgraph.vertex_count(sg) == 0 && return target
    ok = theta_feasible(sg, θ)
    if ok
        edges = Subgraph.edge_count(fg, sg)
        if !target.feasible || edges > target.edges
            target.feasible = true
            target.edges = edges
            target.fit = Float64(fitness(fg, sg, θ))
        end
    elseif !target.feasible
        fit = Float64(fitness(fg, sg, θ))
        if fit > target.fit
            target.fit = fit
        end
    end
    return target
end

function candidates_subgraph(candidates::Vector{DegreeNode})
    U = Set{Int}()
    V = Set{Int}()
    for c in candidates
        if c.is_u
            push!(U, c.id)
        else
            push!(V, c.id)
        end
    end
    return SubGraph(U, V)
end

"""
True when no k-defective completion of `S` inside `C` can beat `target`.

`upper_bound` (the same function branch-and-bound uses) supplies the best
reachable side sizes and edge count. A θ-feasible incumbent is beaten only by
a θ-feasible completion with strictly more edges. An infeasible incumbent is
beaten by any θ-feasible completion, or by an infeasible one with higher
`instance_fitness` (`min_side^2`).
"""
function bound_cannot_beat(fg::FrozenBipartite, S::SubGraph, C::SubGraph,
    k::Int, S_missing::Int, θ::Int, target::BoundTarget)
    ensure_membership!(S, fg)
    bind_membership!(C, fg)
    upper_u, upper_v, upper_e = upper_bound(S, C, fg, k, S_missing)
    can_be_feasible = upper_u >= θ && upper_v >= θ
    if target.feasible
        return !can_be_feasible || upper_e <= target.edges
    end
    can_be_feasible && return false
    return min(upper_u, upper_v)^2 <= target.fit
end

function ant_bound_prunes(fg::FrozenBipartite, ant::Ant, k::Int, θ::Int, target::BoundTarget)
    return bound_cannot_beat(fg, ant.explored, candidates_subgraph(ant.candidates),
        k, ant.missing, θ, target)
end

"""Subtract this ant's construction deposits from `colony` and forget them."""
function rollback_ant_pheromone!(colony::ColonyPheromones, ant::Ant)
    species = colony.species[ant.species]
    for (is_u, id, dep) in ant.deposits
        node = Node(is_u, id)
        add_pheromone!(species, node, -dep)
        add_pheromone!(colony.shared, node, -dep * SHARED_PHEROMONE_FACTOR)
    end
    empty!(ant.deposits)
    return colony
end

# Start and end index are inclusive
function advance_ants!(fg::FrozenBipartite, pheromones::ColonyPheromones, pheromone::Int,
    ants::Vector{Ant}, k::Int, θ::Int, ant_chunk; prefer_smaller_side::Bool=true,
    neighbor_scope_limit::Bool=true,
    trace_target::Union{Nothing,SubGraph}=nothing,
    deposit_scale=nothing,
    prune_by_bound::Bool=false,
    bound_target::Union{Nothing,BoundTarget}=nothing,
    greedy_fill::Symbol=:off,
    fitness=instance_fitness,
    eta_exp::Float64=Float64(ETA_EXP),
    deposit_online::Bool=true)
    # Thread-local storage to prevent data races
    additions = zero_colony_pheromones(fg, length(pheromones.species))
    local_invalid_ants = Int[]
    
    for idx in ant_chunk
        # advance_ant! mutates the ant and adds to local `additions`
        if !advance_ant!(fg, pheromones, additions, pheromone, ants[idx], k, θ;
            prefer_smaller_side=prefer_smaller_side,
            neighbor_scope_limit=neighbor_scope_limit,
            ant_id=idx, trace_target=trace_target,
            deposit_scale=deposit_scale,
            prune_by_bound=prune_by_bound,
            bound_target=bound_target,
            greedy_fill=greedy_fill,
            fitness=fitness,
            eta_exp=eta_exp,
            deposit_online=deposit_online)
            push!(local_invalid_ants, idx)
        end
    end

    return additions, local_invalid_ants
end

"""
Desirability: τ · η^`eta_exp`

η = d_S + d_G · σ(−|S|). `eta_exp` defaults to `ETA_EXP` (`src/constants.jl`).
"""
function node_desirability(pheromones::ColonyPheromones, fg::FrozenBipartite,
    node::DegreeNode, species::Int, exp_sg_vertex_count::Float64;
    eta_exp::Float64=Float64(ETA_EXP))
    τ = max(effective_pheromone(pheromones, Node(node.is_u, node.id), species), eps(Float64))
    deg_G = node.is_u ? degree_u(fg, node.id) : degree_v(fg, node.id)
    η = node.deg + deg_G * exp_sg_vertex_count
    return τ * η^eta_exp
end

"""Lay `amount` on every vertex of `sg` (species trail, and shared × `SHARED_PHEROMONE_FACTOR`)."""
function deposit_solution_pheromone!(pheromones::ColonyPheromones, sg::SubGraph,
    species::Int, amount::Float64)
    amount == 0 && return pheromones
    for u in sg.U
        node = Node(true, u)
        add_pheromone!(pheromones.species[species], node, amount)
        add_pheromone!(pheromones.shared, node, amount * SHARED_PHEROMONE_FACTOR)
    end
    for v in sg.V
        node = Node(false, v)
        add_pheromone!(pheromones.species[species], node, amount)
        add_pheromone!(pheromones.shared, node, amount * SHARED_PHEROMONE_FACTOR)
    end
    return pheromones
end

"""
Candidates in `candidates` that are neighbors of `last` in `fg`.

Compact ACO graphs use dense ids 1:n, so CSR neighbor indices index masks /
candidate ids directly (same assumption as `reduce_candidates!`). Returns an
empty vector when `last` is null or no neighbor remains in `candidates`.
"""
function neighbor_restricted_candidates(fg::FrozenBipartite, last::Node,
    candidates::Vector{DegreeNode})
    last.id == -1 && return DegreeNode[]

    if last.is_u
        ui = get(fg.u_index, last.id, nothing)
        ui === nothing && return DegreeNode[]
        is_nbr = falses(length(fg.v_ids))
        @inbounds for k in neighbor_range_u(fg, ui)
            is_nbr[fg.v_adj[k]] = true
        end
        out = DegreeNode[]
        sizehint!(out, min(length(candidates), count(is_nbr)))
        @inbounds for c in candidates
            if !c.is_u && is_nbr[c.id]
                push!(out, c)
            end
        end
        return out
    else
        vi = get(fg.v_index, last.id, nothing)
        vi === nothing && return DegreeNode[]
        is_nbr = falses(length(fg.u_ids))
        @inbounds for k in neighbor_range_v(fg, vi)
            is_nbr[fg.u_adj[k]] = true
        end
        out = DegreeNode[]
        sizehint!(out, min(length(candidates), count(is_nbr)))
        @inbounds for c in candidates
            if c.is_u && is_nbr[c.id]
                push!(out, c)
            end
        end
        return out
    end
end

# Returns false if the ant has no further moves
function advance_ant!(fg::FrozenBipartite, pheromones::ColonyPheromones, additions::ColonyPheromones,
    pheromone::Int, ant::Ant, k::Int, θ::Int; prefer_smaller_side::Bool=true,
    neighbor_scope_limit::Bool=true,
    ant_id::Int=0, trace_target::Union{Nothing,SubGraph}=nothing,
    deposit_scale=nothing,
    prune_by_bound::Bool=false,
    bound_target::Union{Nothing,BoundTarget}=nothing,
    greedy_fill::Symbol=:off,
    fitness=instance_fitness,
    eta_exp::Float64=Float64(ETA_EXP),
    deposit_online::Bool=true)
    depth = Subgraph.vertex_count(ant.explored)
    missing = ant.missing
    candidates = ant.candidates

    if ACO_TRACE
        println("  "^depth, "ant=$ant_id species=$(ant.species) depth=$depth  ",
                "S.U=", sorted_str(ant.explored.U), " S.V=", sorted_str(ant.explored.V),
                "  missing=$missing/$k  |C|=$(length(candidates))")
        if trace_target !== nothing
            ou, ov = target_overlap(ant.explored, trace_target)
            target_in_C = count(c -> (c.is_u && (c.id in trace_target.U)) ||
                                     (!c.is_u && (c.id in trace_target.V)), candidates)
            remaining_U = length(setdiff(trace_target.U, ant.explored.U))
            remaining_V = length(setdiff(trace_target.V, ant.explored.V))
            println("  "^depth, "  target_hit=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))",
                    "  target_still_in_C=$target_in_C  target_remaining=$remaining_U+$remaining_V")
        end
    end

    if prune_by_bound && bound_target !== nothing &&
        ant_bound_prunes(fg, ant, k, θ, bound_target)
        ant.bound_pruned = true
        if ACO_TRACE
            println("  "^depth, "-> STOP (upper bound cannot beat incumbent)")
        end
        return false
    end

    if isempty(candidates)
        if ACO_TRACE
            msg = "  "^depth * "-> STOP (no candidates)"
            if trace_target !== nothing
                ou, ov = target_overlap(ant.explored, trace_target)
                msg *= "  final_target_hit=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))"
            end
            println(msg)
        end
        greedy_fill === :end && record_end_fill!(ant, fg, k, θ, fitness)
        return false
    end

    # Optionally prefer neighbors of the last visited node that are still in C.
    # If that intersection is empty (null last, or every nbr already in S / pruned),
    # or neighbor_scope_limit is off, sample the full candidate set.
    used_neighbor_pool = false
    if neighbor_scope_limit
        pool = neighbor_restricted_candidates(fg, ant.last_visited, candidates)
        used_neighbor_pool = !isempty(pool)
        if !used_neighbor_pool
            pool = candidates
        end
    else
        pool = candidates
    end

    # Soft bias: when exactly one side of S already has ≥θ, boost candidates on
    # the under-θ (smaller) side instead of hard-filtering the other side out.
    prefer_u = prefer_smaller_side ? prefer_smaller_side_prefer_u(ant.explored, θ) : nothing
    if ACO_TRACE && prefer_u !== nothing
        println("  "^depth, "  prefer_smaller_side: boost ",
                prefer_u ? "U" : "V", " ×", PREFER_SMALLER_SIDE_MULTIPLIER,
                "  (|U|=$(length(ant.explored.U)) |V|=$(length(ant.explored.V)) θ=$θ)")
    end

    if ACO_TRACE
        if !neighbor_scope_limit
            println("  "^depth, "  neighbor-scope-limit off → full |C|=$(length(candidates))")
            println("  "^depth, "  candidates=", _trace_degree_nodes(candidates))
        elseif used_neighbor_pool
            println("  "^depth, "  neighbor-pool |N∩C|=$(length(pool)) of |C|=$(length(candidates))",
                    "  last=", ant.last_visited.is_u ? "u" : "v", ant.last_visited.id)
            println("  "^depth, "  pool=", _trace_degree_nodes(pool))
        else
            println("  "^depth, "  neighbor-pool empty → full |C|=$(length(candidates))")
            println("  "^depth, "  candidates=", _trace_degree_nodes(candidates))
        end
    end

    exp_subgraph_vertex_count = 1 / (1 + exp(Subgraph.vertex_count(ant.explored)))

    score = node -> begin
        d = node_desirability(pheromones, fg, node, ant.species, exp_subgraph_vertex_count;
            eta_exp=eta_exp)
        if prefer_u !== nothing && node.is_u == prefer_u
            d *= PREFER_SMALLER_SIDE_MULTIPLIER
        end
        d
    end

    next_with_deg = linear_sample_one(score, pool)

    next = Node(next_with_deg)

    if ACO_TRACE
        # Expensive: score every candidate so we can see how peaked the distribution is.
        dmin = Inf
        dmax = -Inf
        nmin = next_with_deg
        nmax = next_with_deg
        n_at_max = 0

        desir = score(next_with_deg)

        for c in pool
            d = score(c)
            if d < dmin
                dmin = d
                nmin = c
            end
            if d > dmax
                dmax = d
                nmax = c
                n_at_max = 1
            elseif d == dmax
                n_at_max += 1
            end
        end
        ratio = dmin > 0 ? dmax / dmin : Inf
        println("  "^depth, "  desir_range min=$(round(dmin; digits=4))",
                " (", nmin.is_u ? "u" : "v", nmin.id, ",deg=$(nmin.deg))",
                "  max=$(round(dmax; digits=4))",
                " (", nmax.is_u ? "u" : "v", nmax.id, ",deg=$(nmax.deg))",
                "  max/min=$(round(ratio; digits=4))",
                "  n_at_max=$n_at_max/$(length(pool))")
        in_target = trace_target !== nothing &&
            ((next.is_u && (next.id in trace_target.U)) ||
             (!next.is_u && (next.id in trace_target.V)))
        println("  "^depth, "  choose ", next.is_u ? "u=" : "v=", next.id,
                "  deg=$(next_with_deg.deg)  desir=$(round(desir; digits=4))",
                in_target ? "  [target]" : (trace_target !== nothing ? "  [off-target]" : ""))
    end

    # Nondegree cost of adding into current S (before the mutation).
    n_opp = next_with_deg.is_u ? length(ant.explored.V) : length(ant.explored.U)
    cost = n_opp - next_with_deg.deg

    Subgraph.add_node!(ant.explored, fg, next.is_u, next.id)
    ant.missing = missing + cost
    ant.last_visited = next
    record_ant_addition!(ant, next)

    if ACO_TRACE
        true_missing = Subgraph.missing_edges(fg, ant.explored)
        ant.missing == true_missing ||
            error("ant missing drift: tracked=$(ant.missing) true=$true_missing")
    end

    reduce_candidates!(ant.candidates, fg, next_with_deg, k - ant.missing,
                       length(ant.explored.U), length(ant.explored.V))

    # The vertex just added can make every completion lose to the incumbent.
    # Leave it undeposited and drop the rest of this ant's trail on rollback.
    if prune_by_bound && bound_target !== nothing &&
        ant_bound_prunes(fg, ant, k, θ, bound_target)
        ant.bound_pruned = true
        if ACO_TRACE
            println("  "^depth, "-> STOP (upper bound cannot beat incumbent)")
        end
        return false
    end

    # Once the side just extended has θ vertices, hold it fixed, pack the other
    # side, then take every zero-cost vertex back on this side. The walk itself
    # is not rewritten; the best snapshot is kept on `ant.filled`.
    if greedy_fill === :step
        side_n = next.is_u ? length(ant.explored.U) : length(ant.explored.V)
        if side_n >= θ
            record_fill!(ant, fg, fill_keeping_side(fg, ant.explored, next.is_u, k), θ, fitness)
        end
    end

    if deposit_online
        dep = Float64(pheromone)
        if deposit_scale !== nothing
            dep *= Float64(deposit_scale(fg, ant.explored))
        end
        add_pheromone!(additions.species[ant.species], next, dep)
        add_pheromone!(additions.shared, next, dep * SHARED_PHEROMONE_FACTOR)
        prune_by_bound && push!(ant.deposits, (next.is_u, next.id, dep))
    end
    return true
end

# Desirability multiplier applied to candidates on the under-θ side.
const PREFER_SMALLER_SIDE_MULTIPLIER = 2.0

"""`:off`, `:end` (once, when the ant stops), or `:step` (after every qualifying add)."""
function normalize_greedy_fill(mode)::Symbol
    sym = mode isa Symbol ? mode : Symbol(lowercase(strip(String(mode))))
    sym in (:off, :end, :step) ||
        throw(ArgumentError("greedy_fill must be off, end, or step; got $mode"))
    return sym
end

"""
Vertices of the opposite side, cheapest missing-edge cost first, that fit in `k`.

Fix one side. Each opposite vertex costs `c = |fixed| − |N ∩ fixed|` and adds
`|fixed| − c` edges, and the vertices share only the defect budget. Taking them
in ascending `c`, skipping any with `c ≥ |fixed|` (zero edges), is the maximum-edge
packing: a more expensive vertex is never worth keeping ahead of a cheaper one.
Compact ids, the same numbering `reduce_candidates!` uses.
"""
function pack_opposite(fg::FrozenBipartite, fixed_is_u::Bool, fixed, k::Int)
    n_fixed = length(fixed)
    n_fixed == 0 && return Int[]
    n_same = fixed_is_u ? length(fg.u_ids) : length(fg.v_ids)
    in_fixed = falses(n_same)
    for id in fixed
        in_fixed[id] = true
    end
    n_opp = fixed_is_u ? length(fg.v_ids) : length(fg.u_ids)
    costs = Vector{Int}(undef, n_opp)
    if fixed_is_u
        @inbounds for vi in 1:n_opp
            hits = 0
            for t in neighbor_range_v(fg, vi)
                in_fixed[fg.u_adj[t]] && (hits += 1)
            end
            costs[vi] = n_fixed - hits
        end
    else
        @inbounds for ui in 1:n_opp
            hits = 0
            for t in neighbor_range_u(fg, ui)
                in_fixed[fg.v_adj[t]] && (hits += 1)
            end
            costs[ui] = n_fixed - hits
        end
    end
    order = sortperm(costs)
    chosen = Int[]
    sizehint!(chosen, n_opp)
    spent = 0
    for i in order
        c = costs[i]
        # Sorted ascending: once a vertex adds no edge, or does not fit, neither does any later one.
        c >= n_fixed && break
        spent + c > k && break
        spent += c
        push!(chosen, i)
    end
    return chosen
end

"""
Same-side vertices outside `fixed` that touch every vertex of `opposite`.

Each one adds `|opposite|` edges and no missing edges, so the opposite packing
stays within `k`.
"""
function free_same_side(fg::FrozenBipartite, fixed_is_u::Bool, fixed, opposite)
    n_opp = length(opposite)
    n_opp == 0 && return Int[]
    n_opp_side = fixed_is_u ? length(fg.v_ids) : length(fg.u_ids)
    in_opp = falses(n_opp_side)
    for id in opposite
        in_opp[id] = true
    end
    n_same = fixed_is_u ? length(fg.u_ids) : length(fg.v_ids)
    in_fixed = falses(n_same)
    for id in fixed
        in_fixed[id] = true
    end
    extras = Int[]
    if fixed_is_u
        @inbounds for ui in 1:n_same
            in_fixed[ui] && continue
            hits = 0
            for t in neighbor_range_u(fg, ui)
                in_opp[fg.v_adj[t]] && (hits += 1)
            end
            hits == n_opp && push!(extras, ui)
        end
    else
        @inbounds for vi in 1:n_same
            in_fixed[vi] && continue
            hits = 0
            for t in neighbor_range_v(fg, vi)
                in_opp[fg.u_adj[t]] && (hits += 1)
            end
            hits == n_opp && push!(extras, vi)
        end
    end
    return extras
end

"""
Keep `fixed_is_u`'s side of `S`, replace the other side with `pack_opposite`,
then add every zero-cost vertex on the kept side.
"""
function fill_keeping_side(fg::FrozenBipartite, S::SubGraph, fixed_is_u::Bool, k::Int)
    fixed = fixed_is_u ? S.U : S.V
    isempty(fixed) && return SubGraph()
    opposite = pack_opposite(fg, fixed_is_u, fixed, k)
    extras = free_same_side(fg, fixed_is_u, fixed, opposite)
    same = Set{Int}(fixed)
    for id in extras
        push!(same, id)
    end
    opp = Set{Int}(opposite)
    return fixed_is_u ? SubGraph(same, opp) : SubGraph(opp, same)
end

"""Remember `cand` when it beats the fill already stored on `ant`."""
function record_fill!(ant::Ant, fg::FrozenBipartite, cand::SubGraph, θ::Int, fitness)
    Subgraph.vertex_count(cand) == 0 && return ant
    if Subgraph.vertex_count(ant.filled) == 0 ||
       better_than_best(fg, cand, ant.filled, θ; fitness=fitness)
        ant.filled = cand
    end
    return ant
end

"""Both orientations of `fill_keeping_side`. The better one is recorded."""
function record_end_fill!(ant::Ant, fg::FrozenBipartite, k::Int, θ::Int, fitness)
    record_fill!(ant, fg, fill_keeping_side(fg, ant.explored, true, k), θ, fitness)
    record_fill!(ant, fg, fill_keeping_side(fg, ant.explored, false, k), θ, fitness)
    return ant
end

"""
Solution to score for this ant. The fill wins only when it beats the walk and
still contains every forced seed.
"""
function kept_ant_solution(ant::Ant, fg::FrozenBipartite, θ::Int, fitness,
    seed_compact::Union{Nothing,SubGraph}, assigned::Union{Nothing,SubGraph})
    walk = ant.explored
    Subgraph.vertex_count(ant.filled) == 0 && return walk
    if seed_compact !== nothing && !subgraph_has_seed(ant.filled, seed_compact)
        return walk
    end
    if assigned !== nothing && !subgraph_has_seed(ant.filled, assigned)
        return walk
    end
    better_than_best(fg, ant.filled, walk, θ; fitness=fitness) || return walk
    return ant.filled
end

"""
When exactly one side of `sg` has ≥θ nodes, return whether to prefer U
(`true`) or V (`false`). Otherwise return `nothing` (no side bias).
"""
function prefer_smaller_side_prefer_u(sg::SubGraph, θ::Int)::Union{Nothing,Bool}
    nU, nV = length(sg.U), length(sg.V)
    ((nU >= θ) == (nV >= θ)) && return nothing
    return nU < nV
end

function seed_ants_from_elites!(ants::Vector{Ant}, best_subgraphs::Vector{SubGraph},
    best_subgraph::SubGraph, best_scores::AbstractVector{<:Real}, n_seed::Int, n_remove::Int,
    fg::FrozenBipartite, k::Int)
    n_seed <= 0 && return
    Subgraph.vertex_count(best_subgraph) == 0 && return

    n = min(n_seed, length(ants))
    for i in 1:n
        s = ants[i].species
        elite = best_scores[s] > 0 && Subgraph.vertex_count(best_subgraphs[s]) > 0 ?
            best_subgraphs[s] : best_subgraph
        Subgraph.vertex_count(elite) == 0 && continue

        seeded = elite_seed_subgraph(elite, n_remove)
        last = if !isempty(seeded.U)
            Node(true, first(seeded.U))
        elseif !isempty(seeded.V)
            Node(false, first(seeded.V))
        else
            Node()
        end
        missing = Subgraph.missing_edges(fg, seeded)
        cands = candidate_set_with_nondegrees(fg, seeded, k)
        ants[i] = Ant(seeded, last, s, cands, missing,
            Tuple{Bool,Int}[], Dict{Int,Int}(), Tuple{Bool,Int,Float64}[], false,
            SubGraph())
    end
end

function elite_seed_subgraph(elite::SubGraph, n_remove::Int)
    sg = SubGraph(copy(elite.U), copy(elite.V))
    n_remove <= 0 && return sg

    nodes = Node[]
    sizehint!(nodes, Subgraph.vertex_count(sg))
    for u in sg.U
        push!(nodes, Node(true, u))
    end
    for v in sg.V
        push!(nodes, Node(false, v))
    end

    n_drop = min(n_remove, length(nodes))
    n_drop == 0 && return sg

    for node in sample(nodes, n_drop; replace=false)
        Subgraph.remove_node!(sg, node.is_u, node.id)
    end
    return sg
end
