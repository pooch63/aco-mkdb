#=
Edge-ACO construction step: same ant / candidate machinery as `src/aco/advance.jl`,
but desirability and deposit use CSR edge trails (`ColonyEdgePheromones`).
=#

# Reuse SHARED_PHEROMONE_FACTOR / PREFER_SMALLER_SIDE_MULTIPLIER from node ACO
# (included first via method.jl / algorithm.jl).

function edge_advance_ants!(fg::FrozenBipartite, pheromones::ColonyEdgePheromones, pheromone::Int,
    ants::Vector{Ant}, k::Int, θ::Int, ant_chunk; prefer_smaller_side::Bool=true,
    neighbor_scope_limit::Bool=true,
    trace_target::Union{Nothing,SubGraph}=nothing)
    additions = zero_colony_edge_pheromones(fg, length(pheromones.species))
    local_invalid_ants = Int[]

    for idx in ant_chunk
        if !edge_advance_ant!(fg, pheromones, additions, pheromone, ants[idx], k, θ;
            prefer_smaller_side=prefer_smaller_side,
            neighbor_scope_limit=neighbor_scope_limit,
            ant_id=idx, trace_target=trace_target)
            push!(local_invalid_ants, idx)
        end
    end

    return additions, local_invalid_ants
end

"""
Desirability: τ_edge · η

τ_edge = mean effective pheromone on edges from the candidate into the opposite
side of S (1.0 when that set is empty / no graph edges).

η = d_S + d_G · σ(-|S|)  (same as node ACO)
"""
function edge_node_desirability(pheromones::ColonyEdgePheromones, fg::FrozenBipartite,
    node::DegreeNode, species::Int, exp_sg_vertex_count::Float64, sg::SubGraph)
    τ = mean_edge_pheromone_into(pheromones, fg, node, species, sg)
    deg_G = node.is_u ? degree_u(fg, node.id) : degree_v(fg, node.id)
    η = node.deg + deg_G * exp_sg_vertex_count
    return τ * η
end

# Returns false if the ant has no further moves
function edge_advance_ant!(fg::FrozenBipartite, pheromones::ColonyEdgePheromones,
    additions::ColonyEdgePheromones, pheromone::Int, ant::Ant, k::Int, θ::Int;
    prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true,
    ant_id::Int=0, trace_target::Union{Nothing,SubGraph}=nothing)
    depth = Subgraph.vertex_count(ant.explored)
    missing = ant.missing
    candidates = ant.candidates

    if EDGES_TRACE
        println("  "^depth, "edge-ant=$ant_id species=$(ant.species) depth=$depth  ",
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

    if isempty(candidates)
        if EDGES_TRACE
            msg = "  "^depth * "-> STOP (no candidates)"
            if trace_target !== nothing
                ou, ov = target_overlap(ant.explored, trace_target)
                msg *= "  final_target_hit=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))"
            end
            println(msg)
        end
        return false
    end

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

    prefer_u = prefer_smaller_side ? prefer_smaller_side_prefer_u(ant.explored, θ) : nothing
    if EDGES_TRACE && prefer_u !== nothing
        println("  "^depth, "  prefer_smaller_side: boost ",
                prefer_u ? "U" : "V", " ×", PREFER_SMALLER_SIDE_MULTIPLIER,
                "  (|U|=$(length(ant.explored.U)) |V|=$(length(ant.explored.V)) θ=$θ)")
    end

    if EDGES_TRACE
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
        d = edge_node_desirability(pheromones, fg, node, ant.species,
            exp_subgraph_vertex_count, ant.explored)
        if prefer_u !== nothing && node.is_u == prefer_u
            d *= PREFER_SMALLER_SIDE_MULTIPLIER
        end
        d
    end

    next_with_deg = linear_sample_one(score, pool)
    next = Node(next_with_deg)

    if EDGES_TRACE
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

    n_opp = next_with_deg.is_u ? length(ant.explored.V) : length(ant.explored.U)
    cost = n_opp - next_with_deg.deg

    # Deposit on cross edges into S *before* mutating (same opposite side after add).
    deposit_edges_into!(additions.species[ant.species], fg, next, ant.explored, pheromone)
    deposit_edges_into!(additions.shared, fg, next, ant.explored,
        pheromone * SHARED_PHEROMONE_FACTOR)

    Subgraph.add_node!(ant.explored, fg, next.is_u, next.id)
    ant.missing = missing + cost
    ant.last_visited = next
    record_ant_addition!(ant, next)

    if EDGES_TRACE
        true_missing = Subgraph.missing_edges(fg, ant.explored)
        ant.missing == true_missing ||
            error("edge-ant missing drift: tracked=$(ant.missing) true=$true_missing")
    end

    reduce_candidates!(ant.candidates, fg, next_with_deg, k - ant.missing,
                       length(ant.explored.U), length(ant.explored.V))

    return true
end
