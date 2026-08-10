mutable struct Ant
    explored::SubGraph
    last_visited::Node
    species::Int
    candidates::Vector{DegreeNode}
    missing::Int
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
    return [Ant(SubGraph(), Node(), mod1(i, num_subspecies), copy(template), 0)
            for i in 1:num_ants]
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

# Start and end index are inclusive
function advance_ants!(fg::FrozenBipartite, pheromones::ColonyPheromones, pheromone::Int,
    ants::Vector{Ant}, k::Int, θ::Int, ant_chunk; prefer_smaller_side::Bool=true,
    trace_target::Union{Nothing,SubGraph}=nothing)
    # Thread-local storage to prevent data races
    additions = zero_colony_pheromones(fg, length(pheromones.species))
    local_invalid_ants = Int[]
    
    for idx in ant_chunk
        # advance_ant! mutates the ant and adds to local `additions`
        if !advance_ant!(fg, pheromones, additions, pheromone, ants[idx], k, θ;
            prefer_smaller_side=prefer_smaller_side, ant_id=idx, trace_target=trace_target)
            push!(local_invalid_ants, idx)
        end
    end

    return additions, local_invalid_ants
end

"""
Desirability: τ · η²

η = d_S + d_G * exp(|S|)
"""
function node_desirability(pheromones::ColonyPheromones, fg::FrozenBipartite,
    node::DegreeNode, species::Int, exp_sg_vertex_count::Float64)
    τ = max(effective_pheromone(pheromones, Node(node.is_u, node.id), species), eps(Float64))
    deg_G = node.is_u ? degree_u(fg, node.id) : degree_v(fg, node.id)
    η = node.deg + deg_G * exp_sg_vertex_count
    return τ^3 * η^2
end

# Returns false if the ant has no further moves
function advance_ant!(fg::FrozenBipartite, pheromones::ColonyPheromones, additions::ColonyPheromones,
    pheromone::Int, ant::Ant, k::Int, θ::Int; prefer_smaller_side::Bool=true,
    ant_id::Int=0, trace_target::Union{Nothing,SubGraph}=nothing)
    depth = Subgraph.vertex_count(ant.explored)
    missing = ant.missing
    candidates = ant.candidates

    if TRACE
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

    if isempty(candidates)
        if TRACE
            msg = "  "^depth * "-> STOP (no candidates)"
            if trace_target !== nothing
                ou, ov = target_overlap(ant.explored, trace_target)
                msg *= "  final_target_hit=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))"
            end
            println(msg)
        end
        return false
    end

    sample_pool = candidates
    if prefer_smaller_side
        before_len = length(candidates)
        sample_pool = prefer_smaller_side_candidates(candidates, ant.explored, θ)
        if TRACE && length(sample_pool) < before_len
            filtered_target = 0
            if trace_target !== nothing
                for c in candidates
                    in_target = (c.is_u && (c.id in trace_target.U)) ||
                                (!c.is_u && (c.id in trace_target.V))
                    in_target || continue
                    kept = any(cand -> cand.is_u == c.is_u && cand.id == c.id, sample_pool)
                    kept || (filtered_target += 1)
                end
            end
            println("  "^depth, "  prefer_smaller_side: |C| $(before_len)→$(length(sample_pool))",
                    filtered_target > 0 ? "  !!! dropped $filtered_target target candidates" : "")
        end
    end

    if TRACE
        println("  "^depth, "  candidates=", _trace_degree_nodes(sample_pool))
    end

    exp_subgraph_vertex_count = 1 / (1 + exp(Subgraph.vertex_count(ant.explored)))

    next_with_deg = softmax_sample_one(
        node -> node_desirability(pheromones, fg, node, ant.species, exp_subgraph_vertex_count),
        sample_pool,
    )

    next = Node(next_with_deg)

    if TRACE
        # Expensive: score every candidate so we can see how peaked the distribution is.
        dmin = Inf
        dmax = -Inf
        nmin = next_with_deg
        nmax = next_with_deg
        n_at_max = 0

        desir = node_desirability(pheromones, fg, next_with_deg, ant.species, exp_subgraph_vertex_count)

        for c in sample_pool
            d = node_desirability(pheromones, fg, c, ant.species, ant.explored)
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
                "  n_at_max=$n_at_max/$(length(sample_pool))")
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

    if TRACE
        true_missing = Subgraph.missing_edges(fg, ant.explored)
        ant.missing == true_missing ||
            error("ant missing drift: tracked=$(ant.missing) true=$true_missing")
    end

    reduce_candidates!(ant.candidates, fg, next_with_deg, k - ant.missing,
                       length(ant.explored.U), length(ant.explored.V))

    add_pheromone!(additions.species[ant.species], next, pheromone)
    add_pheromone!(additions.shared, next, pheromone * SHARED_PHEROMONE_FACTOR)
    return true
end

function prefer_smaller_side_candidates(candidates::Vector{DegreeNode}, sg::SubGraph, θ::Int)
    nU, nV = length(sg.U), length(sg.V)
    (min(nU, nV) >= θ || nU == nV) && return candidates

    prefer_u = nU < nV
    filtered = DegreeNode[c for c in candidates if c.is_u == prefer_u]
    return isempty(filtered) ? candidates : filtered
end

function seed_ants_from_elites!(ants::Vector{Ant}, best_subgraphs::Vector{SubGraph},
    best_subgraph::SubGraph, best_scores::Vector{Int}, n_seed::Int, n_remove::Int,
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
        ants[i] = Ant(seeded, last, s, cands, missing)
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
