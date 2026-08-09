#= Ant colony optimization to find the maximum biclique =#

using Base.Threads

const __ACO_JL__ = false

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__TABU_JL__) || include("tabu.jl")

# Flip to true to dump per-step construction / pruning decisions (like opponent.jl).
# Best used with parallelize=false — ACO forces that when TRACE is on.
const TRACE = false

const ELITE_PHEROMONE_FACTOR = 2
const SHARED_PHEROMONE_FACTOR = 0.25

isdefined(@__MODULE__, :sorted_str) || (sorted_str(s::Set{Int}) = "{" * join(sort(collect(s)), ",") * "}")

"""Map an original-id subgraph into compact dense ids; report ids dropped by reduction (by Claude)."""
function compactify_subgraph(r::GraphRemapping, sg::SubGraph)
    u_map = Dict{Int,Int}(orig => i for (i, orig) in enumerate(r.u_original))
    v_map = Dict{Int,Int}(orig => i for (i, orig) in enumerate(r.v_original))
    dropped_U = sort(Int[u for u in sg.U if !haskey(u_map, u)])
    dropped_V = sort(Int[v for v in sg.V if !haskey(v_map, v)])
    kept = SubGraph(
        Set(u_map[u] for u in sg.U if haskey(u_map, u)),
        Set(v_map[v] for v in sg.V if haskey(v_map, v)),
    )
    return kept, dropped_U, dropped_V
end

function target_overlap(sg::SubGraph, target::SubGraph)
    return length(intersect(sg.U, target.U)), length(intersect(sg.V, target.V))
end

function _trace_degree_nodes(nodes::Vector{DegreeNode}; limit::Int=12)
    isempty(nodes) && return "{}"
    shown = nodes[1:min(limit, length(nodes))]
    parts = String[
        string(n.is_u ? "u" : "v", n.id, "(d=", n.deg, ")")
        for n in shown
    ]
    extra = length(nodes) > limit ? ",…(+$(length(nodes) - limit))" : ""
    return "{" * join(parts, ",") * extra * "}"
end

# Prune nodes with shared pheromone below threshold every this many colony
# iterations. 0 disables pruning.
# const PRUNE_EVERY = 10
const PRUNE_EVERY = 0

"""
Default shared-pheromone prune threshold:
`1 / (1 + exp(-ln(n_nodes) / num_ants))`.

Fewer ants => higher threshold => more aggressive pruning. More ants => threshold
"""
function default_prune_threshold(num_ants::Int, n_nodes::Int)
    num_ants >= 1 || throw(ArgumentError("num_ants must be >= 1, got $num_ants"))
    n = max(n_nodes, 1)
    return 1 / (1 + exp(-log(n) / num_ants))
end

struct Pheromones
    U::Vector{Float64}
    V::Vector{Float64}
end

Pheromones(nU::Int, nV::Int) = Pheromones(ones(nU), ones(nV))
Pheromones(fg::FrozenBipartite) = Pheromones(length(fg.u_ids), length(fg.v_ids))
zero_pheromones(fg::FrozenBipartite) = Pheromones(zeros(length(fg.u_ids)), zeros(length(fg.v_ids)))

get_pheromone(pheromones::Pheromones, node::Node) = (node.is_u ? pheromones.U : pheromones.V)[node.id]
add_pheromone!(pheromones::Pheromones, node::Node, pheromone::Real) = ((node.is_u ? pheromones.U : pheromones.V)[node.id] += pheromone)

function evaporate_pheromones!(pheromones::Pheromones, evaporation::Float64)
    pheromones.U .*= evaporation
    pheromones.V .*= evaporation
end

struct ColonyPheromones
    shared::Pheromones
    species::Vector{Pheromones}
end

function ColonyPheromones(fg::FrozenBipartite, num_subspecies::Int)
    ColonyPheromones(Pheromones(fg), [zero_pheromones(fg) for _ in 1:num_subspecies])
end

function zero_colony_pheromones(fg::FrozenBipartite, num_subspecies::Int)
    ColonyPheromones(zero_pheromones(fg), [zero_pheromones(fg) for _ in 1:num_subspecies])
end

function evaporate_pheromones!(colony::ColonyPheromones, evaporation::Float64)
    evaporate_pheromones!(colony.shared, evaporation)
    for species_pheromones in colony.species
        evaporate_pheromones!(species_pheromones, evaporation)
    end
end

function merge_pheromones!(colony::ColonyPheromones, additions::ColonyPheromones)
    colony.shared.U .+= additions.shared.U
    colony.shared.V .+= additions.shared.V
    for s in eachindex(colony.species)
        colony.species[s].U .+= additions.species[s].U
        colony.species[s].V .+= additions.species[s].V
    end
end

function effective_pheromone(colony::ColonyPheromones, node::Node, species::Int)
    τ = get_pheromone(colony.shared, node) + get_pheromone(colony.species[species], node)
    for s in eachindex(colony.species)
        s == species && continue
        τ -= get_pheromone(colony.species[s], node)
    end
    return τ
end

function slice_pheromones(p::Pheromones, keep_U::AbstractVector{Int}, keep_V::AbstractVector{Int})
    return Pheromones(p.U[keep_U], p.V[keep_V])
end

function slice_colony_pheromones(colony::ColonyPheromones, keep_U::AbstractVector{Int},
    keep_V::AbstractVector{Int})
    return ColonyPheromones(
        slice_pheromones(colony.shared, keep_U, keep_V),
        [slice_pheromones(sp, keep_U, keep_V) for sp in colony.species],
    )
end

function remap_subgraph_dense(sg::SubGraph, u_old_to_new::Dict{Int,Int}, v_old_to_new::Dict{Int,Int})
    return SubGraph(
        Set(u_old_to_new[u] for u in sg.U),
        Set(v_old_to_new[v] for v in sg.V),
    )
end

function prune_low_pheromone(fg::FrozenBipartite, remapping::GraphRemapping,
    colony::ColonyPheromones, threshold::Float64,
    best_subgraphs::Vector{SubGraph}, best_subgraph::SubGraph;
    iter::Int=0)

    nU = length(fg.u_ids)
    nV = length(fg.v_ids)

    protect_U = copy(best_subgraph.U)
    protect_V = copy(best_subgraph.V)
    for sg in best_subgraphs
        union!(protect_U, sg.U)
        union!(protect_V, sg.V)
    end

    keep_U = Int[i for i in eachindex(fg.u_ids)
                 if colony.shared.U[i] >= threshold || (i in protect_U)]
    keep_V = Int[i for i in eachindex(fg.v_ids)
                 if colony.shared.V[i] >= threshold || (i in protect_V)]

    # If either side would empty, refuse the prune (treat as nothing removed).
    if isempty(keep_U) || isempty(keep_V)
        removed_U, removed_V = 0, 0
        keep_U, keep_V = collect(1:nU), collect(1:nV)
    else
        removed_U = nU - length(keep_U)
        removed_V = nV - length(keep_V)
    end

    println("ACO prune iter=$iter: removed $removed_U U + $removed_V V " *
            "(|U|=$nU→$(nU - removed_U), |V|=$nV→$(nV - removed_V), τ=$(round(threshold; digits=4)))")

    if removed_U == 0 && removed_V == 0
        return fg, remapping, colony, best_subgraphs, best_subgraph
    end

    new_fg = induce_frozen(fg, keep_U, keep_V)
    new_remapping = GraphRemapping(remapping.u_original[keep_U], remapping.v_original[keep_V])
    new_colony = slice_colony_pheromones(colony, keep_U, keep_V)

    u_old_to_new = Dict{Int,Int}(old => new for (new, old) in enumerate(keep_U))
    v_old_to_new = Dict{Int,Int}(old => new for (new, old) in enumerate(keep_V))
    new_bests = [remap_subgraph_dense(sg, u_old_to_new, v_old_to_new) for sg in best_subgraphs]
    new_best = remap_subgraph_dense(best_subgraph, u_old_to_new, v_old_to_new)

    return new_fg, new_remapping, new_colony, new_bests, new_best
end

mutable struct Ant
    explored::SubGraph
    last_visited::Node
    species::Int
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

function seed_ants_from_elites!(ants::Vector{Ant}, best_subgraphs::Vector{SubGraph},
    best_subgraph::SubGraph, best_scores::Vector{Int}, n_seed::Int, n_remove::Int)
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
        ants[i] = Ant(seeded, last, s)
    end
end

function prefer_smaller_side_candidates(candidates::Vector{DegreeNode}, sg::SubGraph, θ::Int)
    nU, nV = length(sg.U), length(sg.V)
    (min(nU, nV) >= θ || nU == nV) && return candidates

    prefer_u = nU < nV
    filtered = DegreeNode[c for c in candidates if c.is_u == prefer_u]
    return isempty(filtered) ? candidates : filtered
end

function aco(g::BipartiteGraph, pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64, k::Int, θ::Int, num_subspecies::Int;
    parallelize::Bool=true, force_gc::Bool=false, iteration_callback=nothing,
    prune_every::Int=PRUNE_EVERY, pheromone_threshold::Union{Float64,Nothing}=nothing,
    tt::Int=2, tabu_patience::Int=3,
    prefer_smaller_side::Bool=true,
    elite_seed::Bool=true,
    elite_seed_ants::Int=3,
    elite_seed_remove::Int=2,
    trace_target::Union{Nothing,SubGraph}=nothing)
    num_subspecies >= 1 || throw(ArgumentError("num_subspecies must be >= 1, got $num_subspecies"))
    prune_every >= 0 || throw(ArgumentError("prune_every must be >= 0, got $prune_every"))
    elite_seed_ants >= 0 || throw(ArgumentError("elite_seed_ants must be >= 0, got $elite_seed_ants"))
    elite_seed_remove >= 0 || throw(ArgumentError("elite_seed_remove must be >= 0, got $elite_seed_remove"))

    if TRACE && parallelize
        println("ACO TRACE: forcing parallelize=false so step logs stay readable")
        parallelize = false
    end

    apply_graph_reductions!(g, k, θ, nothing, nothing, true, ReductionMode.all_reductions)
    
    fg = freeze(g)

    println(length(fg.u_ids), " ", length(fg.v_ids))

    compact_fg, remapping = compact_frozen(fg)
    pheromones = ColonyPheromones(compact_fg, num_subspecies)

    # Compact-space watch target (optional). Dropped nodes mean reduction already
    # removed part of the known optimum — ACO can never recover those.
    target_compact::Union{Nothing,SubGraph} = nothing
    if TRACE && trace_target !== nothing
        target_compact, dropped_U, dropped_V = compactify_subgraph(remapping, trace_target)
        println("ACO TRACE target: original |U|=$(length(trace_target.U)) |V|=$(length(trace_target.V)) → " *
                "compact |U|=$(length(target_compact.U)) |V|=$(length(target_compact.V))")
        println("  compact U=", sorted_str(target_compact.U), " V=", sorted_str(target_compact.V))
        if !isempty(dropped_U) || !isempty(dropped_V)
            println("  !!! reduction dropped target nodes before search: " *
                    "U=$(dropped_U) V=$(dropped_V)")
        end
    end

    ants = [Ant(SubGraph(), Node(), mod1(i, num_subspecies)) for i in 1:num_ants]
    invalid_ants = Set{Int}()

    best_scores = fill(0, num_subspecies)
    best_subgraphs = [SubGraph() for _ in 1:num_subspecies]
    best_score::Int = 0
    best_subgraph::SubGraph = SubGraph()

    explored_ants = [ant for ant in ants]

    for iter in 1:num_iterations
        TRACE && println("==== ACO iter $iter/$num_iterations  best_score=$best_score " *
                         "|U|=$(length(best_subgraph.U)) |V|=$(length(best_subgraph.V)) ====")

        if elite_seed
            seed_ants_from_elites!(ants, best_subgraphs, best_subgraph, best_scores,
                elite_seed_ants, elite_seed_remove)
            if TRACE
                for i in 1:min(elite_seed_ants, length(ants))
                    sg = ants[i].explored
                    msg = "  seed ant=$i species=$(ants[i].species) " *
                          "U=$(sorted_str(sg.U)) V=$(sorted_str(sg.V))"
                    if target_compact !== nothing
                        ou, ov = target_overlap(sg, target_compact)
                        msg *= "  target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))"
                    end
                    println(msg)
                end
            end
        end

        # Track which ants are still exploring
        active_ants = collect(1:num_ants)

        while !isempty(active_ants)
            if parallelize
                # Divide the remaining ants evenly across available threads
                chunk_size = max(1, cld(length(active_ants), nthreads()))
                
                # 1. Spawn tasks for each chunk
                tasks = map(Iterators.partition(active_ants, chunk_size)) do chunk
                    Threads.@spawn advance_ants!(compact_fg, pheromones, pheromone, ants, k, θ, chunk;
                        prefer_smaller_side=prefer_smaller_side, trace_target=target_compact)
                end
                
                # 2. Wait for all threads to finish their step
                results = fetch.(tasks)
            else
                # Process all active ants sequentially on the main thread.
                # Wrapped in an array to match the structure of multithreaded `results`.
                results = [advance_ants!(compact_fg, pheromones, pheromone, ants, k, θ, active_ants;
                    prefer_smaller_side=prefer_smaller_side, trace_target=target_compact)]
            end
            
            # 3. Safely merge the results on the main thread
            for (local_additions, local_invalids) in results
                # Merge the local additions into the global pheromone tracker
                merge_pheromones!(pheromones, local_additions)
                
                # Remove dead ants from the active pool for the next iteration
                setdiff!(active_ants, local_invalids)
            end
        end

        evaporate_pheromones!(pheromones, evaporation)

        # Softmax-select a few finished ants by instance fitness, tabu-repair
        # them, then reinforce every node in the repaired subgraph.
        for s in 1:num_subspecies
            eligible = [ant for ant in ants if ant.last_visited.id != -1 && ant.species == s]
            n_elite = min(3, length(eligible))
            n_elite == 0 && continue

            elites = softmax_sample(
                ant -> instance_fitness(compact_fg, ant.explored, θ),
                eligible,
                n_elite,
            )
            for ant in elites
                pre_score = instance_fitness(compact_fg, ant.explored, θ)
                TRACE && println("  elite species=$s pre-repair score=$pre_score " *
                                 "U=$(sorted_str(ant.explored.U)) V=$(sorted_str(ant.explored.V))")
                tabu_repair!(compact_fg, ant.explored, k, θ, tt, tabu_patience)
                if TRACE
                    post_score = instance_fitness(compact_fg, ant.explored, θ)
                    msg = "  elite species=$s post-repair score=$post_score " *
                          "U=$(sorted_str(ant.explored.U)) V=$(sorted_str(ant.explored.V))"
                    if target_compact !== nothing
                        ou, ov = target_overlap(ant.explored, target_compact)
                        msg *= "  target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))"
                    end
                    println(msg)
                end
                for u in ant.explored.U
                    node = Node(true, u)
                    add_pheromone!(pheromones.species[s], node, pheromone * ELITE_PHEROMONE_FACTOR)
                    add_pheromone!(pheromones.shared, node, pheromone * ELITE_PHEROMONE_FACTOR * SHARED_PHEROMONE_FACTOR)
                end
                for v in ant.explored.V
                    node = Node(false, v)
                    add_pheromone!(pheromones.species[s], node, pheromone * ELITE_PHEROMONE_FACTOR)
                    add_pheromone!(pheromones.shared, node, pheromone * ELITE_PHEROMONE_FACTOR * SHARED_PHEROMONE_FACTOR)
                end
            end
        end

        for ant in ants
            score = instance_fitness(compact_fg, ant.explored, θ)
            s = ant.species
            if score > best_scores[s]
                best_scores[s] = score
                # Copy so later colony mutations / seeds never alias the stored best.
                best_subgraphs[s] = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
            end
            if score > best_score
                best_score = score
                best_subgraph = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
                if TRACE
                    msg = "  NEW BEST score=$best_score U=$(sorted_str(best_subgraph.U)) V=$(sorted_str(best_subgraph.V))"
                    if target_compact !== nothing
                        ou, ov = target_overlap(best_subgraph, target_compact)
                        msg *= "  target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))"
                    end
                    println(msg)
                end
            end
        end

        if TRACE && target_compact !== nothing
            ou, ov = target_overlap(best_subgraph, target_compact)
            println("  iter=$iter end: best_score=$best_score " *
                    "target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))")
        end

        if prune_every > 0 && iter % prune_every == 0 && iter < num_iterations
            n_nodes = length(compact_fg.u_ids) + length(compact_fg.v_ids)
            τ = pheromone_threshold === nothing ?
                default_prune_threshold(num_ants, n_nodes) : pheromone_threshold
            if TRACE && target_compact !== nothing
                # Capture pre-prune keep sets so we can see if the target is culled.
                protect_U = copy(best_subgraph.U)
                protect_V = copy(best_subgraph.V)
                for sg in best_subgraphs
                    union!(protect_U, sg.U)
                    union!(protect_V, sg.V)
                end
                lost_U = sort(Int[u for u in target_compact.U
                                  if pheromones.shared.U[u] < τ && !(u in protect_U)])
                lost_V = sort(Int[v for v in target_compact.V
                                  if pheromones.shared.V[v] < τ && !(v in protect_V)])
                if !isempty(lost_U) || !isempty(lost_V)
                    println("  !!! pheromone prune will drop target nodes: U=$lost_U V=$lost_V")
                end
            end
            compact_fg, remapping, pheromones, best_subgraphs, best_subgraph =
                prune_low_pheromone(compact_fg, remapping, pheromones, τ,
                                    best_subgraphs, best_subgraph; iter=iter)
            if target_compact !== nothing
                # Dense ids may have changed; rebuild target against new remapping
                # from the original-id watch set when available.
                if trace_target !== nothing
                    target_compact, _, _ = compactify_subgraph(remapping, trace_target)
                end
            end
        end

        invalid_ants = Set{Int}()
        ants = [Ant(SubGraph(), Node(), mod1(i, num_subspecies)) for i in 1:num_ants]

        force_gc && GC.gc()

        if iteration_callback !== nothing
            iteration_callback(iter, best_subgraph, compact_fg, remapping) || break
        end
    end

    remapped = [remap_subgraph(remapping, best_subgraphs[s]) for s in 1:num_subspecies]
    for s in 1:num_subspecies
        sg = remapped[s]
        println("ACO subspecies $s best: |U|=$(length(sg.U)) |V|=$(length(sg.V)) score=$(best_scores[s])")
        if TRACE && trace_target !== nothing
            ou = length(intersect(sg.U, trace_target.U))
            ov = length(intersect(sg.V, trace_target.V))
            println("  target_hit (original ids)=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))")
        end
    end
    return remapped
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
Desirability: τ³ · η²

η = d_S + d_G / (1 + |S|)
"""
function node_desirability(pheromones::ColonyPheromones, fg::FrozenBipartite,
    node::DegreeNode, species::Int, sg::SubGraph)
    τ = max(effective_pheromone(pheromones, Node(node.is_u, node.id), species), eps(Float64))
    deg_G = node.is_u ? degree_u(fg, node.id) : degree_v(fg, node.id)
    η = node.deg + deg_G / (1 + exp(-Subgraph.vertex_count(sg)))
    # η = node.deg + deg_G / (1 + Subgraph.vertex_count(sg))
    return τ^3 * η^2
end

# Returns false if the ant has no further moves
function advance_ant!(fg::FrozenBipartite, pheromones::ColonyPheromones, additions::ColonyPheromones,
    pheromone::Int, ant::Ant, k::Int, θ::Int; prefer_smaller_side::Bool=true,
    ant_id::Int=0, trace_target::Union{Nothing,SubGraph}=nothing)
    # Room for algorithmic improvement: have ants instead just start off of where
    # they are instead of looking across the entire graph

    depth = Subgraph.vertex_count(ant.explored)
    missing = Subgraph.missing_edges(fg, ant.explored)
    budget = k - missing
    candidates = candidate_set_with_nondegrees(fg, ant.explored, k)

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

    if length(candidates) == 0
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

    if prefer_smaller_side
        before = candidates
        candidates = prefer_smaller_side_candidates(candidates, ant.explored, θ)
        if TRACE && length(candidates) < length(before)
            filtered_target = 0
            if trace_target !== nothing
                    for c in before
                    in_target = (c.is_u && (c.id in trace_target.U)) ||
                                (!c.is_u && (c.id in trace_target.V))
                    in_target || continue
                    kept = any(cand -> cand.is_u == c.is_u && cand.id == c.id, candidates)
                    kept || (filtered_target += 1)
                end
            end
            println("  "^depth, "  prefer_smaller_side: |C| $(length(before))→$(length(candidates))",
                    filtered_target > 0 ? "  !!! dropped $filtered_target target candidates" : "")
        end
    end

    if TRACE
        println("  "^depth, "  candidates=", _trace_degree_nodes(candidates))
    end

    next_with_deg = softmax_sample(
        node -> node_desirability(pheromones, fg, node, ant.species, ant.explored),
        candidates,
        1,
    )[1]

    next = Node(next_with_deg)
    desir = node_desirability(pheromones, fg, next_with_deg, ant.species, ant.explored)

    if TRACE
        # Expensive: score every candidate so we can see how peaked the distribution is.
        dmin = Inf
        dmax = -Inf
        nmin = next_with_deg
        nmax = next_with_deg
        n_at_max = 0
        for c in candidates
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
                "  n_at_max=$n_at_max/$(length(candidates))")
        in_target = trace_target !== nothing &&
            ((next.is_u && (next.id in trace_target.U)) ||
             (!next.is_u && (next.id in trace_target.V)))
        println("  "^depth, "  choose ", next.is_u ? "u=" : "v=", next.id,
                "  deg=$(next_with_deg.deg)  desir=$(round(desir; digits=4))",
                in_target ? "  [target]" : (trace_target !== nothing ? "  [off-target]" : ""))
    end

    Subgraph.add_node!(ant.explored, fg, next.is_u, next.id)
    
    ant.last_visited = next

    add_pheromone!(additions.species[ant.species], next, pheromone)
    add_pheromone!(additions.shared, next, pheromone * SHARED_PHEROMONE_FACTOR)
    return true
end
