#= Ant colony optimization to find the maximum biclique =#

using Base.Threads

const __ACO_JL__ = true

include("pheromone.jl")
include("advance.jl")

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

# MAX-MIN Ant System (MMAS) bounds on each species pheromone map only.
# Shared pheromone is left unbounded. Linear selection / single-ant deposit
# are intentionally not part of this change.

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

function aco(g::BipartiteGraph, pheromone::Int, num_ants::Int, num_iterations::Int, evaporation::Float64, k::Int, θ::Int, num_subspecies::Int;
    parallelize::Bool=true, force_gc::Bool=false, iteration_callback=nothing,
    pheromone_min::Union{Float64,Nothing}=nothing,
    pheromone_max::Union{Float64,Nothing}=nothing,
    tt::Int=2, tabu_patience::Int=3,
    prefer_smaller_side::Bool=true,
    elite_seed::Bool=true,
    elite_seed_ants::Int=3,
    elite_seed_remove::Int=2,
    trace_target::Union{Nothing,SubGraph}=nothing)
    num_subspecies >= 1 || throw(ArgumentError("num_subspecies must be >= 1, got $num_subspecies"))
    elite_seed_ants >= 0 || throw(ArgumentError("elite_seed_ants must be >= 0, got $elite_seed_ants"))
    elite_seed_remove >= 0 || throw(ArgumentError("elite_seed_remove must be >= 0, got $elite_seed_remove"))

    if TRACE && parallelize
        println("ACO TRACE: forcing parallelize=false so step logs stay readable")
        parallelize = false
    end

    apply_graph_reductions!(g, k, θ, nothing, nothing, true, ReductionMode.all_reductions)
    
    fg = freeze(g)

    println("Size of reduced graphs", length(fg.u_ids), " ", length(fg.v_ids))
    println(2955 in fg.u_ids, 42570 in fg.v_ids)

    compact_fg, remapping = compact_frozen(fg)
    pheromones = ColonyPheromones(compact_fg, num_subspecies)

    best_scores = fill(0, num_subspecies)
    best_subgraphs = [SubGraph() for _ in 1:num_subspecies]
    best_score::Int = 0
    best_subgraph::SubGraph = SubGraph()

    # Seed MMAS τ_max from the θ-heuristic solution size so the ceiling starts
    # higher than the empty-best default (best_n=2).
    heuristic_sg = theta_based_heuristic(compact_fg, k, θ; return_invalid=true)
    if Subgraph.vertex_count(heuristic_sg) > 0
        heuristic_score = instance_fitness(compact_fg, heuristic_sg, θ)
        best_subgraph = SubGraph(copy(heuristic_sg.U), copy(heuristic_sg.V))
        best_score = heuristic_score
        for s in 1:num_subspecies
            best_subgraphs[s] = SubGraph(copy(heuristic_sg.U), copy(heuristic_sg.V))
            best_scores[s] = heuristic_score
        end
        println("ACO θ-heuristic seed: |U|=$(length(heuristic_sg.U)) |V|=$(length(heuristic_sg.V)) " *
                "score=$heuristic_score vertices=$(Subgraph.vertex_count(heuristic_sg))")
    end

    n_nodes = length(compact_fg.u_ids) + length(compact_fg.v_ids)
    τ_mins = Vector{Float64}(undef, num_subspecies)
    τ_maxs = Vector{Float64}(undef, num_subspecies)
    update_species_pheromone_bounds!(τ_mins, τ_maxs, pheromone, n_nodes, evaporation, best_subgraphs;
        pheromone_min=pheromone_min, pheromone_max=pheromone_max)
    clamp_species_pheromones!(pheromones, τ_mins, τ_maxs)
    println("ACO MMAS species bounds (init): " *
            join(["s$s=[$(round(τ_mins[s]; digits=6)),$(round(τ_maxs[s]; digits=6))]"
                  for s in 1:num_subspecies], " "))
    # println("ACO MMAS species bounds: τ_min=$(round(τ_min; digits=6)) τ_max=$(round(τ_max; digits=6))")
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

    ants = new_ants(compact_fg, num_ants, num_subspecies)
    invalid_ants = Set{Int}()

    explored_ants = [ant for ant in ants]

    for iter in 1:num_iterations
        TRACE && println("==== ACO iter $iter/$num_iterations  best_score=$best_score " *
                         "|U|=$(length(best_subgraph.U)) |V|=$(length(best_subgraph.V)) ====")

        if elite_seed
            seed_ants_from_elites!(ants, best_subgraphs, best_subgraph, best_scores,
                elite_seed_ants, elite_seed_remove, compact_fg, k)
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
                # Cap species trails during construction so mid-iter selection stays bounded.
                clamp_species_pheromones!(pheromones, τ_mins, τ_maxs)
                
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
                TRACE && println("  elite species=$s pre-repair score=$(instance_fitness(compact_fg, ant.explored, θ)) " *
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
                tabu = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
                tabu_repair!(compact_fg, tabu, k, θ, tt, tabu_patience)
                tabu_score = instance_fitness(compact_fg, tabu, θ)

                best_score = max(score, tabu_score)
                best_subgraph = score > tabu_score ? SubGraph(copy(ant.explored.U), copy(ant.explored.V)) : tabu
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

        # Refresh MMAS bounds from each subspecies' best, then floor/cap trails.
        n_nodes = length(compact_fg.u_ids) + length(compact_fg.v_ids)
        update_species_pheromone_bounds!(τ_mins, τ_maxs, pheromone, n_nodes, evaporation, best_subgraphs;
            pheromone_min=pheromone_min, pheromone_max=pheromone_max)
        clamp_species_pheromones!(pheromones, τ_mins, τ_maxs)

        if TRACE && target_compact !== nothing
            ou, ov = target_overlap(best_subgraph, target_compact)
            println("  iter=$iter end: best_score=$best_score " *
                    "target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))")
        end

        invalid_ants = Set{Int}()
        ants = new_ants(compact_fg, num_ants, num_subspecies)

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
