#= Edge-trail ant colony optimization for maximum k-defective biclique =#

using Base.Threads

const __EDGES_ACO_JL__ = true

# Node ACO must be loaded first (Ant helpers, better_than_best, SHARED_PHEROMONE_FACTOR, …).
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath(@__DIR__, "..", "aco", "algorithm.jl"))

include("pheromone.jl")
include("advance.jl")

# Flip to true to dump per-step construction decisions (forces parallelize=false).
const EDGES_TRACE = false

"""
Edge-ACO: same colony loop as `aco`, but pheromone lives on graph edges.

Desirability of a candidate uses the mean edge-τ into the opposite side of S;
deposit reinforces those cross edges (and elite deposit reinforces the induced
edge set of elite subgraphs).
"""
function edges_aco(g::BipartiteGraph, pheromone::Int, num_ants::Int, num_iterations::Int,
    evaporation::Float64, k::Int, θ::Int, num_subspecies::Int;
    parallelize::Bool=true, force_gc::Bool=false, iteration_callback=nothing,
    pheromone_min::Union{Float64,Nothing}=nothing,
    pheromone_max::Union{Float64,Nothing}=nothing,
    tt::Int=2, tabu_patience::Int=3,
    prefer_smaller_side::Bool=false,
    neighbor_scope_limit::Bool=true,
    elite_seed::Bool=true,
    elite_seed_ants::Int=3,
    elite_seed_remove::Int=2,
    elite_pheromone::Bool=false,
    aco_tabu::Bool=false,
    mmas::Bool=false,
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    trace_target::Union{Nothing,SubGraph}=nothing,
    seed_nodes::Union{Nothing,SubGraph}=SubGraph(Set(), Set()),
    construction_stats::Union{Nothing,Base.RefValue}=nothing)
    num_subspecies >= 1 || throw(ArgumentError("num_subspecies must be >= 1, got $num_subspecies"))
    elite_seed_ants >= 0 || throw(ArgumentError("elite_seed_ants must be >= 0, got $elite_seed_ants"))
    elite_seed_remove >= 0 || throw(ArgumentError("elite_seed_remove must be >= 0, got $elite_seed_remove"))

    if EDGES_TRACE && parallelize
        println("edges-ACO EDGES_TRACE: forcing parallelize=false so step logs stay readable")
        parallelize = false
    end

    apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)

    fg = freeze(g)

    println("Prefer smaller side: $prefer_smaller_side")
    println("Neighbor scope limit: $neighbor_scope_limit")
    println("Size of reduced graphs", length(fg.u_ids), " ", length(fg.v_ids))

    compact_fg, remapping = compact_frozen(fg)
    pheromones = ColonyEdgePheromones(compact_fg, num_subspecies)
    n_edges = length(compact_fg.v_adj)
    println("edges-ACO trails on $n_edges CSR edges (not vertices)")

    seed_compact::Union{Nothing,SubGraph} = nothing
    if seed_nodes !== nothing && Subgraph.vertex_count(seed_nodes) > 0
        seed_compact, dropped_U, dropped_V = compactify_subgraph(remapping, seed_nodes)
        println("edges-ACO seed_nodes: original U=$(sorted_str(seed_nodes.U)) V=$(sorted_str(seed_nodes.V)) → " *
                "compact U=$(sorted_str(seed_compact.U)) V=$(sorted_str(seed_compact.V))")
        if !isempty(dropped_U) || !isempty(dropped_V)
            println("  seed nodes dropped by reduction: U=$dropped_U V=$dropped_V")
        end
        if Subgraph.vertex_count(seed_compact) == 0
            seed_compact = nothing
        end
    end

    best_scores = fill(0, num_subspecies)
    best_subgraphs = [SubGraph() for _ in 1:num_subspecies]
    best_iterations = fill(0, num_subspecies)
    best_times = fill(0.0, num_subspecies)
    best_score::Int = 0
    best_subgraph::SubGraph = SubGraph()
    best_iteration::Int = 0
    best_time::Float64 = 0.0

    if seed_compact !== nothing
        seed_score = instance_fitness(compact_fg, seed_compact, θ)
        best_subgraph = SubGraph(copy(seed_compact.U), copy(seed_compact.V))
        best_score = seed_score
        best_iteration = 0
        best_time = 0.0
        for s in 1:num_subspecies
            best_subgraphs[s] = SubGraph(copy(seed_compact.U), copy(seed_compact.V))
            best_scores[s] = seed_score
            best_iterations[s] = 0
            best_times[s] = 0.0
        end
        println("edges-ACO forced-seed incumbent: U=$(sorted_str(best_subgraph.U)) " *
                "V=$(sorted_str(best_subgraph.V)) score=$seed_score")
    end

    τ_mins = Vector{Float64}(undef, num_subspecies)
    τ_maxs = Vector{Float64}(undef, num_subspecies)
    if mmas
        mmas_init = [SubGraph() for _ in 1:num_subspecies]
        heuristic_sg = theta_based_heuristic(compact_fg, k, θ; return_invalid=true)
        if Subgraph.vertex_count(heuristic_sg) > 0 &&
           (seed_compact === nothing || subgraph_has_seed(heuristic_sg, seed_compact))
            heuristic_score = instance_fitness(compact_fg, heuristic_sg, θ)
            for s in 1:num_subspecies
                mmas_init[s] = SubGraph(copy(heuristic_sg.U), copy(heuristic_sg.V))
            end
            println("edges-ACO θ-heuristic → MMAS init only: |U|=$(length(heuristic_sg.U)) " *
                    "|V|=$(length(heuristic_sg.V)) score=$heuristic_score " *
                    "edges=$(Subgraph.edge_count(compact_fg, heuristic_sg))")
        elseif Subgraph.vertex_count(heuristic_sg) > 0
            println("edges-ACO θ-heuristic skipped for MMAS init (|U|=$(length(heuristic_sg.U)) " *
                    "|V|=$(length(heuristic_sg.V))): missing forced seed_nodes")
        end
        update_species_edge_pheromone_bounds!(τ_mins, τ_maxs, pheromone, n_edges, evaporation,
            mmas_init, compact_fg; pheromone_min=pheromone_min, pheromone_max=pheromone_max)
        clamp_species_edge_pheromones!(pheromones, τ_mins, τ_maxs)
        println("edges-ACO MMAS species bounds (init): " *
                join(["s$s=[$(round(τ_mins[s]; digits=6)),$(round(τ_maxs[s]; digits=6))]"
                      for s in 1:num_subspecies], " "))
    else
        fill!(τ_mins, 0.0)
        fill!(τ_maxs, Inf)
        println("edges-ACO MMAS disabled: edge pheromone unbounded [0, +∞) " *
                "(θ-heuristic not run for edges-ACO)")
    end
    println("edges-ACO ablations: elite_pheromone=$elite_pheromone tabu=$aco_tabu mmas=$mmas")

    target_compact::Union{Nothing,SubGraph} = nothing
    if EDGES_TRACE && trace_target !== nothing
        target_compact, dropped_U, dropped_V = compactify_subgraph(remapping, trace_target)
        println("edges-ACO EDGES_TRACE target: original |U|=$(length(trace_target.U)) |V|=$(length(trace_target.V)) → " *
                "compact |U|=$(length(target_compact.U)) |V|=$(length(target_compact.V))")
        println("  compact U=", sorted_str(target_compact.U), " V=", sorted_str(target_compact.V))
        if !isempty(dropped_U) || !isempty(dropped_V)
            println("  !!! reduction dropped target nodes before search: " *
                    "U=$(dropped_U) V=$(dropped_V)")
        end
    end

    ants = new_ants(compact_fg, num_ants, num_subspecies)
    seed_compact !== nothing && seed_ants_with_subgraph!(ants, compact_fg, k, seed_compact)

    pooled_missing_at_size = Dict{Int, Vector{Int}}()
    last_iteration_orders = Vector{Tuple{Bool,Int}}[]

    t0 = time_ns()

    for iter in 1:num_iterations
        EDGES_TRACE && println("==== edges-ACO iter $iter/$num_iterations  best_score=$best_score " *
                         "|U|=$(length(best_subgraph.U)) |V|=$(length(best_subgraph.V)) ====")

        if elite_seed && seed_compact === nothing
            seed_ants_from_elites!(ants, best_subgraphs, best_subgraph, best_scores,
                elite_seed_ants, elite_seed_remove, compact_fg, k)
            if EDGES_TRACE
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
        elseif elite_seed && seed_compact !== nothing && iter == 1
            println("edges-ACO elite_seed disabled while seed_nodes forced-inclusion is active")
        end

        active_ants = collect(1:num_ants)

        while !isempty(active_ants)
            if parallelize
                chunk_size = max(1, cld(length(active_ants), nthreads()))
                tasks = map(Iterators.partition(active_ants, chunk_size)) do chunk
                    Threads.@spawn edge_advance_ants!(compact_fg, pheromones, pheromone, ants, k, θ, chunk;
                        prefer_smaller_side=prefer_smaller_side,
                        neighbor_scope_limit=neighbor_scope_limit,
                        trace_target=target_compact)
                end
                results = fetch.(tasks)
            else
                results = [edge_advance_ants!(compact_fg, pheromones, pheromone, ants, k, θ, active_ants;
                    prefer_smaller_side=prefer_smaller_side,
                    neighbor_scope_limit=neighbor_scope_limit,
                    trace_target=target_compact)]
            end

            for (local_additions, local_invalids) in results
                merge_edge_pheromones!(pheromones, local_additions)
                mmas && clamp_species_edge_pheromones!(pheromones, τ_mins, τ_maxs)
                setdiff!(active_ants, local_invalids)
            end
        end

        evaporate_edge_pheromones!(pheromones, evaporation)

        if elite_pheromone || aco_tabu
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
                    EDGES_TRACE && println("  elite species=$s pre-repair score=$(instance_fitness(compact_fg, ant.explored, θ)) " *
                                     "U=$(sorted_str(ant.explored.U)) V=$(sorted_str(ant.explored.V))")
                    if aco_tabu
                        tabu_repair!(compact_fg, ant.explored, k, θ, tt, tabu_patience)
                        seed_compact !== nothing && merge_seed!(ant.explored, seed_compact)
                        if EDGES_TRACE
                            post_score = instance_fitness(compact_fg, ant.explored, θ)
                            msg = "  elite species=$s post-repair score=$post_score " *
                                  "U=$(sorted_str(ant.explored.U)) V=$(sorted_str(ant.explored.V))"
                            if target_compact !== nothing
                                ou, ov = target_overlap(ant.explored, target_compact)
                                msg *= "  target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))"
                            end
                            println(msg)
                        end
                    end
                    if elite_pheromone
                        deposit_induced_edges!(pheromones.species[s], compact_fg, ant.explored,
                            pheromone * ELITE_PHEROMONE_FACTOR)
                        deposit_induced_edges!(pheromones.shared, compact_fg, ant.explored,
                            pheromone * ELITE_PHEROMONE_FACTOR * SHARED_PHEROMONE_FACTOR)
                    end
                end
            end
        end

        for ant in ants
            if seed_compact !== nothing && !subgraph_has_seed(ant.explored, seed_compact)
                continue
            end
            score = instance_fitness(compact_fg, ant.explored, θ)
            s = ant.species
            if better_than_best(compact_fg, ant.explored, best_subgraphs[s], θ)
                best_scores[s] = score
                best_subgraphs[s] = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
                best_iterations[s] = iter
                best_times[s] = (time_ns() - t0) / 1e9
            end
            if better_than_best(compact_fg, ant.explored, best_subgraph, θ)
                if aco_tabu
                    tabu = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
                    tabu_repair!(compact_fg, tabu, k, θ, tt, tabu_patience)
                    if seed_compact !== nothing
                        merge_seed!(tabu, seed_compact)
                    end
                    if (seed_compact === nothing || subgraph_has_seed(tabu, seed_compact)) &&
                       !better_than_best(compact_fg, ant.explored, tabu, θ)
                        best_score = instance_fitness(compact_fg, tabu, θ)
                        best_subgraph = tabu
                    else
                        best_score = score
                        best_subgraph = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
                    end
                else
                    best_score = score
                    best_subgraph = SubGraph(copy(ant.explored.U), copy(ant.explored.V))
                end
                best_iteration = iter
                best_time = (time_ns() - t0) / 1e9
                if EDGES_TRACE
                    msg = "  NEW BEST score=$best_score edges=$(Subgraph.edge_count(compact_fg, best_subgraph)) " *
                          "θ_ok=$(theta_feasible(best_subgraph, θ)) iter=$best_iteration " *
                          "t=$(round(best_time; digits=4))s " *
                          "U=$(sorted_str(best_subgraph.U)) V=$(sorted_str(best_subgraph.V))"
                    if target_compact !== nothing
                        ou, ov = target_overlap(best_subgraph, target_compact)
                        msg *= "  target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))"
                    end
                    println(msg)
                end
            end
        end

        if mmas
            n_edges = length(compact_fg.v_adj)
            update_species_edge_pheromone_bounds!(τ_mins, τ_maxs, pheromone, n_edges, evaporation,
                best_subgraphs, compact_fg; pheromone_min=pheromone_min, pheromone_max=pheromone_max)
            clamp_species_edge_pheromones!(pheromones, τ_mins, τ_maxs)
        end

        if EDGES_TRACE && target_compact !== nothing
            ou, ov = target_overlap(best_subgraph, target_compact)
            println("  iter=$iter end: best_score=$best_score " *
                    "target_hit=$ou/$(length(target_compact.U)),$ov/$(length(target_compact.V))")
        end

        if construction_stats !== nothing
            for ant in ants
                for (size, missing_count) in ant.missing_at_size
                    push!(get!(pooled_missing_at_size, size, Int[]), missing_count)
                end
            end
            last_iteration_orders = [copy(ant.addition_order) for ant in ants]
        end

        ants = new_ants(compact_fg, num_ants, num_subspecies)
        seed_compact !== nothing && seed_ants_with_subgraph!(ants, compact_fg, k, seed_compact)

        force_gc && GC.gc()

        if iteration_callback !== nothing
            elapsed_s = (time_ns() - t0) / 1e9
            iteration_callback(iter, best_subgraph, compact_fg, remapping, elapsed_s) || break
        end
    end

    remapped = [remap_subgraph(remapping, best_subgraphs[s]) for s in 1:num_subspecies]
    if seed_nodes !== nothing && seed_compact !== nothing
        for s in 1:num_subspecies
            subgraph_has_seed(best_subgraphs[s], seed_compact) ||
                error("edges-ACO forced seed missing from subspecies $s best after search")
        end
        subgraph_has_seed(best_subgraph, seed_compact) ||
            error("edges-ACO forced seed missing from global best after search")
    end
    for s in 1:num_subspecies
        sg = remapped[s]
        src = if Subgraph.vertex_count(best_subgraphs[s]) == 0
            " (no edges-ACO solution)"
        elseif best_iterations[s] == 0
            " (forced-seed incumbent; ants never improved)"
        else
            ""
        end
        println("edges-ACO subspecies $s best: |U|=$(length(sg.U)) |V|=$(length(sg.V)) " *
                "score=$(best_scores[s]) found_at_iter=$(best_iterations[s]) " *
                "found_at_t=$(round(best_times[s]; digits=4))s$src")
        if EDGES_TRACE && trace_target !== nothing
            ou = length(intersect(sg.U, trace_target.U))
            ov = length(intersect(sg.V, trace_target.V))
            println("  target_hit (original ids)=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))")
        end
    end
    src = if Subgraph.vertex_count(best_subgraph) == 0
        " — no edges-ACO solution found"
    elseif best_iteration == 0
        " — still the forced-seed incumbent (ants never improved)"
    else
        ""
    end
    println("edges-ACO global best found at iteration $best_iteration " *
            "(t=$(round(best_time; digits=4))s, score=$best_score)$src")
    if construction_stats !== nothing
        construction_stats[] = (;
            missing_at_size=pooled_missing_at_size,
            last_iteration_orders=last_iteration_orders,
        )
    end
    return remapped, best_iterations, best_times, pheromones, remapping
end
