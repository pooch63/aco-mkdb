#= Ant colony optimization to find the maximum biclique =#

using Base.Threads

const __ACO_JL__ = true

isdefined(@__MODULE__, :__CONSTANTS_JL__) || include(joinpath(@__DIR__, "..", "constants.jl"))
include("pheromone.jl")
include("advance.jl")

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__TABU_JL__) || include(joinpath("..", "tabu.jl"))

# Flip to true to dump per-step construction / pruning decisions (like opponent.jl).
# Best used with parallelize=false — ACO forces that when ACO_TRACE is on.
const ACO_TRACE = false

# Ablation defaults (overridable via aco() kwargs / CLI / vary.bash env).
# Softmax-select a few finished ants by fitness and deposit elite pheromone on them.
# Tabu repair on elite ants before deposit, and on new global bests.
# MAX-MIN Ant System: clamp shared + each species trail to [τ_min, τ_max].
# When false, pheromone is unbounded ([0, +∞)).

const ELITE_PHEROMONE_FACTOR = 2
const SHARED_PHEROMONE_FACTOR = 0.25

# MAX-MIN Ant System (MMAS) bounds on shared and species pheromone maps.
# `best_only_pheromone` skips per-step deposits and, after evaporation, lays
# pheromone only on the iteration-best solution.

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

@inline theta_feasible(sg::SubGraph, θ::Int) = length(sg.U) ≥ θ && length(sg.V) ≥ θ

"""
Whether `cand` should replace `curr` as a tracked best.

θ-feasible solutions compete by edge count. Fitness is used only when the
incumbent is also not θ-feasible (so an infeasible search can still progress).
A θ-feasible candidate always beats an infeasible incumbent.

`fitness` defaults to `instance_fitness`; diffusion ACO passes a cohesion-aware
score so low-variance subgraphs are preferred while still searching for θ.
"""
function better_than_best(fg::FrozenBipartite, cand::SubGraph, curr::SubGraph, θ::Int;
    fitness=instance_fitness)
    cand_ok = theta_feasible(cand, θ)
    curr_ok = theta_feasible(curr, θ)
    if cand_ok
        curr_ok || return true
        return Subgraph.edge_count(fg, cand) > Subgraph.edge_count(fg, curr)
    elseif curr_ok
        return false
    else
        return fitness(fg, cand, θ) > fitness(fg, curr, θ)
    end
end

"""
Best finished ant this round, by the same rule as the colony incumbent.

Bound-pruned ants and ants that lost a forced seed are skipped. The returned
subgraph is `kept_ant_solution` (greedy fill when it wins). `(nothing, empty)`
when every ant was discarded.
"""
function pick_iteration_best(ants::Vector{Ant}, fg::FrozenBipartite, θ::Int, fitness,
    seed_compact::Union{Nothing,SubGraph}, seeds_compact::Vector{SubGraph},
    num_ants::Int, iter::Int)
    best_ant = nothing
    best_kept = SubGraph()
    for (i, ant) in enumerate(ants)
        ant.bound_pruned && continue
        assigned = round_robin_seed(seeds_compact, num_ants, i, iter)
        if seed_compact !== nothing && !subgraph_has_seed(ant.explored, seed_compact)
            continue
        end
        if assigned !== nothing && !subgraph_has_seed(ant.explored, assigned)
            continue
        end
        kept = kept_ant_solution(ant, fg, θ, fitness, seed_compact, assigned)
        if best_ant === nothing || better_than_best(fg, kept, best_kept, θ; fitness=fitness)
            best_ant = ant
            best_kept = kept
        end
    end
    return best_ant, best_kept
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
    prefer_smaller_side::Bool=false,
    # When true, sample from last node's neighbors ∩ C when nonempty; else full C.
    neighbor_scope_limit::Bool=true,
    elite_seed::Bool=true,
    elite_seed_ants::Int=3,
    elite_seed_remove::Int=2,
    # Paper ablations (default off — baseline ACO does not use these).
    elite_pheromone::Bool=false,  # elitist pheromone emit on softmax elites
    aco_tabu::Bool=false,         # tabu repair on elites / new global bests
    mmas::Bool=false,             # MAX-MIN Ant System τ bounds
    # Halt an ant when `upper_bound` cannot beat the incumbent, and subtract
    # the pheromone that ant deposited during this construction.
    prune_by_bound::Bool=false,
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    trace_target::Union{Nothing,SubGraph}=nothing,
    # Original-id nodes to plant into every ant at the start of each construction.
    # Experimental default for the known pair under test; pass `nothing` to disable.
    seed_nodes::Union{Nothing,SubGraph}=SubGraph(Set(), Set()),
    # One original-id subgraph per seed. Ant i on iteration t is planted with
    # ant_seeds[mod1(i + (t-1)*num_ants, length)]. `nothing` / empty disables.
    # Cannot be combined with a non-empty `seed_nodes`.
    ant_seeds::Union{Nothing,AbstractVector{<:SubGraph}}=nothing,
    construction_stats::Union{Nothing,Base.RefValue}=nothing,
    # `:off` leaves the walk as built. `:end` packs each side once when the ant
    # stops. `:step` packs after every add onto a side that already has ≥θ
    # vertices. The best snapshot is what the colony keeps.
    greedy_fill::Union{Symbol,AbstractString}=:off,
    # Exponent on heuristic desirability η in τ · η^eta_exp (default ETA_EXP).
    eta_exp::Real=ETA_EXP,
    # When true, ants do not deposit while walking. After the round evaporates,
    # only the iteration-best solution receives pheromone.
    best_only_pheromone::Bool=false,
    # Optional `(compact_fg) -> (; fitness, deposit_scale?)` hook for variants
    # (e.g. diffusion ACO). `fitness(fg, sg, θ)` replaces `instance_fitness`;
    # `deposit_scale(fg, sg)` multiplies per-step pheromone deposit (default 1).
    make_scoring=nothing)
    num_subspecies >= 1 || throw(ArgumentError("num_subspecies must be >= 1, got $num_subspecies"))
    elite_seed_ants >= 0 || throw(ArgumentError("elite_seed_ants must be >= 0, got $elite_seed_ants"))
    elite_seed_remove >= 0 || throw(ArgumentError("elite_seed_remove must be >= 0, got $elite_seed_remove"))
    greedy_fill = normalize_greedy_fill(greedy_fill)
    eta_exp = Float64(eta_exp)
    isfinite(eta_exp) || throw(ArgumentError("eta_exp must be finite, got $eta_exp"))
    if best_only_pheromone && elite_pheromone
        throw(ArgumentError(
            "best_only_pheromone deposits only the iteration-best solution and cannot be combined with elite_pheromone"))
    end

    if ACO_TRACE && parallelize
        println("ACO ACO_TRACE: forcing parallelize=false so step logs stay readable")
        parallelize = false
    end

    apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)

    fg = freeze(g)

    println("Prefer smaller side: $prefer_smaller_side")
    println("Neighbor scope limit: $neighbor_scope_limit")
    println("Size of reduced graphs", length(fg.u_ids), " ", length(fg.v_ids))

    compact_fg, remapping = compact_frozen(fg)
    pheromones = ColonyPheromones(compact_fg, num_subspecies)

    fitness = instance_fitness
    deposit_scale = nothing
    if make_scoring !== nothing
        scoring = applicable(make_scoring, compact_fg, remapping) ?
            make_scoring(compact_fg, remapping) : make_scoring(compact_fg)
        fitness = scoring.fitness
        deposit_scale = hasproperty(scoring, :deposit_scale) ? scoring.deposit_scale : nothing
    end

    if ant_seeds !== nothing && !isempty(ant_seeds) &&
       seed_nodes !== nothing && Subgraph.vertex_count(seed_nodes) > 0
        throw(ArgumentError("aco ant_seeds and seed_nodes cannot both be set"))
    end

    # Map original-id seed nodes into compact space (dropped ⇒ already removed by reduction).
    seed_compact::Union{Nothing,SubGraph} = nothing
    if seed_nodes !== nothing && Subgraph.vertex_count(seed_nodes) > 0
        seed_compact, dropped_U, dropped_V = compactify_subgraph(remapping, seed_nodes)
        println("ACO seed_nodes: original U=$(sorted_str(seed_nodes.U)) V=$(sorted_str(seed_nodes.V)) → " *
                "compact U=$(sorted_str(seed_compact.U)) V=$(sorted_str(seed_compact.V))")
        if !isempty(dropped_U) || !isempty(dropped_V)
            println("  seed nodes dropped by reduction: U=$dropped_U V=$dropped_V")
        end
        if Subgraph.vertex_count(seed_compact) == 0
            seed_compact = nothing
        end
    end

    # Per-ant seeds (split ACO's one colony). Each kept seed is one forced start;
    # round-robin assigns them across ants and iterations.
    seeds_compact = SubGraph[]
    if ant_seeds !== nothing && !isempty(ant_seeds)
        for seed in ant_seeds
            kept, dropped_U, dropped_V = compactify_subgraph(remapping, seed)
            if Subgraph.vertex_count(kept) == 0
                println("ACO ant seed dropped by reduction: U=$dropped_U V=$dropped_V")
                continue
            end
            push!(seeds_compact, kept)
        end
        println("ACO ant seeds: $(length(seeds_compact)) of $(length(ant_seeds)) kept; " *
                "round-robin across $num_ants ants × $num_iterations iterations")
    end

    best_scores = fill(0.0, num_subspecies)
    best_subgraphs = [SubGraph() for _ in 1:num_subspecies]
    # Iteration / wall time at which each subspecies / global best was last improved
    # by an ant. 0 / 0.0 only if a forced-seed incumbent was installed pre-loop
    # (θ-heuristic is never tracked as best).
    best_iterations = fill(0, num_subspecies)
    best_times = fill(0.0, num_subspecies)
    best_score::Float64 = 0.0
    best_subgraph::SubGraph = SubGraph()
    best_iteration::Int = 0
    best_time::Float64 = 0.0

    # Forced-inclusion incumbent. A single seed_nodes subgraph must appear in
    # every reported best. Per-ant seeds install the best seed among them; each
    # ant then keeps its own seed, and the global best keeps whichever it grew from.
    forced_incumbent::Union{Nothing,SubGraph} = nothing
    if !isempty(seeds_compact)
        forced_incumbent = seeds_compact[1]
        for seed in seeds_compact
            better_than_best(compact_fg, seed, forced_incumbent, θ; fitness=fitness) &&
                (forced_incumbent = seed)
        end
    elseif seed_compact !== nothing
        forced_incumbent = seed_compact
    end
    if forced_incumbent !== nothing
        seed_score = Float64(fitness(compact_fg, forced_incumbent, θ))
        best_subgraph = SubGraph(copy(forced_incumbent.U), copy(forced_incumbent.V))
        best_score = seed_score
        best_iteration = 0
        best_time = 0.0
        for s in 1:num_subspecies
            best_subgraphs[s] = SubGraph(copy(forced_incumbent.U), copy(forced_incumbent.V))
            best_scores[s] = seed_score
            best_iterations[s] = 0
            best_times[s] = 0.0
        end
        src = isempty(seeds_compact) ? "" : " (best of $(length(seeds_compact)) ant seeds)"
        println("ACO forced-seed incumbent: U=$(sorted_str(best_subgraph.U)) " *
                "V=$(sorted_str(best_subgraph.V)) score=$seed_score$src")
    end

    n_nodes = length(compact_fg.u_ids) + length(compact_fg.v_ids)
    τ_mins = Vector{Float64}(undef, num_subspecies)
    τ_maxs = Vector{Float64}(undef, num_subspecies)
    if mmas
        # θ-heuristic feeds MMAS τ bounds only — never the tracked incumbent.
        # Prefer a seed-feasible heuristic so the ceiling reflects admissible quality;
        # otherwise fall back to empty-best default (best_n=2) via empty subgraphs.
        mmas_init = [SubGraph() for _ in 1:num_subspecies]
        heuristic_sg = theta_based_heuristic(compact_fg, k, θ; return_invalid=true)
        if Subgraph.vertex_count(heuristic_sg) > 0 &&
           (seed_compact === nothing || subgraph_has_seed(heuristic_sg, seed_compact))
            heuristic_score = fitness(compact_fg, heuristic_sg, θ)
            for s in 1:num_subspecies
                mmas_init[s] = SubGraph(copy(heuristic_sg.U), copy(heuristic_sg.V))
            end
            println("ACO θ-heuristic → MMAS init only: |U|=$(length(heuristic_sg.U)) " *
                    "|V|=$(length(heuristic_sg.V)) score=$heuristic_score " *
                    "vertices=$(Subgraph.vertex_count(heuristic_sg))")
        elseif Subgraph.vertex_count(heuristic_sg) > 0
            println("ACO θ-heuristic skipped for MMAS init (|U|=$(length(heuristic_sg.U)) " *
                    "|V|=$(length(heuristic_sg.V))): missing forced seed_nodes")
        end
        update_species_pheromone_bounds!(τ_mins, τ_maxs, pheromone, n_nodes, evaporation, mmas_init;
            pheromone_min=pheromone_min, pheromone_max=pheromone_max)
        clamp_species_pheromones!(pheromones, τ_mins, τ_maxs)
        println("ACO MMAS species bounds (init): " *
                join(["s$s=[$(round(τ_mins[s]; digits=6)),$(round(τ_maxs[s]; digits=6))]"
                      for s in 1:num_subspecies], " "))
    else
        fill!(τ_mins, 0.0)
        fill!(τ_maxs, Inf)
        println("ACO MMAS disabled: species pheromone unbounded [0, +∞) " *
                "(θ-heuristic not run for ACO)")
    end
    println("ACO ablations: elite_pheromone=$elite_pheromone tabu=$aco_tabu mmas=$mmas bound_prune=$prune_by_bound greedy_fill=$greedy_fill eta_exp=$eta_exp best_only_pheromone=$best_only_pheromone")
    # println("ACO MMAS species bounds: τ_min=$(round(τ_min; digits=6)) τ_max=$(round(τ_max; digits=6))")
    # Compact-space watch target (optional). Dropped nodes mean reduction already
    # removed part of the known optimum — ACO can never recover those.
    target_compact::Union{Nothing,SubGraph} = nothing
    if ACO_TRACE && trace_target !== nothing
        target_compact, dropped_U, dropped_V = compactify_subgraph(remapping, trace_target)
        println("ACO ACO_TRACE target: original |U|=$(length(trace_target.U)) |V|=$(length(trace_target.V)) → " *
                "compact |U|=$(length(target_compact.U)) |V|=$(length(target_compact.V))")
        println("  compact U=", sorted_str(target_compact.U), " V=", sorted_str(target_compact.V))
        if !isempty(dropped_U) || !isempty(dropped_V)
            println("  !!! reduction dropped target nodes before search: " *
                    "U=$(dropped_U) V=$(dropped_V)")
        end
    end

    function plant_seeds!(ants, iter::Int)
        if !isempty(seeds_compact)
            seed_ants_round_robin!(ants, compact_fg, k, seeds_compact, iter)
        elseif seed_compact !== nothing
            seed_ants_with_subgraph!(ants, compact_fg, k, seed_compact)
        end
        return ants
    end

    ants = new_ants(compact_fg, num_ants, num_subspecies)
    plant_seeds!(ants, 1)
    invalid_ants = Set{Int}()

    explored_ants = [ant for ant in ants]

    pooled_missing_at_size = Dict{Int, Vector{Int}}()
    last_iteration_orders = Vector{Tuple{Bool,Int}}[]

    # Wall-clock origin for time-to-best (excludes reduction / MMAS setup).
    t0 = time_ns()
    n_bound_pruned = 0

    for iter in 1:num_iterations
        ACO_TRACE && println("==== ACO iter $iter/$num_iterations  best_score=$best_score " *
                         "|U|=$(length(best_subgraph.U)) |V|=$(length(best_subgraph.V)) ====")

        # Elite partial-restart overwrites forced seed_nodes; skip it while
        # forced inclusion is active so every ant keeps the required vertices.
        forced_seeds = seed_compact !== nothing || !isempty(seeds_compact)
        if elite_seed && !forced_seeds
            seed_ants_from_elites!(ants, best_subgraphs, best_subgraph, best_scores,
                elite_seed_ants, elite_seed_remove, compact_fg, k)
            if ACO_TRACE
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
        elseif elite_seed && forced_seeds && iter == 1
            println("ACO elite_seed disabled while forced seed inclusion is active")
        end

        # Track which ants are still exploring
        active_ants = collect(1:num_ants)
        bound_target = prune_by_bound ?
            bound_target_from(compact_fg, best_subgraph, θ, fitness) : nothing

        while !isempty(active_ants)
            if parallelize
                # Divide the remaining ants evenly across available threads
                chunk_size = max(1, cld(length(active_ants), nthreads()))
                
                # 1. Spawn tasks for each chunk
                tasks = map(Iterators.partition(active_ants, chunk_size)) do chunk
                    Threads.@spawn advance_ants!(compact_fg, pheromones, pheromone, ants, k, θ, chunk;
                        prefer_smaller_side=prefer_smaller_side,
                        neighbor_scope_limit=neighbor_scope_limit,
                        trace_target=target_compact,
                        deposit_scale=deposit_scale,
                        prune_by_bound=prune_by_bound,
                        bound_target=bound_target,
                        greedy_fill=greedy_fill,
                        fitness=fitness,
                        eta_exp=eta_exp,
                        deposit_online=!best_only_pheromone)
                end

                # 2. Wait for all threads to finish their step
                results = fetch.(tasks)
            else
                # Process all active ants sequentially on the main thread.
                # Wrapped in an array to match the structure of multithreaded `results`.
                results = [advance_ants!(compact_fg, pheromones, pheromone, ants, k, θ, active_ants;
                    prefer_smaller_side=prefer_smaller_side,
                    neighbor_scope_limit=neighbor_scope_limit,
                    trace_target=target_compact,
                    deposit_scale=deposit_scale,
                    prune_by_bound=prune_by_bound,
                    bound_target=bound_target,
                    greedy_fill=greedy_fill,
                    fitness=fitness,
                    eta_exp=eta_exp,
                    deposit_online=!best_only_pheromone)]
            end
            
            # 3. Safely merge the results on the main thread
            for (local_additions, local_invalids) in results
                # Merge the local additions into the global pheromone tracker
                merge_pheromones!(pheromones, local_additions)
                if prune_by_bound
                    for idx in local_invalids
                        ant = ants[idx]
                        if ant.bound_pruned
                            # Deposits from earlier steps are already in `pheromones`.
                            # This step did not deposit. Subtract so the ant leaves no trail.
                            rollback_ant_pheromone!(pheromones, ant)
                            n_bound_pruned += 1
                        else
                            raise_bound_target!(bound_target, compact_fg, ant.explored, θ, fitness)
                        end
                    end
                end
                # Cap species trails during construction so mid-iter selection stays bounded.
                mmas && clamp_species_pheromones!(pheromones, τ_mins, τ_maxs)
                
                # Remove dead ants from the active pool for the next iteration
                setdiff!(active_ants, local_invalids)
            end
        end

        evaporate_pheromones!(pheromones, evaporation)

        # Softmax-select a few finished ants by instance fitness; optionally
        # tabu-repair them and/or reinforce every node in the (repaired) subgraph.
        if elite_pheromone || aco_tabu
            for s in 1:num_subspecies
                eligible = [ant for ant in ants if ant.last_visited.id != -1 &&
                    ant.species == s && !ant.bound_pruned]
                n_elite = min(3, length(eligible))
                n_elite == 0 && continue

                elites = softmax_sample(
                    ant -> fitness(compact_fg, ant.explored, θ),
                    eligible,
                    n_elite,
                )
                for ant in elites
                    ACO_TRACE && println("  elite species=$s pre-repair score=$(fitness(compact_fg, ant.explored, θ)) " *
                                     "U=$(sorted_str(ant.explored.U)) V=$(sorted_str(ant.explored.V))")
                    if aco_tabu
                        tabu_repair!(compact_fg, ant.explored, k, θ, tt, tabu_patience)
                        seed_compact !== nothing && merge_seed!(ant.explored, seed_compact)
                        if !isempty(seeds_compact)
                            idx = findfirst(===(ant), ants)
                            assigned = idx === nothing ? nothing :
                                round_robin_seed(seeds_compact, num_ants, idx, iter)
                            assigned !== nothing && merge_seed!(ant.explored, assigned)
                        end
                        if ACO_TRACE
                            post_score = fitness(compact_fg, ant.explored, θ)
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
                        elite_dep = Float64(pheromone) * ELITE_PHEROMONE_FACTOR
                        if deposit_scale !== nothing
                            elite_dep *= Float64(deposit_scale(compact_fg, ant.explored))
                        end
                        for u in ant.explored.U
                            node = Node(true, u)
                            add_pheromone!(pheromones.species[s], node, elite_dep)
                            add_pheromone!(pheromones.shared, node, elite_dep * SHARED_PHEROMONE_FACTOR)
                        end
                        for v in ant.explored.V
                            node = Node(false, v)
                            add_pheromone!(pheromones.species[s], node, elite_dep)
                            add_pheromone!(pheromones.shared, node, elite_dep * SHARED_PHEROMONE_FACTOR)
                        end
                    end
                end
            end
        end

        # Online deposits already happened (and were evaporated above) unless
        # best-only mode deferred them. The iteration-best trail is laid after
        # evaporation so it is what the next round samples.
        if best_only_pheromone
            winner, kept = pick_iteration_best(ants, compact_fg, θ, fitness,
                seed_compact, seeds_compact, num_ants, iter)
            if winner !== nothing && Subgraph.vertex_count(kept) > 0
                dep = Float64(pheromone)
                if deposit_scale !== nothing
                    dep *= Float64(deposit_scale(compact_fg, kept))
                end
                deposit_solution_pheromone!(pheromones, kept, winner.species, dep)
                if ACO_TRACE
                    println("  best-only deposit species=$(winner.species) " *
                            "|U|=$(length(kept.U)) |V|=$(length(kept.V)) dep=$dep")
                end
            end
        end

        for (i, ant) in enumerate(ants)
            # A bound-pruned ant is discarded, including any partial subgraph.
            ant.bound_pruned && continue
            # Ants only add vertices, so a forced seed at construction start stays
            # in explored; still skip any ant that lost it (e.g. future repair).
            assigned = round_robin_seed(seeds_compact, num_ants, i, iter)
            if seed_compact !== nothing && !subgraph_has_seed(ant.explored, seed_compact)
                continue
            end
            if assigned !== nothing && !subgraph_has_seed(ant.explored, assigned)
                continue
            end
            kept = kept_ant_solution(ant, compact_fg, θ, fitness, seed_compact, assigned)
            score = Float64(fitness(compact_fg, kept, θ))
            s = ant.species
            # Subspecies / global incumbents: θ-feasible by edges; fitness only if
            # the current best is also not θ-feasible. (Elitism above is unchanged.)
            if better_than_best(compact_fg, kept, best_subgraphs[s], θ; fitness=fitness)
                best_scores[s] = score
                # Copy so later colony mutations / seeds never alias the stored best.
                best_subgraphs[s] = SubGraph(copy(kept.U), copy(kept.V))
                best_iterations[s] = iter
                best_times[s] = (time_ns() - t0) / 1e9
            end
            if better_than_best(compact_fg, kept, best_subgraph, θ; fitness=fitness)
                if aco_tabu
                    tabu = SubGraph(copy(kept.U), copy(kept.V))
                    tabu_repair!(compact_fg, tabu, k, θ, tt, tabu_patience)
                    if seed_compact !== nothing
                        merge_seed!(tabu, seed_compact)
                    end
                    assigned !== nothing && merge_seed!(tabu, assigned)
                    # Prefer repaired if seed-ok and not strictly worse than unrepaired.
                    seed_ok = (seed_compact === nothing || subgraph_has_seed(tabu, seed_compact)) &&
                              (assigned === nothing || subgraph_has_seed(tabu, assigned))
                    if seed_ok &&
                       !better_than_best(compact_fg, kept, tabu, θ; fitness=fitness)
                        best_score = Float64(fitness(compact_fg, tabu, θ))
                        best_subgraph = tabu
                    else
                        best_score = score
                        best_subgraph = SubGraph(copy(kept.U), copy(kept.V))
                    end
                else
                    best_score = score
                    best_subgraph = SubGraph(copy(kept.U), copy(kept.V))
                end
                best_iteration = iter
                best_time = (time_ns() - t0) / 1e9
                if ACO_TRACE
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

        # Refresh MMAS bounds from each subspecies' best, then floor/cap trails.
        if mmas
            n_nodes = length(compact_fg.u_ids) + length(compact_fg.v_ids)
            update_species_pheromone_bounds!(τ_mins, τ_maxs, pheromone, n_nodes, evaporation, best_subgraphs;
                pheromone_min=pheromone_min, pheromone_max=pheromone_max)
            clamp_species_pheromones!(pheromones, τ_mins, τ_maxs)
        end

        if ACO_TRACE && target_compact !== nothing
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
        plant_seeds!(ants, iter + 1)

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
                error("ACO forced seed missing from subspecies $s best after search")
        end
        subgraph_has_seed(best_subgraph, seed_compact) ||
            error("ACO forced seed missing from global best after search")
    end
    if !isempty(seeds_compact)
        contains_seed(sg) = any(seed -> subgraph_has_seed(sg, seed), seeds_compact)
        for s in 1:num_subspecies
            contains_seed(best_subgraphs[s]) ||
                error("ACO ant seed missing from subspecies $s best after search")
        end
        contains_seed(best_subgraph) ||
            error("ACO ant seed missing from global best after search")
    end
    for s in 1:num_subspecies
        sg = remapped[s]
        src = if Subgraph.vertex_count(best_subgraphs[s]) == 0
            " (no ACO solution)"
        elseif best_iterations[s] == 0
            " (forced-seed incumbent; ants never improved)"
        else
            ""
        end
        println("ACO subspecies $s best: |U|=$(length(sg.U)) |V|=$(length(sg.V)) " *
                "score=$(best_scores[s]) found_at_iter=$(best_iterations[s]) " *
                "found_at_t=$(round(best_times[s]; digits=4))s$src")
        if ACO_TRACE && trace_target !== nothing
            ou = length(intersect(sg.U, trace_target.U))
            ov = length(intersect(sg.V, trace_target.V))
            println("  target_hit (original ids)=$ou/$(length(trace_target.U)),$ov/$(length(trace_target.V))")
        end
    end
    src = if Subgraph.vertex_count(best_subgraph) == 0
        " — no ACO solution found"
    elseif best_iteration == 0
        " — still the forced-seed incumbent (ants never improved)"
    else
        ""
    end
    println("ACO global best found at iteration $best_iteration " *
            "(t=$(round(best_time; digits=4))s, score=$best_score)$src")
    if prune_by_bound
        println("ACO bound prune: halted $n_bound_pruned ant-construction(s) " *
                "whose upper bound could not beat the incumbent")
    end
    if construction_stats !== nothing
        construction_stats[] = (;
            missing_at_size=pooled_missing_at_size,
            last_iteration_orders=last_iteration_orders,
        )
    end
    return remapped, best_iterations, best_times, pheromones, remapping
end

"""
Collect effective pheromone for every compact node under `species`.
"""
function all_node_pheromones(colony::ColonyPheromones, species::Int)
    nU = length(colony.shared.U)
    nV = length(colony.shared.V)
    τ = Vector{Float64}(undef, nU + nV)
    for i in 1:nU
        τ[i] = effective_pheromone(colony, Node(true, i), species)
    end
    for i in 1:nV
        τ[nU + i] = effective_pheromone(colony, Node(false, i), species)
    end
    return τ
end

"""
Given final ACO trails and a reference solution in *original* vertex ids
(e.g. the pivot optimum), find the minimum pheromone among solution nodes
that survived reduction, and report what fraction of all nodes sit strictly
below that floor — i.e. how many could be cut while keeping the solution intact.
"""
function pheromone_cut_stats(colony::ColonyPheromones, remapping::GraphRemapping,
    solution::SubGraph; species::Int=1)
    species >= 1 || throw(ArgumentError("species must be >= 1, got $species"))
    species <= length(colony.species) ||
        throw(ArgumentError("species $species > num subspecies $(length(colony.species))"))

    u_map = Dict{Int,Int}(orig => i for (i, orig) in enumerate(remapping.u_original))
    v_map = Dict{Int,Int}(orig => i for (i, orig) in enumerate(remapping.v_original))

    all_τ = all_node_pheromones(colony, species)
    n_nodes = length(all_τ)
    global_min_τ = isempty(all_τ) ? nothing : minimum(all_τ)
    global_max_τ = isempty(all_τ) ? nothing : maximum(all_τ)

    sol_entries = []
    dropped_U = Int[]
    dropped_V = Int[]
    for u in solution.U
        if haskey(u_map, u)
            cid = u_map[u]
            push!(sol_entries, (; is_u=true, orig_id=u, compact_id=cid,
                τ=effective_pheromone(colony, Node(true, cid), species)))
        else
            push!(dropped_U, u)
        end
    end
    for v in solution.V
        if haskey(v_map, v)
            cid = v_map[v]
            push!(sol_entries, (; is_u=false, orig_id=v, compact_id=cid,
                τ=effective_pheromone(colony, Node(false, cid), species)))
        else
            push!(dropped_V, v)
        end
    end

    isempty(sol_entries) && return (;
        n_nodes,
        n_solution_kept=0,
        dropped_U=sort(dropped_U),
        dropped_V=sort(dropped_V),
        min_τ=nothing,
        min_node=nothing,
        n_below=0,
        cut_fraction=0.0,
        percentile=nothing,
        global_min_τ,
        global_max_τ,
        n_at_global_min=0,
        on_trail_floor=false,
    )

    min_entry = argmin(e -> e.τ, sol_entries)
    min_τ = min_entry.τ
    n_below = count(τ -> τ < min_τ, all_τ)
    # Percentile of min_τ among all nodes: fraction with τ ≤ min_τ.
    n_at_or_below = count(τ -> τ <= min_τ, all_τ)
    percentile = 100.0 * n_at_or_below / n_nodes
    cut_fraction = n_below / n_nodes
    # Shared trails start at 1 and evaporate; unvisited nodes pile up on that floor.
    # If the solution's weakest node is still on the global min trail, ACO never
    # reinforced it and a pheromone-only cut cannot safely remove anything.
    rtol = 1e-9 + 1e-6 * abs(something(global_min_τ, 0.0))
    on_trail_floor = global_min_τ !== nothing && abs(min_τ - global_min_τ) <= rtol
    n_at_global_min = global_min_τ === nothing ? 0 :
        count(τ -> abs(τ - global_min_τ) <= rtol, all_τ)

    return (;
        n_nodes,
        n_solution_kept=length(sol_entries),
        dropped_U=sort(dropped_U),
        dropped_V=sort(dropped_V),
        min_τ,
        min_node=min_entry,
        n_below,
        cut_fraction,
        percentile,
        global_min_τ,
        global_max_τ,
        n_at_global_min,
        on_trail_floor,
    )
end

function print_pheromone_cut_stats(stats; label::String="solution")
    println()
    println("ACO-reduce analysis (vs $label):")
    if !isempty(stats.dropped_U) || !isempty(stats.dropped_V)
        println("  reduction dropped solution nodes before ACO: " *
                "U=$(stats.dropped_U) V=$(stats.dropped_V)")
    end
    if stats.min_τ === nothing
        println("  no solution nodes remained after reduction; cannot score cut threshold")
        return
    end
    side = stats.min_node.is_u ? "U" : "V"
    println("  nodes on reduced graph     : $(stats.n_nodes)")
    println("  solution nodes kept        : $(stats.n_solution_kept)")
    println("  trail range                : [$(round(stats.global_min_τ; digits=6)), " *
            "$(round(stats.global_max_τ; digits=6))]")
    println("  min τ in solution          : $(round(stats.min_τ; digits=6)) " *
            "($side id=$(stats.min_node.orig_id), compact=$(stats.min_node.compact_id))")
    println("  τ percentile (≤ min)       : $(round(stats.percentile; digits=2))%")
    println("  nodes with τ < min         : $(stats.n_below) " *
            "($(round(100 * stats.cut_fraction; digits=2))% cuttable)")
    if stats.on_trail_floor
        println("  !!! solution sits on the unevaporated/unvisited trail floor " *
                "($(stats.n_at_global_min)/$(stats.n_nodes) nodes share τ≈$(round(stats.global_min_τ; digits=6))).")
        println("      ACO never reinforced this solution, so a τ-threshold cut cannot " *
                "separate it from the bulk of the graph (0% cuttable is expected).")
        println("      Try more --iterations / --ants so ACO rediscovers it on its own.")
    end
end
