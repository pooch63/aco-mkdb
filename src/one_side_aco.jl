#=
One-side ACO: search a maximum-edge biclique on the smaller part, then
greedy k-repair.

Phase 1 (k = 0). Ants add vertices only on the smaller side. The opposite
side of the biclique is the intersection of those neighborhoods. A vertex
may be added only when the new intersection still has at least θ vertices.
The walk is therefore a θ-feasible biclique as soon as the selected side
also has ≥ θ vertices; edge count is |S| · |∩ N(S)|.

Phase 2 (budget k). Fix one side of that biclique and pack the other with
`fill_keeping_side` (cheapest missing-edge cost first, then free same-side
vertices). Repeat with the sides swapped. Return whichever of the two
repairs (or the unrepaired biclique) is best under `better_than_best`.
=#

using Base.Threads

const __ONE_SIDE_ACO_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath("aco", "algorithm.jl"))

"""Ant that grows a set on one side; opposite side is the neighborhood intersection."""
mutable struct OneSideAnt
    selected::BitVector
    intersection::BitVector
    n_sel::Int
    n_inter::Int
    candidates::Vector{Int}
    cand_hits::Vector{Int}
    last::Int
    species::Int
    deposits::Vector{Int}
end

function hits_into_intersection(fg::FrozenBipartite, search_is_u::Bool, id::Int,
    intersection::BitVector)
    hits = 0
    if search_is_u
        @inbounds for k in neighbor_range_u(fg, id)
            intersection[fg.v_adj[k]] && (hits += 1)
        end
    else
        @inbounds for k in neighbor_range_v(fg, id)
            intersection[fg.u_adj[k]] && (hits += 1)
        end
    end
    return hits
end

function shrink_intersection!(intersection::BitVector, fg::FrozenBipartite,
    search_is_u::Bool, id::Int)
    keep = falses(length(intersection))
    if search_is_u
        @inbounds for k in neighbor_range_u(fg, id)
            vi = fg.v_adj[k]
            keep[vi] = intersection[vi]
        end
    else
        @inbounds for k in neighbor_range_v(fg, id)
            ui = fg.u_adj[k]
            keep[ui] = intersection[ui]
        end
    end
    copy!(intersection, keep)
    return count(intersection)
end

function new_one_side_ant(fg::FrozenBipartite, search_is_u::Bool, θ::Int, species::Int)
    n_same = search_is_u ? length(fg.u_ids) : length(fg.v_ids)
    n_opp = search_is_u ? length(fg.v_ids) : length(fg.u_ids)
    candidates = Int[]
    cand_hits = Int[]
    sizehint!(candidates, n_same)
    sizehint!(cand_hits, n_same)
    @inbounds for id in 1:n_same
        deg = search_is_u ? length(neighbor_range_u(fg, id)) : length(neighbor_range_v(fg, id))
        if deg >= θ
            push!(candidates, id)
            push!(cand_hits, deg)
        end
    end
    return OneSideAnt(falses(n_same), trues(n_opp), 0, n_opp,
        candidates, cand_hits, 0, species, Int[])
end

function one_side_subgraph(search_is_u::Bool, selected::BitVector, intersection::BitVector)
    same = Set{Int}()
    opp = Set{Int}()
    @inbounds for i in eachindex(selected)
        selected[i] && push!(same, i)
    end
    @inbounds for i in eachindex(intersection)
        intersection[i] && push!(opp, i)
    end
    return search_is_u ? SubGraph(same, opp) : SubGraph(opp, same)
end

one_side_subgraph(ant::OneSideAnt, search_is_u::Bool) =
    one_side_subgraph(search_is_u, ant.selected, ant.intersection)

"""
Greedy k-repair: pack each orientation of `sg` and return the best of
{sg, fix-U, fix-V} under `better_than_best`.
"""
function greedy_k_repair(fg::FrozenBipartite, sg::SubGraph, k::Int, θ::Int)
    Subgraph.vertex_count(sg) == 0 && return sg
    best = sg
    for fixed_is_u in (true, false)
        repaired = fill_keeping_side(fg, sg, fixed_is_u, k)
        if Subgraph.vertex_count(repaired) > 0 &&
           better_than_best(fg, repaired, best, θ)
            best = repaired
        end
    end
    return best
end

function one_side_desirability(pheromones::ColonyPheromones, search_is_u::Bool,
    id::Int, hits::Int, species::Int, eta_exp::Float64)
    τ = max(effective_pheromone(pheromones, Node(search_is_u, id), species), eps(Float64))
    η = Float64(max(hits, 0))
    return τ * η^eta_exp
end

"""One construction step. Returns false when the ant has no legal add left."""
function advance_one_side_ant!(ant::OneSideAnt, fg::FrozenBipartite,
    pheromones::ColonyPheromones, additions::ColonyPheromones, search_is_u::Bool,
    θ::Int, pheromone::Int, eta_exp::Float64)
    isempty(ant.candidates) && return false

    idxs = collect(eachindex(ant.candidates))
    score = i -> one_side_desirability(pheromones, search_is_u, ant.candidates[i],
        ant.cand_hits[i], ant.species, eta_exp)
    pick = linear_sample_one(score, idxs)
    id = ant.candidates[pick]

    ant.selected[id] = true
    ant.n_sel += 1
    ant.n_inter = shrink_intersection!(ant.intersection, fg, search_is_u, id)
    ant.n_inter >= θ || error("one-side ACO added a vertex that dropped |∩N| below θ")
    ant.last = id

    write = 1
    @inbounds for i in eachindex(ant.candidates)
        c = ant.candidates[i]
        c == id && continue
        hits = hits_into_intersection(fg, search_is_u, c, ant.intersection)
        if hits >= θ
            ant.candidates[write] = c
            ant.cand_hits[write] = hits
            write += 1
        end
    end
    resize!(ant.candidates, write - 1)
    resize!(ant.cand_hits, write - 1)

    dep = Float64(pheromone)
    node = Node(search_is_u, id)
    add_pheromone!(additions.species[ant.species], node, dep)
    add_pheromone!(additions.shared, node, dep * SHARED_PHEROMONE_FACTOR)
    push!(ant.deposits, id)
    return true
end

function construct_one_side_ant!(ant::OneSideAnt, fg::FrozenBipartite,
    pheromones::ColonyPheromones, additions::ColonyPheromones, search_is_u::Bool,
    θ::Int, pheromone::Int, eta_exp::Float64)
    while advance_one_side_ant!(ant, fg, pheromones, additions, search_is_u, θ,
        pheromone, eta_exp)
    end
    return ant
end

"""
    one_side_aco(g, pheromone, num_ants, num_iterations, evaporation, k, θ, num_subspecies)

Search a θ-feasible biclique on the smaller side, then greedy-repair with `k`.
"""
function one_side_aco(g::BipartiteGraph, pheromone::Int, num_ants::Int,
    num_iterations::Int, evaporation::Float64, k::Int, θ::Int,
    num_subspecies::Int=1;
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    eta_exp::Real=ETA_EXP,
    parallelize::Bool=false,
    verbose::Bool=true)
    num_ants >= 1 || throw(ArgumentError("num_ants must be ≥ 1, got $num_ants"))
    num_iterations >= 1 || throw(ArgumentError("num_iterations must be ≥ 1, got $num_iterations"))
    num_subspecies >= 1 || throw(ArgumentError("num_subspecies must be ≥ 1, got $num_subspecies"))
    0.0 < evaporation <= 1.0 || throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    eta_exp = Float64(eta_exp)
    isfinite(eta_exp) || throw(ArgumentError("eta_exp must be finite, got $eta_exp"))

    apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    fg = freeze(g)
    compact_fg, remapping = compact_frozen(fg)
    nU = length(compact_fg.u_ids)
    nV = length(compact_fg.v_ids)
    search_is_u = nU <= nV

    empty_result = (;
        best=SubGraph(),
        biclique=SubGraph(),
        search_is_u,
        n_same=search_is_u ? nU : nV,
        n_opp=search_is_u ? nV : nU,
        iterations_to_best=nothing,
        time_to_best_s=nothing,
        biclique_edges=0,
        repaired_edges=0,
    )

    if nU < θ || nV < θ
        verbose && println("One-side ACO: reduced graph smaller than θ=$θ (|U|=$nU |V|=$nV)")
        return empty_result
    end

    verbose && println("One-side ACO: search $(search_is_u ? "U" : "V") " *
                       "n_same=$(search_is_u ? nU : nV) n_opp=$(search_is_u ? nV : nU) " *
                       "ants=$num_ants iterations=$num_iterations θ=$θ then k-repair k=$k")

    pheromones = ColonyPheromones(compact_fg, num_subspecies)
    best_biclique = Ref(SubGraph())
    best_repaired = Ref(SubGraph())
    best_iteration = Ref(0)
    best_time = Ref(0.0)
    t0 = time_ns()

    function run_colony()
        for iter in 1:num_iterations
            ants = [new_one_side_ant(compact_fg, search_is_u, θ, mod1(i, num_subspecies))
                    for i in 1:num_ants]
            additions = [zero_colony_pheromones(compact_fg, num_subspecies) for _ in 1:num_ants]

            if parallelize && nthreads() > 1
                @threads :dynamic for i in 1:num_ants
                    construct_one_side_ant!(ants[i], compact_fg, pheromones, additions[i],
                        search_is_u, θ, pheromone, eta_exp)
                end
            else
                for i in 1:num_ants
                    construct_one_side_ant!(ants[i], compact_fg, pheromones, additions[i],
                        search_is_u, θ, pheromone, eta_exp)
                end
            end

            for add in additions
                merge_pheromones!(pheromones, add)
            end
            evaporate_pheromones!(pheromones, evaporation)

            for ant in ants
                sg = one_side_subgraph(ant, search_is_u)
                Subgraph.vertex_count(sg) == 0 && continue
                if better_than_best(compact_fg, sg, best_biclique[], θ)
                    best_biclique[] = SubGraph(copy(sg.U), copy(sg.V))
                end
                repaired = greedy_k_repair(compact_fg, sg, k, θ)
                if better_than_best(compact_fg, repaired, best_repaired[], θ)
                    best_repaired[] = repaired
                    best_iteration[] = iter
                    best_time[] = (time_ns() - t0) / 1e9
                end
            end
        end
    end

    verbose ? run_colony() : redirect_stdout(run_colony, devnull)
    best_biclique = best_biclique[]
    best_repaired = best_repaired[]
    best_iteration = best_iteration[]
    best_time = best_time[]

    if Subgraph.vertex_count(best_repaired) == 0 && Subgraph.vertex_count(best_biclique) > 0
        best_repaired = greedy_k_repair(compact_fg, best_biclique, k, θ)
    end
    if Subgraph.vertex_count(best_repaired) == 0
        verbose && println("One-side ACO: no θ-feasible biclique")
        return empty_result
    end

    best = remap_subgraph(remapping, best_repaired)
    biclique = remap_subgraph(remapping, best_biclique)
    biclique_edges = Subgraph.edge_count(compact_fg, best_biclique)
    repaired_edges = Subgraph.edge_count(compact_fg, best_repaired)
    verbose && println("One-side ACO best: biclique |U|=$(length(biclique.U)) " *
                       "|V|=$(length(biclique.V)) edges=$biclique_edges → " *
                       "repaired |U|=$(length(best.U)) |V|=$(length(best.V)) " *
                       "edges=$repaired_edges")
    return (;
        best,
        biclique,
        search_is_u,
        n_same=search_is_u ? nU : nV,
        n_opp=search_is_u ? nV : nU,
        iterations_to_best=best_iteration,
        time_to_best_s=best_time,
        biclique_edges,
        repaired_edges,
    )
end
