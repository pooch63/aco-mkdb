#=
Super-heuristic: alternate super-reduction with the θ-heuristic, and call ACO
only after θ stops raising the edge lower bound.

1. Run `super_reduce!` (defect weights, common neighbors, neighborhood weights).
2. Run the θ-heuristic. If its edge count beats the incumbent, run stage 2
   again with that solution, then the full reduction, and repeat θ.
3. When θ cannot improve, run ACO once. If ACO beats the incumbent, run the
   full reduction with that solution and return to step 2.
4. Stop after `patience` consecutive cycles with no new best, or `max_cycles`.
=#

const __SUPER_HEURISTIC_JL__ = true

isdefined(@__MODULE__, :__SUPER_REDUCE_JL__) || include("super-reduce.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__TABU_JL__) || include("tabu.jl")
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath("aco", "algorithm.jl"))

struct SuperHeuristicTrace
    sol::SubGraph
    best_edges::Int
    theta_calls::Int
    aco_calls::Int
    theta_improvements::Int
    aco_improvements::Int
    cycles::Int
    stall::Int
    time_to_best_s::Float64
    cycle_of_best::Int
    trace::Vector{String}
end

function _copy_solution(sg::SubGraph)
    return SubGraph(copy(sg.U), copy(sg.V))
end

"""Edge count of a θ-feasible k-defective biclique, or 0 when `sg` is not one."""
function _feasible_edges(g::BipartiteGraph, sg::SubGraph, k::Int, θ::Int)
    length(sg.U) >= θ && length(sg.V) >= θ || return 0
    for u in sg.U
        haskey(g.adjU, u) || return 0
    end
    for v in sg.V
        haskey(g.adjV, v) || return 0
    end
    fg = freeze(g)
    Subgraph.missing_edges(fg, sg) <= k || return 0
    return Subgraph.edge_count(fg, sg)
end

function _run_theta(g::BipartiteGraph, k::Int, θ::Int)
    if length(g.adjU) < θ || length(g.adjV) < θ
        return SubGraph(), 0
    end
    fg = freeze(g)
    sg = theta_based_heuristic(fg, k, θ; return_invalid=false)
    edges = _feasible_edges(g, sg, k, θ)
    return edges == 0 ? SubGraph() : _copy_solution(sg), edges
end

function _run_aco(g::BipartiteGraph, k::Int, θ::Int, ants::Int, iterations::Int,
        pheromone::Int, evaporation::Float64, prefer_smaller_side::Bool,
        neighbor_scope_limit::Bool, eta_exp::Float64)
    if length(g.adjU) < θ || length(g.adjV) < θ
        return SubGraph(), 0
    end
    sols, _, _, _, _ = aco(g, pheromone, ants, iterations, evaporation, k, θ, 1;
        parallelize=false,
        prefer_smaller_side=prefer_smaller_side,
        neighbor_scope_limit=neighbor_scope_limit,
        elite_seed=true,
        eta_exp=eta_exp,
        reduction=ReductionMode.none)
    best = SubGraph()
    best_edges = 0
    for sg in sols
        edges = _feasible_edges(g, sg, k, θ)
        if edges > best_edges
            best_edges = edges
            best = _copy_solution(sg)
        end
    end
    return best, best_edges
end

function _graph_brief(g::BipartiteGraph)
    return "nU=$(length(g.adjU)) nV=$(length(g.adjV)) |E|=$(_edge_count(g))"
end

"""
    super_heuristic!(g, k, θ; patience=2, max_cycles=8, ants=100, iterations=5, ...)

Mutates `g` by repeated `super_reduce!`. Returns the best θ-feasible solution
found and how many θ-heuristic and ACO calls produced it.
"""
function super_heuristic!(g::BipartiteGraph, k::Int, θ::Int;
        patience::Int=2, max_cycles::Int=8,
        ants::Int=100, iterations::Int=5, pheromone::Int=1,
        evaporation::Float64=0.95,
        prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true,
        eta_exp::Real=ETA_EXP, verbose::Bool=true)
    patience >= 1 || throw(ArgumentError("patience must be ≥ 1, got $patience"))
    max_cycles >= 1 || throw(ArgumentError("max_cycles must be ≥ 1, got $max_cycles"))
    ηexp = Float64(eta_exp)

    best = SubGraph()
    best_edges = 0
    incumbent::Union{Nothing,SubGraph} = nothing
    theta_calls = 0
    aco_calls = 0
    theta_improvements = 0
    aco_improvements = 0
    stall = 0
    cycles = 0
    t0 = time()
    time_to_best = 0.0
    cycle_of_best = 0
    trace = String[]

    function note(msg::AbstractString)
        push!(trace, String(msg))
        verbose && println("[super-heuristic] ", msg)
        return nothing
    end

    note("initial super-reduce  $(_graph_brief(g))")
    super_reduce!(g, k, θ; incumbent=incumbent)
    note("after initial super-reduce  $(_graph_brief(g))")

    while cycles < max_cycles
        cycles += 1
        improved = false
        note("cycle $cycles  $(_graph_brief(g))")

        # θ until it cannot raise the edge lower bound.
        theta_passes = 0
        while theta_passes < 32
            theta_passes += 1
            theta_calls += 1
            sg, edges = _run_theta(g, k, θ)
            if edges > best_edges
                best_edges = edges
                best = sg
                incumbent = sg
                time_to_best = time() - t0
                cycle_of_best = cycles
                theta_improvements += 1
                improved = true
                note("theta edges=$edges  new-best  $(_graph_brief(g))")
                super_reduce_stage2!(g, k, θ; incumbent=incumbent)
                super_reduce!(g, k, θ; incumbent=incumbent)
                note("after theta reduce  $(_graph_brief(g))")
            else
                note("theta edges=$edges  best=$best_edges  no improvement")
                break
            end
        end

        aco_calls += 1
        sg, edges = _run_aco(g, k, θ, ants, iterations, pheromone, evaporation,
            prefer_smaller_side, neighbor_scope_limit, ηexp)
        if edges > best_edges
            best_edges = edges
            best = sg
            incumbent = sg
            time_to_best = time() - t0
            cycle_of_best = cycles
            aco_improvements += 1
            improved = true
            note("aco edges=$edges  new-best  $(_graph_brief(g))")
            super_reduce!(g, k, θ; incumbent=incumbent)
            note("after aco reduce  $(_graph_brief(g))")
        else
            note("aco edges=$edges  best=$best_edges  no improvement")
        end

        if improved
            stall = 0
        else
            stall += 1
            stall >= patience && break
        end
    end

    note("stop  cycles=$cycles  stall=$stall  best_edges=$best_edges  " *
         "theta_calls=$theta_calls  aco_calls=$aco_calls  " *
         "theta_improvements=$theta_improvements  aco_improvements=$aco_improvements")
    return SuperHeuristicTrace(best, best_edges, theta_calls, aco_calls,
        theta_improvements, aco_improvements, cycles, stall, time_to_best,
        cycle_of_best, trace)
end
