using Test
using Random
using StatsBase

include(joinpath(@__DIR__, "suite.jl"))
include(joinpath(@__DIR__, "..", "tabu.jl"))

const SEED = parse_seed()
const N = parse_N(20)
Random.seed!(SEED)

const TT = 2
const TABU_PATIENCE = 3
const REPAIR_LABELS = ("greedy", "tabu")

"""
Seed repair from a single max-degree vertex so both solvers share the same start.
"""
function initial_singleton(g::FrozenBipartite)
    G = SubGraph(Set(u for u in g.u_ids), Set(v for v in g.v_ids))
    is_u, node = argmax_nodes((u, n) -> u ? degree_u(g, n) : degree_v(g, n), G)
    return is_u ? SubGraph(Set([node]), Set{Int}()) : SubGraph(Set{Int}(), Set([node]))
end

function solve_greedy(g::FrozenBipartite, k::Int, θ::Int)
    sg = initial_singleton(g)
    greedily_add!(g, sg, k)
    return sg
end

function solve_tabu(g::FrozenBipartite, k::Int, θ::Int)
    sg = initial_singleton(g)
    tabu_repair!(g, sg, k, TT, TABU_PATIENCE)
    return sg
end

"""
Trials where tabu fitness is strictly worse than greedy (starting from the same node).
"""
function tabu_underperforms(summary::SuiteSummary)
    bad = Int[]
    for trial in 1:summary.N
        info = summary.trials[trial]
        g = build_frozen(info.edges, info.nU, info.nV)
        greedy_r = summary.stats["greedy"].results[trial]
        tabu_r = summary.stats["tabu"].results[trial]
        greedy_fit = instance_fitness(g, SubGraph(greedy_r.U, greedy_r.V))
        tabu_fit = instance_fitness(g, SubGraph(tabu_r.U, tabu_r.V))
        if tabu_fit < greedy_fit
            push!(bad, trial)
        end
    end
    return bad
end

@testset "tabu repair matches or beats greedy repair" begin
    solvers = Dict{String,Function}(
        "greedy" => solve_greedy,
        "tabu" => solve_tabu,
    )
    summary = run_graph_suite(N=N, seed=SEED, solvers=solvers)
    underperforms = tabu_underperforms(summary)

    if !isempty(underperforms)
        println()
        println("Tabu underperformed greedy on $(length(underperforms)) trial(s).")
        for trial in underperforms
            info = summary.trials[trial]
            g = build_frozen(info.edges, info.nU, info.nV)
            greedy_r = summary.stats["greedy"].results[trial]
            tabu_r = summary.stats["tabu"].results[trial]
            println("  trial $trial  greedy_fit=$(instance_fitness(g, SubGraph(greedy_r.U, greedy_r.V)))  tabu_fit=$(instance_fitness(g, SubGraph(tabu_r.U, tabu_r.V)))")
            print_solver_mismatch(summary, trial; labels=REPAIR_LABELS)
        end
    end

    print_suite_summary(summary; labels=REPAIR_LABELS)
    @test isempty(underperforms)
end
