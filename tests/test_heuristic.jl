using Test

include(joinpath(@__DIR__, "suite.jl"))

const SEED = parse_seed()
const N = parse_N(10)
Random.seed!(SEED)

function solve_heuristic(g::FrozenBipartite, k::Int, θ::Int)
    return initial_heuristic(g, k, θ)
end


@testset "heuristic suite regression" begin
    summary = run_graph_suite(N=N, seed=SEED, solve_fn=solve_heuristic)
    print_suite_summary(summary)
    s = only_stats(summary)

    @test s.valid <= summary.N
    @test s.optimal <= s.valid
    @test s.worst_ratio >= 0.0
    @test s.worst_ratio <= 1.0
end
