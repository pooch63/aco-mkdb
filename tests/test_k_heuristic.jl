using Test

include(joinpath(@__DIR__, "suite.jl"))
isdefined(@__MODULE__, :__K_HEURISTIC_JL__) || include(joinpath(@__DIR__, "..", "k_heuristic.jl"))

const SAVE_PATH = parse_save()

function solve_k_heuristic(g::FrozenBipartite, k::Int, θ::Int)
    return k_based_heuristic(g, k, θ; return_invalid=true)
end

# Benchmark the k-based deletion heuristic against a branch-and-bound oracle.
# Does not fail on suboptimality — use --save= and tests/compare.jl to compare
# algorithms across identical graph seeds.
#
# Examples:
#   julia tests/test_k_heuristic.jl --seed=1 --N=5 --save=k_heuristic.json
#   julia tests/test_k_heuristic.jl --nU=1000:2000 --nV=1000:2000 --N=3 --save=out.json

println("k-heuristic benchmark")

summary = run_graph_suite(solve_fn=solve_k_heuristic, algorithm="k_heuristic")

print_suite_summary(summary)

if SAVE_PATH !== nothing
    save_suite_json(SAVE_PATH, summary)
end

@testset "k-heuristic benchmark smoke checks" begin
    s = only_stats(summary)
    @test s.valid <= summary.N
    @test s.optimal <= s.valid
    @test length(s.results) == summary.N
end
