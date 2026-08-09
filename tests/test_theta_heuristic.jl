using Test

include(joinpath(@__DIR__, "suite.jl"))
isdefined(@__MODULE__, :__THETA_HEURISTIC_JL__) || include(joinpath(@__DIR__, "..", "src", "theta_heuristic.jl"))

const SAVE_PATH = parse_save()
const INCREMENTAL = "--incremental" in ARGS

function solve_heuristic(g::FrozenBipartite, k::Int, θ::Int)
    return theta_based_heuristic(g, k, θ; incremental=INCREMENTAL, return_invalid=true)
end

# Benchmark the initial heuristic against a branch-and-bound oracle.
# Does not fail on suboptimality — use --save= and tests/compare.jl to compare
# algorithms across identical graph seeds.
#
# Examples:
#   julia tests/test_theta_heuristic.jl --seed=1 --N=5 --save=heuristic.json
#   julia tests/test_theta_heuristic.jl --nU=1000:2000 --nV=1000:2000 --N=3 --save=out.json
#   julia tests/test_theta_heuristic.jl --incremental --seed=1 --N=5
#   julia tests/test_theta_heuristic.jl --jitter=2 --seed=1 --N=5 --no-optimum

println("Heuristic benchmark" * (INCREMENTAL ? " (incremental)" : ""))

summary = run_graph_suite(solve_fn=solve_heuristic, algorithm="heuristic")

if SAVE_PATH !== nothing
    save_suite_json(SAVE_PATH, summary)
end

@testset "heuristic benchmark smoke checks" begin
    s = only_stats(summary)
    @test s.valid <= summary.N
    @test s.optimal <= s.valid
    @test length(s.results) == summary.N
end

print_suite_summary(summary)
