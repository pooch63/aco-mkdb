using Test

include(joinpath(@__DIR__, "suite.jl"))

const SEED = parse_seed()
const N = parse_N(5)
const ORACLE = parse_oracle(OracleMode.branch_oracle)
const SAVE_PATH = parse_save()
const NU_RANGE = parse_int_range("nU", 100:500)
const NV_RANGE = parse_int_range("nV", 100:500)
const EDGE_PROB = parse_float_flag("edge-prob", 0.3)
const THETA_MAX = parse_int_flag("theta-max", 20)
const K_MAX = parse_int_flag("k-max", 4)
Random.seed!(SEED)

function solve_heuristic(g::FrozenBipartite, k::Int, θ::Int)
    println("Heuristic is offic being run")
    return initial_heuristic(g, k, θ; return_invalid=true)
end

# Benchmark the initial heuristic against a branch-and-bound oracle.
# Does not fail on suboptimality — use --save= and tests/compare.jl to compare
# algorithms across identical graph seeds.
#
# Examples:
#   julia tests/test_heuristic.jl --seed=1 --N=5 --save=heuristic.json
#   julia tests/test_heuristic.jl --nU=1000:2000 --nV=1000:2000 --N=3 --save=out.json

println("Heuristic benchmark")
println("  oracle=$(ORACLE)  nU=$(NU_RANGE)  nV=$(NV_RANGE)  edge_prob=$(EDGE_PROB)  θ_max=$(THETA_MAX)  k_max=$(K_MAX)")

summary = run_graph_suite(
    N=N,
    seed=SEED,
    solve_fn=solve_heuristic,
    oracle=ORACLE,
    algorithm="heuristic",
    nU_range=NU_RANGE,
    nV_range=NV_RANGE,
    edge_prob=EDGE_PROB,
    θ_max=THETA_MAX,
    k_max=K_MAX,
)

print_suite_summary(summary)

if SAVE_PATH !== nothing
    save_suite_json(SAVE_PATH, summary)
end

@testset "heuristic benchmark smoke checks" begin
    s = only_stats(summary)
    @test s.valid <= summary.N
    @test s.optimal <= s.valid
    @test length(s.results) == summary.N
end
