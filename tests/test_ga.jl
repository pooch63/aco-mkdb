using Test
using Random

include(joinpath(@__DIR__, "suite.jl"))
# ga.jl pulls fitness/tabu/search; GRAPH defaults off so Makie is not required.
include(joinpath(@__DIR__, "..", "ga.jl"))

const SEED = parse_seed()
const N = parse_N(5)
const ORACLE = parse_oracle(OracleMode.branch_oracle)
const SAVE_PATH = parse_save()
const NU_RANGE = parse_int_range("nU", 100:500)
const NV_RANGE = parse_int_range("nV", 100:500)
const EDGE_PROB = parse_float_flag("edge-prob", 0.3)
const THETA_MAX = parse_int_flag("theta-max", 20)
const K_MAX = parse_int_flag("k-max", 4)

# Match load.jl defaults unless overridden.
const GA_POP = parse_int_flag("ga-N", 10)
const GA_O = parse_int_flag("ga-O", 2)
const GA_GENERATIONS = parse_int_flag("ga-gens", 500)
const GA_K_MUTATE = parse_float_flag("ga-k-mutate", 0.02)

Random.seed!(SEED)

function solve_ga(g::FrozenBipartite, k::Int, θ::Int)
    # Reset GA globals that accumulate across generations/trials.
    global U = Set{Int}()
    global V = Set{Int}()
    mutable_graph = build_mutable_graph(g)
    return ga(mutable_graph, k, θ, GA_POP, GA_O, GA_K_MUTATE, GA_GENERATIONS; repair=RepairMode.mixed)
end

# Benchmark the genetic algorithm against a branch-and-bound oracle.
# Does not fail on suboptimality — use --save= and tests/compare.jl to compare
# against heuristic (or other) runs that share the same --seed and --N.
#
# Examples:
#   julia tests/test_ga.jl --seed=1 --N=5 --save=ga.json
#   julia tests/test_ga.jl --nU=1000:2000 --nV=1000:2000 --ga-gens=100 --save=ga.json

println("GA benchmark")
println("  oracle=$(ORACLE)  nU=$(NU_RANGE)  nV=$(NV_RANGE)  edge_prob=$(EDGE_PROB)  θ_max=$(THETA_MAX)  k_max=$(K_MAX)")
println("  ga: N=$(GA_POP) O=$(GA_O) gens=$(GA_GENERATIONS) k_mutate=$(GA_K_MUTATE)")

summary = run_graph_suite(
    N=N,
    seed=SEED,
    solve_fn=solve_ga,
    oracle=ORACLE,
    algorithm="ga",
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

@testset "ga benchmark smoke checks" begin
    s = only_stats(summary)
    @test s.valid <= summary.N
    @test s.optimal <= s.valid
    @test length(s.results) == summary.N
end
