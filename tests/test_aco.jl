using Test
using Base.Threads

include(joinpath(@__DIR__, "suite.jl"))
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(@__DIR__, "..", "src", "method.jl"))

const SAVE_PATH = parse_save()
const TIME_MODE = "--time" in ARGS

# Match load.jl defaults unless overridden.
const ACO_PHEROMONE = parse_int_flag("pheremone", 1)
const ACO_ANTS = parse_int_flag("ants", 10)
const ACO_ITERATIONS = parse_int_flag("iterations", 10)
const ACO_EVAPORATION = parse_float_flag("evaporation", 0.95)
const ACO_SUBSPECIES = parse_int_flag("subspecies", 1)
const ACO_PREFER_SMALLER_SIDE = parse_bool_flag("prefer-smaller-side", true)
const ACO_NEIGHBOR_SCOPE_LIMIT = parse_bool_flag("neighbor-scope-limit", true)
const ACO_ELITE_SEED = parse_bool_flag("elite-seed", true)
const ACO_ELITE_SEED_ANTS = parse_int_flag("elite-seed-ants", 3)
const ACO_ELITE_SEED_REMOVE = parse_int_flag("elite-seed-remove", 2)

function parse_parallelize(default::Bool=true)
    for arg in ARGS
        if startswith(arg, "--parallelize=")
            val = split(arg, "=", limit=2)[2]
            if val == "none" || val == "false"
                return false
            elseif val == "true" || val == "threads"
                return true
            else
                error("Unknown --parallelize=$val (expected none, false, true, or threads)")
            end
        end
    end
    return default
end

const PARALLELIZE = parse_parallelize(true)

# Accumulated wall time across trials when --time is set.
const TIMINGS = Float64[]

function make_aco_method()
    return ACOMethod(;
        pheromone=ACO_PHEROMONE,
        ants=ACO_ANTS,
        iterations=ACO_ITERATIONS,
        evaporation=ACO_EVAPORATION,
        subspecies=ACO_SUBSPECIES,
        parallelize=PARALLELIZE,
        prefer_smaller_side=ACO_PREFER_SMALLER_SIDE,
        neighbor_scope_limit=ACO_NEIGHBOR_SCOPE_LIMIT,
        elite_seed=ACO_ELITE_SEED,
        elite_seed_ants=ACO_ELITE_SEED_ANTS,
        elite_seed_remove=ACO_ELITE_SEED_REMOVE,
    )
end

function solve_aco(g::FrozenBipartite, k::Int, θ::Int)
    method = make_aco_method()
    if TIME_MODE
        t = @elapsed begin
            sol = as_suite_solver(method)(g, k, θ)
        end
        push!(TIMINGS, t)
        println("  aco time: $(round(t; digits=3))s  (parallelize=$(PARALLELIZE), threads=$(nthreads()))")
        return sol
    else
        return as_suite_solver(method)(g, k, θ)
    end
end

# Benchmark ACO against a branch-and-bound oracle.
# Does not fail on suboptimality — use --save= and tests/compare.jl to compare
# algorithms across identical graph seeds.
#
# Use -t N (or -t auto) so --parallelize can actually use multiple threads.
# Compare speed with two runs that share --seed and --N:
#   julia -t auto tests/test_aco.jl --time --seed=1 --N=5
#   julia -t auto tests/test_aco.jl --time --parallelize=none --seed=1 --N=5
#
# Examples:
#   julia -t auto tests/test_aco.jl --seed=1 --N=5 --save=aco.json
#   julia -t auto tests/test_aco.jl --nU=1000:2000 --nV=1000:2000 --ants=20 --iterations=50 --save=out.json
#   julia -t auto tests/test_aco.jl --time --seed=1 --N=3
#   julia tests/test_aco.jl --time --parallelize=none --seed=1 --N=3
#   julia -t auto tests/test_aco.jl --jitter=2 --seed=1 --N=5 --no-optimum

println("ACO benchmark" * (PARALLELIZE ? " (parallel)" : " (sequential)"))
println("  aco: ants=$(ACO_ANTS) iterations=$(ACO_ITERATIONS) pheromone=$(ACO_PHEROMONE) evaporation=$(ACO_EVAPORATION) subspecies=$(ACO_SUBSPECIES)")
println("  prefer_smaller_side=$(ACO_PREFER_SMALLER_SIDE)  neighbor_scope_limit=$(ACO_NEIGHBOR_SCOPE_LIMIT)  elite_seed=$(ACO_ELITE_SEED) (ants=$(ACO_ELITE_SEED_ANTS), remove=$(ACO_ELITE_SEED_REMOVE))")
println("  parallelize=$(PARALLELIZE)  threads=$(nthreads())")
TIME_MODE && println("  timing enabled (--time)")

summary = run_graph_suite(solve_fn=solve_aco, algorithm="aco")

if TIME_MODE && !isempty(TIMINGS)
    println()
    println("==================== TIMING =====================")
    println("Trials timed         : $(length(TIMINGS))")
    println("Total ACO time       : $(round(sum(TIMINGS); digits=3))s")
    println("Average ACO time     : $(round(sum(TIMINGS) / length(TIMINGS); digits=3))s")
    println("Min / max ACO time   : $(round(minimum(TIMINGS); digits=3))s / $(round(maximum(TIMINGS); digits=3))s")
    println("parallelize=$(PARALLELIZE)  threads=$(nthreads())")
    println("=================================================")
end

if SAVE_PATH !== nothing
    save_suite_json(SAVE_PATH, summary)
end

@testset "aco benchmark smoke checks" begin
    s = only_stats(summary)
    @test s.valid <= summary.N
    @test s.optimal <= s.valid
    @test length(s.results) == summary.N
end

print_suite_summary(summary)
