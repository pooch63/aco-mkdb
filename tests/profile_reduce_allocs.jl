# Allocation profile for reduce_graph! only.
#
# Usage:
#   # Step A — overview (%gc, alloc count):
#   julia tests/profile_reduce_allocs.jl
#
#   # Step B — line-level .mem files (run AFTER step A once, or alone):
#   julia --track-allocation=user tests/profile_reduce_allocs.jl
#   # then inspect src/reduction.jl.mem (and graph.jl.mem)
#
# Same CLI flags as suite / test_reduction_modes --time:
#   --seed=  --nU=  --nV=  --edge-prob=  --theta-max=  --k-max=

include(joinpath(@__DIR__, "suite.jl"))
isdefined(@__MODULE__, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))

using Profile

function main()
    defaults = suite_cli_defaults()
    # Prefer a single reproducible graph: --seed=X with N-style fixed seed.
    seed = defaults.seed
    nU_range = defaults.nU_range
    nV_range = defaults.nV_range
    edge_prob = defaults.edge_prob
    θ_max = defaults.θ_max
    k_max = defaults.k_max

    edges, nU, nV, k, θ = random_graph_from_seed(seed;
        nU_range=nU_range, nV_range=nV_range, edge_prob=edge_prob,
        θ_max=θ_max, k_max=k_max)
    fg = build_frozen(edges, nU, nV)

    println("Allocation profile: reduce_graph!")
    println("  seed=$seed  nU=$nU  nV=$nV  |E|=$(length(edges))  k=$k  θ=$θ")
    println("  nU_range=$nU_range  nV_range=$nV_range  edge_prob=$edge_prob")
    println()

    # Warmup — exclude compilation from both @time and .mem
    g_warm = build_mutable_graph(fg)
    reduce_graph!(g_warm, k, θ, nU, nV)

    # Clear allocation counters so .mem only covers the measured call
    Profile.clear_malloc_data()

    g = build_mutable_graph(fg)
    println("Measured call (@time):")
    @time reduce_graph!(g, k, θ, nU, nV)
    println()
    println("If run with --track-allocation=user, open:")
    println("  src/reduction.jl.mem")
    println("  src/graph.jl.mem")
    println("Right-hand column = bytes allocated on that line.")
end

main()
