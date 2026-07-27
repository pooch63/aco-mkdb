using Test

include(joinpath(@__DIR__, "suite.jl"))

const SEED = parse_seed()
const N = parse_N(20)
Random.seed!(SEED)

const REDUCTION_LABELS = ("none", "lo", "hi")
const REDUCTION_MODES = (ReductionMode.none, ReductionMode.simple, ReductionMode.progressive)

function solve_reduction_mode(mode::ReductionMode.T)
    return function (g::FrozenBipartite, k::Int, θ::Int)
        mutable_graph = build_mutable_graph(g)
        return find_kmdb(mutable_graph, true, BranchMode.pivot, k, θ, mode)
    end
end

function compare_reduction_settings(; N=20, seed=nothing)
    solvers = Dict{String,Function}(
        label => solve_reduction_mode(mode)
        for (label, mode) in zip(REDUCTION_LABELS, REDUCTION_MODES)
    )
    summary = run_graph_suite(N=N, seed=seed, solvers=solvers, oracle=OracleMode.brute_force,
        nU_range=3:6, nV_range=3:6, edge_prob=0.5, θ_max=nothing, k_max=4)

    mismatches = Int[]
    for trial in 1:summary.N
        edge_counts = (summary.stats[label].results[trial].edges for label in REDUCTION_LABELS)
        if length(Set(edge_counts)) > 1
            push!(mismatches, trial)
        end
    end

    return summary, mismatches
end

@testset "reduction settings preserve the same optimal solution" begin
    summary, mismatches = compare_reduction_settings(N=N, seed=SEED)

    if !isempty(mismatches)
        println()
        println("Reduction settings disagreed on $(length(mismatches)) trial(s).")
        for trial in mismatches
            print_solver_mismatch(summary, trial; labels=REDUCTION_LABELS)
        end
    end

    print_suite_summary(summary; labels=REDUCTION_LABELS)

    @test isempty(mismatches)
end
