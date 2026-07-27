using Test

include(joinpath(@__DIR__, "suite.jl"))

const SEED = parse_seed()
const N = parse_N(20)
Random.seed!(SEED)

const REDUCTION_LABELS = ("none", "lo", "hi")
const REDUCTION_MODES = (ReductionMode.none, ReductionMode.simple, ReductionMode.progressive)

function build_mutable_graph(g::FrozenBipartite)
    mutable_graph = BipartiteGraph{Nothing}()
    for u in g.u_ids
        add_u!(mutable_graph, u)
    end
    for v in g.v_ids
        add_v!(mutable_graph, v)
    end
    for u_idx in eachindex(g.u_ids)
        u = g.u_ids[u_idx]
        for k in neighbor_range_u(g, u_idx)
            v = g.v_ids[g.v_adj[k]]
            add_edge!(mutable_graph, u, v, nothing)
        end
    end
    return mutable_graph
end

function solve_reduction_mode(mode::ReductionMode)
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
    summary = run_graph_suite(N=N, seed=seed, solvers=solvers)

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
