using Test

include(joinpath(@__DIR__, "suite.jl"))

function parse_mode()
    if "--binary" in ARGS
        return binary
    else
        return pivot
    end
end

function parse_reduction()
    for arg in ARGS
        if startswith(arg, "--reduce=")
            val = split(arg, "=", limit=2)[2]
            if val == "none"
                return none
            elseif val == "simple"
                return simple
            elseif val == "progressive"
                return progressive
            else
                error("Unknown reduction mode: $val (expected none, simple, or progressive)")
            end
        end
    end
    return progressive
end

const SEED = parse_seed()
const N = parse_N(20)
Random.seed!(SEED)
const MODE = parse_mode()
const REDUCTION = parse_reduction()

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

function solve_mkdb(g::FrozenBipartite, k::Int, θ::Int)
    mutable_graph = build_mutable_graph(g)
    return find_kmdb!(mutable_graph, true, MODE, k, θ, REDUCTION)
end

"""
Exact-search correctness vs brute force.

When no θ-feasible k-MDB exists (`opt_edges == 0`), the solver should return empty.
Otherwise it must return a valid solution with exactly `opt_edges` edges.
"""
function trial_matches(info::TrialInfo, r::TrialResult)
    if info.opt_edges == 0
        return isempty(r.U) && isempty(r.V)
    end
    return r.valid && r.edges == info.opt_edges
end

function mismatched_trials(summary::SuiteSummary)
    s = only_stats(summary)
    return [trial for trial in 1:summary.N
            if !trial_matches(summary.trials[trial], s.results[trial])]
end

function print_mismatched_trials(summary::SuiteSummary, mismatches)
    for trial in mismatches
        print_trial_detail(summary, trial)
    end
end

@testset "find_kmdb! matches brute force" begin
    println("Branch mode: $MODE")
    println("Reduction: $REDUCTION")

    summary = run_graph_suite(N=N, seed=SEED, solve_fn=solve_mkdb)
    mismatches = mismatched_trials(summary)

    if !isempty(mismatches)
        println()
        println("find_kmdb! disagreed with brute force on $(length(mismatches)) trial(s).")
        print_mismatched_trials(summary, mismatches)
    end

    print_suite_summary(summary)
    @test isempty(mismatches)
end
