using Test

include(joinpath(@__DIR__, "suite.jl"))

function parse_mode()
    if "--binary" in ARGS
        return BranchMode.binary
    else
        return BranchMode.pivot
    end
end

function parse_reduction()
    for arg in ARGS
        if startswith(arg, "--reduce=")
            val = split(arg, "=", limit=2)[2]
            if val == "none"
                return ReductionMode.none
            elseif val == "simple"
                return ReductionMode.simple
            elseif val == "progressive"
                return ReductionMode.progressive
            elseif val == "all"
                return ReductionMode.all_reductions
            else
                error("Unknown reduction mode: $val (expected none, simple, or progressive)")
            end
        end
    end
    return ReductionMode.progressive
end

const SEED = parse_seed()
const N = parse_N(20)
Random.seed!(SEED)
const MODE = parse_mode()
const REDUCTION = parse_reduction()
const SAVE_PATH = parse_save()

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

    summary = run_graph_suite(N=N, seed=SEED, solve_fn=solve_mkdb, oracle=OracleMode.brute_force,
        algorithm="mkdb")
    mismatches = mismatched_trials(summary)

    if !isempty(mismatches)
        println()
        println("find_kmdb! disagreed with brute force on $(length(mismatches)) trial(s).")
        print_mismatched_trials(summary, mismatches)
    end

    print_suite_summary(summary)
    if SAVE_PATH !== nothing
        save_suite_json(SAVE_PATH, summary)
    end
    @test isempty(mismatches)
end
