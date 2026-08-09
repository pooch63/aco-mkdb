using Test

include(joinpath(@__DIR__, "suite.jl"))

const TIME_MODE = "--time" in ARGS
const REDUCTION_LABELS = ("none", "lo", "hi")
const REDUCTION_MODES = (ReductionMode.none, ReductionMode.simple, ReductionMode.progressive)

function solve_reduction_mode(mode::ReductionMode.T)
    return function (g::FrozenBipartite, k::Int, θ::Int)
        mutable_graph = build_mutable_graph(g)
        sols = find_kmdb(mutable_graph, true, BranchMode.pivot, k, θ, mode)
        return isempty(sols) ? SubGraph() : first(sols)
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

function _graph_kwargs_from_cli()
    defaults = suite_cli_defaults()
    return (
        nU_range = defaults.nU_range,
        nV_range = defaults.nV_range,
        edge_prob = defaults.edge_prob,
        θ_max = defaults.θ_max,
        k_max = defaults.k_max,
    )
end

function _count_edges(g::BipartiteGraph)
    return sum(length(nbrs) for nbrs in values(g.adjU); init=0)
end

"""
Time reduction on large random graphs. Uses the same CLI flags as `suite.jl`:

  --N=  --seed=  --nU=  --nV=  --edge-prob=  --theta-max=  --k-max=

Examples:
  julia tests/test_reduction_modes.jl --time
  julia tests/test_reduction_modes.jl --time --nU=2000:3000 --nV=2000:3000 --N=3 --seed=1
"""
function benchmark_reduction(; N::Int, seed, graph_kwargs...)
    graph = Dict{Symbol,Any}(pairs(_graph_kwargs_from_cli()))
    for (k, v) in pairs(graph_kwargs)
        graph[k] = v
    end

    Random.seed!(seed)
    graph_seeds = Vector{UInt64}(undef, N)
    for trial in 1:N
        graph_seeds[trial] = N == 1 ? UInt64(seed) : rand(UInt64)
    end

    println("Reduction timing benchmark")
    println("  trials=$(N)  seed=$(seed)")
    println("  nU=$(graph[:nU_range])  nV=$(graph[:nV_range])  edge_prob=$(graph[:edge_prob])  θ_max=$(graph[:θ_max])  k_max=$(graph[:k_max])")
    println()

    totals = Dict(
        :reduce => 0.0,
        :freeze => 0.0,
        :none => 0.0,
        :lo => 0.0,
        :hi => 0.0,
    )

    for trial in 1:N
        graph_seed = graph_seeds[trial]
        edges, nU, nV, k, θ, _, _ = random_graph_from_seed(graph_seed; graph...)
        fg = build_frozen(edges, nU, nV)
        nE = length(edges)

        println("--- trial $trial / $N  (--seed=$graph_seed --N=1) ---")
        println("  input: nU=$nU  nV=$nV  |E|=$nE  k=$k  θ=$θ")

        g = build_mutable_graph(fg)
        t_reduce = @elapsed reduce_graph!(g, k, θ, nU, nV)
        nU_r, nV_r, nE_r = length(g.adjU), length(g.adjV), _count_edges(g)
        t_freeze = @elapsed freeze(g)

        totals[:reduce] += t_reduce
        totals[:freeze] += t_freeze
        println("  reduce_graph!       : $(round(t_reduce; digits=3))s  -> |U|=$nU_r |V|=$nV_r |E|=$nE_r")
        println("  freeze (after lo)   : $(round(t_freeze; digits=3))s")

        for (label, mode) in zip(REDUCTION_LABELS, REDUCTION_MODES)
            g_mode = build_mutable_graph(fg)
            t_mode = @elapsed result = apply_graph_reductions!(g_mode, k, θ, nU, nV, false, mode)
            totals[Symbol(label)] += t_mode
            nE_out = sum(degree_u(result, u) for u in result.u_ids; init=0)
            println("  apply ($label)        : $(round(t_mode; digits=3))s  -> |U|=$(length(result.u_ids)) |V|=$(length(result.v_ids)) |E|=$nE_out")
        end
        println()
    end

    if N > 1
        println("--- averages over $N trials ---")
        println("  reduce_graph!       : $(round(totals[:reduce] / N; digits=3))s")
        println("  freeze (after lo)   : $(round(totals[:freeze] / N; digits=3))s")
        for label in REDUCTION_LABELS
            println("  apply ($label)        : $(round(totals[Symbol(label)] / N; digits=3))s")
        end
    end
end

if TIME_MODE
    defaults = suite_cli_defaults()
    benchmark_reduction(N=defaults.N, seed=defaults.seed)
else
    const SEED = parse_seed()
    const N = parse_N(20)
    Random.seed!(SEED)

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
end
