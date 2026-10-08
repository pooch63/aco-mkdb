"""
Pheromone-effectiveness probe on a few real-world graphs.

The paper ablation (A=100, T=5, every-ant deposit) is nearly flat in Δτ.
This file isolates whether τ can matter at all: elite_seed is off so the
only inter-epoch memory is pheromone, T is long enough for a trail to form,
and the arms are no-pheromone / every-ant / best-only / best-only with η^0.

Usage:
  julia tests/quartile/aco.jl                     # 4-graph probe (not the suite)
  julia tests/quartile/aco.jl amazon/boxes        # one specific graph
  julia tests/quartile/aco.jl --N=3               # first 3 graphs by |E|
  julia tests/quartile/aco.jl --prefix=konect-small --N=3
"""

using Random
using Statistics

include(joinpath(@__DIR__, "graph_selection.jl"))

isdefined(@__MODULE__, :__PATHS_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "pulse.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "method.jl"))
isdefined(@__MODULE__, :__LOAD_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "bin", "load.jl"))

"""
Compute what quartile would have to be set to include every node in the biclique
(as well as every edge in the biclique) based on pulse edge weights.
"""
function compute_biclique_quartile_thresholds(g::BipartiteGraph, chosen_U::Vector{Int},
                                              chosen_V::Vector{Int},
                                              missing_edges::Set{Tuple{Int,Int}};
                                              iterations::Int=20)
    fg = freeze(g)
    edge_weights = pulse_edge_weights(fg, iterations)

    edge_list = Tuple{Float64, Int, Int}[]
    for ui in 1:length(fg.u_ids)
        u_orig = fg.u_ids[ui]
        for k_edge in neighbor_range_u(fg, ui)
            v_orig = fg.v_ids[fg.v_adj[k_edge]]
            push!(edge_list, (edge_weights[k_edge], u_orig, v_orig))
        end
    end
    sort!(edge_list, by=x->x[1], rev=true)
    total_edges = length(edge_list)

    edge_rank = Dict{Tuple{Int,Int}, Int}()
    for (rank, (_, u, v)) in enumerate(edge_list)
        edge_rank[(u, v)] = rank
    end

    # For each biclique node, find the rank of its best incident edge in g
    node_best_ranks = Dict{Int, Int}()
    for u in chosen_U
        ranks = [edge_rank[(u, v)] for v in g.adjU[u] if haskey(edge_rank, (u, v))]
        node_best_ranks[u] = isempty(ranks) ? total_edges : minimum(ranks)
    end
    for v in chosen_V
        ranks = [edge_rank[(u, v)] for u in g.adjV[v] if haskey(edge_rank, (u, v))]
        node_best_ranks[v] = isempty(ranks) ? total_edges : minimum(ranks)
    end

    max_node_rank = maximum(values(node_best_ranks))
    min_q_nodes   = max_node_rank / total_edges

    plant_edges = [(u, v) for u in chosen_U for v in chosen_V
                   if (u, v) ∉ missing_edges]
    edge_ranks   = [get(edge_rank, e, total_edges) for e in plant_edges]
    max_edge_rank = isempty(edge_ranks) ? 0 : maximum(edge_ranks)
    min_q_edges   = max_edge_rank / total_edges

    return (; total_edges, max_node_rank, min_q_nodes, node_best_ranks,
              max_edge_rank, min_q_edges, plant_edges)
end

const K = 2
const θ = 5

# Small enough to finish; long enough that a trail can exist. elite_seed is
# off in every arm so a quality gap is attributable to τ, not to replanting
# the incumbent.
const PROBE_ANTS = 20
const PROBE_ITERS = 20
const PROBE_SEEDS = 1:2
const PROBE_EVAP = 0.95

# Default probe set: two easy ceilings, one late-ITB graph, one graph where
# the recorded settings ablation already showed a best-only vs Δτ=0 gap.
# This is not the 38-graph suite.
const PHEROMONE_PROBE_GRAPHS = [
    "amazon/boxes",
    "konect-small/moreno",
    "konect-small/moreno_blogs",
    "konect/bitcoin",
]

const ARMS = [
    (id="no_ph",      pheromone=0, evaporation=1.00, best_only=false, eta_exp=2.0),
    (id="every_ant",  pheromone=1, evaporation=PROBE_EVAP, best_only=false, eta_exp=2.0),
    (id="best_only",  pheromone=1, evaporation=PROBE_EVAP, best_only=true,  eta_exp=2.0),
    (id="best_η0",    pheromone=1, evaporation=PROBE_EVAP, best_only=true,  eta_exp=0.0),
]

function probe_graph_list()
    user_chose = any(a -> !startswith(a, "-") ||
                           startswith(a, "--N=") ||
                           startswith(a, "--prefix="), ARGS)
    user_chose && return selected_graphs()

    graphs = NamedTuple{(:key, :path, :edges), Tuple{String,String,Int}}[]
    for key in PHEROMONE_PROBE_GRAPHS
        path = resolve_graph_path(key; data_root=_QUARTILE_DATA_ROOT)
        edges = isfile(path) ? count_indexed_edges(path) : 0
        push!(graphs, (key=key, path=path, edges=edges))
    end
    return graphs
end

function run_arm(g::BipartiteGraph, arm, seed::Int)
    g_run = deepcopy(g)
    Random.seed!(seed)
    sols, best_iters, best_times, pheromones, remapping = redirect_stdout(devnull) do
        aco(
            g_run, arm.pheromone, PROBE_ANTS, PROBE_ITERS, arm.evaporation, K, θ, 1;
            reduction=ReductionMode.simple,
            parallelize=false,
            prefer_smaller_side=false,
            neighbor_scope_limit=true,
            elite_seed=false,
            best_only_pheromone=arm.best_only,
            eta_exp=arm.eta_exp,
        )
    end
    sol = sols[1]
    fg = freeze(g_run)
    edges = Subgraph.edge_count(fg, sol)
    feasible = length(sol.U) ≥ θ && length(sol.V) ≥ θ
    trail = pheromone_cut_stats(pheromones, remapping, sol; species=1)
    return (;
        edges, feasible,
        nU=length(sol.U), nV=length(sol.V),
        itb=best_iters[1],
        time=best_times[1],
        τ_min=trail.global_min_τ,
        τ_max=trail.global_max_τ,
        on_floor=trail.on_trail_floor,
        cut_pct=100 * trail.cut_fraction,
    )
end

function test_pheromone_on(dataset_key::AbstractString, path::AbstractString,
                           edge_count::Int)
    println("\n── $dataset_key ──")
    println("  Loading: $path  (|E|=$edge_count)")

    g, edge_count = load_bipartite_graph(path)
    println("  Graph: |U|=$(length(g.adjU))  |V|=$(length(g.adjV))  |E|=$edge_count")

    rng = MersenneTwister(1)
    chosen_U, chosen_V, inserted, missing_edges, existing =
        inject_biclique!(g, θ, θ, K, rng)
    edge_count += inserted
    println("  Injected biclique: u=$θ v=$θ k=$K  (existing=$existing inserted=$inserted missing=$(length(missing_edges)))")

    thresh = compute_biclique_quartile_thresholds(g, chosen_U, chosen_V,
                                                  missing_edges; iterations=20)
    println("  Min quartile for all NODES: $(round(100*thresh.min_q_nodes; digits=2))%  (rank $(thresh.max_node_rank)/$(thresh.total_edges))")
    println("  Min quartile for all EDGES: $(round(100*thresh.min_q_edges; digits=2))%  (rank $(thresh.max_edge_rank)/$(thresh.total_edges))")
    println("  Arms: A=$PROBE_ANTS  T=$PROBE_ITERS  elite_seed=false  F_N  CNN")
    println("  $(rpad("arm", 12)) $(lpad("seed", 4)) $(lpad("|E|", 5)) θ  $(lpad("ITB", 3)) $(lpad("t(s)", 7))  τ-range  floor  cut%")

    by_arm = Dict{String, Vector{NamedTuple}}()
    for arm in ARMS
        rows = NamedTuple[]
        for seed in PROBE_SEEDS
            row = run_arm(g, arm, seed)
            push!(rows, row)
            τlo = row.τ_min === nothing ? NaN : row.τ_min
            τhi = row.τ_max === nothing ? NaN : row.τ_max
            println("  $(rpad(arm.id, 12)) $(lpad(string(seed), 4)) $(lpad(string(row.edges), 5)) $(row.feasible ? "Y" : "n")  $(lpad(string(row.itb), 3)) $(lpad(string(round(row.time; digits=3)), 7))  [$(round(τlo; digits=3)), $(round(τhi; digits=3))]  $(row.on_floor ? "Y" : "n")  $(round(row.cut_pct; digits=1))")
        end
        by_arm[arm.id] = rows
    end

    println("  Means over seeds $(collect(PROBE_SEEDS)):")
    no_ph = by_arm["no_ph"]
    no_ph_mean = mean(r.edges for r in no_ph)
    for arm in ARMS
        rows = by_arm[arm.id]
        μE = mean(r.edges for r in rows)
        feas = mean(r.feasible for r in rows)
        μITB = mean(r.itb for r in rows)
        delta = μE - no_ph_mean
        println("    $(rpad(arm.id, 12)) |E|=$(round(μE; digits=1))  Δvs no_ph=$(lpad(string(round(delta; digits=1)), 6))  feas=$(round(100*feas; digits=0))%  ITB=$(round(μITB; digits=1))")
    end
    println("  ✓ passed")
    return true
end

# ── Run ────────────────────────────────────────────────────────────────────────

graphs = probe_graph_list()

if isempty(graphs)
    println("No graphs found. Check that data/ is populated and --prefix= is correct.")
    exit(1)
end

println("Pheromone probe on $(length(graphs)) graph(s).")
println("This is not the 38-graph suite.")

failed = String[]
for g in graphs
    try
        test_pheromone_on(g.key, g.path, g.edges)
    catch e
        println("  ✗ FAILED: $(g.key): $e")
        showerror(stdout, e, catch_backtrace())
        println()
        push!(failed, g.key)
    end
end

println()
if isempty(failed)
    println("All $(length(graphs)) pheromone-probe run(s) finished.")
else
    println("$(length(failed)) / $(length(graphs)) run(s) FAILED:")
    for key in failed
        println("  $key")
    end
    exit(1)
end
