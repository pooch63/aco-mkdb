"""
Quartile reduction tests on real-world graphs.

Usage:
  julia tests/quartile/reduction.jl               # first 5 graphs (ascending |E|)
  julia tests/quartile/reduction.jl amazon/boxes  # one specific graph
  julia tests/quartile/reduction.jl --N=10        # first 10 graphs
  julia tests/quartile/reduction.jl --prefix=konect-small --N=3
"""

include(joinpath(@__DIR__, "graph_selection.jl"))

isdefined(@__MODULE__, :__PATHS_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "pulse.jl"))
include(joinpath(@__DIR__, "..", "suite.jl"))

const K = 2
const θ = 5

"""
Test quartile edge-weight reduction on a single real-world graph.

Verifies:
  1. The reduction returns non-negative counts.
  2. No isolated vertices remain in the mutable graph after reduction.
  3. The result of `apply_graph_reductions!` is a valid `FrozenBipartite`.
"""
function test_quartile_reduction_on(dataset_key::AbstractString, path::AbstractString,
                                    edge_count::Int)
    println("\n── $dataset_key ──")
    println("  Path : $path")
    println("  |E|  : $edge_count")

    g_raw, _ = load_bipartite_graph(path)

    # ── 1. Direct quartile_edge_reduction! ───────────────────────────────────
    g1 = deepcopy(g_raw)
    n_edges_kept, n_u_kept, n_v_kept = quartile_edge_reduction!(g1, 0.5, θ)

    println("  After quartile_edge_reduction!(q=0.5):")
    println("    |U|=$n_u_kept  |V|=$n_v_kept  |E|=$n_edges_kept")

    @assert n_u_kept >= 0 "n_u_kept < 0 on $dataset_key"
    @assert n_v_kept >= 0 "n_v_kept < 0 on $dataset_key"
    @assert n_edges_kept >= 0 "n_edges_kept < 0 on $dataset_key"

    for (u, nbrs) in g1.adjU
        isempty(nbrs) && error("Isolated U-vertex $u remains after reduction on $dataset_key")
    end
    for (v, nbrs) in g1.adjV
        isempty(nbrs) && error("Isolated V-vertex $v remains after reduction on $dataset_key")
    end

    # ── 2. apply_graph_reductions! (quartile mode) ───────────────────────────
    g2 = deepcopy(g_raw)
    nU = length(g2.adjU)
    nV = length(g2.adjV)
    result = apply_graph_reductions!(g2, K, θ, nU, nV, false, ReductionMode.quartile)

    r_nU = length(result.u_ids)
    r_nV = length(result.v_ids)
    r_E  = sum(degree_u(result, u) for u in result.u_ids; init=0)
    println("  After apply_graph_reductions!(quartile):")
    println("    |U|=$r_nU  |V|=$r_nV  |E|=$r_E")

    result isa FrozenBipartite ||
        error("apply_graph_reductions! did not return FrozenBipartite on $dataset_key")
    @assert r_nU >= 0
    @assert r_nV >= 0

    println("  ✓ passed")
end

# ── Run ────────────────────────────────────────────────────────────────────────

graphs = selected_graphs()

if isempty(graphs)
    println("No graphs found. Check that data/ is populated and --prefix= is correct.")
    exit(1)
end

println("Testing quartile reduction on $(length(graphs)) graph(s) (ascending |E|):")

failed = String[]
for g in graphs
    try
        test_quartile_reduction_on(g.key, g.path, g.edges)
    catch e
        println("  ✗ FAILED: $(g.key): $e")
        push!(failed, g.key)
    end
end

println()
if isempty(failed)
    println("All $(length(graphs)) quartile-reduction test(s) passed.")
else
    println("$(length(failed)) / $(length(graphs)) test(s) FAILED:")
    for key in failed
        println("  $key")
    end
    exit(1)
end
