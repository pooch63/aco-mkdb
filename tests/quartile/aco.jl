"""
Quartile ACO tests on real-world graphs.

Usage:
  julia tests/quartile/aco.jl                     # first 5 graphs (ascending |E|)
  julia tests/quartile/aco.jl amazon/boxes        # one specific graph
  julia tests/quartile/aco.jl --N=10              # first 10 graphs
  julia tests/quartile/aco.jl --prefix=konect-small --N=3
"""

using Random

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

"""
Check whether the injected biclique (nodes and edges) is present in a reduced graph.
"""
function check_plant_presence(g_reduced::BipartiteGraph, chosen_U::Vector{Int},
                               chosen_V::Vector{Int},
                               plant_edges::Vector{Tuple{Int,Int}})
    u_present = [u for u in chosen_U if haskey(g_reduced.adjU, u)]
    v_present = [v for v in chosen_V if haskey(g_reduced.adjV, v)]
    nodes_present = (length(u_present) == length(chosen_U)) &&
                    (length(v_present) == length(chosen_V))

    surviving_edges = [e for e in plant_edges
                       if haskey(g_reduced.adjU, e[1]) &&
                          (e[2] in g_reduced.adjU[e[1]])]
    edges_present = length(surviving_edges) == length(plant_edges)

    return (; nodes_present, u_present, v_present, edges_present,
              surviving_edges, total_plant_edges=length(plant_edges))
end

const K = 2
const θ = 5

"""
Run the quartile-vs-standard ACO comparison on a single real-world graph.
"""
function test_quartile_aco_on(dataset_key::AbstractString, path::AbstractString,
                               edge_count::Int)
    println("\n── $dataset_key ──")
    println("  Loading: $path  (|E|=$edge_count)")

    g, edge_count = load_bipartite_graph(path)
    println("  Graph: |U|=$(length(g.adjU))  |V|=$(length(g.adjV))  |E|=$edge_count")

    # Inject planted biclique
    rng = MersenneTwister(1)
    chosen_U, chosen_V, inserted, missing_edges, existing =
        inject_biclique!(g, θ, θ, K, rng)
    edge_count += inserted
    println("  Injected biclique: u=$θ v=$θ k=$K  (existing=$existing inserted=$inserted missing=$(length(missing_edges)))")
    println("    planted U: $chosen_U")
    println("    planted V: $chosen_V")

    # Compute pulse thresholds
    thresh = compute_biclique_quartile_thresholds(g, chosen_U, chosen_V,
                                                  missing_edges; iterations=20)
    println("  Min quartile for all NODES: $(round(100*thresh.min_q_nodes; digits=2))%  (rank $(thresh.max_node_rank)/$(thresh.total_edges))")
    println("  Min quartile for all EDGES: $(round(100*thresh.min_q_edges; digits=2))%  (rank $(thresh.max_edge_rank)/$(thresh.total_edges))")

    # ── Quartile ACO ─────────────────────────────────────────────────────────
    g_quartile = deepcopy(g)
    sols_q, best_iters_q, best_times_q, _, _ = aco(
        g_quartile, 1, 5, 2, 0.05, K, θ, 1;
        reduction=ReductionMode.quartile, parallelize=false
    )
    sol_q   = sols_q[1]
    q_fg    = freeze(g_quartile)
    q_edges = Subgraph.edge_count(q_fg, sol_q)
    q_plant = check_plant_presence(g_quartile, chosen_U, chosen_V, thresh.plant_edges)

    plant_sg     = SubGraph(Set(chosen_U), Set(chosen_V))
    q_plant_miss = Subgraph.missing_edges(q_fg, plant_sg)

    println("  Quartile ACO: |U|=$(length(sol_q.U))  |V|=$(length(sol_q.V))  edges=$q_edges  iter=$(best_iters_q[1])  time=$(round(best_times_q[1]; digits=3))s")
    println("    Plant intact in g_quartile: $(q_plant.edges_present && q_plant_miss <= K ? "YES" : "NO")  (edges=$(Subgraph.edge_count(q_fg, plant_sg))/$(length(thresh.plant_edges))  missing=$q_plant_miss)")

    # ── Standard ACO ─────────────────────────────────────────────────────────
    g_standard = deepcopy(g)
    sols_s, best_iters_s, best_times_s, _, _ = aco(
        g_standard, 1, 5, 2, 0.05, K, θ, 1;
        reduction=ReductionMode.simple, parallelize=false
    )
    sol_s   = sols_s[1]
    s_fg    = freeze(g_standard)
    s_edges = Subgraph.edge_count(s_fg, sol_s)
    s_plant = check_plant_presence(g_standard, chosen_U, chosen_V, thresh.plant_edges)

    println("  Standard ACO: |U|=$(length(sol_s.U))  |V|=$(length(sol_s.V))  edges=$s_edges  iter=$(best_iters_s[1])  time=$(round(best_times_s[1]; digits=3))s")
    println("    Plant in g_standard: $(s_plant.nodes_present ? "YES" : "NO")  (U=$(length(s_plant.u_present))/$θ  V=$(length(s_plant.v_present))/$θ  edges=$(Subgraph.edge_count(s_fg, plant_sg))/$(length(thresh.plant_edges)))")

    # ── Comparison ───────────────────────────────────────────────────────────
    plant_U = Set(chosen_U)
    plant_V = Set(chosen_V)
    println("  Comparison:")
    println("    Quartile edges=$q_edges  overlap |U|=$(length(intersect(sol_q.U, plant_U)))/$θ  |V|=$(length(intersect(sol_q.V, plant_V)))/$θ")
    println("    Standard edges=$s_edges  overlap |U|=$(length(intersect(sol_s.U, plant_U)))/$θ  |V|=$(length(intersect(sol_s.V, plant_V)))/$θ")
    println("  ✓ passed")

    return true
end

# ── Run ────────────────────────────────────────────────────────────────────────

graphs = selected_graphs()

if isempty(graphs)
    println("No graphs found. Check that data/ is populated and --prefix= is correct.")
    exit(1)
end

println("Testing quartile ACO on $(length(graphs)) graph(s) (ascending |E|):")

failed = String[]
for g in graphs
    try
        test_quartile_aco_on(g.key, g.path, g.edges)
    catch e
        println("  ✗ FAILED: $(g.key): $e")
        push!(failed, g.key)
    end
end

println()
if isempty(failed)
    println("All $(length(graphs)) quartile-ACO test(s) passed.")
else
    println("$(length(failed)) / $(length(graphs)) test(s) FAILED:")
    for key in failed
        println("  $key")
    end
    exit(1)
end
