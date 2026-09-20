using Random

include(joinpath(@__DIR__, "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "src", "io.jl"))
include(joinpath(@__DIR__, "..", "src", "pulse.jl"))
include(joinpath(@__DIR__, "..", "src", "method.jl"))
isdefined(@__MODULE__, :__LOAD_JL__) || include(joinpath(@__DIR__, "..", "bin", "load.jl"))

"""
Compute what quartile would have to be set to include every node in the biclique
(as well as every edge in the biclique) based on pulse edge weights.
"""
function compute_biclique_quartile_thresholds(g::BipartiteGraph, chosen_U::Vector{Int}, chosen_V::Vector{Int}, missing_edges::Set{Tuple{Int,Int}}; iterations::Int=20)
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
    min_q_nodes = max_node_rank / total_edges
    
    plant_edges = [(u, v) for u in chosen_U for v in chosen_V if (u, v) ∉ missing_edges]
    edge_ranks = [get(edge_rank, e, total_edges) for e in plant_edges]
    max_edge_rank = isempty(edge_ranks) ? 0 : maximum(edge_ranks)
    min_q_edges = max_edge_rank / total_edges
    
    return (; total_edges, max_node_rank, min_q_nodes, node_best_ranks, max_edge_rank, min_q_edges, plant_edges)
end

"""
Check whether the injected biclique (nodes and edges) is present in a reduced graph.
"""
function check_plant_presence(g_reduced::BipartiteGraph, chosen_U::Vector{Int}, chosen_V::Vector{Int}, plant_edges::Vector{Tuple{Int,Int}})
    u_present = [u for u in chosen_U if haskey(g_reduced.adjU, u)]
    v_present = [v for v in chosen_V if haskey(g_reduced.adjV, v)]
    nodes_present = (length(u_present) == length(chosen_U)) && (length(v_present) == length(chosen_V))
    
    surviving_edges = [e for e in plant_edges if haskey(g_reduced.adjU, e[1]) && (e[2] in g_reduced.adjU[e[1]])]
    edges_present = length(surviving_edges) == length(plant_edges)
    
    return (; nodes_present, u_present, v_present, edges_present, surviving_edges, total_plant_edges=length(plant_edges))
end

function parse_dataset_arg(default::String="amazon/boxes")
    for arg in ARGS
        startswith(arg, "-") && continue
        return arg
    end
    return default
end

"""
Test quartile reduction with ACO on a real dataset.
"""
function test_quartile_aco(dataset_name::AbstractString=parse_dataset_arg())
    file_path = resolve_graph_path(dataset_name)
    println("Loading graph from: $file_path (dataset: $dataset_name)")
    
    g, edge_count = load_bipartite_graph(file_path)
    println("Original graph: |U|=$(length(g.adjU)), |V|=$(length(g.adjV)), |E|=$edge_count")
    
    k = 2
    θ = 5
    
    # Inject planted biclique before pulse and ACO
    rng = MersenneTwister(1)
    chosen_U, chosen_V, inserted, missing_edges, existing =
        inject_biclique!(g, θ, θ, k, rng)
    edge_count += inserted
    println("\nInjected biclique: u=$θ, v=$θ, k=$k (existing=$existing, inserted=$inserted, missing=$(length(missing_edges)))")
    println("  planted U: $chosen_U")
    println("  planted V: $chosen_V")
    println("  graph with plant: |U|=$(length(g.adjU)), |V|=$(length(g.adjV)), |E|=$edge_count")
    
    # Calculate what quartile would be required to include every node (and edge) in the biclique
    pulse_iters = 20
    thresh = compute_biclique_quartile_thresholds(g, chosen_U, chosen_V, missing_edges; iterations=pulse_iters)
    println("\n=== Quartile Threshold Analysis for Injected Biclique ===")
    println("  Minimum quartile to include EVERY NODE in the biclique: $(round(thresh.min_q_nodes; digits=4)) ($(round(100*thresh.min_q_nodes; digits=2))%)")
    println("    (highest rank among nodes' first incident edges: rank $(thresh.max_node_rank) / $(thresh.total_edges))")
    println("  Minimum quartile to include EVERY EDGE in the biclique: $(round(thresh.min_q_edges; digits=4)) ($(round(100*thresh.min_q_edges; digits=2))%)")
    println("    (highest rank among biclique edges: rank $(thresh.max_edge_rank) / $(thresh.total_edges))")
    
    # Test with quartile reduction
    println("\n=== Testing ACO with quartile reduction ===")
    g_quartile = deepcopy(g)
    
    # Run ACO on the graph using quartile reduction mode
    sols_quartile, best_iters, best_times, pheromones, remapping = aco(
        g_quartile,
        1,  # pheromone
        5,  # num_ants
        2,  # num_iterations
        0.05,  # evaporation
        k,
        θ,
        1,  # num_subspecies
        reduction=ReductionMode.quartile,
        parallelize=false
    )
    sol_quartile = sols_quartile[1]
    best_iter = best_iters[1]
    best_time = best_times[1]
    
    # Log whether injected biclique is present in the quartile-reduced graph
    q_fg = freeze(g_quartile)
    plant_sg = SubGraph(Set(chosen_U), Set(chosen_V))
    q_plant_edges = Subgraph.edge_count(q_fg, plant_sg)
    q_plant_missing = Subgraph.missing_edges(q_fg, plant_sg)
    q_plant = check_plant_presence(g_quartile, chosen_U, chosen_V, thresh.plant_edges)
    
    println("\nInjected biclique verification in quartile-reduced graph:")
    println("  All U nodes present in g_quartile: $(all(u -> haskey(g_quartile.adjU, u), chosen_U)) ($([u for u in chosen_U if haskey(g_quartile.adjU, u)]))")
    println("  All V nodes present in g_quartile: $(all(v -> haskey(g_quartile.adjV, v), chosen_V)) ($([v for v in chosen_V if haskey(g_quartile.adjV, v)]))")
    println("  Biclique edges present in g_quartile: $q_plant_edges / $(length(thresh.plant_edges))")
    println("  Biclique missing edges in g_quartile: $q_plant_missing (budget k=$k)")
    println("  Is full planted biclique intact in g_quartile? $(q_plant.edges_present && q_plant_missing <= k ? "YES" : "NO")")
    
    println("\nACO with quartile reduction result:")
    println("  |U|=$(length(sol_quartile.U)), |V|=$(length(sol_quartile.V))")
    println("  Found at iteration: $best_iter")
    println("  Time: $(round(best_time; digits=3))s")
    
    # Test with standard reduction for comparison
    println("\n=== Testing ACO with standard reduction ===")
    g_standard = deepcopy(g)
    
    sols_standard, best_iters_std, best_times_std, pheromones_std, remapping_std = aco(
        g_standard,
        1,  # pheromone
        5,  # num_ants
        2,  # num_iterations
        0.05,  # evaporation
        k,
        θ,
        1,  # num_subspecies
        reduction=ReductionMode.simple,
        parallelize=false
    )
    sol_standard = sols_standard[1]
    best_iter_std = best_iters_std[1]
    best_time_std = best_times_std[1]
    
    println("\nACO with standard reduction result:")
    println("  |U|=$(length(sol_standard.U)), |V|=$(length(sol_standard.V))")
    println("  Found at iteration: $best_iter_std")
    println("  Time: $(round(best_time_std; digits=3))s")
    
    # Log whether injected biclique is present in standard-reduced graph
    s_fg = freeze(g_standard)
    s_plant_edges = Subgraph.edge_count(s_fg, plant_sg)
    s_plant = check_plant_presence(g_standard, chosen_U, chosen_V, thresh.plant_edges)
    println("\nInjected biclique in standard-reduced graph:")
    println("  All nodes present: $(s_plant.nodes_present ? "YES" : "NO") (|U|: $(length(s_plant.u_present))/$θ, |V|: $(length(s_plant.v_present))/$θ)")
    println("  Biclique edges present: $s_plant_edges / $(length(thresh.plant_edges))")
    
    println("\n=== Comparison ===")
    q_edges = Subgraph.edge_count(freeze(g_quartile), sol_quartile)
    s_edges = Subgraph.edge_count(freeze(g_standard), sol_standard)
    println("Quartile edges: $q_edges")
    println("Standard edges: $s_edges")
    plant_U = Set(chosen_U)
    plant_V = Set(chosen_V)
    println("Quartile plant overlap: |U ∩ plant_U|=$(length(intersect(sol_quartile.U, plant_U)))/$θ, |V ∩ plant_V|=$(length(intersect(sol_quartile.V, plant_V)))/$θ")
    println("Standard plant overlap: |U ∩ plant_U|=$(length(intersect(sol_standard.U, plant_U)))/$θ, |V ∩ plant_V|=$(length(intersect(sol_standard.V, plant_V)))/$θ")
    
    return true
end

test_quartile_aco()
