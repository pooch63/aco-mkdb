#=
Pulse-based propagation algorithm for bipartite graphs.

Every iteration:
1. Each vertex on U sends pulses to all its neighbors in V
2. Each vertex on V sends pulses to all its neighbors in U
3. When a pulse hits a vertex, its count increments by exactly 1
4. When a pulse travels an edge, the edge count increases by the vertex weight

After 20 iterations, log the mean edge weight in the planted biclique vs average edge weight.
=#

const __PULSE_JL__ = true

using Random
using Statistics

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__IO_JL__) || include("io.jl")

"""
Run pulse propagation on a frozen bipartite graph.

Arguments:
- fg: FrozenBipartite graph
- iterations: Number of pulse iterations (default 20)
- injected_biclique: Optional SubGraph representing a planted biclique to analyze

Returns:
- Dictionary with statistics including mean edge weights in biclique vs overall
"""
function pulse_propagation(fg::FrozenBipartite, iterations::Int=20,
                           injected_biclique::Union{Nothing,SubGraph}=nothing)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    
    # Vertex weights (start at 1.0)
    u_weights = ones(Float64, nU)
    v_weights = ones(Float64, nV)
    
    # Edge weights (start at 0.0)
    # Use CSR slot indexing for edges
    edge_weights = zeros(Float64, length(fg.v_adj))
    
    println("Starting pulse propagation on graph with $nU U-nodes, $nV V-nodes, $(length(edge_weights)) edges")
    println("Running for $iterations iterations")
    
    if injected_biclique !== nothing
        println("Planted biclique: |U|=$(length(injected_biclique.U)), |V|=$(length(injected_biclique.V))")
    end
    
    for iter in 1:iterations
        # Phase 1: U sends pulses to V
        for ui in 1:nU
            u_weight = u_weights[ui]
            for k in neighbor_range_u(fg, ui)
                vi = fg.v_adj[k]
                # Pulse hits V vertex
                v_weights[vi] += 1.0
                # Edge count increases by vertex weight
                edge_weights[k] += u_weight
            end
        end
        
        # Phase 2: V sends pulses to U
        for vi in 1:nV
            v_weight = v_weights[vi]
            for k in neighbor_range_v(fg, vi)
                ui = fg.u_adj[k]
                # Pulse hits U vertex
                u_weights[ui] += 1.0
                # Edge count increases by vertex weight
                # Note: u_adj stores the reverse CSR slot, need to map to forward slot
                # For simplicity, we'll track both directions separately
            end
        end
        
        if iter % 5 == 0
            avg_u_weight = mean(u_weights)
            avg_v_weight = mean(v_weights)
            avg_edge_weight = mean(edge_weights)
            println("Iteration $iter: avg U weight=$avg_u_weight, avg V weight=$avg_v_weight, avg edge weight=$avg_edge_weight")
        end
    end
    
    # Calculate statistics
    avg_edge_weight = mean(edge_weights)
    
    if injected_biclique !== nothing
        # Map original biclique vertex IDs to dense indices
        u_index_map = fg.u_index
        v_index_map = fg.v_index
        
        biclique_edge_weights = Float64[]
        biclique_edges = 0
        
        for u in injected_biclique.U
            ui = get(u_index_map, u, nothing)
            ui === nothing && continue
            for v in injected_biclique.V
                vi = get(v_index_map, v, nothing)
                vi === nothing && continue
                
                # Find the edge slot
                slot = nothing
                for k in neighbor_range_u(fg, ui)
                    if fg.v_adj[k] == vi
                        slot = k
                        break
                    end
                end
                
                if slot !== nothing
                    push!(biclique_edge_weights, edge_weights[slot])
                    biclique_edges += 1
                end
            end
        end
        
        if !isempty(biclique_edge_weights)
            mean_biclique_edge_weight = mean(biclique_edge_weights)
            println("=== Pulse Propagation Results ===")
            println("Total edges in graph: $(length(edge_weights))")
            println("Average edge weight (overall): $avg_edge_weight")
            println("Edges in planted biclique: $biclique_edges")
            println("Mean edge weight in planted biclique: $mean_biclique_edge_weight")
            println("Ratio (biclique / overall): $(mean_biclique_edge_weight / avg_edge_weight)")
            
            return Dict{String,Any}(
                "iterations" => iterations,
                "avg_u_weight" => mean(u_weights),
                "avg_v_weight" => mean(v_weights),
                "avg_edge_weight" => avg_edge_weight,
                "biclique_edges" => biclique_edges,
                "mean_biclique_edge_weight" => mean_biclique_edge_weight,
                "ratio" => mean_biclique_edge_weight / avg_edge_weight,
                "u_weights" => u_weights,
                "v_weights" => v_weights,
                "edge_weights" => edge_weights,
            )
        else
            println("Warning: No edges found in planted biclique (vertices may have been removed by reduction)")
            return Dict{String,Any}(
                "iterations" => iterations,
                "avg_u_weight" => mean(u_weights),
                "avg_v_weight" => mean(v_weights),
                "avg_edge_weight" => avg_edge_weight,
                "biclique_edges" => 0,
                "mean_biclique_edge_weight" => nothing,
                "ratio" => nothing,
            )
        end
    else
        println("=== Pulse Propagation Results (no biclique) ===")
        println("Average edge weight: $avg_edge_weight")
        
        return Dict{String,Any}(
            "iterations" => iterations,
            "avg_u_weight" => mean(u_weights),
            "avg_v_weight" => mean(v_weights),
            "avg_edge_weight" => avg_edge_weight,
        )
    end
end

"""
Convenience wrapper that loads a graph, optionally injects a biclique, and runs pulse propagation.
"""
function pulse_experiment(dataset_name::String; p::Int=5, q::Int=5, k::Int=2,
                          iterations::Int=20, seed::Union{Int,Nothing}=nothing)
    # Load graph
    file_path = resolve_graph_path(dataset_name)
    println("Loading graph from: $file_path")
    
    g, edge_count = load_bipartite_graph(file_path)
    fg = freeze(g)
    
    # Inject biclique if parameters provided
    injected_biclique = nothing
    if p > 0 && q > 0
        println("Injecting biclique: p=$p, q=$q, k=$k")
        rng = seed === nothing ? MersenneTwister() : MersenneTwister(seed)
        
        # Sample vertices
        u_candidates = collect(keys(g.adjU))
        v_candidates = collect(keys(g.adjV))
        
        if length(u_candidates) >= p && length(v_candidates) >= q
            shuffle!(rng, u_candidates)
            shuffle!(rng, v_candidates)
            
            chosen_u = u_candidates[1:p]
            chosen_v = v_candidates[1:q]
            
            # Add all edges except k missing
            all_pairs = [(u, v) for u in chosen_u, v in chosen_v]
            existing = Set{Tuple{Int,Int}}()
            for (u, v) in all_pairs
                if v in g.adjU[u]
                    push!(existing, (u, v))
                end
            end
            
            absent = [(u, v) for (u, v) in all_pairs if (u, v) ∉ existing]
            
            if length(absent) >= k
                shuffle!(rng, absent)
                missing_edges = Set(absent[1:k])
                
                for (u, v) in all_pairs
                    if (u, v) ∉ existing && (u, v) ∉ missing_edges
                        add_edge!(g, u, v, nothing)
                    end
                end
                
                injected_biclique = SubGraph(Set(chosen_u), Set(chosen_v))
                println("Biclique injected successfully")
            else
                println("Warning: Not enough absent edges to create k-defective biclique")
            end
        else
            println("Warning: Not enough vertices for biclique injection")
        end
        
        # Re-freeze after injection
        fg = freeze(g)
    end
    
    # Run pulse propagation
    return pulse_propagation(fg, iterations, injected_biclique)
end

function main()
    if isempty(ARGS)
        println(stderr, "Usage: julia src/pulse.jl <dataset_name> [--p=N] [--q=M] [--k=K] [--iterations=I] [--seed=S]")
        println(stderr, "Example: julia src/pulse.jl amazon/boxes --p=5 --q=5 --k=2 --iterations=20")
        exit(1)
    end

    dataset_name = ARGS[1]
    
    # Parse optional flags
    p = 5
    q = 5
    k = 2
    iterations = 20
    seed = nothing

    for arg in ARGS[2:end]
        if startswith(arg, "--p=")
            p = parse(Int, arg[4:end])
        elseif startswith(arg, "--q=")
            q = parse(Int, arg[4:end])
        elseif startswith(arg, "--k=")
            k = parse(Int, arg[4:end])
        elseif startswith(arg, "--iterations=")
            iterations = parse(Int, arg[13:end])
        elseif startswith(arg, "--seed=")
            seed = parse(Int, arg[7:end])
        end
    end

    pulse_experiment(dataset_name; p=p, q=q, k=k, iterations=iterations, seed=seed)
end

"""
Run pulse propagation and return edge weights for quartile-based reduction.

Arguments:
- fg: FrozenBipartite graph
- iterations: Number of pulse iterations (default 20)

Returns:
- Vector of edge weights (parallel to fg.v_adj / fg.edge_data)
"""
function pulse_edge_weights(fg::FrozenBipartite, iterations::Int=20)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    
    # Vertex weights (start at 1.0)
    u_weights = ones(Float64, nU)
    v_weights = ones(Float64, nV)
    
    # Edge weights (start at 0.0)
    edge_weights = zeros(Float64, length(fg.v_adj))
    
    for iter in 1:iterations
        # Phase 1: U sends pulses to V
        for ui in 1:nU
            u_weight = u_weights[ui]
            for k in neighbor_range_u(fg, ui)
                vi = fg.v_adj[k]
                # Pulse hits V vertex
                v_weights[vi] += 1.0
                # Edge count increases by vertex weight
                edge_weights[k] += u_weight
            end
        end
        
        # Phase 2: V sends pulses to U
        for vi in 1:nV
            v_weight = v_weights[vi]
            for k in neighbor_range_v(fg, vi)
                ui = fg.u_adj[k]
                # Pulse hits U vertex
                u_weights[ui] += 1.0
            end
        end
    end
    
    return edge_weights
end

"""
Apply quartile-based edge weight reduction to a BipartiteGraph.

Arguments:
- g: BipartiteGraph to reduce (will be modified in place)
- quartile: Fraction of top edges to keep (default 0.2 for top 20%)
- iterations: Number of pulse iterations (default 20)

Returns:
- Number of edges kept
- Number of vertices kept on each side
"""
function quartile_edge_reduction!(g::BipartiteGraph, quartile::Float64=0.2, iterations::Int=20)
    quartile > 0.0 || throw(ArgumentError("quartile must be > 0, got $quartile"))
    quartile <= 1.0 || throw(ArgumentError("quartile must be <= 1, got $quartile"))
    
    # First freeze to get CSR structure for pulse propagation
    fg = freeze(g)
    
    # Run pulse propagation to get edge weights
    edge_weights = pulse_edge_weights(fg, iterations)
    
    # Find the threshold for the top quartile
    # Create list of (weight, u_orig, v_orig) tuples
    edge_list = Tuple{Float64, Int, Int}[]
    for ui in 1:length(fg.u_ids)
        u_orig = fg.u_ids[ui]
        for k in neighbor_range_u(fg, ui)
            v_orig = fg.v_ids[fg.v_adj[k]]
            push!(edge_list, (edge_weights[k], u_orig, v_orig))
        end
    end
    
    # Sort by weight descending
    sort!(edge_list, by=x->x[1], rev=true)
    
    n_edges = length(edge_list)
    n_keep = max(1, floor(Int, quartile * n_edges))
    
    println("Quartile reduction: keeping top $(round(100*quartile; digits=1))% edges")
    println("  Total edges: $n_edges, keeping: $n_keep")
    
    # Mark which edges to keep (exactly n_keep edges)
    keep_edges = Set{Tuple{Int,Int}}()
    for i in 1:n_keep
        _, u_orig, v_orig = edge_list[i]
        push!(keep_edges, (u_orig, v_orig))
    end
    
    # Remove edges below threshold
    edges_removed = 0
    for (u, nbrs) in copy(g.adjU)
        for v in copy(nbrs)
            if (u, v) ∉ keep_edges
                rem_edge_structural!(g, u, v)
                edges_removed += 1
            end
        end
    end
    
    # Remove isolated vertices (no edges remaining)
    u_removed = 0
    v_removed = 0
    for u in collect(keys(g.adjU))
        if isempty(g.adjU[u])
            rem_u_structural!(g, u)
            u_removed += 1
        end
    end
    for v in collect(keys(g.adjV))
        if isempty(g.adjV[v])
            rem_v_structural!(g, v)
            v_removed += 1
        end
    end
    
    n_edges_kept = length(keep_edges)
    n_u_kept = length(g.adjU)
    n_v_kept = length(g.adjV)
    
    println("  Removed $edges_removed edges, $u_removed U-nodes, $v_removed V-nodes")
    println("  Kept $n_edges_kept edges, $n_u_kept U-nodes, $n_v_kept V-nodes")
    
    return n_edges_kept, n_u_kept, n_v_kept
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
