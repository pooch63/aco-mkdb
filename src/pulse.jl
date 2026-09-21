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
isdefined(@__MODULE__, :__NEIGHBORHOOD_JL__) || include("neighborhood.jl")


const PULSE_FORMULA_REGISTRY = Dict{String,Function}()

"""Register a pulse weighting formula under a lowercase key."""
function register_pulse_formula!(name::AbstractString, fn::Function)
    key = lowercase(strip(String(name)))
    isempty(key) && throw(ArgumentError("pulse formula name must be non-empty"))
    PULSE_FORMULA_REGISTRY[key] = fn
    return key
end

list_pulse_formulas() = sort!(collect(keys(PULSE_FORMULA_REGISTRY)))

function make_pulse_formula(name::AbstractString)
    key = lowercase(strip(String(name)))
    fn = get(PULSE_FORMULA_REGISTRY, key, nothing)
    if fn === nothing
        known = join(list_pulse_formulas(), ", ")
        throw(ArgumentError("Unknown pulse formula '$name'. Registered: $known"))
    end
    return fn
end

function _pulse_weight_standard(u_weight::Float64, v_weight::Float64, u_deg::Int, v_deg::Int)
    return u_deg > 0 ? u_weight / u_deg : 0.0
end

function _pulse_weight_degree_sq(u_weight::Float64, v_weight::Float64, u_deg::Int, v_deg::Int)
    return u_deg > 0 ? u_weight / (u_deg * u_deg) : 0.0
end

function _pulse_weight_product(u_weight::Float64, v_weight::Float64, u_deg::Int, v_deg::Int)
    if u_deg <= 0 || v_deg <= 0
        return 0.0
    end
    return (u_weight * v_weight) / (u_deg * v_deg)
end

function _pulse_weight_uniform(u_weight::Float64, v_weight::Float64, u_deg::Int, v_deg::Int)
    return 1.0
end

register_pulse_formula!("degree_sq", _pulse_weight_degree_sq)
register_pulse_formula!("product", _pulse_weight_product)
register_pulse_formula!("standard", _pulse_weight_standard)
register_pulse_formula!("uniform", _pulse_weight_uniform)

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
                           injected_biclique::Union{Nothing,SubGraph}=nothing;
                           formula::AbstractString="standard")
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
    
    pulse_fn = make_pulse_formula(formula)

    for iter in 1:iterations
        # Phase 1: U sends pulses to V
        for ui in 1:nU
            u_weight = u_weights[ui]
            u_deg = length(neighbor_range_u(fg, ui))
            for k in neighbor_range_u(fg, ui)
                vi = fg.v_adj[k]
                v_weight = v_weights[vi]
                v_deg = length(neighbor_range_v(fg, vi))
                # Pulse hits V vertex
                v_weights[vi] += 1.0
                # Edge count increases by the selected pulse formula
                edge_weights[k] += pulse_fn(u_weight, v_weight, u_deg, v_deg)
            end
        end
        
        # Phase 2: V sends pulses to U
        for vi in 1:nV
            v_weight = v_weights[vi]
            v_deg = length(neighbor_range_v(fg, vi))
            for k in neighbor_range_v(fg, vi)
                ui = fg.u_adj[k]
                u_weight = u_weights[ui]
                u_deg = length(neighbor_range_u(fg, ui))
                # Pulse hits U vertex
                u_weights[ui] += 1.0
                # The reverse pass keeps the same propagation semantics without changing edge weights.
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
function pulse_edge_weights(fg::FrozenBipartite, iterations::Int=20; formula::AbstractString="standard")
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    
    # Vertex weights (start at 1.0)
    u_weights = ones(Float64, nU)
    v_weights = ones(Float64, nV)
    
    # Edge weights (start at 0.0)
    edge_weights = zeros(Float64, length(fg.v_adj))
    pulse_fn = make_pulse_formula(formula)
    
    for iter in 1:iterations
        # Phase 1: U sends pulses to V
        for ui in 1:nU
            u_weight = u_weights[ui]
            u_deg = length(neighbor_range_u(fg, ui))
            for k in neighbor_range_u(fg, ui)
                vi = fg.v_adj[k]
                v_weight = v_weights[vi]
                v_deg = length(neighbor_range_v(fg, vi))
                # Pulse hits V vertex
                v_weights[vi] += 1.0
                # Edge count increases by the selected pulse formula
                edge_weights[k] += pulse_fn(u_weight, v_weight, u_deg, v_deg)
            end
        end
        
        # Phase 2: V sends pulses to U
        for vi in 1:nV
            v_weight = v_weights[vi]
            v_deg = length(neighbor_range_v(fg, vi))
            for k in neighbor_range_v(fg, vi)
                ui = fg.u_adj[k]
                u_weight = u_weights[ui]
                u_deg = length(neighbor_range_u(fg, ui))
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
function quartile_edge_reduction!(g::BipartiteGraph, quartile::Float64=0.2, iterations::Int=20;
    formula::AbstractString="standard")
    quartile > 0.0 || throw(ArgumentError("quartile must be > 0, got $quartile"))
    quartile <= 1.0 || throw(ArgumentError("quartile must be <= 1, got $quartile"))
    
    # First freeze to get CSR structure for pulse propagation
    fg = freeze(g)
    
    # Run pulse propagation to get edge weights
    edge_weights = pulse_edge_weights(fg, iterations; formula=formula)
    
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

"""
Apply neighborhood-Jaccard edge-score reduction to a BipartiteGraph.

Computes S_C(u, v) = C_U(u, v) · C_V(u, v) for every edge, where C_U / C_V
are the mean Jaccard similarities of a vertex's neighbour-set to each of its
co-neighbours' neighbour-sets.  Keeps the top `top_fraction` fraction of edges
by S_C score, then removes isolated vertices.

Arguments:
- g: BipartiteGraph to reduce (modified in-place)
- top_fraction: fraction of highest-scoring edges to keep (default 0.2)

Returns:
- Number of edges kept
- Number of U-vertices kept
- Number of V-vertices kept
"""
function neighborhood_edge_reduction!(g::BipartiteGraph, top_fraction::Float64=0.2; θ::Int=5, theta::Int=θ)
    actual_θ = theta != 5 ? theta : θ
    top_fraction > 0.0 || throw(ArgumentError("top_fraction must be > 0, got $top_fraction"))
    top_fraction <= 1.0 || throw(ArgumentError("top_fraction must be <= 1, got $top_fraction"))

    fg = freeze(g)
    scores = neighborhood_scores(fg; θ=actual_θ)

    # Build (score, u_orig, v_orig) list
    edge_list = Tuple{Float64, Int, Int}[]
    sizehint!(edge_list, length(fg.v_adj))
    for ui in 1:length(fg.u_ids)
        u_orig = fg.u_ids[ui]
        for k in neighbor_range_u(fg, ui)
            v_orig = fg.v_ids[fg.v_adj[k]]
            push!(edge_list, (scores[k], u_orig, v_orig))
        end
    end

    sort!(edge_list, by=x->x[1], rev=true)

    n_edges = length(edge_list)
    n_keep  = max(1, floor(Int, top_fraction * n_edges))

    println("Neighborhood reduction: keeping top $(round(100*top_fraction; digits=1))% edges by S_C score")
    println("  Total edges: $n_edges, keeping: $n_keep")

    keep_edges = Set{Tuple{Int,Int}}()
    for i in 1:n_keep
        _, u_orig, v_orig = edge_list[i]
        push!(keep_edges, (u_orig, v_orig))
    end

    edges_removed = 0
    for (u, nbrs) in copy(g.adjU)
        for v in copy(nbrs)
            if (u, v) ∉ keep_edges
                rem_edge_structural!(g, u, v)
                edges_removed += 1
            end
        end
    end

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
    n_u_kept     = length(g.adjU)
    n_v_kept     = length(g.adjV)

    println("  Removed $edges_removed edges, $u_removed U-nodes, $v_removed V-nodes")
    println("  Kept $n_edges_kept edges, $n_u_kept U-nodes, $n_v_kept V-nodes")

    return n_edges_kept, n_u_kept, n_v_kept
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
