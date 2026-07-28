const __PARALLEL_TABU_JL__ = true

isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__TABU_JL__) || include("tabu.jl")

"""
Seed a population the same way GA does (softmax N−1 degree-weighted nodes plus
the θ-heuristic), then run `tabu_repair!` on each member until patience is
exhausted. Logs and returns the highest-fitness and largest-vertex solutions.
"""
function parallel_tabu(g::BipartiteGraph, k::Int, θ::Int, N::Int;
    tt::Int=3, tabu_patience::Int=10,
    use_heuristic::Bool=true, reduction::ReductionMode.T=ReductionMode.all_reductions,
    num_U::Union{Int, Nothing}=nothing, num_V::Union{Int, Nothing}=nothing)

    @assert θ > k "θ must be greater than k"

    fg = apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, reduction)

    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        empty = SubGraph(Set(), Set())
        return (best_fitness=empty, most_vertices=empty)
    end

    if N > length(fg.u_ids) + length(fg.v_ids)
        N = length(fg.u_ids) + length(fg.v_ids)
        @warn "N cannot exceed the number of nodes in optimized graph. Automatically clamping N to $(N)"
    end

    return parallel_tabu_search(fg, k, θ, N, tt, tabu_patience)
end

function parallel_tabu_search(fg::FrozenBipartite, k::Int, θ::Int, N::Int,
    tt::Int, tabu_patience::Int)

    G = SubGraph(Set(u for u in fg.u_ids), Set(v for v in fg.v_ids))
    # Choose softmax-distributed N−1 nodes (same as GA)
    nodes = softmax_sample_nodes((u, n) -> u ? degree_u(fg, n) : degree_v(fg, n), G, N )

    instances = Instance[]
    for node in nodes
        set = Set(node.id)
        empty = Set()
        subgraph = node.is_u ? SubGraph(set, empty) : SubGraph(empty, set)
        push!(instances, Instance(subgraph, k))
    end

    # push!(instances, Instance(theta_based_heuristic(fg, k, θ; return_invalid=true), k))

    for (i, instance) in enumerate(instances)
        println("Tabu instance $i / $(length(instances))  (|U|,|V|)=($(length(instance.subgraph.U)), $(length(instance.subgraph.V)))")
        tabu_repair!(fg, instance.subgraph, instance.k, θ, tt, tabu_patience)
        fit = instance_fitness(fg, instance.subgraph, θ)
        verts = Subgraph.vertex_count(instance.subgraph)
        println("  done  fitness=$fit  vertices=$verts  (|U|,|V|)=($(length(instance.subgraph.U)), $(length(instance.subgraph.V)))")
    end

    best_fit = argmax(inst -> instance_fitness(fg, inst.subgraph, θ), instances)
    most_verts = argmax(inst -> Subgraph.vertex_count(inst.subgraph), instances)

    best_fit_score = instance_fitness(fg, best_fit.subgraph, θ)
    most_verts_count = Subgraph.vertex_count(most_verts.subgraph)
    most_verts_fit = instance_fitness(fg, most_verts.subgraph, θ)

    println("Best fitness: score=$best_fit_score  vertices=$(Subgraph.vertex_count(best_fit.subgraph))  (|U|,|V|)=($(length(best_fit.subgraph.U)), $(length(best_fit.subgraph.V)))")
    @show best_fit.subgraph
    println("Most vertices: count=$most_verts_count  fitness=$most_verts_fit  (|U|,|V|)=($(length(most_verts.subgraph.U)), $(length(most_verts.subgraph.V)))")
    @show most_verts.subgraph

    return (best_fitness=best_fit.subgraph, most_vertices=most_verts.subgraph)
end
