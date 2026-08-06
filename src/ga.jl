using Base.Threads

const __GA_JL__ = true

isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__TABU_JL__) || include("tabu.jl")

using EnumX
using Random
using StatsBase

const COLLECT_METRICS = true
# Off by default so headless / benchmark runs do not pull in Makie.
# Set GA_GRAPH=1 (or flip this const) to plot metrics after search.
const GRAPH = get(ENV, "GA_GRAPH", "") == "1"

if GRAPH
    include("plot_metrics.jl")
end

global U::Set{Int} = Set()
global V::Set{Int} = Set()

@enumx RepairMode greedy tabu mixed

# Fraction of unique (U ∪ V) node ids across the population relative to total
# vertex slots. E.g. for SubGraph({1,2},{1,3}) and SubGraph({2,3},{4,2}):
#   (|{1,2,3}| + |{1,2,3,4}|) / 8 = 7/8
function population_diversity(instances::Vector{Instance})
    unique_u = Set{Int}()
    unique_v = Set{Int}()
    total = 0
    for instance in instances
        union!(unique_u, instance.subgraph.U)
        union!(unique_v, instance.subgraph.V)
        total += length(instance.subgraph.U) + length(instance.subgraph.V)
    end
    return total == 0 ? 0.0 : (length(unique_u) + length(unique_v)) / total
end

# If the number of entrigeneres in g.adjU is not equal to the number of nodes or same for V,
# e.g., there are some gaps in node IDs, you'll need to pass the maximum node ID for each side
function ga(g::BipartiteGraph, k::Int, θ::Int, N::Int, O::Int, k_mutate::Float64, generations::Int;
    H::Int = 2,
    use_heuristic::Bool=true, reduction::ReductionMode.T=ReductionMode.all_reductions,
    repair::RepairMode.T = RepairMode.tabu, tt::Int=2, tabu_patience::Int=3,
    num_U::Union{Int, Nothing}=nothing, num_V::Union{Int, Nothing}=nothing)

    @assert θ > k "θ must be greater than k"

    fg = apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, reduction)

    # If there's not enough nodes remaining on either side, then we
    # already know it's an invalid solution
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph(Set(), Set())
    end

    # We can't have more generations than there are nodes
    if N > length(fg.u_ids) + length(fg.v_ids)
        N = length(fg.u_ids) + length(fg.v_ids)
        @warn "N cannot exceed the number of nodes in optimized graph. Automatically clamping N to $(N)"
    end

    sol = search(
        fg,
        k,
        θ,
        N,
        H,
        O,
        k_mutate,
        repair,
        tt,
        tabu_patience,
        generations
    )

    println("After graph, reductions, G=(U,V)=($(length(fg.u_ids)), $(length(fg.v_ids)))")

    println(length(U), " ", length(V))
    @show U
    @show V

    # fg.u_ids = U
    # fg.v_ids = V

    return sol
end

function search(fg::FrozenBipartite{T}, k::Int, θ::Int, N::Int, H::Int, O::Int, k_mutate::Float64,
    use_heuristic::Bool, repair::RepairMode.T, tt::Int, tabu_patience::Int, generations::Int) where {T}
    G = SubGraph(Set(u for u in fg.u_ids), Set(v for v in fg.v_ids))
    # Choose softmax-distributed N nodes
    nodes = softmax_sample_nodes((u, n) -> u ? degree_u(fg, n) : degree_v(fg, n), G, use_heuristic ? N-1 : N)

    instances = Instance[]
    for node in nodes
        set = Set(node.id)
        empty = Set()
        subgraph = node.is_u ? SubGraph(set, empty) : SubGraph(empty, set)
        push!(instances, Instance(subgraph, k))
    end

    # Add the heuristic as another
    use_heuristic && push!(instances, Instance(theta_based_heuristic(fg, k, θ; return_invalid=true), k))

    # Room for algorithmic improvement: Keep track of parents that mutated so we don't
    # have parents breed together twice

    # Room for algorithmic improvement: Keep track of the absolute best instance we've ever seen
    # so if diversity makes the population collapse, we can just return the best one

    diversity_history = Float64[]
    best_fitness_history = Float64[]

    for gen in 1:generations
        instances = evolve(instances, fg, k, θ, N, H, O, k_mutate, repair, tt, tabu_patience)
        best = argmax(instance -> instance_fitness(fg, instance.subgraph, θ), instances)
        best_fitness = instance_fitness(fg, best.subgraph, θ)

        union!(U, best.subgraph.U)
        union!(V, best.subgraph.V)

        if COLLECT_METRICS
            diversity = population_diversity(instances)
            push!(diversity_history, diversity)
            push!(best_fitness_history, Float64(best_fitness))
            @show gen, diversity, best_fitness
        end

        # @show best
        # @show instance_fitness(fg, best.subgraph)
    end

    if COLLECT_METRICS && GRAPH
        plot_metrics(diversity_history, best_fitness_history)
    end

    return argmax(instance -> instance_fitness(fg, instance.subgraph, θ), instances).subgraph
end

function evolve(instances::Vector{Instance}, fg::FrozenBipartite, max_k::Int, θ::Int, N::Int,
    H::Int, O::Int, k_mutate::Float64, repair::RepairMode.T, tt::Int, tabu_patience::Int)
    next = Instance[]

    @threads for _ in 1:N-2
        male, female = softmax_sample(instance -> instance_fitness(fg, instance.subgraph, θ), instances, 2)
        
        k = rand(1, 2) == 1 ? male.k : female.k

        if rand() < k_mutate
            if rand(1, 2) == 1
                k = min(max_k, k + 1)
            else
                k = max(0, k - 1)
            end
        end

        @show k

        offspring = crossover(fg, male.subgraph, female.subgraph, k, θ, repair, tt, tabu_patience)
        @show instance_fitness(fg, offspring, θ)
        push!(next, Instance(offspring, k))
    end

    elites = softmax_sample(instance -> instance_fitness(fg, instance.subgraph, θ), instances, 2)
    
    for elite in elites
        push!(next, Instance(deepcopy(elite.subgraph), elite.k))
    end

    opt = softmax_sample(instance -> instance_fitness(fg, instance.subgraph, θ), next, O)
    for offspring in opt
        H_opt!(fg, offspring.subgraph, offspring.k, θ, H, repair, tt, tabu_patience)
    end

    return next
end

# Room for algorithmic improvement: could each subgraph keep a k value of the maximum
# missing edges it will allow, which could range from 0 to k (or maybe even 2k), so that way
# there are some graphs that start super conservatively and might then hit solutions that
# more liberal graphs get stopped at
function crossover(fg::FrozenBipartite, male::SubGraph, female::SubGraph, k::Int, θ::Int, repair::RepairMode.T, tt::Int, tabu_patience::Int)
    S = subgraph_intersection(male, female)

    next::SubGraph = S

    # If the intersection is null, try taking the crossover of the candidate set
    if Subgraph.vertex_count(S) == 0
        candidate = Subgraph.add(male, female)

        next = SubGraph(Set(), Set())
        
        # Room for algorithmic improvement: should we build from the degree of nodes within the union of their candidates,
        # or from the degree of the union of the graphs themselves? Or both?
        while Subgraph.missing_edges(fg, next) < k && Subgraph.vertex_count(candidate) > 0
            is_u, node = argmax_nodes((u, n) -> degree_in_subgraph(fg, u, n, candidate), candidate)
            Subgraph.add_node!(next, is_u, node)
            Subgraph.remove_node!(candidate, is_u, node)
        end
    end

    # Because the two parents' k values might be different, we might need to make the graph valid for that k again
    while Subgraph.missing_edges(fg, next) > k
        is_u, node = argmax_nodes((u, n) -> nondegree_in_subgraph(fg, u, n, next), next)
        Subgraph.remove_node!(next, is_u, node)
    end

    println("Before greedily add, next=", Subgraph.vertex_count(next))

    last = deepcopy(next)
    greedily_add!(fg, next, k)
    println("Greedy score: $(instance_fitness(fg, next, θ))")
    next = deepcopy(last)
    tabu_repair!(fg, next, k, θ, tt, tabu_patience)
    println("Tabu score: $(instance_fitness(fg, next, θ))")
    next = last

    if repair == RepairMode.greedy
        greedily_add!(fg, next, k)
    elseif repair == RepairMode.tabu
        tabu_repair!(fg, next, k, θ, tt, tabu_patience)
    elseif repair == RepairMode.mixed
        greedy_instance = deepcopy(next)
        tabu_instance = deepcopy(next)
        mixed_instance = deepcopy(next)
        
        greedily_add!(fg, greedy_instance, k)
        tabu_repair!(fg, tabu_instance, k, θ, tt, tabu_patience)

        greedily_add!(fg, mixed__instance, k)
        tabu_repair!(fg, mixed_instance, k, θ, tt, tabu_patience)

        greedy_score = instance_fitness(fg, greedy_instance, θ)
        tabu_score = instance_fitness(fg, tabu_instance, θ)
        mixed_score = instance_fitness(fg, mixed_instance, θ)

        if greedy_score > tabu_score && greedy_score > mixed_score
            next = greedy_instance
        elseif tabu_score > greedy_score && tabu_score > mixed_score
            next = tabu_instance
        end
    end

    println("After greedily add, next=", Subgraph.vertex_count(next))

    return next
end

# Remove H vertices from S, softmaxed by their nondegree with C
# ROOM FOR IMPROVEMENT: Should we do nondegree with C, nondegree with S, or nondegree with everything?
function H_opt!(fg::FrozenBipartite, offspring::SubGraph, k::Int, θ::Int, H::Int, repair::RepairMode.T, tt::Int, tabu_patience::Int)
    C = candidate_set(fg, offspring, k)
    nodes = softmax_sample_nodes((u, n) -> nondegree_in_subgraph(fg, u, n, C), offspring, min(H, Subgraph.vertex_count(offspring)))

    for node in nodes
        Subgraph.remove_node!(offspring, node.is_u, node.id)
    end

    if repair == RepairMode.greedy
        greedily_add!(fg, offspring, k)
    elseif repair == RepairMode.tabu
        tabu_repair!(fg, offspring, k, θ, tt, tabu_patience)
    end
end

function subgraph_intersection(sg1::SubGraph, sg2::SubGraph)
    U = intersect(sg1.U, sg2.U)
    V = intersect(sg1.V, sg2.V)
    return SubGraph(U, V)
end

