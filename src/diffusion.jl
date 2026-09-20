#=
=================================================================================
Diffusion-guided vertex ACO for maximum k-defective edge biclique.

1. Assign every vertex an initial heat value equal to its degree.
2. For `diffusion_iters` rounds, replace each vertex's value with a weighted
   average of its own value and its neighbors' values, where neighbor edges
   (u, v) are weighted by c(u, v) = |n2(u) ∩ n(v)| (the number of common nodes
   that this edge represents, computed via 2-pass scan as in src/reduction.jl).
   Vertices that settle near the same value tend to lie in the same dense
   topological region of the bipartite graph.
3. Run standard vertex-trail ACO (`src/aco/`), but:
   - instance reward / elite fitness is boosted when the subgraph's diffused
     values have low standard deviation (cohesive ≈ likely co-biclique);
   - per-step pheromone deposit is scaled by the same cohesion factor.

Method id: `diffusion`.
=================================================================================
=#

const __DIFFUSION_JL__ = true

isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath("aco", "algorithm.jl"))

using Random

"""
Cohesion of diffused values on `sg`: 1 / (1 + std). Equal values → 1;
high spread → closer to 0.
"""
function diffusion_cohesion(sg::SubGraph, u_vals::Vector{Float64}, v_vals::Vector{Float64})
    n = length(sg.U) + length(sg.V)
    n <= 1 && return 1.0

    # Population std of the subgraph's diffused labels.
    mean = 0.0
    @inbounds for u in sg.U
        mean += u_vals[u]
    end
    @inbounds for v in sg.V
        mean += v_vals[v]
    end
    mean /= n

    var = 0.0
    @inbounds for u in sg.U
        d = u_vals[u] - mean
        var += d * d
    end
    @inbounds for v in sg.V
        d = v_vals[v] - mean
        var += d * d
    end
    std = sqrt(var / n)
    return 1.0 / (1.0 + std)
end

"""
Instance fitness × cohesion: larger when the subgraph is both well-sized
(θ-aware fitness) and topologically cohesive under the diffused field.
"""
function diffusion_fitness(fg::FrozenBipartite, sg::SubGraph, θ::Int,
    u_vals::Vector{Float64}, v_vals::Vector{Float64})
    return Float64(instance_fitness(fg, sg, θ)) *
        diffusion_cohesion(sg, u_vals, v_vals)
end

"""
Precompute edge weights for diffusion based on common nodes.
For an edge (u, v), let n(v) be the neighbors of v, and n2(u) be the 2-hop
neighbors of u via other neighbors. c(u, v) = |n2(u) ∩ n(v)| counts the number
of common nodes (w ∈ n(v) \\ {u}) that share at least one other neighbor with u
besides v (i.e. forming a 4-cycle / biclique motif with (u, v)).

Computed using the same 2-pass stamp-and-count process as in `src/reduction.jl`.
Returns `(u_edge_weights, v_edge_weights, u_total_weight, v_total_weight)`.
"""
function compute_diffusion_weights(fg::FrozenBipartite)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    nE = length(fg.v_adj)

    u_edge_weights = zeros(Float64, nE)
    v_edge_weights = zeros(Float64, nE)
    u_total_weight = ones(Float64, nU)
    v_total_weight = ones(Float64, nV)

    stamp_u = zeros(Int, nU)
    count_u = zeros(Int, nU)
    cur_stamp = 0

    # U -> V direction: for each u, compute weights for its outgoing edges to v
    @inbounds for ui in 1:nU
        nbr_range = neighbor_range_u(fg, ui)
        isempty(nbr_range) && continue

        cur_stamp += 1
        # Pass 1: count 2-hop occurrences of U-side vertices from ui
        for k in nbr_range
            vi = fg.v_adj[k]
            for l in neighbor_range_v(fg, vi)
                w = fg.u_adj[l]
                if stamp_u[w] != cur_stamp
                    stamp_u[w] = cur_stamp
                    count_u[w] = 1
                else
                    count_u[w] += 1
                end
            end
        end

        # Pass 2: for each neighbor v, count common nodes w in n(v) with count_u[w] >= 2 (w != ui)
        sum_w = 0.0
        for k in nbr_range
            vi = fg.v_adj[k]
            c_uv = 0.0
            for l in neighbor_range_v(fg, vi)
                w = fg.u_adj[l]
                if w != ui && stamp_u[w] == cur_stamp && count_u[w] >= 2
                    c_uv += 1.0
                end
            end
            u_edge_weights[k] = c_uv
            sum_w += c_uv
        end
        u_total_weight[ui] = 1.0 + sum_w
    end

    stamp_v = zeros(Int, nV)
    count_v = zeros(Int, nV)
    cur_stamp = 0

    # V -> U direction: for each v, compute weights for its outgoing edges to u
    @inbounds for vi in 1:nV
        nbr_range = neighbor_range_v(fg, vi)
        isempty(nbr_range) && continue

        cur_stamp += 1
        # Pass 1: count 2-hop occurrences of V-side vertices from vi
        for k in nbr_range
            ui = fg.u_adj[k]
            for l in neighbor_range_u(fg, ui)
                w = fg.v_adj[l]
                if stamp_v[w] != cur_stamp
                    stamp_v[w] = cur_stamp
                    count_v[w] = 1
                else
                    count_v[w] += 1
                end
            end
        end

        # Pass 2: for each neighbor u, count common nodes w in n(u) with count_v[w] >= 2 (w != vi)
        sum_w = 0.0
        for k in nbr_range
            ui = fg.u_adj[k]
            c_vu = 0.0
            for l in neighbor_range_u(fg, ui)
                w = fg.v_adj[l]
                if w != vi && stamp_v[w] == cur_stamp && count_v[w] >= 2
                    c_vu += 1.0
                end
            end
            v_edge_weights[k] = c_vu
            sum_w += c_vu
        end
        v_total_weight[vi] = 1.0 + sum_w
    end

    return u_edge_weights, v_edge_weights, u_total_weight, v_total_weight
end

"""
One Jacobi-style diffusion step: each vertex becomes the weighted average of
itself and its graph neighbors based on common-node edge weights.
Isolated vertices (or vertices with zero neighbor weights) keep their value.
"""
function _diffusion_step!(fg::FrozenBipartite, u_vals::Vector{Float64},
    v_vals::Vector{Float64}, u_next::Vector{Float64}, v_next::Vector{Float64},
    u_edge_weights::Vector{Float64}, v_edge_weights::Vector{Float64},
    u_total_weight::Vector{Float64}, v_total_weight::Vector{Float64})
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)

    @inbounds for ui in 1:nU
        s = u_vals[ui]
        for k in neighbor_range_u(fg, ui)
            s += u_edge_weights[k] * v_vals[fg.v_adj[k]]
        end
        u_next[ui] = s / u_total_weight[ui]
    end
    @inbounds for vi in 1:nV
        s = v_vals[vi]
        for k in neighbor_range_v(fg, vi)
            s += v_edge_weights[k] * u_vals[fg.u_adj[k]]
        end
        v_next[vi] = s / v_total_weight[vi]
    end
    return nothing
end

function _diffusion_step!(fg::FrozenBipartite, u_vals::Vector{Float64},
    v_vals::Vector{Float64}, u_next::Vector{Float64}, v_next::Vector{Float64})
    u_ew, v_ew, u_tw, v_tw = compute_diffusion_weights(fg)
    return _diffusion_step!(fg, u_vals, v_vals, u_next, v_next, u_ew, v_ew, u_tw, v_tw)
end

"""
Population standard deviation of vertex heat values across U and V.
"""
function _heat_std(u_vals::Vector{Float64}, v_vals::Vector{Float64})
    n = length(u_vals) + length(v_vals)
    n == 0 && return 0.0
    mean = 0.0
    @inbounds for u in u_vals
        mean += u
    end
    @inbounds for v in v_vals
        mean += v
    end
    mean /= n

    var = 0.0
    @inbounds for u in u_vals
        d = u - mean
        var += d * d
    end
    @inbounds for v in v_vals
        d = v - mean
        var += d * d
    end
    return sqrt(var / n)
end

"""
Extract U and V node collections from various biclique representations.
"""
function _extract_uv(injected_biclique)
    injected_biclique === nothing && return nothing, nothing
    if injected_biclique isa SubGraph
        return injected_biclique.U, injected_biclique.V
    elseif hasproperty(injected_biclique, :U) && hasproperty(injected_biclique, :V)
        return getproperty(injected_biclique, :U), getproperty(injected_biclique, :V)
    elseif injected_biclique isa Tuple && length(injected_biclique) == 2
        return injected_biclique[1], injected_biclique[2]
    elseif haskey(injected_biclique, :U) && haskey(injected_biclique, :V)
        return injected_biclique[:U], injected_biclique[:V]
    elseif haskey(injected_biclique, "U") && haskey(injected_biclique, "V")
        return injected_biclique["U"], injected_biclique["V"]
    end
    return nothing, nothing
end

"""
Look up dense 1-based index of vertex u on the U side of fg.
"""
function _lookup_u_index(fg::FrozenBipartite, u::Int, remapping=nothing)
    if remapping !== nothing && hasproperty(remapping, :u_original)
        idx = findfirst(==(u), remapping.u_original)
        idx !== nothing && return idx
    end
    if haskey(fg.u_index, u)
        return fg.u_index[u]
    end
    if 1 <= u <= length(fg.u_ids)
        return u
    end
    return nothing
end

"""
Look up dense 1-based index of vertex v on the V side of fg.
"""
function _lookup_v_index(fg::FrozenBipartite, v::Int, remapping=nothing)
    if remapping !== nothing && hasproperty(remapping, :v_original)
        idx = findfirst(==(v), remapping.v_original)
        idx !== nothing && return idx
    end
    if haskey(fg.v_index, v)
        return fg.v_index[v]
    end
    if 1 <= v <= length(fg.v_ids)
        return v
    end
    return nothing
end

"""
Sample sample_n unique vertex indices from 1:total_n.
"""
function _sample_vertex_indices(rng::AbstractRNG, total_n::Int, sample_n::Int)
    sample_n = min(sample_n, total_n)
    sample_n <= 0 && return Int[]
    if sample_n == total_n
        return collect(1:total_n)
    elseif sample_n < total_n ÷ 2
        sampled = Set{Int}()
        sizehint!(sampled, sample_n)
        while length(sampled) < sample_n
            push!(sampled, rand(rng, 1:total_n))
        end
        return collect(sampled)
    else
        return Random.randperm(rng, total_n)[1:sample_n]
    end
end

"""
Initialize U/V values with vertex degrees and diffuse for `iterations` steps.
Returns `(u_vals, v_vals)` indexed by compact vertex id (1..nU / 1..nV).
"""
function diffuse_vertex_values(fg::FrozenBipartite; iterations::Int=20,
    rng::AbstractRNG=Random.default_rng(), log_std::Bool=true, io::IO=stdout,
    injected_biclique=nothing, remapping=nothing)
    iterations >= 0 || throw(ArgumentError("iterations must be ≥ 0, got $iterations"))
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    u_vals = [Float64(length(neighbor_range_u(fg, ui))) for ui in 1:nU]
    v_vals = [Float64(length(neighbor_range_v(fg, vi))) for vi in 1:nV]
    u_next = similar(u_vals)
    v_next = similar(v_vals)

    u_edge_weights, v_edge_weights, u_total_weight, v_total_weight =
        compute_diffusion_weights(fg)

    if log_std
        std_val = _heat_std(u_vals, v_vals)
        println(io, "Diffusion iter 0/$iterations: heat std=$std_val")
    end

    for iter in 1:iterations
        _diffusion_step!(fg, u_vals, v_vals, u_next, v_next,
            u_edge_weights, v_edge_weights, u_total_weight, v_total_weight)
        u_vals, u_next = u_next, u_vals
        v_vals, v_next = v_next, v_vals
        if log_std
            std_val = _heat_std(u_vals, v_vals)
            println(io, "Diffusion iter $iter/$iterations: heat std=$std_val")
        end
    end

    if log_std && injected_biclique !== nothing
        stats = compute_diffusion_heat_stats(u_vals, v_vals, injected_biclique, fg;
            rng=rng, remapping=remapping)
        n_inj = stats["biclique_nodes"]
        if n_inj > 0
            mean_inj = stats["biclique_mean"]
            mean_all = stats["full_graph_mean"]
            mean_rand = stats["sampled_graph_mean"]
            println(io, "Diffusion heat mean: injected=$mean_inj vs overall=$mean_all vs random_sample=$mean_rand (n=$n_inj)")
        else
            mean_all = stats["full_graph_mean"]
            println(io, "Diffusion heat mean: injected vertices not found in graph vs overall=$mean_all")
        end
    end

    return u_vals, v_vals
end

function _vec_std(vals::Vector{Float64})
    n = length(vals)
    n <= 1 && return 0.0
    mean = sum(vals) / n
    var = sum((x - mean)^2 for x in vals) / n
    return sqrt(var)
end

"""
Compute heat statistics comparing a biclique (injected plant or discovered subgraph)
against the full graph and a random sample of equal size.
"""
function compute_diffusion_heat_stats(u_vals::Vector{Float64}, v_vals::Vector{Float64},
    biclique, fg::FrozenBipartite; rng::AbstractRNG=Random.default_rng(), remapping=nothing)
    total_n = length(u_vals) + length(v_vals)
    mean_all = total_n > 0 ? (sum(u_vals) + sum(v_vals)) / total_n : 0.0
    std_all = _heat_std(u_vals, v_vals)

    inj_u, inj_v = _extract_uv(biclique)
    inj_vals = Float64[]
    if inj_u !== nothing && inj_v !== nothing
        for u in inj_u
            ui = _lookup_u_index(fg, u, remapping)
            if ui !== nothing && 1 <= ui <= length(u_vals)
                push!(inj_vals, u_vals[ui])
            end
        end
        for v in inj_v
            vi = _lookup_v_index(fg, v, remapping)
            if vi !== nothing && 1 <= vi <= length(v_vals)
                push!(inj_vals, v_vals[vi])
            end
        end
    end

    n_inj = length(inj_vals)
    if n_inj == 0
        return Dict{String,Any}(
            "biclique_nodes" => 0,
            "total_nodes" => total_n,
            "full_graph_mean" => mean_all,
            "total_graph_mean" => mean_all,
            "full_graph_std" => std_all,
            "biclique_mean" => nothing,
            "biclique_heat_mean" => nothing,
            "sampled_graph_mean" => nothing,
            "percent_change" => nothing,
            "difference" => nothing,
        )
    end

    mean_inj = sum(inj_vals) / n_inj
    std_inj = _vec_std(inj_vals)

    sampled_indices = _sample_vertex_indices(rng, total_n, n_inj)
    sample_sum = 0.0
    sample_vals = Float64[]
    for idx in sampled_indices
        val = idx <= length(u_vals) ? u_vals[idx] : v_vals[idx - length(u_vals)]
        push!(sample_vals, val)
        sample_sum += val
    end
    mean_rand = length(sampled_indices) > 0 ? sample_sum / length(sampled_indices) : 0.0
    std_rand = _vec_std(sample_vals)

    diff = mean_inj - mean_all
    pct_change = mean_all != 0.0 ? (diff / mean_all) * 100.0 : 0.0
    diff_rand = mean_inj - mean_rand
    pct_change_rand = mean_rand != 0.0 ? (diff_rand / mean_rand) * 100.0 : 0.0

    return Dict{String,Any}(
        "biclique_mean" => mean_inj,
        "biclique_heat_mean" => mean_inj,
        "full_graph_mean" => mean_all,
        "total_graph_mean" => mean_all,
        "sampled_graph_mean" => mean_rand,
        "biclique_std" => std_inj,
        "full_graph_std" => std_all,
        "sampled_graph_std" => std_rand,
        "percent_change" => pct_change,
        "difference" => diff,
        "percent_change_vs_sampled" => pct_change_rand,
        "difference_vs_sampled" => diff_rand,
        "biclique_nodes" => n_inj,
        "total_nodes" => total_n,
    )
end

"""
Compute diffusion vertex values and heat stats for fg given an optional injected biclique
and/or solution biclique.
"""
function get_diffusion_heat_stats(fg::FrozenBipartite; iterations::Int=20,
    injected_biclique=nothing, solution_biclique=nothing,
    rng::AbstractRNG=Random.default_rng(), remapping=nothing)
    u_vals, v_vals = diffuse_vertex_values(fg; iterations=iterations, rng=rng,
        log_std=false, remapping=remapping)
    total_n = length(u_vals) + length(v_vals)
    mean_all = total_n > 0 ? (sum(u_vals) + sum(v_vals)) / total_n : 0.0
    std_all = _heat_std(u_vals, v_vals)

    inj_stats = injected_biclique !== nothing ?
        compute_diffusion_heat_stats(u_vals, v_vals, injected_biclique, fg; rng=rng, remapping=remapping) : nothing
    sol_stats = solution_biclique !== nothing ?
        compute_diffusion_heat_stats(u_vals, v_vals, solution_biclique, fg; rng=rng, remapping=remapping) : nothing

    primary = if inj_stats !== nothing && get(inj_stats, "biclique_nodes", 0) > 0
        inj_stats
    elseif sol_stats !== nothing && get(sol_stats, "biclique_nodes", 0) > 0
        sol_stats
    else
        Dict{String,Any}(
            "biclique_nodes" => 0,
            "total_nodes" => total_n,
            "full_graph_mean" => mean_all,
            "total_graph_mean" => mean_all,
            "full_graph_std" => std_all,
            "biclique_mean" => nothing,
            "biclique_heat_mean" => nothing,
            "sampled_graph_mean" => nothing,
            "percent_change" => nothing,
            "difference" => nothing,
        )
    end

    res = copy(primary)
    res["iterations"] = iterations
    if inj_stats !== nothing
        res["injected"] = inj_stats
    end
    if sol_stats !== nothing
        res["solution"] = sol_stats
    end
    return res
end

"""
Build the ACO scoring hook from a precomputed (or freshly diffused) field.
"""
function diffusion_scoring(compact_fg::FrozenBipartite; diffusion_iters::Int=20,
    rng::AbstractRNG=Random.default_rng(), injected_biclique=nothing,
    remapping=nothing, log_std::Bool=true, io::IO=stdout)
    u_vals, v_vals = diffuse_vertex_values(compact_fg;
        iterations=diffusion_iters, rng=rng, log_std=log_std, io=io,
        injected_biclique=injected_biclique, remapping=remapping)
    fitness = (fg, sg, θ) -> diffusion_fitness(fg, sg, θ, u_vals, v_vals)
    deposit_scale = (fg, sg) -> diffusion_cohesion(sg, u_vals, v_vals)
    return (; fitness, deposit_scale, u_vals, v_vals)
end

"""
Vertex-trail ACO with a diffusion-cohesion reward.

Same kwargs as `aco`, plus `diffusion_iters` (default 20).
"""
function diffusion_aco(g::BipartiteGraph, pheromone::Int, num_ants::Int,
    num_iterations::Int, evaporation::Float64, k::Int, θ::Int, num_subspecies::Int;
    diffusion_iters::Int=20,
    parallelize::Bool=true, force_gc::Bool=false, iteration_callback=nothing,
    pheromone_min::Union{Float64,Nothing}=nothing,
    pheromone_max::Union{Float64,Nothing}=nothing,
    tt::Int=2, tabu_patience::Int=3,
    prefer_smaller_side::Bool=false,
    neighbor_scope_limit::Bool=true,
    elite_seed::Bool=true,
    elite_seed_ants::Int=3,
    elite_seed_remove::Int=2,
    elite_pheromone::Bool=false,
    aco_tabu::Bool=false,
    mmas::Bool=false,
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    trace_target::Union{Nothing,SubGraph}=nothing,
    seed_nodes::Union{Nothing,SubGraph}=SubGraph(Set(), Set()),
    construction_stats::Union{Nothing,Base.RefValue}=nothing,
    injected_biclique=nothing,
    log_std::Bool=true,
    io::IO=stdout)
    diffusion_iters >= 0 || throw(ArgumentError(
        "diffusion_iters must be ≥ 0, got $diffusion_iters"))
    println("Diffusion ACO: diffusion_iters=$diffusion_iters (vertex pheromone)")

    target_plant = injected_biclique !== nothing ? injected_biclique : trace_target
    return aco(g, pheromone, num_ants, num_iterations, evaporation, k, θ, num_subspecies;
        parallelize=parallelize,
        force_gc=force_gc,
        iteration_callback=iteration_callback,
        pheromone_min=pheromone_min,
        pheromone_max=pheromone_max,
        tt=tt,
        tabu_patience=tabu_patience,
        prefer_smaller_side=prefer_smaller_side,
        neighbor_scope_limit=neighbor_scope_limit,
        elite_seed=elite_seed,
        elite_seed_ants=elite_seed_ants,
        elite_seed_remove=elite_seed_remove,
        elite_pheromone=elite_pheromone,
        aco_tabu=aco_tabu,
        mmas=mmas,
        reduction=reduction,
        trace_target=trace_target,
        seed_nodes=seed_nodes,
        construction_stats=construction_stats,
        make_scoring=(fg, rem=nothing) -> diffusion_scoring(fg;
            diffusion_iters=diffusion_iters,
            injected_biclique=target_plant,
            remapping=rem,
            log_std=log_std,
            io=io))
end
