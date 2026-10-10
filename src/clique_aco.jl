#=
Clique ACO: vertex ACO on the unipartite compatibility graph.

`super-reduce.jl` joins two vertices on the smaller side when they share at
least `θ − k` opposite neighbors. That projection is one undirected graph.
An ant grows a single set on it. Desirability is the vertex-ACO rule
`τ · η^eta_exp` with `η = d_S + d_G · σ(−|S|)`: `d_S` is the number of
neighbors already in the set, `d_G` is the degree in the projection, and
`σ(−|S|) = 1 / (1 + exp(|S|))`.

After the first vertex, the candidate list is the neighbors of the vertex
just added. A neighbor stays only when putting it in the set would leave at
most `k` missing projection edges (non-edges inside the set). The same `k`
is the biclique defect budget used to pack the opposite side of each
prefix. The ant keeps the prefix whose packing has the best biclique, then
`fill_keeping_side` adds every zero-defect vertex on the searched side.
=#

using Base.Threads

const __CLIQUE_ACO_JL__ = true

isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath("aco", "algorithm.jl"))
isdefined(@__MODULE__, :__SUPER_REDUCE_JL__) || include("super-reduce.jl")
isdefined(@__MODULE__, :__ORACLE_ACO_JL__) || include("oracle_aco.jl")

"""Compatibility graph. `side_id[i]` is the compact id on the searched side."""
struct CliqueGraph
    search_is_u::Bool
    side_id::Vector{Int}
    adj::Vector{Vector{Int}}
    deg::Vector{Int}
    edges::Int
end

CliqueGraph(search_is_u::Bool) = CliqueGraph(search_is_u, Int[], Vector{Int}[], Int[], 0)

"""Pheromone on projection vertices: one shared trail plus one trail per species."""
struct CliqueTrails
    shared::Vector{Float64}
    species::Vector{Vector{Float64}}
end

function CliqueTrails(n::Int, nspecies::Int; init::Float64=1.0)
    return CliqueTrails(fill(init, n), [zeros(Float64, n) for _ in 1:nspecies])
end

zero_clique_trails(n::Int, nspecies::Int) = CliqueTrails(n, nspecies; init=0.0)

function clique_tau(trails::CliqueTrails, idx::Int, species::Int)
    τ = trails.shared[idx] + trails.species[species][idx]
    for s in eachindex(trails.species)
        s == species && continue
        τ -= trails.species[s][idx]
    end
    return τ
end

function merge_clique_trails!(dst::CliqueTrails, src::CliqueTrails)
    dst.shared .+= src.shared
    for s in eachindex(dst.species)
        dst.species[s] .+= src.species[s]
    end
    return dst
end

function evaporate_clique_trails!(trails::CliqueTrails, evaporation::Float64)
    trails.shared .*= evaporation
    for s in trails.species
        s .*= evaporation
    end
    return trails
end

mutable struct CliqueAnt
    in_S::BitVector
    deg_S::Vector{Int}
    order::Vector{Int}
    n::Int
    missing::Int
    last::Int
    species::Int
    done::Bool
    best_len::Int
    best_edges::Int
    best_feasible::Bool
    best_fit::Int
end

function CliqueAnt(n::Int, species::Int)
    return CliqueAnt(falses(n), zeros(Int, n), Int[], 0, 0, 0, species, false,
        0, -1, false, -1)
end

function side_neighborhoods(fg::FrozenBipartite, search_is_u::Bool)
    if search_is_u
        n = length(fg.u_ids)
        adj = Dict{Int,Set{Int}}()
        for u in 1:n
            s = Set{Int}()
            for t in neighbor_range_u(fg, u)
                push!(s, fg.v_adj[t])
            end
            adj[u] = s
        end
        return adj
    end
    n = length(fg.v_ids)
    adj = Dict{Int,Set{Int}}()
    for v in 1:n
        s = Set{Int}()
        for t in neighbor_range_v(fg, v)
            push!(s, fg.u_adj[t])
        end
        adj[v] = s
    end
    return adj
end

"""
Projection used by `super_reduce!`: an edge when two vertices on the smaller
side share at least `θ − k` opposite neighbors. Isolates are dropped.
"""
function compatibility_graph(fg::FrozenBipartite, k::Int, θ::Int)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    search_is_u = nU <= nV
    n = search_is_u ? nU : nV
    n == 0 && return CliqueGraph(search_is_u)
    τ = θ - k
    neighborhoods = side_neighborhoods(fg, search_is_u)
    if τ <= 0
        ids = collect(1:n)
        adj = [Int[] for _ in 1:n]
        for i in 1:n, j in (i + 1):n
            push!(adj[i], j)
            push!(adj[j], i)
        end
        deg = [n - 1 for _ in 1:n]
        return CliqueGraph(search_is_u, ids, adj, deg, n * (n - 1) ÷ 2)
    end
    ids = Int[]
    for i in 1:n
        length(neighborhoods[i]) >= τ && push!(ids, i)
    end
    isempty(ids) && return CliqueGraph(search_is_u)
    wadj, _, _, _ = _weighted_projection(ids, neighborhoods, θ, τ)
    keep = Int[]
    for i in eachindex(wadj)
        isempty(wadj[i]) || push!(keep, i)
    end
    isempty(keep) && return CliqueGraph(search_is_u)
    old_to_new = zeros(Int, length(ids))
    for (neu, old) in enumerate(keep)
        old_to_new[old] = neu
    end
    side_id = [ids[old] for old in keep]
    adj = [Int[] for _ in keep]
    for (neu, old) in enumerate(keep)
        for j in keys(wadj[old])
            nj = old_to_new[j]
            nj == 0 && continue
            push!(adj[neu], nj)
        end
        sort!(adj[neu])
    end
    deg = [length(a) for a in adj]
    return CliqueGraph(search_is_u, side_id, adj, deg, sum(deg) ÷ 2)
end

"""Missing projection edges inside `members` (1-based clique indices)."""
function clique_induced_missing(cg::CliqueGraph, members::AbstractVector{Int})
    n = length(cg.deg)
    missing = 0
    for a in members
        seen = falses(n)
        for j in cg.adj[a]
            seen[j] = true
        end
        for b in members
            b <= a && continue
            seen[b] || (missing += 1)
        end
    end
    return missing
end

"""
Neighbors of the vertex just added that still fit in the defect budget.
With no vertex added yet, every vertex is a candidate.
"""
function clique_candidates(cg::CliqueGraph, ant::CliqueAnt, k::Int)
    out = Int[]
    if ant.last == 0
        @inbounds for i in eachindex(cg.deg)
            ant.in_S[i] && continue
            ant.missing + (ant.n - ant.deg_S[i]) <= k && push!(out, i)
        end
        return out
    end
    @inbounds for j in cg.adj[ant.last]
        ant.in_S[j] && continue
        ant.missing + (ant.n - ant.deg_S[j]) <= k && push!(out, j)
    end
    return out
end

"""Append `v`. `v` must already be legal for the current budget."""
function commit_clique_vertex!(ant::CliqueAnt, cg::CliqueGraph, v::Int)
    cost = ant.n - ant.deg_S[v]
    ant.missing += cost
    ant.n += 1
    ant.in_S[v] = true
    ant.last = v
    push!(ant.order, v)
    for j in cg.adj[v]
        ant.deg_S[j] += 1
    end
    return ant
end

"""`true` when this packing should replace the one recorded on `ant`."""
function pack_beats(nL::Int, nR::Int, edges::Int, θ::Int, ant::CliqueAnt)
    feas = nL >= θ && nR >= θ
    if feas
        ant.best_feasible || return true
        return edges > ant.best_edges
    elseif ant.best_feasible
        return false
    end
    m = min(nL, nR)
    fit = m * m
    return fit > ant.best_fit
end

function remember_pack!(ant::CliqueAnt, nL::Int, nR::Int, edges::Int, θ::Int)
    pack_beats(nL, nR, edges, θ, ant) || return ant
    ant.best_len = ant.n
    ant.best_edges = edges
    ant.best_feasible = nL >= θ && nR >= θ
    m = min(nL, nR)
    ant.best_fit = m * m
    return ant
end

function clique_desirability(trails::CliqueTrails, cg::CliqueGraph, ant::CliqueAnt,
    v::Int, eta_exp::Float64)
    τ = max(clique_tau(trails, v, ant.species), eps(Float64))
    σ = 1.0 / (1.0 + exp(ant.n))
    η = ant.deg_S[v] + cg.deg[v] * σ
    return τ * η^eta_exp
end

"""One construction step. Returns false when the ant has no legal add left."""
function step_clique_ant!(ant::CliqueAnt, cg::CliqueGraph, fg::FrozenBipartite,
    st::OracleState, trails::CliqueTrails, additions::CliqueTrails,
    k::Int, θ::Int, pheromone::Int, eta_exp::Float64)
    ant.done && return false
    cands = clique_candidates(cg, ant, k)
    if isempty(cands)
        ant.done = true
        return false
    end
    v = linear_sample_one(i -> clique_desirability(trails, cg, ant, i, eta_exp), cands)
    commit_clique_vertex!(ant, cg, v)
    add_left!(st, fg, cg.side_id[v])
    edges, nsel = oracle_score(st, k)
    remember_pack!(ant, st.nL, nsel, edges, θ)
    dep = Float64(pheromone)
    additions.species[ant.species][v] += dep
    additions.shared[v] += dep * SHARED_PHEROMONE_FACTOR
    return true
end

function prefix_biclique(fg::FrozenBipartite, cg::CliqueGraph, order::Vector{Int},
    len::Int, k::Int)
    len == 0 && return SubGraph()
    same = Set{Int}(cg.side_id[order[i]] for i in 1:len)
    bare = cg.search_is_u ? SubGraph(same, Set{Int}()) : SubGraph(Set{Int}(), same)
    return fill_keeping_side(fg, bare, cg.search_is_u, k)
end

"""
    clique_aco(g, pheromone, num_ants, num_iterations, evaporation, k, θ, num_subspecies)

Build the compatibility graph and run vertex ACO on it.
"""
function clique_aco(g::BipartiteGraph, pheromone::Int, num_ants::Int,
    num_iterations::Int, evaporation::Float64, k::Int, θ::Int,
    num_subspecies::Int=1;
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    eta_exp::Real=ETA_EXP,
    parallelize::Bool=false,
    verbose::Bool=true)
    num_ants >= 1 || throw(ArgumentError("num_ants must be ≥ 1, got $num_ants"))
    num_iterations >= 1 || throw(ArgumentError("num_iterations must be ≥ 1, got $num_iterations"))
    num_subspecies >= 1 || throw(ArgumentError("num_subspecies must be ≥ 1, got $num_subspecies"))
    0.0 < evaporation <= 1.0 || throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    k >= 0 || throw(ArgumentError("k must be ≥ 0, got $k"))
    eta_exp = Float64(eta_exp)
    isfinite(eta_exp) || throw(ArgumentError("eta_exp must be finite, got $eta_exp"))

    fg0 = apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    compact_fg, remapping = compact_frozen(fg0)
    nU = length(compact_fg.u_ids)
    nV = length(compact_fg.v_ids)
    search_is_u = nU <= nV

    empty_result = (;
        best=SubGraph(),
        search_is_u,
        clique_vertices=0,
        clique_edges=0,
        iterations_to_best=nothing,
        time_to_best_s=nothing,
        build_s=0.0,
        edges=0,
    )
    if nU < θ || nV < θ
        verbose && println("Clique ACO: reduced graph smaller than θ=$θ (|U|=$nU |V|=$nV)")
        return empty_result
    end

    t_build = time_ns()
    cg = compatibility_graph(compact_fg, k, θ)
    build_s = (time_ns() - t_build) / 1e9
    n = length(cg.side_id)
    if n == 0
        verbose && println("Clique ACO: compatibility graph is empty")
        return (; empty_result..., search_is_u=cg.search_is_u, build_s)
    end

    verbose && println("Clique ACO: search $(cg.search_is_u ? "U" : "V") " *
                       "clique_n=$n clique_edges=$(cg.edges) " *
                       "ants=$num_ants iterations=$num_iterations k=$k θ=$θ")

    trails = CliqueTrails(n, num_subspecies)
    best = SubGraph()
    best_iteration = 0
    best_time = 0.0
    t0 = time_ns()

    for iter in 1:num_iterations
        ants = [CliqueAnt(n, mod1(i, num_subspecies)) for i in 1:num_ants]
        states = [OracleState(compact_fg, cg.search_is_u) for _ in 1:num_ants]
        additions = [zero_clique_trails(n, num_subspecies) for _ in 1:num_ants]
        active = fill(true, num_ants)
        while any(active)
            for add in additions
                fill!(add.shared, 0.0)
                for s in add.species
                    fill!(s, 0.0)
                end
            end
            if parallelize && nthreads() > 1
                @threads :dynamic for i in 1:num_ants
                    active[i] || continue
                    step_clique_ant!(ants[i], cg, compact_fg, states[i],
                        trails, additions[i], k, θ, pheromone, eta_exp)
                end
            else
                for i in 1:num_ants
                    active[i] || continue
                    step_clique_ant!(ants[i], cg, compact_fg, states[i],
                        trails, additions[i], k, θ, pheromone, eta_exp)
                end
            end
            for i in 1:num_ants
                active[i] || continue
                merge_clique_trails!(trails, additions[i])
                active[i] = !ants[i].done
            end
        end
        evaporate_clique_trails!(trails, evaporation)

        for ant in ants
            ant.best_len == 0 && continue
            filled = prefix_biclique(compact_fg, cg, ant.order, ant.best_len, k)
            if better_than_best(compact_fg, filled, best, θ)
                best = filled
                best_iteration = iter
                best_time = (time_ns() - t0) / 1e9
            end
        end
    end

    edges = Subgraph.vertex_count(best) == 0 ? 0 : Subgraph.edge_count(compact_fg, best)
    reported = Subgraph.vertex_count(best) == 0 ? SubGraph() : remap_subgraph(remapping, best)
    verbose && println("Clique ACO best: |U|=$(length(reported.U)) |V|=$(length(reported.V)) " *
                       "edges=$edges  build=$(round(build_s; digits=3))s")
    return (;
        best=reported,
        search_is_u=cg.search_is_u,
        clique_vertices=n,
        clique_edges=cg.edges,
        iterations_to_best=best_iteration == 0 ? nothing : best_iteration,
        time_to_best_s=best_iteration == 0 ? nothing : best_time,
        build_s,
        edges,
    )
end
