#=
Clique ACO: vertex ACO on the unipartite compatibility graph.

`super-reduce.jl` joins two vertices on the smaller side when they share at
least `θ − k` opposite neighbors. That projection is one undirected graph.
An ant grows a single set on it.

The first vertex is sampled the way vertex ACO samples an empty subgraph:
weight `τ · d^η`, where `d` is the bipartite degree. That degree is also the
greedy-fill edge count of a one-vertex side. Every later candidate is a
neighbor, in the projection, of the vertex just added. Its weight is
`τ · e^η`, where `e` is the greedy-fill edge count of the side with that
vertex added.

The fill is not rebuilt per candidate. Each opposite vertex stores its
missing-edge count `c = |L| − |N ∩ L|`. Adding a candidate leaves neighbors
at the same `c` and raises `c` by one for every non-neighbor; the pack is
read off that updated map. A neighbor is sampled only when that pack has
more edges than the current side. The ant stops when no neighbor grows the
fill, keeps that prefix, then `fill_keeping_side` adds every zero-defect
vertex on the searched side.
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
    cand_buf::Vector{Int}
    weight_buf::Vector{Float64}
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
    return CliqueAnt(falses(n), zeros(Int, n), Int[], Int[], Float64[],
        0, 0, 0, species, false, 0, -1, false, -1)
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
function compatibility_graph(fg::FrozenBipartite, k::Int, θ::Int;
        search_is_u::Union{Nothing,Bool}=nothing)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    if search_is_u === nothing
        search_is_u = nU <= nV
    end
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
Neighbors of the vertex just added, written into `ant.cand_buf`. With nothing
added yet, every vertex is a candidate, so the first sample can follow global
degree. Returns how many candidates were written.
"""
function clique_candidates!(cg::CliqueGraph, ant::CliqueAnt)
    buf = ant.cand_buf
    empty!(buf)
    if ant.last == 0
        @inbounds for i in eachindex(cg.deg)
            ant.in_S[i] || push!(buf, i)
        end
        return length(buf)
    end
    @inbounds for j in cg.adj[ant.last]
        ant.in_S[j] || push!(buf, j)
    end
    return length(buf)
end

clique_candidates(cg::CliqueGraph, ant::CliqueAnt) = (clique_candidates!(cg, ant); copy(ant.cand_buf))

"""
Greedy-fill edge count of `L ∪ {u}` without committing `u`.

`u` is a compact id on the searched side. Opposite vertices adjacent to `u`
keep their missing-edge count; every other opposite vertex's count rises by
one. The pack is the same ascending-cost fill as `oracle_score`.
"""
function oracle_edges_if_added(st::OracleState, fg::FrozenBipartite, u::Int, k::Int)
    nL = st.nL
    nh = st.buf_hist
    len = nL + 1
    @inbounds for i in 1:len
        nh[i] = 0
    end
    hits = st.hits
    @inbounds if st.search_is_u
        for t in neighbor_range_u(fg, u)
            c = nL - hits[fg.v_adj[t]]
            nh[c + 1] += 1
        end
    else
        for t in neighbor_range_v(fg, u)
            c = nL - hits[fg.u_adj[t]]
            nh[c + 1] += 1
        end
    end
    nL2 = nL + 1
    budget = k
    edges = 0
    maxc = min(k, nL)
    hist = st.hist
    @inbounds for c in 0:maxc
        avail = c == 0 ? nh[1] : nh[c + 1] + hist[c] - nh[c]
        avail <= 0 && continue
        c > 0 && budget < c && break
        take = c == 0 ? avail : min(avail, budget ÷ c)
        take <= 0 && continue
        edges += take * (nL2 - c)
        budget -= take * c
        budget == 0 && break
    end
    return edges
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

"""`τ · e^η`, where `e` is the greedy-fill edge count after adding clique index `v`."""
function clique_desirability(trails::CliqueTrails, cg::CliqueGraph, ant::CliqueAnt,
    st::OracleState, fg::FrozenBipartite, v::Int, k::Int, eta_exp::Float64)
    edges = oracle_edges_if_added(st, fg, cg.side_id[v], k)
    edges <= 0 && return 0.0
    τ = max(clique_tau(trails, v, ant.species), eps(Float64))
    return τ * Float64(edges)^eta_exp
end

"""Neighbor counts at costs `0:k` for every searched-side vertex. Row `c+1` is cost `c`."""
function init_fill_counts!(counts::Matrix{Int}, fg::FrozenBipartite, search_is_u::Bool)
    fill!(counts, 0)
    n = size(counts, 2)
    @inbounds if search_is_u
        for u in 1:n
            counts[1, u] = length(neighbor_range_u(fg, u))
        end
    else
        for u in 1:n
            counts[1, u] = length(neighbor_range_v(fg, u))
        end
    end
    return counts
end

"""Greedy-fill edges of `L ∪ {u}` from the maintained neighbor-cost counts."""
function fill_edges_from_counts(counts::Matrix{Int}, hist::Vector{Int}, u::Int, nL::Int, k::Int)
    maxc = min(k, nL)
    nL2 = nL + 1
    budget = k
    edges = 0
    @inbounds for c in 0:maxc
        avail = c == 0 ? counts[1, u] : counts[c + 1, u] + hist[c] - counts[c, u]
        avail <= 0 && continue
        c > 0 && budget < c && break
        take = c == 0 ? avail : min(avail, budget ÷ c)
        take <= 0 && continue
        edges += take * (nL2 - c)
        budget -= take * c
        budget == 0 && break
    end
    return edges
end

"""Move every non-neighbor of compact id `x` up one cost in `counts`. Call before `add_left!`."""
function raise_nonneighbors!(counts::Matrix{Int}, st::OracleState, fg::FrozenBipartite,
        x::Int, k::Int, mark::Vector{Int}, stamp::Int)
    nL = st.nL
    hits = st.hits
    @inbounds if st.search_is_u
        for t in neighbor_range_u(fg, x)
            mark[fg.v_adj[t]] = stamp
        end
        for y in 1:st.nR
            mark[y] == stamp && continue
            c = nL - hits[y]
            (c < 0 || c > k) && continue
            for t in neighbor_range_v(fg, y)
                s = fg.u_adj[t]
                counts[c + 1, s] -= 1
                c < k && (counts[c + 2, s] += 1)
            end
        end
    else
        for t in neighbor_range_v(fg, x)
            mark[fg.u_adj[t]] = stamp
        end
        for y in 1:st.nR
            mark[y] == stamp && continue
            c = nL - hits[y]
            (c < 0 || c > k) && continue
            for t in neighbor_range_u(fg, y)
                s = fg.v_adj[t]
                counts[c + 1, s] -= 1
                c < k && (counts[c + 2, s] += 1)
            end
        end
    end
    return counts
end

"""
Sample a neighbor whose greedy fill has strictly more edges than the current
side. Returns 0 when every neighbor stays flat or shrinks the fill.
"""
function sample_clique_vertex(ant::CliqueAnt, cg::CliqueGraph, fg::FrozenBipartite,
    st::OracleState, counts::Matrix{Int}, trails::CliqueTrails, k::Int, eta_exp::Float64)
    nc = clique_candidates!(cg, ant)
    nc == 0 && return 0
    cands = ant.cand_buf
    current, _ = oracle_score(st, k)
    weights = ant.weight_buf
    if length(weights) < nc
        resize!(weights, nc)
    end
    nL = st.nL
    hist = st.hist
    total = 0.0
    @inbounds for i in 1:nc
        edges = fill_edges_from_counts(counts, hist, cg.side_id[cands[i]], nL, k)
        if edges <= current
            weights[i] = 0.0
            continue
        end
        τ = max(clique_tau(trails, cands[i], ant.species), eps(Float64))
        w = τ * Float64(edges)^eta_exp
        weights[i] = w
        total += w
    end
    total <= 0.0 && return 0
    r = rand() * total
    cum = 0.0
    @inbounds for i in 1:nc
        cum += weights[i]
        cum >= r && return cands[i]
    end
    return cands[nc]
end

"""One construction step. Returns false when the ant has no positive-fill add left."""
function step_clique_ant!(ant::CliqueAnt, cg::CliqueGraph, fg::FrozenBipartite,
    st::OracleState, counts::Matrix{Int}, mark::Vector{Int}, stamp::Base.RefValue{Int},
    trails::CliqueTrails, additions::CliqueTrails,
    k::Int, θ::Int, pheromone::Int, eta_exp::Float64)
    ant.done && return false
    v = sample_clique_vertex(ant, cg, fg, st, counts, trails, k, eta_exp)
    if v == 0
        ant.done = true
        return false
    end
    commit_clique_vertex!(ant, cg, v)
    sid = cg.side_id[v]
    stamp[] += 1
    if stamp[] == typemax(Int)
        fill!(mark, 0)
        stamp[] = 1
    end
    raise_nonneighbors!(counts, st, fg, sid, k, mark, stamp[])
    add_left!(st, fg, sid)
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
        n_side = cg.search_is_u ? length(compact_fg.u_ids) : length(compact_fg.v_ids)
        n_opp = cg.search_is_u ? length(compact_fg.v_ids) : length(compact_fg.u_ids)
        counts = [init_fill_counts!(zeros(Int, k + 1, n_side), compact_fg, cg.search_is_u)
                  for _ in 1:num_ants]
        marks = [zeros(Int, n_opp) for _ in 1:num_ants]
        stamps = [Ref(0) for _ in 1:num_ants]
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
                    step_clique_ant!(ants[i], cg, compact_fg, states[i], counts[i],
                        marks[i], stamps[i], trails, additions[i], k, θ, pheromone, eta_exp)
                end
            else
                for i in 1:num_ants
                    active[i] || continue
                    step_clique_ant!(ants[i], cg, compact_fg, states[i], counts[i],
                        marks[i], stamps[i], trails, additions[i], k, θ, pheromone, eta_exp)
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
