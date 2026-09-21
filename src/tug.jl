#=
=================================================================================
Tug-of-War (Alternating Best-Response) Heuristic for k-Defective Bicliques (k-MDB).

Solves:
    maximize |E(U, V)| subject to |bar{E}(U, V)| <= k and |U| >= θ, |V| >= θ

Key mechanisms:
1. Degree peeling: iterative removal of vertices with degree < θ - k.
2. Seed scoring & ablation:
   - :sc     => S_C(u, v) neighborhood-Jaccard score
   - :degree => deg(u) * deg(v) degree product
   - :random => uniform random edge selection
3. Multi-scale core formation: {2, floor(θ/2), θ} sized crews using co-neighbor
   Jaccard similarity.
4. Cheapest-first best-response: hires candidates from cheapest to most expensive
   subject to meeting θ and staying within defect budget k.
5. Alternation loop: alternating U -> V -> U until no edge improvement.
6. Defect-based shaking: kicks out vertices causing the most defects, forbids
   them for one pass, and re-recruits to escape local optima.
=================================================================================
=#

const __TUG_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__NEIGHBORHOOD_JL__) || include("neighborhood.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")

using Random

# ─────────────────────────────────────────────────────────────────────────────
# 1. Degree Peeling (Iterative (θ - k)-core reduction)
# ─────────────────────────────────────────────────────────────────────────────

"""
    peel_degrees!(g::BipartiteGraph, min_degree::Int) -> BipartiteGraph

Iteratively removes vertices with degree < `min_degree`. In any valid k-defective
biclique with |U|, |V| >= θ, every vertex must have degree at least θ - k.
"""
function peel_degrees!(g::BipartiteGraph, min_degree::Int)
    min_degree <= 0 && return g

    q_u = Int[]
    q_v = Int[]

    for (u, nbrs) in g.adjU
        if length(nbrs) < min_degree
            push!(q_u, u)
        end
    end
    for (v, nbrs) in g.adjV
        if length(nbrs) < min_degree
            push!(q_v, v)
        end
    end

    while !isempty(q_u) || !isempty(q_v)
        while !isempty(q_u)
            u = pop!(q_u)
            haskey(g.adjU, u) || continue
            nbrs = collect(g.adjU[u])
            rem_u!(g, u)
            for v in nbrs
                if haskey(g.adjV, v) && length(g.adjV[v]) < min_degree
                    push!(q_v, v)
                end
            end
        end

        while !isempty(q_v)
            v = pop!(q_v)
            haskey(g.adjV, v) || continue
            nbrs = collect(g.adjV[v])
            rem_v!(g, v)
            for u in nbrs
                if haskey(g.adjU, u) && length(g.adjU[u]) < min_degree
                    push!(q_u, u)
                end
            end
        end
    end

    return g
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Dense Graph Helpers & Best Response
# ─────────────────────────────────────────────────────────────────────────────

"""
    best_response_v_dense(fg, U_dense, k, θ; forbidden_V=nothing) -> Union{Vector{Int}, Nothing}

Given fixed U vertices, recruit V vertices cheapest-first.
Cost of v is |U| - deg_U(v).
Requires hiring at least θ vertices with total cost <= k.
Hires as many vertices as budget allows. Returns dense V indices or nothing.
"""
function best_response_v_dense(fg::FrozenBipartite, U_dense::Vector{Int}, k::Int, θ::Int;
                               forbidden_V::Union{Nothing,Set{Int}}=nothing)
    nV = length(fg.v_ids)
    u_len = length(U_dense)
    u_len == 0 && return nothing

    deg_in_U = zeros(Int, nV)
    @inbounds for ui in U_dense
        for slot in neighbor_range_u(fg, ui)
            vi = fg.v_adj[slot]
            deg_in_U[vi] += 1
        end
    end

    candidates = Tuple{Int, Int, Int}[] # (cost, -global_deg, vi)
    sizehint!(candidates, nV)
    @inbounds for vi in 1:nV
        if forbidden_V !== nothing && vi in forbidden_V
            continue
        end
        cost = u_len - deg_in_U[vi]
        gdeg = length(neighbor_range_v(fg, vi))
        push!(candidates, (cost, -gdeg, vi))
    end

    length(candidates) < θ && return nothing
    sort!(candidates)

    cost_theta = 0
    @inbounds for i in 1:θ
        cost_theta += candidates[i][1]
    end
    cost_theta > k && return nothing

    cum_cost = cost_theta
    hired_count = θ
    @inbounds for i in (θ + 1):length(candidates)
        c = candidates[i][1]
        c == u_len && break # 0 edges to U, adds nothing to objective
        if cum_cost + c <= k
            cum_cost += c
            hired_count = i
        else
            break
        end
    end

    res = Vector{Int}(undef, hired_count)
    @inbounds for i in 1:hired_count
        res[i] = candidates[i][3]
    end
    return res
end

"""
    best_response_u_dense(fg, V_dense, k, θ; forbidden_U=nothing) -> Union{Vector{Int}, Nothing}

Given fixed V vertices, recruit U vertices cheapest-first.
"""
function best_response_u_dense(fg::FrozenBipartite, V_dense::Vector{Int}, k::Int, θ::Int;
                               forbidden_U::Union{Nothing,Set{Int}}=nothing)
    nU = length(fg.u_ids)
    v_len = length(V_dense)
    v_len == 0 && return nothing

    deg_in_V = zeros(Int, nU)
    @inbounds for vi in V_dense
        for slot in neighbor_range_v(fg, vi)
            ui = fg.u_adj[slot]
            deg_in_V[ui] += 1
        end
    end

    candidates = Tuple{Int, Int, Int}[] # (cost, -global_deg, ui)
    sizehint!(candidates, nU)
    @inbounds for ui in 1:nU
        if forbidden_U !== nothing && ui in forbidden_U
            continue
        end
        cost = v_len - deg_in_V[ui]
        gdeg = length(neighbor_range_u(fg, ui))
        push!(candidates, (cost, -gdeg, ui))
    end

    length(candidates) < θ && return nothing
    sort!(candidates)

    cost_theta = 0
    @inbounds for i in 1:θ
        cost_theta += candidates[i][1]
    end
    cost_theta > k && return nothing

    cum_cost = cost_theta
    hired_count = θ
    @inbounds for i in (θ + 1):length(candidates)
        c = candidates[i][1]
        c == v_len && break
        if cum_cost + c <= k
            cum_cost += c
            hired_count = i
        else
            break
        end
    end

    res = Vector{Int}(undef, hired_count)
    @inbounds for i in 1:hired_count
        res[i] = candidates[i][3]
    end
    return res
end

"""
    count_edges_and_defects_dense(fg, U_dense, V_dense) -> (edges, missing_edges)
"""
function count_edges_and_defects_dense(fg::FrozenBipartite, U_dense::Vector{Int}, V_dense::Vector{Int})
    nV = length(fg.v_ids)
    deg_in_U = zeros(Int, nV)
    @inbounds for ui in U_dense
        for slot in neighbor_range_u(fg, ui)
            vi = fg.v_adj[slot]
            deg_in_U[vi] += 1
        end
    end
    edges = 0
    @inbounds for vi in V_dense
        edges += deg_in_U[vi]
    end
    missing_edges = length(U_dense) * length(V_dense) - edges
    return edges, missing_edges
end

"""
    vertex_defects_dense(fg, U_dense, V_dense) -> (def_u, def_v)
"""
function vertex_defects_dense(fg::FrozenBipartite, U_dense::Vector{Int}, V_dense::Vector{Int})
    nV = length(fg.v_ids)
    nU = length(fg.u_ids)
    deg_in_U = zeros(Int, nV)
    @inbounds for ui in U_dense
        for slot in neighbor_range_u(fg, ui)
            deg_in_U[fg.v_adj[slot]] += 1
        end
    end
    deg_in_V = zeros(Int, nU)
    @inbounds for vi in V_dense
        for slot in neighbor_range_v(fg, vi)
            deg_in_V[fg.u_adj[slot]] += 1
        end
    end

    def_u = [length(V_dense) - deg_in_V[ui] for ui in U_dense]
    def_v = [length(U_dense) - deg_in_U[vi] for vi in V_dense]

    return def_u, def_v
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Core Formation (Similarity Crews)
# ─────────────────────────────────────────────────────────────────────────────

function form_u_core(fg::FrozenBipartite, u_seed::Int, v_seed::Int, core_size::Int;
                     cnt_v::Vector{Int8}, dirty_v::Vector{Int})
    core = [u_seed]
    core_size <= 1 && return core

    nbrs_v = [fg.u_adj[k] for k in neighbor_range_v(fg, v_seed) if fg.u_adj[k] != u_seed]

    cand_sim = Tuple{Float64, Int, Int}[]
    for u in nbrs_v
        sim = _jaccard_u(fg, cnt_v, dirty_v, u_seed, u)
        deg = length(neighbor_range_u(fg, u))
        push!(cand_sim, (sim, deg, u))
    end

    sort!(cand_sim, by=x -> (x[1], x[2]), rev=true)
    for i in 1:min(core_size - 1, length(cand_sim))
        push!(core, cand_sim[i][3])
    end

    if length(core) < core_size
        cand_others = Tuple{Float64, Int, Int}[]
        in_core = Set(core)
        for u in 1:length(fg.u_ids)
            u in in_core && continue
            sim = _jaccard_u(fg, cnt_v, dirty_v, u_seed, u)
            sim > 0.0 || continue
            deg = length(neighbor_range_u(fg, u))
            push!(cand_others, (sim, deg, u))
        end
        sort!(cand_others, by=x -> (x[1], x[2]), rev=true)
        for i in 1:min(core_size - length(core), length(cand_others))
            push!(core, cand_others[i][3])
        end
    end

    return core
end

function form_v_core(fg::FrozenBipartite, u_seed::Int, v_seed::Int, core_size::Int;
                     cnt_u::Vector{Int8}, dirty_u::Vector{Int})
    core = [v_seed]
    core_size <= 1 && return core

    nbrs_u = [fg.v_adj[k] for k in neighbor_range_u(fg, u_seed) if fg.v_adj[k] != v_seed]

    cand_sim = Tuple{Float64, Int, Int}[]
    for v in nbrs_u
        sim = _jaccard_v(fg, cnt_u, dirty_u, v_seed, v)
        deg = length(neighbor_range_v(fg, v))
        push!(cand_sim, (sim, deg, v))
    end

    sort!(cand_sim, by=x -> (x[1], x[2]), rev=true)
    for i in 1:min(core_size - 1, length(cand_sim))
        push!(core, cand_sim[i][3])
    end

    if length(core) < core_size
        cand_others = Tuple{Float64, Int, Int}[]
        in_core = Set(core)
        for v in 1:length(fg.v_ids)
            v in in_core && continue
            sim = _jaccard_v(fg, cnt_u, dirty_u, v_seed, v)
            sim > 0.0 || continue
            deg = length(neighbor_range_v(fg, v))
            push!(cand_others, (sim, deg, v))
        end
        sort!(cand_others, by=x -> (x[1], x[2]), rev=true)
        for i in 1:min(core_size - length(core), length(cand_others))
            push!(core, cand_others[i][3])
        end
    end

    return core
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Alternation Loop & Shaking
# ─────────────────────────────────────────────────────────────────────────────

function alternate_recruitment_dense(fg::FrozenBipartite, initial_U::Vector{Int}, k::Int, θ::Int;
                                     max_rounds::Int=50)
    V = best_response_v_dense(fg, initial_U, k, θ)
    V === nothing && return nothing

    U = best_response_u_dense(fg, V, k, θ)
    U === nothing && return nothing

    edges, missing_e = count_edges_and_defects_dense(fg, U, V)
    missing_e > k && return nothing

    for _ in 1:max_rounds
        V_next = best_response_v_dense(fg, U, k, θ)
        V_next === nothing && break
        U_next = best_response_u_dense(fg, V_next, k, θ)
        U_next === nothing && break

        new_edges, new_missing = count_edges_and_defects_dense(fg, U_next, V_next)
        if new_missing <= k && new_edges > edges
            U = U_next
            V = V_next
            edges = new_edges
        else
            break
        end
    end

    return (U, V, edges)
end

function alternate_recruitment_v_first_dense(fg::FrozenBipartite, initial_V::Vector{Int}, k::Int, θ::Int;
                                             max_rounds::Int=50)
    U = best_response_u_dense(fg, initial_V, k, θ)
    U === nothing && return nothing

    V = best_response_v_dense(fg, U, k, θ)
    V === nothing && return nothing

    edges, missing_e = count_edges_and_defects_dense(fg, U, V)
    missing_e > k && return nothing

    for _ in 1:max_rounds
        U_next = best_response_u_dense(fg, V, k, θ)
        U_next === nothing && break
        V_next = best_response_v_dense(fg, U_next, k, θ)
        V_next === nothing && break

        new_edges, new_missing = count_edges_and_defects_dense(fg, U_next, V_next)
        if new_missing <= k && new_edges > edges
            U = U_next
            V = V_next
            edges = new_edges
        else
            break
        end
    end

    return (U, V, edges)
end

function shake_and_improve(fg::FrozenBipartite, U::Vector{Int}, V::Vector{Int},
                           cur_edges::Int, k::Int, θ::Int; max_shakes::Int=3)
    best_U = U
    best_V = V
    best_edges = cur_edges

    for _ in 1:max_shakes
        def_u, def_v = vertex_defects_dense(fg, best_U, best_V)
        max_u_def = isempty(def_u) ? 0 : maximum(def_u)
        max_v_def = isempty(def_v) ? 0 : maximum(def_v)

        (max_u_def == 0 && max_v_def == 0) && break

        improved = false

        # Shake U side: remove worst defect node, forbid for 1 round
        if max_u_def > 0
            worst_u_idx = argmax(def_u)
            worst_u = best_U[worst_u_idx]
            U_shaken = filter(!=(worst_u), best_U)
            if !isempty(U_shaken)
                V_temp = best_response_v_dense(fg, U_shaken, k, θ)
                if V_temp !== nothing
                    U_temp = best_response_u_dense(fg, V_temp, k, θ; forbidden_U=Set([worst_u]))
                    if U_temp !== nothing
                        alt_res = alternate_recruitment_dense(fg, U_temp, k, θ)
                        if alt_res !== nothing
                            new_U, new_V, new_edges = alt_res
                            if new_edges > best_edges
                                best_U = new_U
                                best_V = new_V
                                best_edges = new_edges
                                improved = true
                            end
                        end
                    end
                end
            end
        end

        # Shake V side: remove worst defect node, forbid for 1 round
        if max_v_def > 0
            worst_v_idx = argmax(def_v)
            worst_v = best_V[worst_v_idx]
            V_shaken = filter(!=(worst_v), best_V)
            if !isempty(V_shaken)
                U_temp = best_response_u_dense(fg, V_shaken, k, θ)
                if U_temp !== nothing
                    V_temp = best_response_v_dense(fg, U_temp, k, θ; forbidden_V=Set([worst_v]))
                    if V_temp !== nothing
                        alt_res = alternate_recruitment_dense(fg, U_temp, k, θ)
                        if alt_res !== nothing
                            new_U, new_V, new_edges = alt_res
                            if new_edges > best_edges
                                best_U = new_U
                                best_V = new_V
                                best_edges = new_edges
                                improved = true
                            end
                        end
                    end
                end
            end
        end

        improved || break
    end

    return best_U, best_V, best_edges
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Top-Level Solver
# ─────────────────────────────────────────────────────────────────────────────

"""
    tug_of_war_solve(fg::FrozenBipartite, k::Int, θ::Int;
                      seed_source::Symbol=:sc,
                      num_seeds::Int=10,
                      max_shakes::Int=3,
                      seed::Union{Nothing,Int}=nothing) -> SubGraph

Solve k-MDB using Tug-of-War alternation.
`seed_source` may be `:sc` (S_C scores), `:degree` (degree product), or `:random`.
"""
function tug_of_war_solve(fg::FrozenBipartite, k::Int, θ::Int;
                          seed_source::Symbol=:sc,
                          num_seeds::Int=10,
                          max_shakes::Int=3,
                          seed::Union{Nothing,Int}=nothing)::SubGraph
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    nE = length(fg.v_adj)

    (nU < θ || nV < θ || nE == 0) && return SubGraph(Set(), Set())

    rng = seed === nothing ? MersenneTwister() : MersenneTwister(seed)

    # Build CSR reverse map: slot -> ui
    edge_u = Vector{Int}(undef, nE)
    @inbounds for ui in 1:nU
        for slot in neighbor_range_u(fg, ui)
            edge_u[slot] = ui
        end
    end

    # Select seed edges according to seed_source
    seed_slots = Int[]
    if seed_source == :sc
        scores = neighborhood_scores(fg; θ=θ)
        perm = sortperm(scores, rev=true)
        seed_slots = perm[1:min(num_seeds, length(perm))]
    elseif seed_source == :degree
        deg_prod = [length(neighbor_range_u(fg, edge_u[slot])) * length(neighbor_range_v(fg, fg.v_adj[slot]))
                    for slot in 1:nE]
        perm = sortperm(deg_prod, rev=true)
        seed_slots = perm[1:min(num_seeds, length(perm))]
    elseif seed_source == :random
        perm = randperm(rng, nE)
        seed_slots = perm[1:min(num_seeds, length(perm))]
    else
        throw(ArgumentError("Unknown seed_source :$seed_source (must be :sc, :degree, or :random)"))
    end

    # Reusable scratch buffers for Jaccard similarity in core formation
    cnt_v = zeros(Int8, nV)
    dirty_v = Int[]
    sizehint!(dirty_v, 64)

    cnt_u = zeros(Int8, nU)
    dirty_u = Int[]
    sizehint!(dirty_u, 64)

    # Core sizes scale with θ: {2, floor(θ/2), θ}
    core_sizes = unique(filter(c -> c >= 2, [2, max(2, div(θ, 2)), θ]))

    best_U = Int[]
    best_V = Int[]
    best_edges = -1

    for slot in seed_slots
        u_seed = edge_u[slot]
        v_seed = fg.v_adj[slot]

        for c in core_sizes
            # 1. Crew on U side
            u_core = form_u_core(fg, u_seed, v_seed, c; cnt_v=cnt_v, dirty_v=dirty_v)
            res_u = alternate_recruitment_dense(fg, u_core, k, θ)
            if res_u !== nothing
                U_cand, V_cand, cand_edges = res_u
                if max_shakes > 0
                    U_cand, V_cand, cand_edges = shake_and_improve(fg, U_cand, V_cand, cand_edges, k, θ; max_shakes=max_shakes)
                end
                if cand_edges > best_edges
                    best_U = U_cand
                    best_V = V_cand
                    best_edges = cand_edges
                end
            end

            # 2. Crew on V side
            v_core = form_v_core(fg, u_seed, v_seed, c; cnt_u=cnt_u, dirty_u=dirty_u)
            res_v = alternate_recruitment_v_first_dense(fg, v_core, k, θ)
            if res_v !== nothing
                U_cand, V_cand, cand_edges = res_v
                if max_shakes > 0
                    U_cand, V_cand, cand_edges = shake_and_improve(fg, U_cand, V_cand, cand_edges, k, θ; max_shakes=max_shakes)
                end
                if cand_edges > best_edges
                    best_U = U_cand
                    best_V = V_cand
                    best_edges = cand_edges
                end
            end
        end
    end

    if best_edges < 0 || length(best_U) < θ || length(best_V) < θ
        return SubGraph(Set(), Set())
    end

    U_orig = Set([fg.u_ids[ui] for ui in best_U])
    V_orig = Set([fg.v_ids[vi] for vi in best_V])
    return SubGraph(U_orig, V_orig)
end
