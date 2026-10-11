#=
Degeneracy clique heuristic on the super-reduced compatibility graph.

`super_reduce!` deletes vertices that cannot sit in a θ-feasible k-defective
biclique. The mapped unipartite graph is the compatibility projection of what
remains: two vertices on the smaller side are adjacent when they share at
least `θ − k` opposite neighbors. A clique in that graph is a mutually
compatible set on one side.

The search is the standard degeneracy recurrence. In a degeneracy ordering
v₁, …, vₙ, N⁺(v) is the neighbors of v that come later, and |N⁺(v)| ≤ d.
Every clique has a unique earliest vertex v with the rest inside N⁺(v), so

    ω(G) = 1 + max over v of ω(G[N⁺(v)])

and each subproblem has at most d vertices. This file takes the greedy
branch of that recurrence (a lower bound, a core prune, a local core, and a
coloring bound), then runs clique ACO only on the vertices of the clique it
found.

    julia src/clique-find.jl
    julia src/clique-find.jl konect-small/maayan-vidal konect-small/moreno_health
=#

const __CLIQUE_FIND_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__REDUCTION_JL__) || include("reduction.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__CLIQUE_ACO_JL__) || include("clique_aco.jl")

using Random

"""Drop self-loops and duplicate neighbors. `adj` is undirected and 1-based."""
function _clean_adj(adj::Vector{Vector{Int}})
    n = length(adj)
    out = [Int[] for _ in 1:n]
    for v in 1:n
        seen = Set{Int}()
        for u in adj[v]
            u == v && continue
            1 <= u <= n || continue
            push!(seen, u)
        end
        out[v] = sort!(collect(seen))
    end
    return out
end

"""
Batagelj–Zaversnik core peel.

`order` is the removal order (earliest first). `core[v]` is the core number,
snapshotted when `v` is peeled.
"""
function _peel(adj::Vector{Vector{Int}})
    n = length(adj)
    deg = [length(adj[i]) for i in 1:n]
    n == 0 && return Int[], Int[]
    maxdeg = maximum(deg)
    bin = zeros(Int, maxdeg + 1)
    for d in deg
        bin[d + 1] += 1
    end
    start = 1
    for d in 0:maxdeg
        num = bin[d + 1]
        bin[d + 1] = start
        start += num
    end
    vert = Vector{Int}(undef, n)
    pos = Vector{Int}(undef, n)
    for v in 1:n
        pos[v] = bin[deg[v] + 1]
        vert[pos[v]] = v
        bin[deg[v] + 1] += 1
    end
    for d in maxdeg:-1:1
        bin[d + 1] = bin[d]
    end
    bin[1] = 1
    core = zeros(Int, n)
    for i in 1:n
        v = vert[i]
        core[v] = deg[v]
        for u in adj[v]
            deg[u] > deg[v] || continue
            du = deg[u]
            pu = pos[u]
            pw = bin[du + 1]
            w = vert[pw]
            if u != w
                pos[u] = pw
                vert[pu] = w
                pos[w] = pu
                vert[pw] = u
            end
            bin[du + 1] += 1
            deg[u] -= 1
        end
    end
    return vert, core
end

"""Neighbors of each vertex that are later in `order` (the peel order)."""
function _later_neighbors(adj::Vector{Vector{Int}}, order::Vector{Int})
    n = length(adj)
    rank = zeros(Int, n)
    for (i, v) in enumerate(order)
        rank[v] = i
    end
    later = [Int[] for _ in 1:n]
    for v in 1:n
        rv = rank[v]
        for u in adj[v]
            rank[u] > rv && push!(later[v], u)
        end
    end
    return later
end

"""
    degeneracy_ordering(adj) -> (order, core, later, degeneracy)

`order` is earliest-peeled first. `later[v]` is N⁺(v). `degeneracy` is the
maximum core number, and |N⁺(v)| ≤ degeneracy for every v.
"""
function degeneracy_ordering(adj::Vector{Vector{Int}})
    clean = _clean_adj(adj)
    order, core = _peel(clean)
    later = _later_neighbors(clean, order)
    d = isempty(core) ? 0 : maximum(core)
    return order, core, later, d
end

"""Induced subgraph on `verts`, compactly renumbered to `1:length(verts)`."""
function _induced(adj::Vector{Vector{Int}}, verts::Vector{Int}, loc::Vector{Int})
    m = length(verts)
    for i in 1:m
        loc[verts[i]] = i
    end
    local_adj = [Int[] for _ in 1:m]
    for i in 1:m
        row = local_adj[i]
        for u in adj[verts[i]]
            j = loc[u]
            j != 0 && push!(row, j)
        end
    end
    for v in verts
        loc[v] = 0
    end
    return local_adj
end

"""Vertices that survive the min-degree peel. Indices are into `local_adj`."""
function _kcore_indices(local_adj::Vector{Vector{Int}}, min_deg::Int)
    m = length(local_adj)
    min_deg <= 0 && return collect(1:m)
    deg = [length(local_adj[i]) for i in 1:m]
    alive = trues(m)
    q = Int[]
    for i in 1:m
        deg[i] < min_deg || continue
        alive[i] = false
        push!(q, i)
    end
    head = 1
    while head <= length(q)
        i = q[head]
        head += 1
        for j in local_adj[i]
            alive[j] || continue
            deg[j] -= 1
            if deg[j] < min_deg
                alive[j] = false
                push!(q, j)
            end
        end
    end
    return [i for i in 1:m if alive[i]]
end

"""Renumber `local_adj` down to the subset `keep` (indices into `local_adj`)."""
function _compact(local_adj::Vector{Vector{Int}}, keep::Vector{Int})
    m = length(keep)
    mapto = zeros(Int, length(local_adj))
    for (neu, old) in enumerate(keep)
        mapto[old] = neu
    end
    out = [Int[] for _ in 1:m]
    for (neu, old) in enumerate(keep)
        row = out[neu]
        for j in local_adj[old]
            nj = mapto[j]
            nj != 0 && push!(row, nj)
        end
    end
    return out
end

"""
Smallest-last greedy coloring. The count is an upper bound on the clique
number of `local_adj`.
"""
function _color_count(local_adj::Vector{Vector{Int}})
    m = length(local_adj)
    m == 0 && return 0
    order, _ = _peel(local_adj)
    color = zeros(Int, m)
    seen = zeros(Int, m)
    stamp = 0
    ncolors = 0
    for idx in m:-1:1
        v = order[idx]
        stamp += 1
        for u in local_adj[v]
            c = color[u]
            c != 0 && (seen[c] = stamp)
        end
        c = 1
        while c <= m && seen[c] == stamp
            c += 1
        end
        color[v] = c
        ncolors = max(ncolors, c)
    end
    return ncolors
end

"""
Greedy clique: repeatedly take the pool vertex with the most neighbors in
the pool, then restrict the pool to that vertex's neighbors.
"""
function _greedy_local(local_adj::Vector{Vector{Int}})
    m = length(local_adj)
    m == 0 && return Int[]
    alive = trues(m)
    deg = [length(a) for a in local_adj]
    clique = Int[]
    while true
        best = 0
        bestd = -1
        for i in 1:m
            alive[i] || continue
            d = deg[i]
            if d > bestd
                bestd = d
                best = i
            end
        end
        best == 0 && break
        push!(clique, best)
        nxt = falses(m)
        for j in local_adj[best]
            alive[j] && (nxt[j] = true)
        end
        alive = nxt
        any(alive) || break
        for i in 1:m
            alive[i] || continue
            d = 0
            for j in local_adj[i]
                alive[j] && (d += 1)
            end
            deg[i] = d
        end
    end
    return clique
end

function _greedy_in(adj::Vector{Vector{Int}}, verts::Vector{Int}, loc::Vector{Int})
    isempty(verts) && return Int[]
    picked = _greedy_local(_induced(adj, verts, loc))
    return [verts[i] for i in picked]
end

"""
Local (LB − 1)-core, then a coloring prune. `nothing` when the neighborhood
cannot contain a clique of size `lb`. Otherwise the greedy clique there.
"""
function _filtered_greedy(adj::Vector{Vector{Int}}, verts::Vector{Int}, lb::Int, loc::Vector{Int})
    local_adj = _induced(adj, verts, loc)
    keep = _kcore_indices(local_adj, lb - 1)
    length(keep) < lb && return nothing
    sub = _compact(local_adj, keep)
    _color_count(sub) <= lb - 1 && return nothing
    surv = [verts[i] for i in keep]
    picked = _greedy_local(sub)
    return [surv[i] for i in picked]
end

"""
    max_clique_heuristic(adj) -> (vertices, degeneracy)

Largest clique found by the degeneracy greedy. `vertices` are 1-based indices
into `adj`. `degeneracy` is the maximum core number (so ω ≤ degeneracy + 1).
"""
function max_clique_heuristic(adj_in::Vector{Vector{Int}})
    adj = _clean_adj(adj_in)
    n = length(adj)
    n == 0 && return Int[], 0
    order, core = _peel(adj)
    later = _later_neighbors(adj, order)
    degeneracy = maximum(core)
    loc = zeros(Int, n)

    best = Int[]
    lb = 0
    for v in Iterators.reverse(order)
        # A clique inside N⁺(v) has size at most |N⁺(v)|, so this v cannot
        # beat the incumbent when that neighborhood is smaller than it.
        length(later[v]) < lb && continue
        cl = _greedy_in(adj, later[v], loc)
        if length(cl) + 1 > lb
            lb = length(cl) + 1
            best = [v; cl]
        end
    end

    # A clique of size lb + 1 needs every member in the lb-core.
    alive = [core[i] >= lb for i in 1:n]
    improved = true
    while improved
        improved = false
        for v in Iterators.reverse(order)
            alive[v] || continue
            nbrs = Int[]
            for u in later[v]
                alive[u] && push!(nbrs, u)
            end
            length(nbrs) < lb && continue
            cl = _filtered_greedy(adj, nbrs, lb, loc)
            cl === nothing && continue
            if length(cl) + 1 > lb
                lb = length(cl) + 1
                best = [v; cl]
                improved = true
                for u in 1:n
                    core[u] < lb && (alive[u] = false)
                end
                break
            end
        end
    end
    return best, degeneracy
end

"""True when every pair in `verts` is an edge of `adj`."""
function _is_clique(adj::Vector{Vector{Int}}, verts::Vector{Int})
    n = length(adj)
    mark = zeros(Int, n)
    stamp = 0
    for v in verts
        stamp += 1
        for u in adj[v]
            mark[u] = stamp
        end
        for u in verts
            u == v && continue
            mark[u] == stamp || return false
        end
    end
    return true
end

"""Keep `keep` on the searched side and drop vertices that no longer have a neighbor."""
function _restrict_to_side!(g::BipartiteGraph, search_is_u::Bool, keep::AbstractSet{Int})
    side = search_is_u ? g.adjU : g.adjV
    for id in collect(keys(side))
        id in keep || rem_node_structural!(g, search_is_u, id)
    end
    for is_u in (true, false)
        adj = is_u ? g.adjU : g.adjV
        for id in collect(keys(adj))
            isempty(adj[id]) && rem_node_structural!(g, is_u, id)
        end
    end
    return g
end

"""
    clique_find(g, k, θ; ants=20, iterations=20, ...) -> NamedTuple

Mutates `g`. Applies `super_reduce!`, builds the compatibility projection,
finds a large clique, deletes every other vertex on that side, and runs
`clique_aco` on what remains. The returned subgraph uses original vertex ids.
"""
function clique_find(g::BipartiteGraph, k::Int, θ::Int;
        ants::Int=20, iterations::Int=20, pheromone::Int=1,
        evaporation::Float64=0.95, eta_exp::Real=2.0, verbose::Bool=false)
    empty = (;
        solution=SubGraph(),
        clique_size=0,
        degeneracy=0,
        search_is_u=true,
        projection_n=0,
        projection_m=0,
        reduced_nU=0,
        reduced_nV=0,
        reduced_m=0,
        restricted_nU=0,
        restricted_nV=0,
        reduce_s=0.0,
        project_s=0.0,
        clique_s=0.0,
        aco_s=0.0,
    )

    t0 = time_ns()
    super_reduce!(g, k, θ)
    reduce_s = (time_ns() - t0) / 1e9
    reduced_nU = length(g.adjU)
    reduced_nV = length(g.adjV)
    reduced_m = _edge_count(g)

    t1 = time_ns()
    fg = freeze(g)
    cg = compatibility_graph(fg, k, θ)
    project_s = (time_ns() - t1) / 1e9
    n = length(cg.side_id)
    if n == 0
        verbose && println("Clique-find: compatibility graph is empty " *
                           "(|U|=$reduced_nU |V|=$reduced_nV)")
        return (; empty..., reduce_s, project_s, reduced_nU, reduced_nV, reduced_m,
            search_is_u=length(fg.u_ids) <= length(fg.v_ids))
    end

    t2 = time_ns()
    members, degeneracy = max_clique_heuristic(cg.adj)
    clique_s = (time_ns() - t2) / 1e9
    _is_clique(cg.adj, members) ||
        error("clique-find returned a vertex set that is not a clique")

    ids = Int[]
    sizehint!(ids, length(members))
    for i in members
        dense = cg.side_id[i]
        push!(ids, cg.search_is_u ? fg.u_ids[dense] : fg.v_ids[dense])
    end
    _restrict_to_side!(g, cg.search_is_u, Set(ids))
    restricted_nU = length(g.adjU)
    restricted_nV = length(g.adjV)
    verbose && println("Clique-find: search $(cg.search_is_u ? "U" : "V") " *
                       "projection_n=$n projection_m=$(cg.edges) degeneracy=$degeneracy " *
                       "clique=$(length(members)) restricted |U|=$restricted_nU |V|=$restricted_nV")

    t3 = time_ns()
    result = clique_aco(g, pheromone, ants, iterations, evaporation, k, θ, 1;
        reduction=ReductionMode.none, eta_exp=eta_exp, parallelize=false, verbose=verbose)
    aco_s = (time_ns() - t3) / 1e9
    return (;
        solution=result.best,
        clique_size=length(members),
        degeneracy,
        search_is_u=cg.search_is_u,
        projection_n=n,
        projection_m=cg.edges,
        reduced_nU,
        reduced_nV,
        reduced_m,
        restricted_nU,
        restricted_nV,
        reduce_s,
        project_s,
        clique_s,
        aco_s,
    )
end

"""Checks the peel and the greedy clique on a few hand-built graphs."""
function clique_find_selftest()
    function adj_of(edges, n)
        adj = [Int[] for _ in 1:n]
        for (u, v) in edges
            push!(adj[u], v)
            push!(adj[v], u)
        end
        return adj
    end

    k5 = [filter(!=(i), collect(1:5)) for i in 1:5]
    verts, d = max_clique_heuristic(k5)
    length(verts) == 5 || error("K5 clique size $(length(verts))")
    d == 4 || error("K5 degeneracy $d")
    _is_clique(_clean_adj(k5), verts) || error("K5 result is not a clique")
    _, core, later, _ = degeneracy_ordering(k5)
    all(==(4), core) || error("K5 cores $core")
    all(length(later[v]) <= 4 for v in 1:5) || error("K5 later-neighbor bound")

    path = adj_of([(1, 2), (2, 3)], 3)
    _, corep, laterp, dp = degeneracy_ordering(path)
    dp == 1 || error("path degeneracy $dp")
    all(==(1), corep) || error("path cores $corep")
    all(length(laterp[v]) <= 1 for v in 1:3) || error("path later-neighbor bound")
    pv, _ = max_clique_heuristic(path)
    length(pv) == 2 || error("path clique size $(length(pv))")

    tri = adj_of([(1, 2), (2, 3), (3, 1), (3, 4)], 4)
    tv, _ = max_clique_heuristic(tri)
    length(tv) == 3 || error("triangle clique size $(length(tv))")
    _is_clique(_clean_adj(tri), tv) || error("triangle result is not a clique")

    # K4 on {1,2,3,4}, triangle on {4,5,6}.
    edges = Tuple{Int,Int}[]
    for i in 1:4, j in (i + 1):4
        push!(edges, (i, j))
    end
    push!(edges, (4, 5), (5, 6), (4, 6))
    mixed = adj_of(edges, 6)
    mv, _ = max_clique_heuristic(mixed)
    length(mv) == 4 || error("mixed clique size $(length(mv))")
    _is_clique(_clean_adj(mixed), mv) || error("mixed result is not a clique")
    return true
end

# ── Comparison driver ────────────────────────────────────────────────────────

const CLIQUE_FIND_K = 2
const CLIQUE_FIND_THETA = 5
# Same budget as the every-ant arm in tests/quartile/aco.jl.
const CLIQUE_FIND_ANTS = 20
const CLIQUE_FIND_ITERS = 20
const CLIQUE_FIND_EVAP = 0.95
const CLIQUE_FIND_ETA = 2.0

# Graphs whose super-reduction keeps both sides. moreno, facebook, and boxes
# lose every vertex at k=2, θ=5, so the projection is empty.
const CLIQUE_FIND_GRAPHS = [
    "konect-small/maayan-vidal",
    "konect-small/moreno_health",
    "konect-small/dnc-corecipient",
]

function _biclique_score(fg::FrozenBipartite, sg::SubGraph, k::Int, θ::Int)
    nU = length(sg.U)
    nV = length(sg.V)
    edges = (nU == 0 || nV == 0) ? 0 : Subgraph.edge_count(fg, sg)
    missing = nU * nV - edges
    feasible = nU >= θ && nV >= θ && missing <= k
    return (; nU, nV, edges, missing, feasible)
end

function _run_theta(fg::FrozenBipartite, k::Int, θ::Int)
    t0 = time_ns()
    sg = theta_based_heuristic(fg, k, θ)
    return sg, (time_ns() - t0) / 1e9
end

function _run_vertex_aco(g::BipartiteGraph, k::Int, θ::Int)
    Random.seed!(1)
    t0 = time_ns()
    sols, _, _, _, _ = redirect_stdout(devnull) do
        aco(g, 1, CLIQUE_FIND_ANTS, CLIQUE_FIND_ITERS, CLIQUE_FIND_EVAP, k, θ, 1;
            parallelize=false,
            prefer_smaller_side=false,
            neighbor_scope_limit=true,
            elite_seed=false,
            best_only_pheromone=false,
            eta_exp=CLIQUE_FIND_ETA,
            reduction=ReductionMode.simple)
    end
    return sols[1], (time_ns() - t0) / 1e9
end

function _fmt_score(label, score, seconds)
    feas = score.feasible ? "yes" : "no"
    return "  $(rpad(label, 14)) |U|=$(lpad(string(score.nU), 5))  |V|=$(lpad(string(score.nV), 5))  " *
           "edges=$(lpad(string(score.edges), 6))  missing=$(lpad(string(score.missing), 4))  " *
           "θ-feas=$(rpad(feas, 3))  time=$(lpad(string(round(seconds; digits=3)), 8))s"
end

function _warmup_methods()
    g = BipartiteGraph{Nothing}()
    for u in 1:8, v in 1:8
        (u == 1 && v <= 2) && continue
        add_edge!(g, u, v, nothing)
    end
    theta_based_heuristic(freeze(g), CLIQUE_FIND_K, CLIQUE_FIND_THETA)
    redirect_stdout(devnull) do
        aco(deepcopy(g), 1, 2, 1, CLIQUE_FIND_EVAP, CLIQUE_FIND_K, CLIQUE_FIND_THETA, 1;
            parallelize=false, prefer_smaller_side=false, neighbor_scope_limit=true,
            elite_seed=false, reduction=ReductionMode.simple, eta_exp=CLIQUE_FIND_ETA)
    end
    clique_find(deepcopy(g), CLIQUE_FIND_K, CLIQUE_FIND_THETA;
        ants=2, iterations=1, verbose=false)
    return nothing
end

function compare_clique_find(graphs::Vector{String}=CLIQUE_FIND_GRAPHS)
    k = CLIQUE_FIND_K
    θ = CLIQUE_FIND_THETA
    data_root = joinpath(@__DIR__, "..", "data")
    println("k=$k  θ=$θ")
    println("ACO budget (both ACO arms): ants=$CLIQUE_FIND_ANTS  " *
            "iterations=$CLIQUE_FIND_ITERS  evaporation=$CLIQUE_FIND_EVAP  η^$CLIQUE_FIND_ETA")
    println("vertex ACO: reduction=simple  F_N  elite_seed=off  (quartile every-ant arm)")
    println("clique-find: super_reduce → degeneracy clique → clique ACO on that clique only")
    println("Times exclude a one-time JIT warmup.")
    _warmup_methods()
    println("warmup done")

    for key in graphs
        path = resolve_graph_path(key; data_root=data_root)
        isfile(path) || error("missing graph file for $key: $path")
        println("\n── $key ──")
        g, nedges = load_bipartite_graph(path)
        println("  |U|=$(length(g.adjU))  |V|=$(length(g.adjV))  |E|=$nedges")
        base = freeze(g)
        # The 8×8 warmup does not cover a large projection. Discard one real
        # pass so the timed call is not the first one on this graph.
        clique_find(deepcopy(g), k, θ; ants=CLIQUE_FIND_ANTS, iterations=CLIQUE_FIND_ITERS,
            evaporation=CLIQUE_FIND_EVAP, eta_exp=CLIQUE_FIND_ETA, verbose=false)
        _run_vertex_aco(deepcopy(g), k, θ)

        sg_theta, t_theta = _run_theta(base, k, θ)
        sc_theta = _biclique_score(base, sg_theta, k, θ)

        sg_aco, t_aco = _run_vertex_aco(deepcopy(g), k, θ)
        sc_aco = _biclique_score(base, sg_aco, k, θ)

        Random.seed!(1)
        found = clique_find(deepcopy(g), k, θ;
            ants=CLIQUE_FIND_ANTS, iterations=CLIQUE_FIND_ITERS,
            evaporation=CLIQUE_FIND_EVAP, eta_exp=CLIQUE_FIND_ETA, verbose=false)
        sc_cf = _biclique_score(base, found.solution, k, θ)
        t_cf = found.reduce_s + found.project_s + found.clique_s + found.aco_s
        side = found.search_is_u ? "U" : "V"
        println("  super-reduce |U|=$(found.reduced_nU) |V|=$(found.reduced_nV) |E|=$(found.reduced_m)")
        println("  projection side=$side n=$(found.projection_n) m=$(found.projection_m) " *
                "degeneracy=$(found.degeneracy) clique=$(found.clique_size)")
        println("  restricted |U|=$(found.restricted_nU) |V|=$(found.restricted_nV)")
        println("  clique-find time: reduce=$(round(found.reduce_s; digits=3))s  " *
                "project=$(round(found.project_s; digits=3))s  " *
                "clique=$(round(found.clique_s; digits=3))s  " *
                "clique_aco=$(round(found.aco_s; digits=3))s")
        println(_fmt_score("clique-find", sc_cf, t_cf))
        println(_fmt_score("theta", sc_theta, t_theta))
        println(_fmt_score("aco", sc_aco, t_aco))
    end
    return nothing
end

function _cli_graphs()
    keys = String[]
    for arg in ARGS
        startswith(arg, "-") && continue
        push!(keys, arg)
    end
    return isempty(keys) ? CLIQUE_FIND_GRAPHS : keys
end

if !isempty(PROGRAM_FILE) && abspath(PROGRAM_FILE) == @__FILE__
    clique_find_selftest()
    compare_clique_find(_cli_graphs())
end
