#=
Oracle ACO: search only the left side. The right side is an exact knapsack.

For a fixed left set L, a right vertex v costs c_L(v) = |L| − |N(v) ∩ L|
missing edges and contributes |L| − c_L(v) real edges. Vertices with cost
above k can never enter the right side, and cheaper vertices are always
worth more, so the maximum-edge right side is the greedy packing of costs
0..k. That score is f(L).

Adding or removing one left vertex changes costs only through its
neighborhood, so the cost histogram is updated in O(deg(u) + |L|) instead of
rebuilding it. Ants use that oracle three ways:

  1. discrepancy = 0, decompose = false
     Stochastic one-side construction. Each step samples a left vertex with
     probability proportional to pheromone × f(L ∪ {u}).
  2. discrepancy = D > 0
     Limited discrepancy search from a pheromone-sampled butterfly (or
     K_{3,3}). The best-ranked extension is free; each worse-ranked choice
     spends one discrepancy. Past a local maximum, continuing downhill also
     spends a discrepancy.
  3. decompose = true
     One subproblem per seed vertex on the search side, in degree order.
     The subproblem is the closed 3-hop of that seed (every θ-feasible
     k-defective biclique through the seed lives there when θ > k), with
     earlier seeds removed. Subproblems share the incumbent edge count, so
     a strong early seed prunes the rest.

Pheromone sits on left vertices and on unordered pairs from the best L.
=#

using Base.Threads

const __ORACLE_ACO_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__FITNESS_JL__) || include("fitness.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__SPLIT_ACO_JL__) || include("split_aco.jl")

# Left-side search state. `hist[c+1]` is how many right vertices currently
# cost c; `degsum` is the sum of their full-graph degrees. Both are maintained
# incrementally. Logical bins are `0:nL` even though the arrays are longer.
mutable struct OracleState
    search_is_u::Bool
    nL::Int
    nR::Int
    nSame::Int
    inL::BitVector
    hits::Vector{Int}
    hist::Vector{Int}
    degsum::Vector{Int}
    members::Vector{Int}
    rdeg::Vector{Int}
    total_deg::Int
    buf_hist::Vector{Int}
    buf_deg::Vector{Int}
    scratch_hist::Vector{Int}
    scratch_deg::Vector{Int}
    mark_same::Vector{Int}
    mark_opp::Vector{Int}
    mark_gen::Int
    mark_gen_same::Int
end

struct OracleMove
    id::Int
    edges::Int
    nR::Int
    alive::Int
    des::Float64
end

mutable struct OracleIncumbent
    edges::Int
    feasible::Bool
    sg::SubGraph
    iter::Int
    time_s::Float64
    seed::Int
    lock::ReentrantLock
end

mutable struct OracleColonyBest
    edges::Int
    feasible::Bool
    L::Vector{Int}
end

mutable struct SidePheromone
    vertex::Vector{Float64}
    pairs::Dict{UInt64,Float64}
    base::Float64
end

struct OracleCtx
    k::Int
    θ::Int
    discrepancy::Int
    branch::Int
    seed_size::Int
    pool::Int
    max_nodes::Int
    amount::Float64
    map_u::Vector{Int}
    map_v::Vector{Int}
    t0::UInt64
end

function OracleState(fg::FrozenBipartite, search_is_u::Bool)
    nSame = search_is_u ? length(fg.u_ids) : length(fg.v_ids)
    nR = search_is_u ? length(fg.v_ids) : length(fg.u_ids)
    rdeg = Vector{Int}(undef, nR)
    total = 0
    if search_is_u
        for v in 1:nR
            d = length(neighbor_range_v(fg, v))
            rdeg[v] = d
            total += d
        end
    else
        for u in 1:nR
            d = length(neighbor_range_u(fg, u))
            rdeg[u] = d
            total += d
        end
    end
    cap = nSame + 2
    hist = zeros(Int, cap)
    degsum = zeros(Int, cap)
    nR > 0 && (hist[1] = nR)
    degsum[1] = total
    return OracleState(
        search_is_u, 0, nR, nSame, falses(nSame), zeros(Int, nR),
        hist, degsum, Int[], rdeg, total,
        zeros(Int, cap), zeros(Int, cap), zeros(Int, cap), zeros(Int, cap),
        zeros(Int, max(nSame, 0)), zeros(Int, max(nR, 0)), 0, 0)
end

function reset_oracle!(st::OracleState)
    st.nL = 0
    fill!(st.inL, false)
    fill!(st.hits, 0)
    empty!(st.members)
    if st.nR > 0
        st.hist[1] = st.nR
    end
    st.degsum[1] = st.total_deg
    return st
end

@inline function _side_degree(fg::FrozenBipartite, search_is_u::Bool, u::Int)
    return search_is_u ? length(neighbor_range_u(fg, u)) : length(neighbor_range_v(fg, u))
end

"""
True when histogram, hit counts, and membership describe the same L.

Used by the unit tests. A vertex's cost is `nL − hits[v]`.
"""
function oracle_state_consistent(st::OracleState)
    nL = st.nL
    0 <= nL <= st.nSame || return false
    length(st.members) == nL || return false
    seen = falses(st.nSame)
    for u in st.members
        1 <= u <= st.nSame || return false
        seen[u] && return false
        seen[u] = true
        st.inL[u] || return false
    end
    for u in 1:st.nSame
        st.inL[u] == seen[u] || return false
    end
    recount = zeros(Int, nL + 1)
    deg = zeros(Int, nL + 1)
    for v in 1:st.nR
        hits = st.hits[v]
        0 <= hits <= nL || return false
        c = nL - hits
        recount[c + 1] += 1
        deg[c + 1] += st.rdeg[v]
    end
    for c in 0:nL
        st.hist[c + 1] == recount[c + 1] || return false
        st.degsum[c + 1] == deg[c + 1] || return false
    end
    return true
end

# Adding u raises |L| by 1. Neighbors of u keep their cost (hits rises with
# |L|); every other right vertex's cost rises by 1. `nh[c+1]` counts
# neighbors whose old cost was c, which is enough to shift the histogram.
function add_left!(st::OracleState, fg::FrozenBipartite, u::Int)
    st.inL[u] && error("oracle ACO added a left vertex that is already in L")
    nL = st.nL
    len = nL + 1
    nh = st.buf_hist
    nd = st.buf_deg
    for i in 1:len
        nh[i] = 0
        nd[i] = 0
    end
    hits = st.hits
    rdeg = st.rdeg
    if st.search_is_u
        for t in neighbor_range_u(fg, u)
            v = fg.v_adj[t]
            c = nL - hits[v]
            nh[c + 1] += 1
            nd[c + 1] += rdeg[v]
            hits[v] += 1
        end
    else
        for t in neighbor_range_v(fg, u)
            v = fg.u_adj[t]
            c = nL - hits[v]
            nh[c + 1] += 1
            nd[c + 1] += rdeg[v]
            hits[v] += 1
        end
    end
    newlen = nL + 2
    sh = st.scratch_hist
    sd = st.scratch_deg
    sh[1] = nh[1]
    sd[1] = nd[1]
    for c in 1:(nL + 1)
        stay_h = c <= nL ? nh[c + 1] : 0
        stay_d = c <= nL ? nd[c + 1] : 0
        sh[c + 1] = stay_h + st.hist[c] - nh[c]
        sd[c + 1] = stay_d + st.degsum[c] - nd[c]
    end
    for i in 1:newlen
        st.hist[i] = sh[i]
        st.degsum[i] = sd[i]
    end
    st.nL = nL + 1
    st.inL[u] = true
    push!(st.members, u)
    return st
end

function remove_left!(st::OracleState, fg::FrozenBipartite)
    st.nL == 0 && error("oracle ACO removed a vertex from an empty left side")
    u = pop!(st.members)
    nL = st.nL
    len = nL + 1
    nh = st.buf_hist
    nd = st.buf_deg
    for i in 1:len
        nh[i] = 0
        nd[i] = 0
    end
    hits = st.hits
    rdeg = st.rdeg
    if st.search_is_u
        for t in neighbor_range_u(fg, u)
            v = fg.v_adj[t]
            c = nL - hits[v]
            nh[c + 1] += 1
            nd[c + 1] += rdeg[v]
            hits[v] -= 1
        end
    else
        for t in neighbor_range_v(fg, u)
            v = fg.u_adj[t]
            c = nL - hits[v]
            nh[c + 1] += 1
            nd[c + 1] += rdeg[v]
            hits[v] -= 1
        end
    end
    # Non-neighbors drop one cost; neighbors stay. New bins are costs 0..nL-1.
    newlen = nL
    sh = st.scratch_hist
    sd = st.scratch_deg
    for c in 0:(nL - 1)
        sh[c + 1] = nh[c + 1] + st.hist[c + 2] - nh[c + 2]
        sd[c + 1] = nd[c + 1] + st.degsum[c + 2] - nd[c + 2]
    end
    for i in 1:newlen
        st.hist[i] = sh[i]
        st.degsum[i] = sd[i]
    end
    st.nL = nL - 1
    st.inL[u] = false
    return u
end

"""Edges and right-side size of the exact packing of `st` under budget `k`."""
function oracle_score(st::OracleState, k::Int)
    nL = st.nL
    nL == 0 && return 0, 0
    budget = k
    edges = 0
    nsel = 0
    maxc = min(k, nL - 1)
    for c in 0:maxc
        avail = st.hist[c + 1]
        avail <= 0 && continue
        if c > 0 && budget < c
            break
        end
        take = c == 0 ? avail : min(avail, budget ÷ c)
        take <= 0 && continue
        edges += take * (nL - c)
        nsel += take
        budget -= take * c
        budget == 0 && break
    end
    return edges, nsel
end

"""Right vertices with cost ≤ k. Adding to L only raises costs, so this shrinks."""
function oracle_alive(st::OracleState, k::Int)
    s = 0
    last = min(k, st.nL)
    for c in 0:last
        s += st.hist[c + 1]
    end
    return s
end

"""
Upper bound on the edge count of any extension of L.

A vertex with cost > k stays ineligible forever. Every edge of an extension
touches some still-eligible right vertex, and that vertex has at most
`rdeg[v]` edges.
"""
function oracle_upper_bound(st::OracleState, k::Int)
    ub = 0
    last = min(k, st.nL)
    for c in 0:last
        ub += st.degsum[c + 1]
    end
    return ub
end

function oracle_solution(fg::FrozenBipartite, st::OracleState, k::Int)
    nL = st.nL
    nL == 0 && return SubGraph()
    buckets = [Int[] for _ in 0:k]
    for v in 1:st.nR
        c = nL - st.hits[v]
        if 0 <= c <= k && c < nL
            push!(buckets[c + 1], v)
        end
    end
    budget = k
    right = Int[]
    for c in 0:k
        avail = buckets[c + 1]
        if c > 0 && budget < c
            break
        end
        take = c == 0 ? length(avail) : min(length(avail), budget ÷ c)
        take <= 0 && continue
        for i in 1:take
            push!(right, avail[i])
        end
        budget -= c * take
        budget == 0 && break
    end
    left = st.members
    if st.search_is_u
        return SubGraph(Set{Int}(left), Set{Int}(right))
    else
        return SubGraph(Set{Int}(right), Set{Int}(left))
    end
end

function alive_touch(st::OracleState, fg::FrozenBipartite, u::Int, k::Int)
    nL = st.nL
    hits = st.hits
    touch = 0
    if st.search_is_u
        for t in neighbor_range_u(fg, u)
            (nL - hits[fg.v_adj[t]]) <= k && (touch += 1)
        end
    else
        for t in neighbor_range_v(fg, u)
            (nL - hits[fg.u_adj[t]]) <= k && (touch += 1)
        end
    end
    return touch
end

@inline function _pair_key(a::Int, b::Int)
    lo, hi = a < b ? (a, b) : (b, a)
    return (UInt64(lo) << 32) | UInt64(hi)
end

function _pair_mean(ph::SidePheromone, members::Vector{Int}, u::Int, n::Int)
    n <= 0 && return ph.base
    s = 0.0
    for i in 1:n
        s += ph.base + get(ph.pairs, _pair_key(members[i], u), 0.0)
    end
    return s / n
end

function _desirability(ph::SidePheromone, st::OracleState, u::Int, edges::Int)
    # `u` is already the last member; pairs are with the previous left vertices.
    τ = ph.vertex[u]
    pair = _pair_mean(ph, st.members, u, st.nL - 1)
    return τ * pair * max(edges, 0)
end

function score_moves(st::OracleState, fg::FrozenBipartite, ph::SidePheromone,
    k::Int, θ::Int, pool::Int)
    n = st.nSame
    ids = Int[]
    keys = Float64[]
    for u in 1:n
        st.inL[u] && continue
        touch = alive_touch(st, fg, u, k)
        touch == 0 && continue
        push!(ids, u)
        push!(keys, ph.vertex[u] * (1 + touch))
    end
    # No neighbor of the live right side. Still try a left vertex when |L| < θ,
    # because a zero-touch add can finish the left side without killing R.
    if isempty(ids) && st.nL < θ
        for u in 1:n
            st.inL[u] && continue
            push!(ids, u)
            push!(keys, ph.vertex[u])
        end
    end
    moves = OracleMove[]
    isempty(ids) && return moves
    chosen = length(ids) > pool ? partialsortperm(keys, 1:pool; rev=true) : eachindex(ids)
    sizehint!(moves, length(chosen))
    for i in chosen
        u = ids[i]
        add_left!(st, fg, u)
        edges, nR = oracle_score(st, k)
        alive = oracle_alive(st, k)
        des = _desirability(ph, st, u, edges)
        remove_left!(st, fg)
        push!(moves, OracleMove(u, edges, nR, alive, des))
    end
    return moves
end

"""
Moves that keep a θ-sized right side possible.

Once L is θ-feasible, only strict edge improvements are kept. With discrepancy
budget left and no improvement available, the ranked downhill steps are kept
and `downhill` is true (taking one spends a discrepancy).
"""
function legal_moves(st::OracleState, fg::FrozenBipartite, ph::SidePheromone,
    ctx::OracleCtx, Dleft::Int)
    raw = score_moves(st, fg, ph, ctx.k, ctx.θ, ctx.pool)
    edges, nR = oracle_score(st, ctx.k)
    feasible = st.nL >= ctx.θ && nR >= ctx.θ
    improving = OracleMove[]
    viable = OracleMove[]
    for m in raw
        m.alive < ctx.θ && continue
        push!(viable, m)
        m.edges > edges && push!(improving, m)
    end
    downhill = false
    chosen = viable
    if feasible
        if !isempty(improving)
            chosen = improving
        elseif Dleft <= 0
            chosen = OracleMove[]
        else
            downhill = true
        end
    end
    sort!(chosen; by=m -> m.des, rev=true)
    return chosen, downhill
end

function _note!(inc::OracleIncumbent, colony::OracleColonyBest, st::OracleState,
    fg::FrozenBipartite, ctx::OracleCtx, iter::Int, seed_index::Int)
    edges, nR = oracle_score(st, ctx.k)
    (st.nL >= ctx.θ && nR >= ctx.θ) || return
    lock(inc.lock) do
        lose_global = inc.feasible && edges <= inc.edges
        lose_local = colony.feasible && edges <= colony.edges
        if lose_global && lose_local
            return
        end
        if !lose_local
            colony.feasible = true
            colony.edges = edges
            colony.L = copy(st.members)
        end
        if !lose_global
            sg = oracle_solution(fg, st, ctx.k)
            U = Set{Int}(ctx.map_u[u] for u in sg.U)
            V = Set{Int}(ctx.map_v[v] for v in sg.V)
            inc.feasible = true
            inc.edges = edges
            inc.sg = SubGraph(U, V)
            inc.iter = iter
            inc.seed = seed_index
            inc.time_s = (time_ns() - ctx.t0) / 1e9
        end
    end
    return
end

function _bump_gen!(st::OracleState, which::Symbol)
    if which === :same
        st.mark_gen_same += 1
        if st.mark_gen_same >= typemax(Int) - 8
            fill!(st.mark_same, 0)
            st.mark_gen_same = 1
        end
        return st.mark_gen_same
    else
        st.mark_gen += 1
        if st.mark_gen >= typemax(Int) - 8
            fill!(st.mark_opp, 0)
            st.mark_gen = 1
        end
        return st.mark_gen
    end
end

function _two_hop(st::OracleState, fg::FrozenBipartite, roots::Vector{Int})
    gen = _bump_gen!(st, :same)
    for u in roots
        st.mark_same[u] = gen
    end
    out = Int[]
    if st.search_is_u
        for u in roots
            for t in neighbor_range_u(fg, u)
                v = fg.v_adj[t]
                for s in neighbor_range_v(fg, v)
                    u2 = fg.u_adj[s]
                    if st.mark_same[u2] != gen
                        st.mark_same[u2] = gen
                        push!(out, u2)
                    end
                end
            end
        end
    else
        for v in roots
            for t in neighbor_range_v(fg, v)
                u = fg.u_adj[t]
                for s in neighbor_range_u(fg, u)
                    v2 = fg.v_adj[s]
                    if st.mark_same[v2] != gen
                        st.mark_same[v2] = gen
                        push!(out, v2)
                    end
                end
            end
        end
    end
    return out
end

"""How many right vertices are adjacent to every id in `start` and to `candidate`."""
function _common_with_all(st::OracleState, fg::FrozenBipartite, start::Vector{Int}, candidate::Int)
    n = length(start)
    n == 0 && return _side_degree(fg, st.search_is_u, candidate)
    base = st.mark_gen
    if base > typemax(Int) - n - 4
        fill!(st.mark_opp, 0)
        base = 0
    end
    depth = 0
    for u in start
        depth += 1
        prev = base + depth - 1
        cur = base + depth
        if st.search_is_u
            for t in neighbor_range_u(fg, u)
                v = fg.v_adj[t]
                if depth == 1 || st.mark_opp[v] == prev
                    st.mark_opp[v] = cur
                end
            end
        else
            for t in neighbor_range_v(fg, u)
                v = fg.u_adj[t]
                if depth == 1 || st.mark_opp[v] == prev
                    st.mark_opp[v] = cur
                end
            end
        end
    end
    need = base + n
    cnt = 0
    if st.search_is_u
        for t in neighbor_range_u(fg, candidate)
            st.mark_opp[fg.v_adj[t]] == need && (cnt += 1)
        end
    else
        for t in neighbor_range_v(fg, candidate)
            st.mark_opp[fg.u_adj[t]] == need && (cnt += 1)
        end
    end
    st.mark_gen = base + n
    return cnt
end

function sample_left_seed(st::OracleState, fg::FrozenBipartite, ph::SidePheromone,
    seed_size::Int, forced::Union{Nothing,Int})
    start = Int[]
    if forced !== nothing
        push!(start, forced)
    else
        ids = Int[]
        for u in 1:st.nSame
            _side_degree(fg, st.search_is_u, u) > 0 && push!(ids, u)
        end
        isempty(ids) && return start
        u1 = linear_sample_one(u -> ph.vertex[u] * (1 + _side_degree(fg, st.search_is_u, u)), ids)
        push!(start, u1)
    end
    target = max(seed_size, 1)
    while length(start) < target
        partners = _two_hop(st, fg, start)
        isempty(partners) && break
        legal = Int[]
        weight = Float64[]
        for u in partners
            common = _common_with_all(st, fg, start, u)
            common >= target || continue
            push!(legal, u)
            push!(weight, ph.vertex[u] * common)
        end
        isempty(legal) && break
        idx = linear_sample_one(i -> weight[i], collect(1:length(legal)))
        push!(start, legal[idx])
    end
    return start
end

function evaporate_side!(ph::SidePheromone, retention::Float64)
    ph.vertex .*= retention
    ph.base *= retention
    for (key, val) in ph.pairs
        ph.pairs[key] = val * retention
    end
    return ph
end

function deposit_left!(ph::SidePheromone, members::Vector{Int}, amount::Float64)
    amount == 0 && return ph
    for u in members
        ph.vertex[u] += amount
    end
    n = length(members)
    for i in 1:n
        ui = members[i]
        for j in (i + 1):n
            key = _pair_key(ui, members[j])
            ph.pairs[key] = get(ph.pairs, key, 0.0) + amount
        end
    end
    return ph
end

function greedy_walk!(st::OracleState, fg::FrozenBipartite, ph::SidePheromone,
    inc::OracleIncumbent, colony::OracleColonyBest, ctx::OracleCtx, iter::Int,
    seed_index::Int, nodes::Ref{Int})
    steps = max(st.nSame - st.nL, 0)
    for _ in 1:steps
        nodes[] += 1
        _note!(inc, colony, st, fg, ctx, iter, seed_index)
        nodes[] >= ctx.max_nodes && return
        if oracle_alive(st, ctx.k) < ctx.θ
            return
        end
        if inc.feasible && oracle_upper_bound(st, ctx.k) <= inc.edges
            return
        end
        chosen, _ = legal_moves(st, fg, ph, ctx, 0)
        isempty(chosen) && return
        pick = linear_sample_one(m -> m.des, chosen)
        add_left!(st, fg, pick.id)
    end
    _note!(inc, colony, st, fg, ctx, iter, seed_index)
    return
end

function lds!(st::OracleState, fg::FrozenBipartite, ph::SidePheromone,
    inc::OracleIncumbent, colony::OracleColonyBest, ctx::OracleCtx, iter::Int,
    seed_index::Int, Dleft::Int, nodes::Ref{Int})
    nodes[] += 1
    _note!(inc, colony, st, fg, ctx, iter, seed_index)
    nodes[] >= ctx.max_nodes && return
    if oracle_alive(st, ctx.k) < ctx.θ
        return
    end
    if inc.feasible && oracle_upper_bound(st, ctx.k) <= inc.edges
        return
    end
    chosen, downhill = legal_moves(st, fg, ph, ctx, Dleft)
    isempty(chosen) && return
    ntry = min(length(chosen), ctx.branch)
    for rank in 1:ntry
        extra = (downhill ? 1 : 0) + (rank == 1 ? 0 : 1)
        extra > Dleft && continue
        add_left!(st, fg, chosen[rank].id)
        lds!(st, fg, ph, inc, colony, ctx, iter, seed_index, Dleft - extra, nodes)
        remove_left!(st, fg)
        nodes[] >= ctx.max_nodes && return
    end
    return
end

function run_one_ant!(st::OracleState, fg::FrozenBipartite, ph::SidePheromone,
    inc::OracleIncumbent, colony::OracleColonyBest, ctx::OracleCtx,
    forced::Union{Nothing,Int}, iter::Int, seed_index::Int)
    reset_oracle!(st)
    seed = sample_left_seed(st, fg, ph, ctx.seed_size, forced)
    for u in seed
        st.inL[u] && continue
        add_left!(st, fg, u)
    end
    nodes = Ref(0)
    if ctx.discrepancy <= 0
        greedy_walk!(st, fg, ph, inc, colony, ctx, iter, seed_index, nodes)
    else
        lds!(st, fg, ph, inc, colony, ctx, iter, seed_index, ctx.discrepancy, nodes)
    end
    return nodes[]
end

function run_oracle_colony!(fg::FrozenBipartite, search_is_u::Bool, ph::SidePheromone,
    inc::OracleIncumbent, ctx::OracleCtx, forced::Union{Nothing,Int}, ants::Int,
    iters::Int, retention::Float64, parallelize::Bool, seed_index::Int)
    n_same = search_is_u ? length(fg.u_ids) : length(fg.v_ids)
    n_same == 0 && return 0
    states = [OracleState(fg, search_is_u) for _ in 1:ants]
    colony = OracleColonyBest(0, false, Int[])
    total = 0
    for iter in 1:iters
        if parallelize && nthreads() > 1 && ants > 1
            counts = zeros(Int, ants)
            @threads for i in 1:ants
                counts[i] = run_one_ant!(states[i], fg, ph, inc, colony, ctx,
                    forced, iter, seed_index)
            end
            total += sum(counts)
        else
            for i in 1:ants
                total += run_one_ant!(states[i], fg, ph, inc, colony, ctx,
                    forced, iter, seed_index)
            end
        end
        evaporate_side!(ph, retention)
        if colony.feasible
            deposit_left!(ph, colony.L, ctx.amount)
        end
    end
    return total
end

function _degree_order(fg::FrozenBipartite, search_is_u::Bool)
    n = search_is_u ? length(fg.u_ids) : length(fg.v_ids)
    order = collect(1:n)
    sort!(order; by=u -> (-_side_degree(fg, search_is_u, u), u))
    return order
end

function _seed_vertices(fg::FrozenBipartite, root::Int, search_is_u::Bool,
    earlier::Set{Int}, k::Int, θ::Int)
    if θ > k
        same, opp = three_hop_neighborhood(fg, root, search_is_u)
    else
        n_same = search_is_u ? length(fg.u_ids) : length(fg.v_ids)
        n_opp = search_is_u ? length(fg.v_ids) : length(fg.u_ids)
        same = collect(1:n_same)
        opp = collect(1:n_opp)
    end
    filter!(x -> x == root || !(x in earlier), same)
    return same, opp
end

function _induce_seed(fg::FrozenBipartite, same, opp, search_is_u::Bool, root::Int)
    same_ids = sort!(collect(Int, same))
    opp_ids = sort!(collect(Int, opp))
    if search_is_u
        keep_u, keep_v = same_ids, opp_ids
        local_root = findfirst(==(root), keep_u)
    else
        keep_u, keep_v = opp_ids, same_ids
        local_root = findfirst(==(root), keep_v)
    end
    if local_root === nothing || isempty(keep_u) || isempty(keep_v)
        return nothing
    end
    induced = induce_frozen(fg, keep_u, keep_v)
    return induced, keep_u, keep_v, local_root
end

function _fresh_pheromone(n::Int)
    return SidePheromone(ones(Float64, n), Dict{UInt64,Float64}(), 1.0)
end

function _empty_oracle_result(search_is_u::Bool, n_same::Int, n_opp::Int)
    return (;
        best=SubGraph(),
        search_is_u,
        n_same,
        n_opp,
        iterations_to_best=nothing,
        time_to_best_s=nothing,
        edges=0,
        seeds_run=0,
        seeds_pruned=0,
        best_seed=0,
        expansions=0,
    )
end

"""
    oracle_aco(g, pheromone, num_ants, num_iterations, evaporation, k, θ)

Search the smaller side. `discrepancy = 0` is the stochastic oracle colony.
`discrepancy > 0` adds limited backtracking. `decompose = true` splits the
graph into per-seed 3-hop subproblems that share one incumbent.
"""
function oracle_aco(g::BipartiteGraph, pheromone::Int, num_ants::Int,
    num_iterations::Int, evaporation::Float64, k::Int, θ::Int;
    discrepancy::Int=0, decompose::Bool=false, branch::Int=4, seed_size::Int=2,
    n_seeds::Int=20, pool::Int=64, max_nodes::Int=2000, parallelize::Bool=false,
    reduction::ReductionMode.T=ReductionMode.all_reductions, verbose::Bool=false)
    num_ants >= 1 || throw(ArgumentError("num_ants must be ≥ 1, got $num_ants"))
    num_iterations >= 1 || throw(ArgumentError("num_iterations must be ≥ 1, got $num_iterations"))
    0.0 < evaporation <= 1.0 || throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    discrepancy >= 0 || throw(ArgumentError("discrepancy must be ≥ 0, got $discrepancy"))
    branch >= 1 || throw(ArgumentError("branch must be ≥ 1, got $branch"))
    seed_size >= 1 || throw(ArgumentError("seed_size must be ≥ 1, got $seed_size"))
    n_seeds >= 0 || throw(ArgumentError("n_seeds must be ≥ 0, got $n_seeds"))
    pool >= 1 || throw(ArgumentError("pool must be ≥ 1, got $pool"))
    max_nodes >= 1 || throw(ArgumentError("max_nodes must be ≥ 1, got $max_nodes"))
    pheromone >= 0 || throw(ArgumentError("pheromone must be ≥ 0, got $pheromone"))

    apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    compact, remapping = compact_frozen(freeze(g))
    nU = length(compact.u_ids)
    nV = length(compact.v_ids)
    search_is_u = nU <= nV
    n_same = search_is_u ? nU : nV
    n_opp = search_is_u ? nV : nU
    if nU < θ || nV < θ
        verbose && println("Oracle ACO: reduced graph smaller than θ=$θ (|U|=$nU |V|=$nV)")
        return _empty_oracle_result(search_is_u, n_same, n_opp)
    end

    t0 = time_ns()
    inc = OracleIncumbent(0, false, SubGraph(), 0, 0.0, 0, ReentrantLock())
    expansions = 0
    seeds_run = 0
    seeds_pruned = 0

    function make_ctx(map_u, map_v)
        return OracleCtx(k, θ, discrepancy, branch, seed_size, pool, max_nodes,
            Float64(pheromone), map_u, map_v, t0)
    end

    if !decompose
        ph = _fresh_pheromone(n_same)
        ctx = make_ctx(compact.u_ids, compact.v_ids)
        verbose && println("Oracle ACO: search $(search_is_u ? "U" : "V") " *
                           "n_same=$n_same n_opp=$n_opp ants=$num_ants " *
                           "iterations=$num_iterations discrepancy=$discrepancy k=$k θ=$θ")
        expansions = run_oracle_colony!(compact, search_is_u, ph, inc, ctx, nothing,
            num_ants, num_iterations, evaporation, parallelize, 0)
        seeds_run = 1
    else
        order = _degree_order(compact, search_is_u)
        nrun = n_seeds == 0 ? length(order) : min(n_seeds, length(order))
        earlier = Set{Int}()
        verbose && println("Oracle ACO decompose: search $(search_is_u ? "U" : "V") " *
                           "seeds=$nrun/$n_same discrepancy=$discrepancy k=$k θ=$θ")
        for si in 1:nrun
            root = order[si]
            same, opp = _seed_vertices(compact, root, search_is_u, earlier, k, θ)
            push!(earlier, root)
            if length(same) < θ || length(opp) < θ
                seeds_pruned += 1
                continue
            end
            built = _induce_seed(compact, same, opp, search_is_u, root)
            if built === nothing
                seeds_pruned += 1
                continue
            end
            induced, keep_u, keep_v, local_root = built
            if inc.feasible && length(induced.v_adj) <= inc.edges
                seeds_pruned += 1
                continue
            end
            n_local = search_is_u ? length(induced.u_ids) : length(induced.v_ids)
            ph = _fresh_pheromone(n_local)
            ctx = make_ctx(keep_u, keep_v)
            seeds_run += 1
            verbose && println("  seed $si/$nrun root=$root |U|=$(length(keep_u)) " *
                               "|V|=$(length(keep_v)) |E|=$(length(induced.v_adj))")
            expansions += run_oracle_colony!(induced, search_is_u, ph, inc, ctx,
                local_root, num_ants, num_iterations, evaporation, parallelize, si)
        end
    end

    if !inc.feasible
        verbose && println("Oracle ACO: no θ-feasible solution")
        empty = _empty_oracle_result(search_is_u, n_same, n_opp)
        return (; empty..., seeds_run, seeds_pruned, expansions)
    end
    best = remap_subgraph(remapping, inc.sg)
    verbose && println("Oracle ACO best: |U|=$(length(best.U)) |V|=$(length(best.V)) " *
                       "edges=$(inc.edges) seed=$(inc.seed)")
    return (;
        best,
        search_is_u,
        n_same,
        n_opp,
        iterations_to_best=inc.iter,
        time_to_best_s=inc.time_s,
        edges=inc.edges,
        seeds_run,
        seeds_pruned,
        best_seed=inc.seed,
        expansions,
    )
end
