include(joinpath(@__DIR__, "..", "graph.jl"))
include(joinpath(@__DIR__, "..", "opponent.jl"))  # wherever upper_bound / nondegree_in_subgraph live

using Random

function parse_seed()
    for arg in ARGS
        if startswith(arg, "--seed=")
            return parse(UInt64, split(arg, "=", limit=2)[2])
        end
    end
    return UInt64(time_ns())
end

const SEED = parse_seed()
Random.seed!(SEED)

# =============================================================================
# REFERENCE 1: direct translation of Lemma 5 (vertex bounds) + Lemma 6 (edge
# bound), computed independently of the two-pointer code under test.
# =============================================================================
function brute_upper_bound(S::SubGraph, C::SubGraph, g::FrozenBipartite, k::Int, S_missing::Int)
    budget = k - S_missing
    @assert budget >= 0 "S already violates the k-defective constraint"

    Ulist = collect(C.U)
    Vlist = collect(C.V)

    dU = [nondegree_in_subgraph(g, true, u, S) for u in Ulist]
    dV = [nondegree_in_subgraph(g, false, v, S) for v in Vlist]

    sort!(dU); sort!(dV)   # ascending, per the lemma's ordering

    prefixU = cumsum(vcat(0, dU))   # prefixU[i+1] = sum of first i (sorted) values
    prefixV = cumsum(vcat(0, dV))

    # --- independent vertex bounds (Lemma 5) ---
    i_max = 0
    for i in 1:length(dU)
        prefixU[i+1] <= budget ? (i_max = i) : break
    end
    j_max = 0
    for j in 1:length(dV)
        prefixV[j+1] <= budget ? (j_max = j) : break
    end

    # --- joint edge bound (Lemma 6): for each feasible i, take the best j ---
    best_e = length(S.U) * length(S.V) - S_missing   # i = j = 0 baseline
    for i in 0:length(dU)
        du_i = prefixU[i+1]
        du_i > budget && break   # prefixU is nondecreasing, nothing feasible past here
        remaining = budget - du_i
        j = 0
        for jj in 1:length(dV)
            prefixV[jj+1] <= remaining ? (j = jj) : break
        end
        e = (length(S.U) + i) * (length(S.V) + j) - S_missing - du_i - prefixV[j+1]
        best_e = max(best_e, e)
    end

    return length(S.U) + i_max, length(S.V) + j_max, best_e
end

# =============================================================================
# REFERENCE 2: brute-force enumeration of every actual completion of (S, C),
# to check that upper_bound is never violated by a real k-defective biclique.
# Only feasible for small |C|, hence the cap below.
# =============================================================================
function true_max_extension(S::SubGraph, C::SubGraph, g::FrozenBipartite, k::Int)
    Ulist = collect(C.U); Vlist = collect(C.V)
    nU, nV = length(Ulist), length(Vlist)
    best_edges, best_u, best_v = -1, 0, 0
    for umask in 0:(2^nU - 1), vmask in 0:(2^nV - 1)
        addU = Set(Ulist[i] for i in 1:nU if (umask >> (i-1)) & 1 == 1)
        addV = Set(Vlist[i] for i in 1:nV if (vmask >> (i-1)) & 1 == 1)
        D = SubGraph(union(S.U, addU), union(S.V, addV))
        if Subgraph.missing_edges(g, D) <= k
            e = Subgraph.edge_count(g, D)
            e > best_edges && (best_edges = e; best_u = length(D.U); best_v = length(D.V))
        end
    end
    return best_u, best_v, best_edges
end

# =============================================================================
# Random (S, C, k) instance generator: S must itself be a feasible partial
# k-defective biclique; C is the (possibly trimmed) remainder.
# =============================================================================
function random_instance(; nU_range=4:6, nV_range=4:6, edge_prob=0.5, k_range=0:3, max_C=12)
    nU, nV = rand(nU_range), rand(nV_range)
    edges = Set{Tuple{Int,Int}}()
    for u in 1:nU, v in 1:nV
        rand() < edge_prob && push!(edges, (u, v))
    end
    g = build_frozen(edges, nU, nV)
    k = rand(k_range)

    S = SubGraph(Set{Int}(), Set{Int}())
    for _ in 1:rand(0:min(nU, 2)), u in [rand(1:nU)]
        cand = SubGraph(union(S.U, Set([u])), S.V)
        Subgraph.missing_edges(g, cand) <= k && (S = cand)
    end
    for _ in 1:rand(0:min(nV, 2)), v in [rand(1:nV)]
        cand = SubGraph(S.U, union(S.V, Set([v])))
        Subgraph.missing_edges(g, cand) <= k && (S = cand)
    end

    remU = setdiff(Set(1:nU), S.U)
    remV = setdiff(Set(1:nV), S.V)
    # optionally trim C so the brute-force enumeration stays cheap
    while length(remU) + length(remV) > max_C
        if length(remU) >= length(remV) && !isempty(remU)
            delete!(remU, rand(collect(remU)))
        elseif !isempty(remV)
            delete!(remV, rand(collect(remV)))
        end
    end
    C = SubGraph(remU, remV)

    return g, S, C, k
end

# =============================================================================
# MAIN: run N trials, checking exact-match and soundness
# =============================================================================
N = 500
for trial in 1:N
    g, S, C, k = random_instance()
    S_missing = Subgraph.missing_edges(g, S)

    fast = upper_bound(S, C, g, k, S_missing)
    ref  = brute_upper_bound(S, C, g, k, S_missing)

    if fast != ref
        println("MISMATCH vs formula on trial $trial (--seed=$SEED)")
        println("  S=$S  C=$C  k=$k  S_missing=$S_missing")
        println("  upper_bound       -> $fast")
        println("  brute_upper_bound -> $ref")
        break
    end

    tu, tv, te = true_max_extension(S, C, g, k)
    bu, bv, be = fast
    if tu > bu || tv > bv || te > be
        println("UNSOUND on trial $trial (--seed=$SEED): bound violated by a real completion")
        println("  S=$S  C=$C  k=$k")
        println("  upper_bound claims  (U<=$bu, V<=$bv, E<=$be)")
        println("  actual best found    (U=$tu,  V=$tv,  E=$te)")
        break
    end
end
println("Done. $N/$N trials passed. Seed used: --seed=$SEED")