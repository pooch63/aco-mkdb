include(joinpath(@__DIR__, "..", "graph.jl"))
include(joinpath(@__DIR__, "..", "opponent.jl"))

# =============================================================================
# DRIVER: brute force + build the real graph + run branch_binary + compare
# =============================================================================

function find_optimum(edges, nU, nV, k, θ)
    function missing_edges_bf(Uset, Vset, edges)
        m = 0
        for u in Uset, v in Vset
            if !((u, v) in edges); m += 1; end
        end
        return m
    end

    best = (0, nothing, nothing)
    for umask in 0:(2^nU - 1), vmask in 0:(2^nV - 1)
        Uset = [u for u in 1:nU if (umask >> (u - 1)) & 1 == 1]
        Vset = [v for v in 1:nV if (vmask >> (v - 1)) & 1 == 1]
        if length(Uset) < θ || length(Vset) < θ; continue; end
        m = missing_edges_bf(Uset, Vset, edges)
        if m <= k
            e = length(Uset) * length(Vset) - m
            if e > best[1]
                best = (e, Uset, Vset)
            end
        end
    end
    return best
end

function build_frozen(edges, nU, nV)
    g = BipartiteGraph{Nothing}()
    for u in 1:nU
        add_u!(g, u)
    end
    for v in 1:nV
        add_v!(g, v)
    end
    for (u, v) in edges
        add_edge!(g, u, v, nothing)
    end
    return freeze(g)
end

edges = Set([
    (1,1),(1,2),(1,3),(1,4),
    (2,1),(2,2),(2,3),(2,4),
    (3,1),(3,2),(3,3),          # (3,4) intentionally missing
    (4,1),                       # noise
    (5,5),                       # noise
])

nU, nV = 5, 5
k, θ = 1, 3

println("="^70)
println("Brute force:")
bf_edges, bf_U, bf_V = find_optimum(edges, nU, nV, k, θ)
println("  edges=$bf_edges  U=$bf_U  V=$bf_V")
println("  (expected: edges=11, U=[1,2,3], V=[1,2,3,4])")

println("="^70)
println("branch_binary trace:")
fg = build_frozen(edges, nU, nV)

S0 = SubGraph(Set{Int}(), Set{Int}())
C0 = SubGraph(Set{Int}(1:nU), Set{Int}(1:nV))
D0 = SubGraph(Set{Int}(), Set{Int}())

D = branch_binary(S0, C0, fg, D0, k, θ)

println("="^70)
println("branch_binary result:")
println("  D.U=", sort(collect(D.U)), "  D.V=", sort(collect(D.V)))
println("  edges=", Subgraph.edge_count(fg, D), "  missing_edges=", Subgraph.missing_edges(fg, D))

println("="^70)
if Subgraph.edge_count(fg, D) == bf_edges
    println("MATCH")
else
    println("MISMATCH: branch_binary=", Subgraph.edge_count(fg, D), " vs brute force=", bf_edges)
end