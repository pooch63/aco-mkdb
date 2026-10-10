using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__SUPER_REDUCE_JL__) || include(joinpath(SRC, "super-reduce.jl"))

const K = 2
const THETA = 5

function add_biclique!(g, us, vs)
    for u in us, v in vs
        add_edge!(g, u, v, nothing)
    end
    return g
end

"""K_{5,5} plus a disjoint K_{3,3}. The triple is θ-infeasible and has compatibility degree 2."""
function biclique_plus_triple()
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    add_biclique!(g, 6:8, 6:8)
    return g
end

"""
K_{5,5} plus a compatibility star on the U side: four private triples, each
shared only with a center. Leaves have degree 1; after they drop, the center does too.
"""
function biclique_plus_star()
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    # U6 is the center; U7..U10 are leaves. V starts at 6.
    v = 6
    for leaf in 7:10
        triple = v:(v + 2)
        for x in triple
            add_edge!(g, 6, x, nothing)
            add_edge!(g, leaf, x, nothing)
        end
        v += 3
    end
    return g
end

function side_ids(g::BipartiteGraph)
    return Set(keys(g.adjU)), Set(keys(g.adjV))
end

@testset "planted biclique survives and the disjoint triple is removed" begin
    g_sup = biclique_plus_triple()
    g_cnn = _copy_bipartite(g_sup)
    super_reduce!(g_sup, K, THETA)
    _apply_cnn!(g_cnn, K, THETA)

    @test side_ids(g_sup) == (Set(1:5), Set(1:5))
    @test side_ids(g_cnn) == (Set(1:8), Set(1:8))
end

@testset "degree cascade drops the center after its leaves" begin
    for cap in (SUPER_REDUCE_EDGE_CAP, -1)
        g = biclique_plus_star()
        super_reduce!(g, K, THETA; edge_cap=cap)
        @test side_ids(g) == (Set(1:5), Set(1:5))
    end
end

@testset "a side shorter than θ cannot host a feasible biclique" begin
    for side in (:smaller, :larger)
        g = BipartiteGraph{Nothing}()
        add_biclique!(g, 1:3, 1:8)
        super_reduce!(g, K, THETA; side=side)
        @test isempty(g.adjU)
        @test isempty(g.adjV)
    end
end

@testset "projecting the larger side still drops the disjoint triple" begin
    g = biclique_plus_star()
    super_reduce!(g, K, THETA; side=:larger)
    @test side_ids(g) == (Set(1:5), Set(1:5))
end

@testset "second pass deletes nothing" begin
    g = biclique_plus_triple()
    super_reduce!(g, K, THETA)
    before = side_ids(g)
    super_reduce!(g, K, THETA)
    @test side_ids(g) == before
end

function missing_edges(g, U, V)
    missing = 0
    for u in U, v in V
        has_edge(g, u, v) || (missing += 1)
    end
    return missing
end

function subsets_at_least(ids, θ)
    out = Vector{Int}[]
    n = length(ids)
    for mask in 1:(1 << n) - 1
        chosen = Int[]
        for i in 1:n
            if mask & (1 << (i - 1)) != 0
                push!(chosen, ids[i])
            end
        end
        length(chosen) >= θ && push!(out, chosen)
    end
    return out
end

"""
Four vertices, one pool of three opposite vertices shared by all of them.
Every pair has codegree 2, which is θ − 3 at θ = 5, so the edge rule drops
the whole projection. The triples themselves still share the pool, so only
the size check is doing the work.
"""
function shared_witness_k4()
    g = BipartiteGraph{Nothing}()
    for v in 1:3, u in 1:4
        add_edge!(g, u, v, nothing)
    end
    return g
end

"""K_5 in which every pair has three private opposite vertices and no triple shares one."""
function private_triple_k5(; u0::Int=1, v0::Int=1)
    g = BipartiteGraph{Nothing}()
    v = v0
    for a in u0:(u0 + 3), b in (a + 1):(u0 + 4)
        for _ in 1:3
            add_edge!(g, a, v, nothing)
            add_edge!(g, b, v, nothing)
            v += 1
        end
    end
    return g
end

@testset "edge rule drops a pair with only θ − 3 projected common neighbors" begin
    g = shared_witness_k4()
    kept, ecut, vcut = _weight_prune(collect(1:4), g.adjU, K, THETA, true, false)
    @test ecut == 6
    @test vcut == 0
    @test Set(kept) == Set(1:4)
end

@testset "triangle weight drops a clique whose triples are disjoint" begin
    g = private_triple_k5()
    kept, ecut, vcut = _weight_prune(collect(1:5), g.adjU, K, THETA, true, true)
    @test isempty(kept)
    @test ecut == 10
    @test vcut == 5

    kept_e, ecut_e, vcut_e = _weight_prune(collect(1:5), g.adjU, K, THETA, true, false)
    @test Set(kept_e) == Set(1:5)
    @test ecut_e == 10
    @test vcut_e == 0

    # Degree in the projection is already θ − 1 and every weight equals k, so rule 2 alone deletes nothing.
    kept_v, ecut_v, vcut_v = _weight_prune(collect(1:5), g.adjU, K, THETA, false, true)
    @test Set(kept_v) == Set(1:5)
    @test ecut_v == 0
    @test vcut_v == 0
end

@testset "a complete biclique survives both weighted rules" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    kept, ecut, vcut = _weight_prune(collect(1:5), g.adjU, K, THETA, true, true)
    @test Set(kept) == Set(1:5)
    @test ecut == 0
    @test vcut == 0

    # C > θ clamps the weight at 0 rather than going negative.
    g6 = BipartiteGraph{Nothing}()
    add_biclique!(g6, 1:6, 1:6)
    kept6, ecut6, vcut6 = _weight_prune(collect(1:6), g6.adjU, K, THETA, true, true)
    @test Set(kept6) == Set(1:6)
    @test ecut6 == 0
    @test vcut6 == 0
end

"""
K_{4,4} plus a triangle-free compatibility gadget on U: parts {5,6,7} and
{8,9,10}, two private opposite vertices per cross pair. At k = 2, θ = 4 the
common-neighbor pass keeps the gadget; the edge rule does not.
"""
function planted_plus_triangle_free()
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:4, 1:4)
    v = 5
    for a in 5:7, b in 8:10
        for _ in 1:2
            add_edge!(g, a, v, nothing)
            add_edge!(g, b, v, nothing)
            v += 1
        end
    end
    return g
end

@testset "triangle-free core is removed only when the edge rule is on" begin
    g = planted_plus_triangle_free()
    super_reduce!(g, 2, 4)
    @test side_ids(g) == (Set(1:4), Set(1:4))

    g_off = planted_plus_triangle_free()
    super_reduce!(g_off, 2, 4; prune_edges=false, prune_vertices=false)
    u_off, _ = side_ids(g_off)
    @test Set(5:10) ⊆ u_off
end

function planted_plus_private_k5()
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    v = 6
    for a in 6:9, b in (a + 1):10
        for _ in 1:3
            add_edge!(g, a, v, nothing)
            add_edge!(g, b, v, nothing)
            v += 1
        end
    end
    return g
end

@testset "weighted rules delete the private-triple clique before the common-neighbor pass" begin
    g = planted_plus_private_k5()
    st = Dict{Symbol,Int}()
    super_reduce!(g, K, THETA; stats=st)
    @test side_ids(g) == (Set(1:5), Set(1:5))
    @test st[:weight_vertices_cut] >= 5
    @test st[:weight_edges_cut] >= 10

    g_off = planted_plus_private_k5()
    st_off = Dict{Symbol,Int}()
    super_reduce!(g_off, K, THETA; stats=st_off, prune_edges=false, prune_vertices=false)
    @test side_ids(g_off) == (Set(1:5), Set(1:5))
    @test st_off[:weight_edges_cut] == 0
    @test st_off[:weight_vertices_cut] == 0
end

"""
Edge {1,2} has three projected common neighbors and a triple of codegree 3,
so the size check and the triangle check both keep it. Those three neighbors
span no projected edge, short of the three edges a 5-clique would need.
"""
function edge_count_witness()
    g = BipartiteGraph{Nothing}()
    v = 1
    for us in ((1, 2, 4), (1, 2, 3), (1, 2, 5))
        for _ in 1:3
            for u in us
                add_edge!(g, u, v, nothing)
            end
            v += 1
        end
    end
    return g
end

@testset "induced edges in the common neighborhood prune an edge the triangle rule keeps" begin
    g = edge_count_witness()
    kept, ecut, vcut = _weight_prune(collect(1:5), g.adjU, K, THETA, true, false)
    @test ecut == 7
    @test vcut == 0
    @test Set(kept) == Set(1:5)
end

"""K_{5,10} beside a disjoint K_{5,5}. The lighter clique is θ-feasible and loses on neighborhood weight."""
function dense_plus_light_biclique()
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:10)
    add_biclique!(g, 6:10, 11:15)
    return g
end

@testset "neighborhood weight keeps the heavier biclique" begin
    g = dense_plus_light_biclique()
    st = Dict{Symbol,Int}()
    super_reduce!(g, K, THETA; stats=st)
    @test side_ids(g) == (Set(1:5), Set(1:10))
    @test st[:mwc_vertices_cut] >= 5

    g_off = dense_plus_light_biclique()
    super_reduce!(g_off, K, THETA; prune_vertices=false)
    u_off, v_off = side_ids(g_off)
    @test Set(6:10) ⊆ u_off
    @test Set(11:15) ⊆ v_off
end

function best_feasible_edges(g, k, θ)
    us = collect(keys(g.adjU))
    vs = collect(keys(g.adjV))
    best = 0
    for U in subsets_at_least(us, θ), V in subsets_at_least(vs, θ)
        miss = missing_edges(g, U, V)
        miss <= k || continue
        e = length(U) * length(V) - miss
        e > best && (best = e)
    end
    return best
end

@testset "the best θ-feasible edge count survives" begin
    Random.seed!(1)
    g0 = BipartiteGraph{Nothing}()
    for u in 1:7, v in 1:7
        rand() < 0.55 && add_edge!(g0, u, v, nothing)
    end
    # Guarantee one feasible biclique so the test is not vacuous.
    add_biclique!(g0, 1:3, 1:3)
    before = best_feasible_edges(g0, 1, 3)
    @test before > 0

    g = _copy_bipartite(g0)
    super_reduce!(g, 1, 3)
    @test best_feasible_edges(g, 1, 3) == before

    g_wide = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:9
        rand() < 0.5 && add_edge!(g_wide, u, v, nothing)
    end
    add_biclique!(g_wide, 1:3, 1:3)
    before_wide = best_feasible_edges(g_wide, 1, 3)
    @test before_wide > 0
    reduced = _copy_bipartite(g_wide)
    super_reduce!(reduced, 1, 3; side=:larger)
    @test best_feasible_edges(reduced, 1, 3) == before_wide
end
