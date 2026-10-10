using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

"""Complete 6×6 biclique — both sides equal, search U by the ≤ tie-break."""
function complete_biclique()
    g = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(g, u, v, nothing)
    end
    return g
end

"""
U is the smaller side (5 vs 8). Perfect 5×5 biclique plus two V vertices
that each miss one U neighbor. k = 2 repair on the U side should add both.
"""
function repair_graph()
    g = BipartiteGraph{Nothing}()
    for u in 1:5, v in 1:5
        add_edge!(g, u, v, nothing)
    end
    # v6 misses u1; v7 misses u2.
    for u in 2:5
        add_edge!(g, u, 6, nothing)
    end
    for u in (1, 3, 4, 5)
        add_edge!(g, u, 7, nothing)
    end
    add_v!(g, 8)
    return g
end

@testset "search side is the smaller part" begin
    g = repair_graph()
    Random.seed!(1)
    r = one_side_aco(deepcopy(g), 1, 8, 3, 0.95, 2, 3, 1;
        reduction=ReductionMode.none, parallelize=false, verbose=false)
    @test r.search_is_u
    @test r.n_same == 5
    @test r.n_opp == 8
end

@testset "complete biclique is recovered then repair cannot grow" begin
    g = complete_biclique()
    Random.seed!(1)
    r = one_side_aco(deepcopy(g), 1, 6, 2, 0.95, 1, 3, 1;
        reduction=ReductionMode.none, parallelize=false, verbose=false)
    @test length(r.biclique.U) == 6
    @test length(r.biclique.V) == 6
    @test r.biclique_edges == 36
    @test r.repaired_edges == 36
    sc = score_solution(freeze(g), r.best, 1, 3)
    @test sc.valid
    @test sc.edges == 36
end

@testset "greedy k-repair beats the biclique when defects help" begin
    g = repair_graph()
    fg, remap = compact_frozen(freeze(g))
    u = Dict(orig => i for (i, orig) in enumerate(remap.u_original))
    v = Dict(orig => i for (i, orig) in enumerate(remap.v_original))
    biclique = SubGraph(Set(u[i] for i in 1:5), Set(v[i] for i in 1:5))
    @test Subgraph.edge_count(fg, biclique) == 25
    repaired = greedy_k_repair(fg, biclique, 2, 3)
    @test Subgraph.missing_edges(fg, repaired) <= 2
    @test Subgraph.edge_count(fg, repaired) > 25
    @test v[6] in repaired.V
    @test v[7] in repaired.V
end

@testset "one_side_aco finds the biclique and repairs it" begin
    g = repair_graph()
    Random.seed!(1)
    r = one_side_aco(deepcopy(g), 1, 12, 4, 0.95, 2, 3, 1;
        reduction=ReductionMode.none, parallelize=false, verbose=false)
    @test r.biclique_edges >= 25
    @test r.repaired_edges >= r.biclique_edges
    sc = score_solution(freeze(g), r.best, 2, 3)
    @test sc.valid
    @test sc.edges >= 25
end

@testset "method wrapper matches paper win rule" begin
    g = complete_biclique()
    k, θ = 1, 3
    Random.seed!(1)
    result = run_method!(OneSideACOMethod(; ants=4, iterations=2, parallelize=false),
        deepcopy(g), k, θ; reduction=ReductionMode.none)
    @test result.name == "one_side_aco"
    @test result.meta["search_side"] == "U"
    sc = score_result(freeze(g), result, k, θ)
    @test sc.valid
    @test sc.edges == 36
end
