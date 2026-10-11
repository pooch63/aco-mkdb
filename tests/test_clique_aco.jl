using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

const K = 2
const THETA = 5

function add_biclique!(g, us, vs)
    for u in us, v in vs
        add_edge!(g, u, v, nothing)
    end
    return g
end

"""K_4 minus the edge {3,4}. Indices are clique ids."""
function almost_clique()
    return CliqueGraph(true, [10, 20, 30, 40],
        [[2, 3, 4], [1, 3, 4], [1, 2], [1, 2]],
        [3, 3, 2, 2], 5)
end

@testset "compatibility graph of a biclique is a complete graph" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    fg, _ = compact_frozen(freeze(g))
    cg = compatibility_graph(fg, K, THETA)
    @test cg.search_is_u
    @test length(cg.side_id) == 5
    @test cg.edges == 10
    @test all(d == 4 for d in cg.deg)
    @test clique_induced_missing(cg, 1:5) == 0
end

@testset "a pair below θ − k common neighbors is not an edge" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:2, 1:5)
    # U3 meets U1 and U2 in only two vertices, short of τ = 3.
    add_edge!(g, 3, 1, nothing)
    add_edge!(g, 3, 2, nothing)
    fg, remap = compact_frozen(freeze(g))
    cg = compatibility_graph(fg, K, THETA)
    @test length(cg.side_id) == 2
    @test cg.edges == 1
    @test Set(remap.u_original[i] for i in cg.side_id) == Set([1, 2])
end

@testset "candidates are neighbors of the last vertex" begin
    cg = almost_clique()
    ant = CliqueAnt(4, 1)
    @test clique_candidates(cg, ant) == [1, 2, 3, 4]
    commit_clique_vertex!(ant, cg, 1)
    commit_clique_vertex!(ant, cg, 4)
    @test ant.missing == 0
    # 3 is not adjacent to 4, so it is not a candidate.
    @test clique_candidates(cg, ant) == [2]
    commit_clique_vertex!(ant, cg, 2)
    @test ant.missing == 0
    @test clique_induced_missing(cg, ant.order) == ant.missing
    # 3 is adjacent to 2, so neighbor scope offers it. The fill, not the
    # projection, decides whether that offer is worth taking.
    @test clique_candidates(cg, ant) == [3]
end

@testset "desirability is τ times the greedy-fill edge count" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    # U6 meets only two of the five opposite vertices.
    add_edge!(g, 6, 1, nothing)
    add_edge!(g, 6, 2, nothing)
    fg, _ = compact_frozen(freeze(g))
    cg = compatibility_graph(fg, K, THETA)
    # τ = θ − k = 3, so U6 is absent and the five full vertices remain.
    @test length(cg.side_id) == 5
    st = OracleState(fg, true)
    ant = CliqueAnt(length(cg.side_id), 1)
    trails = CliqueTrails(length(cg.side_id), 1)
    # One vertex packs its five neighbors: the fill edge count is the degree.
    @test oracle_edges_if_added(st, fg, cg.side_id[1], K) == 5
    @test clique_desirability(trails, cg, ant, st, fg, 1, K, 2.0) ≈ 25.0
    add_left!(st, fg, cg.side_id[1])
    commit_clique_vertex!(ant, cg, 1)
    # A second universal vertex keeps all five opposite vertices at cost 0.
    @test oracle_edges_if_added(st, fg, cg.side_id[2], K) == 10
    committed, _ = oracle_score(st, K)
    @test committed == 5
end

@testset "a non-neighbor's missing count rises by one" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, [1], 1:5)
    add_edge!(g, 2, 1, nothing)
    add_edge!(g, 2, 2, nothing)
    add_edge!(g, 2, 3, nothing)
    fg, remap = compact_frozen(freeze(g))
    st = OracleState(fg, true)
    u1 = findfirst(==(1), remap.u_original)
    u2 = findfirst(==(2), remap.u_original)
    add_left!(st, fg, u1)
    # k = 0 keeps only cost-0 opposite vertices. U2 misses V4 and V5, so those
    # two counts rise from 0 to 1 and drop out. The three shared neighbors
    # stay at cost 0 and contribute 2 edges each.
    @test oracle_edges_if_added(st, fg, u2, 0) == 6
    add_left!(st, fg, u2)
    edges, nsel = oracle_score(st, 0)
    @test edges == 6
    @test nsel == 3
end

@testset "a complete biclique is recovered from the clique" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:6, 1:6)
    Random.seed!(1)
    r = clique_aco(deepcopy(g), 1, 1, 1, 0.95, 1, 3, 1;
        reduction=ReductionMode.none, parallelize=false, verbose=false)
    @test r.search_is_u
    @test r.clique_vertices == 6
    @test r.clique_edges == 15
    @test length(r.best.U) == 6
    @test length(r.best.V) == 6
    @test r.edges == 36
    sc = score_solution(freeze(g), r.best, 1, 3)
    @test sc.valid
    @test sc.edges == 36
end

@testset "clique-aco stays within the defect budget" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    # U6 shares three neighbors with each of U1..U5, so it is in the projection,
    # but it is not joined to a private vertex U7 that only meets U1.
    for v in 1:3
        add_edge!(g, 6, v, nothing)
    end
    Random.seed!(1)
    r = clique_aco(deepcopy(g), 1, 8, 3, 0.95, K, THETA, 1;
        reduction=ReductionMode.none, parallelize=false, verbose=false)
    sc = score_solution(freeze(g), r.best, K, THETA)
    @test sc.k_valid
    @test sc.missing <= K
end

@testset "registered clique-aco solves a biclique" begin
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:6, 1:6)
    Random.seed!(1)
    m = make_method("clique-aco"; ants=2, iterations=1, parallelize=false)
    result = run_method!(m, g, 1, 3; reduction=ReductionMode.none, verbose=false)
    @test result.name == "clique_aco"
    sc = score_solution(freeze(g), result.sol, 1, 3)
    @test sc.valid
    @test sc.edges == 36
    @test result.meta["clique_vertices"] == 6
end
