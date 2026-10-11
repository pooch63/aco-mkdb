using Test

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__DOUBLE_GREEDY_JL__) || include(joinpath(SRC, "double-greedy.jl"))
isdefined(@__MODULE__, :__THETA_HEURISTIC_JL__) || include(joinpath(SRC, "theta_heuristic.jl"))

function complete_biclique(nU::Int, nV::Int)
    g = BipartiteGraph{Nothing}()
    for u in 1:nU, v in 1:nV
        add_edge!(g, u, v, nothing)
    end
    return g
end

function edges_of(fg::FrozenBipartite, sg::SubGraph)
    return Subgraph.edge_count(fg, sg), Subgraph.missing_edges(fg, sg)
end

@testset "double-greedy" begin
    @testset "grows past θ on a complete biclique" begin
        g = complete_biclique(6, 6)
        fg = freeze(g)
        r = double_greedy(fg, 0, 3)
        @test r.seed_is_u
        @test r.seed_id == 1
        @test r.sol.U == Set(1:6)
        @test r.sol.V == Set(1:6)
        e, miss = edges_of(fg, r.sol)
        @test e == 36
        @test miss == 0

        theta = theta_based_heuristic(fg, 0, 3)
        @test length(theta.U) == 3
        @test length(theta.V) == 6
        @test Subgraph.edge_count(fg, theta) == 18
    end

    @testset "seeds the higher-degree side" begin
        # Every V has degree 4 and every U has degree 3.
        g = complete_biclique(4, 3)
        fg = freeze(g)
        r = double_greedy(fg, 0, 2)
        @test !r.seed_is_u
        @test r.seed_id == 1
        @test r.sol.U == Set(1:4)
        @test r.sol.V == Set(1:3)
        e, miss = edges_of(fg, r.sol)
        @test e == 12
        @test miss == 0

        theta = theta_based_heuristic(fg, 0, 2)
        @test Subgraph.edge_count(fg, theta) == 6
    end

    @testset "refuses a vertex that would exceed k" begin
        g = BipartiteGraph{Nothing}()
        for v in 1:4
            add_edge!(g, 1, v, nothing)
        end
        add_edge!(g, 2, 1, nothing)
        add_edge!(g, 2, 2, nothing)
        add_edge!(g, 3, 1, nothing)
        add_edge!(g, 3, 3, nothing)
        add_edge!(g, 3, 4, nothing)
        fg = freeze(g)
        r = double_greedy(fg, 1, 2)
        @test r.seed_is_u
        @test r.seed_id == 1
        @test r.sol.U == Set([1, 3])
        @test r.sol.V == Set(1:4)
        e, miss = edges_of(fg, r.sol)
        @test e == 7
        @test miss == 1
        @test miss <= 1
    end

    @testset "short side wins even when the other side adds more edges" begin
        # U1 is the unique degree-3 vertex, so the seed is U1 plus V1–V3.
        # After U2, U is still the short side. U3 adds no edges; V4 adds one.
        # The short side is filled first, which spends the defect budget and
        # leaves V4 infeasible.
        g = BipartiteGraph{Nothing}()
        add_u!(g, 3)
        for v in 1:3
            add_edge!(g, 1, v, nothing)
        end
        add_edge!(g, 2, 1, nothing)
        add_edge!(g, 2, 4, nothing)
        fg = freeze(g)
        r = double_greedy(fg, 5, 4)
        @test r.sol.U == Set([1, 2, 3])
        @test r.sol.V == Set([1, 2, 3])
        e, miss = edges_of(fg, r.sol)
        @test e == 4
        @test miss == 5
    end

    @testset "trims the long side down to θ to free defect budget" begin
        # U1 is the unique degree-5 vertex, so the seed is U1 plus V1–V5.
        # U2 misses only V5. With k = 0 that one defect is dropped, and V stays
        # at 4, which is still at least θ. U3 cannot be trimmed down to k = 0
        # without taking V below θ.
        g = BipartiteGraph{Nothing}()
        for v in 1:5
            add_edge!(g, 1, v, nothing)
        end
        for v in 1:4
            add_edge!(g, 2, v, nothing)
        end
        add_edge!(g, 3, 1, nothing)
        fg = freeze(g)
        r = double_greedy(fg, 0, 2)
        @test r.sol.U == Set([1, 2])
        @test r.sol.V == Set(1:4)
        @test length(r.sol.U) >= 2
        @test length(r.sol.V) >= 2
        e, miss = edges_of(fg, r.sol)
        @test e == 8
        @test miss == 0
    end

    @testset "empty graph" begin
        fg = freeze(BipartiteGraph{Nothing}())
        r = double_greedy(fg, 2, 5)
        @test isempty(r.sol.U)
        @test isempty(r.sol.V)
        @test r.additions == 0
    end
end
