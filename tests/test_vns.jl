using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__VNS_JL__) || include(joinpath(SRC, "vns.jl"))
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

@testset "vns seed + grow on complete biclique" begin
    g = BipartiteGraph{Nothing}()
    for u in 1:8, v in 1:8
        add_edge!(g, u, v, nothing)
    end
    fg = freeze(g)
    S = vns_seed_edge(fg)
    @test length(S.U) == 1
    @test length(S.V) == 1
    vns_greedily_grow!(fg, S, 1, 3)
    @test Subgraph.edge_count(fg, S) == length(S.U) * length(S.V)
    @test length(S.U) >= 3 && length(S.V) >= 3
end

@testset "vns finds θ-feasible on dense graph" begin
    Random.seed!(1)
    g = BipartiteGraph{Nothing}()
    for u in 1:10, v in 1:10
        add_edge!(g, u, v, nothing)
    end
    k, θ = 2, 4
    result = run_method!(VNSMethod(; kmax=8), deepcopy(g), k, θ;
        reduction=ReductionMode.none)
    sc = score_result(freeze(g), result, k, θ)
    @test sc.k_valid
    @test sc.theta_feasible
    @test sc.edges >= θ * θ
end

@testset "vns respects k-defect on sparse graph" begin
    Random.seed!(2)
    g = BipartiteGraph{Nothing}()
    # Star-like: one hub u connected to all v, plus a few extra edges.
    for v in 1:6
        add_edge!(g, 1, v, nothing)
    end
    for u in 2:5
        add_edge!(g, u, 1, nothing)
        add_edge!(g, u, 2, nothing)
    end
    k, θ = 2, 3
    result = run_method!(VNSMethod(; kmax=6), deepcopy(g), k, θ;
        reduction=ReductionMode.none)
    sc = score_result(freeze(g), result, k, θ)
    @test sc.k_valid
end
