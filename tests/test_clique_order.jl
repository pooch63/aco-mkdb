using Test

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

@testset "clique-order selftest" begin
    @test clique_order_selftest()
end

@testset "clique-order method" begin
    g = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(g, u, v, nothing)
    end
    base = freeze(g)
    result = run_method!(make_method("clique-order"), deepcopy(g), 2, 5;
        reduction=ReductionMode.none)
    @test result.name == "clique-order"
    @test result.meta["clique_size"] == 6
    @test result.meta["kept"] == 6
    @test result.meta["search_side"] == "U"
    sc = score_result(base, result, 2, 5)
    @test sc.k_valid
    @test sc.theta_feasible
    @test sc.edges == 36
    @test sc.missing == 0
end
