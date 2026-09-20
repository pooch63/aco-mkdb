using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__RETRY_JL__) || include(joinpath(SRC, "retry.jl"))
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

@testset "retry neighbor candidates respect k" begin
    g = BipartiteGraph{Nothing}()
    # Complete 3×3 minus one edge (1,3).
    for u in 1:3, v in 1:3
        (u == 1 && v == 3) && continue
        add_edge!(g, u, v, nothing)
    end
    fg = freeze(g)
    S = SubGraph(Set([1]), Set([1, 2]))
    ensure_membership!(S, fg)
    # Last node L = u=1; neighbors in V are {1,2} (not 3). Both already in S → C empty.
    C = retry_neighbor_candidates(fg, S, Node(true, 1), 0)
    @test isempty(C)

    # From v=1, opposite U-neighbors excluding those in S: {2,3} (1 already in).
    C2 = retry_neighbor_candidates(fg, S, Node(false, 1), 0)
    @test Set(n.id for n in C2) == Set([2, 3])
    @test all(n.is_u for n in C2)
end

@testset "retry finds θ-feasible on dense graph" begin
    Random.seed!(1)
    g = BipartiteGraph{Nothing}()
    for u in 1:10, v in 1:10
        add_edge!(g, u, v, nothing)
    end
    k, θ = 2, 4
    result = run_method!(RetryMethod(; beta=0.05, max_steps=5_000), deepcopy(g), k, θ;
        reduction=ReductionMode.none)
    sc = score_result(freeze(g), result, k, θ)
    @test sc.k_valid
    @test sc.theta_feasible
    @test sc.edges >= θ * θ
    @test result.name == "retry"
    @test result.meta["beta"] == 0.05
    @test result.meta["p"] == 1.0
end

@testset "retry respects k-defect on sparse graph" begin
    Random.seed!(2)
    g = BipartiteGraph{Nothing}()
    for v in 1:6
        add_edge!(g, 1, v, nothing)
    end
    for u in 2:5
        add_edge!(g, u, 1, nothing)
        add_edge!(g, u, 2, nothing)
    end
    k, θ = 2, 3
    result = run_method!(RetryMethod(; beta=0.04, p=0.5, max_steps=2_000), deepcopy(g), k, θ;
        reduction=ReductionMode.none)
    sc = score_result(freeze(g), result, k, θ)
    @test sc.k_valid
end

@testset "retry backtrack count is in 1..n" begin
    Random.seed!(3)
    for n in 1:20, _ in 1:50
        r = retry_backtrack_count(n, 1.0)
        @test 1 <= r <= n
    end
end

@testset "retry registry" begin
    m = make_method("retry"; beta=0.2, p=0.75, max_steps=100)
    @test m isa RetryMethod
    @test m.beta == 0.2
    @test m.p == 0.75
    @test m.max_steps == 100
    @test "retry" in list_methods()
end
