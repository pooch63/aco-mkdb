using Test

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

@testset "method registry" begin
    known = list_methods()
    @test "aco" in known
    @test "edges" in known
    @test "diffusion" in known
    @test "ga" in known
    @test "heuristic" in known
    @test "pivot" in known
    @test "tabu" in known
    @test "vns" in known
    @test "retry" in known
    @test "opponent" in known

    m = make_method("heuristic")
    @test m isa HeuristicMethod
    @test method_id(m) == "heuristic"

    aco = make_method("aco"; ants=2, iterations=1, parallelize=false)
    @test aco isa ACOMethod
    @test aco.ants == 2

    edges = make_method("edges"; ants=2, iterations=1, parallelize=false)
    @test edges isa EdgesMethod
    @test edges.ants == 2

    diffusion = make_method("diffusion"; ants=2, iterations=1, parallelize=false,
        diffusion_iters=5)
    @test diffusion isa DiffusionMethod
    @test diffusion.ants == 2
    @test diffusion.diffusion_iters == 5

    vns = make_method("vns"; kmax=5)
    @test vns isa VNSMethod
    @test vns.kmax == 5

    retry = make_method("retry"; beta=0.15, p=0.8, max_steps=500)
    @test retry isa RetryMethod
    @test retry.beta == 0.15
    @test retry.p == 0.8
    @test retry.max_steps == 500
end

@testset "method smoke on tiny biclique" begin
    g = BipartiteGraph{Int}()
    for u in 1:6, v in 1:6
        add_edge!(g, u, v, u * 100 + v)
    end
    k, θ = 1, 3
    reduction = ReductionMode.none

    h = run_method!(HeuristicMethod(), deepcopy(g), k, θ; reduction=reduction)
    @test h isa MethodResult
    @test h.name == "heuristic"
    sc = score_result(freeze(g), h, k, θ)
    @test sc.k_valid

    a = run_method!(ACOMethod(; ants=2, iterations=1, parallelize=false),
        deepcopy(g), k, θ; reduction=reduction)
    @test a isa MethodResult
    @test !isempty(a.all_sols)

    e = run_method!(EdgesMethod(; ants=2, iterations=1, parallelize=false),
        deepcopy(g), k, θ; reduction=reduction)
    @test e isa MethodResult
    @test e.name == "edges"
    @test !isempty(e.all_sols)
    esc = score_result(freeze(g), e, k, θ)
    @test esc.k_valid

    d = run_method!(DiffusionMethod(; ants=2, iterations=1, parallelize=false,
            diffusion_iters=5),
        deepcopy(g), k, θ; reduction=reduction)
    @test d isa MethodResult
    @test d.name == "diffusion"
    @test !isempty(d.all_sols)
    @test d.meta["diffusion_iters"] == 5
    dsc = score_result(freeze(g), d, k, θ)
    @test dsc.k_valid

    v = run_method!(VNSMethod(; kmax=5), deepcopy(g), k, θ; reduction=reduction)
    @test v isa MethodResult
    @test v.name == "vns"
    vsc = score_result(freeze(g), v, k, θ)
    @test vsc.k_valid
    @test vsc.edges > 0

    r = run_method!(RetryMethod(; beta=0.04, max_steps=1_000), deepcopy(g), k, θ;
        reduction=reduction)
    @test r isa MethodResult
    @test r.name == "retry"
    rsc = score_result(freeze(g), r, k, θ)
    @test rsc.k_valid
    @test rsc.edges > 0

    # Paper-style beats: same edges → not a win.
    @test !beats(h, h, freeze(g), k, θ)
    @test compare_results(h, h, freeze(g), k, θ) == :tie
end

@testset "register custom method" begin
    struct DummyMethod <: SolveMethod end
    method_id(::DummyMethod) = "dummy"
    function run_method!(::DummyMethod, g::BipartiteGraph, k::Int, θ::Int; reduction=ReductionMode.none, kwargs...)
        return MethodResult("dummy", SubGraph())
    end
    register_method!("dummy", (; kwargs...) -> DummyMethod())
    @test "dummy" in list_methods()
    @test make_method("dummy") isa DummyMethod
    r = run_method!(make_method("dummy"), BipartiteGraph{Int}(), 1, 2)
    @test r.name == "dummy"
end
