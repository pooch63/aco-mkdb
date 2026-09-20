using Test
using Random
using Statistics

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

@testset "diffusion cohesion" begin
    g = BipartiteGraph{Int}()
    for u in 1:4, v in 1:4
        add_edge!(g, u, v, 0)
    end
    fg, _ = compact_frozen(freeze(g))
    u_vals = fill(0.25, length(fg.u_ids))
    v_vals = fill(0.25, length(fg.v_ids))
    sg = SubGraph(Set([1, 2]), Set([1, 2]))
    @test diffusion_cohesion(sg, u_vals, v_vals) ≈ 1.0

    u_vals[1] = -0.5
    u_vals[2] = 0.5
    v_vals[1] = -0.5
    v_vals[2] = 0.5
    c = diffusion_cohesion(sg, u_vals, v_vals)
    @test c < 1.0
    @test c > 0.0

    fit = diffusion_fitness(fg, sg, 2, u_vals, v_vals)
    base = Float64(instance_fitness(fg, sg, 2))
    @test fit ≈ base * c
end

@testset "diffuse_vertex_values" begin
    Random.seed!(1)
    g = BipartiteGraph{Int}()
    for u in 1:3, v in 1:3
        add_edge!(g, u, v, 0)
    end
    fg, _ = compact_frozen(freeze(g))
    u0, v0 = diffuse_vertex_values(fg; iterations=0, rng=MersenneTwister(1))
    @test u0 == [3.0, 3.0, 3.0]
    @test v0 == [3.0, 3.0, 3.0]

    u1, v1 = diffuse_vertex_values(fg; iterations=5, rng=MersenneTwister(1))
    @test length(u1) == length(fg.u_ids)
    @test length(v1) == length(fg.v_ids)
    # After smoothing on a complete biclique, values should be closer together.
    @test std(vcat(u1, v1)) <= std(vcat(u0, v0)) + 1e-12

    # Verify standard deviation logging across iterations
    io = IOBuffer()
    diffuse_vertex_values(fg; iterations=3, rng=MersenneTwister(1), log_std=true, io=io)
    out = String(take!(io))
    @test occursin("Diffusion iter 0/3: heat std=", out)
    @test occursin("Diffusion iter 1/3: heat std=", out)
    @test occursin("Diffusion iter 2/3: heat std=", out)
    @test occursin("Diffusion iter 3/3: heat std=", out)

    io_silent = IOBuffer()
    diffuse_vertex_values(fg; iterations=3, rng=MersenneTwister(1), log_std=false, io=io_silent)
    @test isempty(String(take!(io_silent)))

    # Verify injected biclique mean comparison logging
    io_inj = IOBuffer()
    plant = SubGraph(Set([1, 2]), Set([1, 2]))
    diffuse_vertex_values(fg; iterations=3, rng=MersenneTwister(1), log_std=true, io=io_inj,
        injected_biclique=plant)
    out_inj = String(take!(io_inj))
    @test occursin("Diffusion heat mean: injected=", out_inj)
    @test occursin("vs overall=", out_inj)
    @test occursin("vs random_sample=", out_inj)
    @test occursin("(n=4)", out_inj)
end

@testset "diffusion ACO smoke" begin
    Random.seed!(42)
    g = BipartiteGraph{Int}()
    for u in 1:6, v in 1:6
        add_edge!(g, u, v, 0)
    end
    r = run_method!(DiffusionMethod(; ants=3, iterations=2, parallelize=false,
            diffusion_iters=3),
        deepcopy(g), 1, 3; reduction=ReductionMode.none)
    @test r.name == "diffusion"
    @test Subgraph.vertex_count(r.sol) > 0
    sc = score_result(freeze(g), r, 1, 3)
    @test sc.k_valid

    io_aco = IOBuffer()
    r2 = run_method!(DiffusionMethod(; ants=2, iterations=1, parallelize=false,
            diffusion_iters=2),
        deepcopy(g), 1, 3; reduction=ReductionMode.none,
        injected_biclique=SubGraph(Set([1, 2]), Set([1, 2])), io=io_aco)
    out_aco = String(take!(io_aco))
    @test occursin("Diffusion heat mean: injected=", out_aco)
    @test occursin("vs overall=", out_aco)
    @test occursin("vs random_sample=", out_aco)
end

@testset "compute_diffusion_weights and weighted diffusion" begin
    # 1. Complete biclique K_{3, 4}
    g_k34 = BipartiteGraph{Int}()
    for u in 1:3, v in 1:4
        add_edge!(g_k34, u, v, 0)
    end
    fg_k34, _ = compact_frozen(freeze(g_k34))
    u_ew, v_ew, u_tw, v_tw = compute_diffusion_weights(fg_k34)

    # For u in U (size 3), each edge (u, v) has |n2(u) \cap n(v)| = 2 other U nodes
    @test all(u_ew .== 2.0)
    # Total weight for each u: 1.0 (self) + 4 neighbors * 2.0 = 9.0
    @test all(u_tw .== 9.0)

    # For v in V (size 4), each edge (v, u) has |n2(v) \cap n(u)| = 3 other V nodes
    @test all(v_ew .== 3.0)
    # Total weight for each v: 1.0 (self) + 3 neighbors * 3.0 = 10.0
    @test all(v_tw .== 10.0)

    # 2. Path graph with no 4-cycles: u1 - v1 - u2 - v2 - u3
    g_path = BipartiteGraph{Int}()
    add_edge!(g_path, 1, 1, 0)
    add_edge!(g_path, 2, 1, 0)
    add_edge!(g_path, 2, 2, 0)
    add_edge!(g_path, 3, 2, 0)
    fg_path, _ = compact_frozen(freeze(g_path))
    u_ew_p, v_ew_p, u_tw_p, v_tw_p = compute_diffusion_weights(fg_path)

    # With no 4-cycles, no edge represents common nodes beyond the edge itself
    @test all(u_ew_p .== 0.0)
    @test all(v_ew_p .== 0.0)
    @test all(u_tw_p .== 1.0)
    @test all(v_tw_p .== 1.0)

    # Diffusion on graph with no 4-cycles leaves values unchanged (self-weight dominates)
    u_init, v_init = diffuse_vertex_values(fg_path; iterations=0, rng=MersenneTwister(42))
    u_diff, v_diff = diffuse_vertex_values(fg_path; iterations=5, rng=MersenneTwister(42))
    @test u_init == u_diff
    @test v_init == v_diff

    # 3. 4-cycle with an attached tail: u1-v1, u1-v2, u2-v1, u2-v2, plus tail u3-v2
    g_tail = BipartiteGraph{Int}()
    add_edge!(g_tail, 1, 1, 0)
    add_edge!(g_tail, 1, 2, 0)
    add_edge!(g_tail, 2, 1, 0)
    add_edge!(g_tail, 2, 2, 0)
    add_edge!(g_tail, 3, 2, 0)
    fg_tail = freeze(g_tail)
    u_ew_t, v_ew_t, u_tw_t, v_tw_t = compute_diffusion_weights(fg_tail)

    # u3 is only connected to v2, shares no other neighbors with v2's other neighbors
    u3_idx = fg_tail.u_index[3]
    for k in neighbor_range_u(fg_tail, u3_idx)
        @test u_ew_t[k] == 0.0
    end
    @test u_tw_t[u3_idx] == 1.0

    # u1 and u2 form a 4-cycle, so their edges to v1 and v2 have common node count = 1.0
    u1_idx = fg_tail.u_index[1]
    for k in neighbor_range_u(fg_tail, u1_idx)
        @test u_ew_t[k] == 1.0
    end
end

