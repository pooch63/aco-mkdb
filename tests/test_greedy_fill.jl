using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

# u1 touches v1, v2, and the trap v3. u2 and u3 touch only v1 and v2.
# With k = 0 the maximum is {u1,u2,u3} × {v1,v2} (6 edges).
function trap_graph()
    g = BipartiteGraph{Nothing}()
    for u in 1:3, v in 1:2
        add_edge!(g, u, v, nothing)
    end
    add_edge!(g, 1, 3, nothing)
    fg, remap = compact_frozen(freeze(g))
    u = Dict(orig => i for (i, orig) in enumerate(remap.u_original))
    v = Dict(orig => i for (i, orig) in enumerate(remap.v_original))
    return fg, u, v
end

@testset "opposite side packs by missing-edge cost" begin
    fg, u, v = trap_graph()
    # u1, u2 fixed. v3 misses u2, so k = 0 leaves it out and k = 1 takes it.
    packed = pack_opposite(fg, true, Set([u[1], u[2]]), 0)
    @test sort(packed) == sort([v[1], v[2]])
    packed_k = pack_opposite(fg, true, Set([u[1], u[2]]), 1)
    @test sort(packed_k) == sort([v[1], v[2], v[3]])
end

@testset "fill keeps the fixed side, packs the other, then takes free vertices" begin
    fg, u, v = trap_graph()
    # Walk stopped at {u1,u2} × {v1}: 2 edges. k = 0.
    walk = SubGraph(Set([u[1], u[2]]), Set([v[1]]))
    filled = fill_keeping_side(fg, walk, true, 0)
    @test filled.U == Set([u[1], u[2], u[3]])
    @test filled.V == Set([v[1], v[2]])
    @test Subgraph.edge_count(fg, filled) == 6
    @test Subgraph.missing_edges(fg, filled) == 0

    # {u1} touches every V, so packing V keeps the trap, and u2 is not free against it.
    trap = SubGraph(Set([u[1]]), Set([v[1], v[2], v[3]]))
    kept_u = fill_keeping_side(fg, trap, true, 0)
    @test kept_u.U == Set([u[1]])
    @test kept_u.V == Set([v[1], v[2], v[3]])
    @test Subgraph.edge_count(fg, kept_u) == 3
end

@testset "end fill keeps the better orientation" begin
    fg, u, v = trap_graph()
    ant = new_ants(fg, 1, 1)[1]
    ant.explored = SubGraph(Set([u[1], u[2]]), Set([v[1]]))
    record_end_fill!(ant, fg, 0, 2, instance_fitness)
    @test ant.filled.U == Set([u[1], u[2], u[3]])
    @test ant.filled.V == Set([v[1], v[2]])
    kept = kept_ant_solution(ant, fg, 2, instance_fitness, nothing, nothing)
    @test Subgraph.edge_count(fg, kept) == 6
    @test Subgraph.edge_count(fg, kept) > Subgraph.edge_count(fg, ant.explored)
end

@testset "colony keeps the fill" begin
    fg = trap_graph()
    g = BipartiteGraph{Nothing}()
    for u in 1:3, v in 1:2
        add_edge!(g, u, v, nothing)
    end
    add_edge!(g, 1, 3, nothing)

    function run(mode)
        Random.seed!(1)
        trial = deepcopy(g)
        sols, _, _, _, _ = aco(trial, 1, 4, 2, 0.95, 0, 2, 1;
            parallelize=false, elite_seed=false, prefer_smaller_side=false,
            neighbor_scope_limit=true, reduction=ReductionMode.none,
            greedy_fill=mode)
        return Subgraph.edge_count(freeze(trial), sols[1])
    end

    off = run(:off)
    @test run(:end) >= off
    @test run(:step) >= off
    @test run(:end) == 6
    @test run(:step) == 6
    @test_throws ArgumentError normalize_greedy_fill("sometimes")
    m = make_method("aco"; greedy_fill="step", ants=1, iterations=1)
    @test m.greedy_fill === :step
end
