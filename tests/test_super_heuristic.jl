using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__SUPER_HEURISTIC_JL__) || include(joinpath(SRC, "super-heuristic.jl"))

function add_biclique!(g, us, vs)
    for u in us, v in vs
        add_edge!(g, u, v, nothing)
    end
    return g
end

"""K_{5,5} plus a disjoint K_{3,3}. The triple is θ-infeasible."""
function biclique_plus_triple()
    g = BipartiteGraph{Nothing}()
    add_biclique!(g, 1:5, 1:5)
    add_biclique!(g, 6:8, 6:8)
    return g
end

@testset "planted biclique survives when an incumbent is supplied" begin
    g = biclique_plus_triple()
    inc = SubGraph(Set(1:5), Set(1:5))
    super_reduce!(g, 2, 5; incumbent=inc)
    @test Set(keys(g.adjU)) == Set(1:5)
    @test Set(keys(g.adjV)) == Set(1:5)
end

@testset "theta raises the edge bound, then ACO is tried until patience" begin
    Random.seed!(1)
    g = biclique_plus_triple()
    traced = super_heuristic!(g, 2, 5; patience=1, max_cycles=6,
        ants=2, iterations=1, verbose=false)
    @test traced.best_edges == 25
    @test length(traced.sol.U) == 5
    @test length(traced.sol.V) == 5
    @test traced.theta_improvements == 1
    @test traced.aco_improvements == 0
    # Cycle 1 improves via θ, so the stall counter resets. One quiet cycle
    # then hits patience, which is a second ACO call and a third θ call.
    @test traced.theta_calls == 3
    @test traced.aco_calls == 2
    @test traced.cycles == 2
end
