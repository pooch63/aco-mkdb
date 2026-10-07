using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

function k33()
    g = BipartiteGraph{Nothing}()
    for u in 1:3, v in 1:3
        add_edge!(g, u, v, nothing)
    end
    fg, remap = compact_frozen(freeze(g))
    u = Dict(orig => i for (i, orig) in enumerate(remap.u_original))
    v = Dict(orig => i for (i, orig) in enumerate(remap.v_original))
    return g, fg, u, v, remap
end

@testset "eta exponent scales desirability" begin
    _, fg, u, _, _ = k33()
    pheromones = ColonyPheromones(fg, 1)
    node = DegreeNode(true, u[1], 4)
    expv = 0.5
    η = 4 + degree_u(fg, u[1]) * expv
    d1 = node_desirability(pheromones, fg, node, 1, expv; eta_exp=1.0)
    d3 = node_desirability(pheromones, fg, node, 1, expv; eta_exp=3.0)
    @test d1 ≈ η
    @test d3 ≈ η^3
    @test d3 ≈ d1^3
end

@testset "online deposit can be turned off" begin
    _, fg, _, _, _ = k33()
    ants = new_ants(fg, 1, 1)
    pheromones = ColonyPheromones(fg, 1)
    additions, invalids = advance_ants!(fg, pheromones, 1, ants, 0, 1, [1];
        neighbor_scope_limit=false, deposit_online=false)
    @test isempty(invalids)
    @test Subgraph.vertex_count(ants[1].explored) == 1
    @test sum(additions.species[1].U) + sum(additions.species[1].V) == 0
    @test sum(additions.shared.U) + sum(additions.shared.V) == 0
    @test isempty(ants[1].deposits)
end

@testset "only the iteration-best solution is reinforced" begin
    _, fg, u, v, _ = k33()
    ants = new_ants(fg, 2, 1)
    ants[1].explored = SubGraph(Set([u[1]]), Set([v[1]]))
    ants[1].last_visited = Node(true, u[1])
    ants[2].explored = SubGraph(Set([u[2], u[3]]), Set([v[2], v[3]]))
    ants[2].last_visited = Node(false, v[2])

    winner, kept = pick_iteration_best(ants, fg, 1, instance_fitness,
        nothing, SubGraph[], 2, 1)
    @test winner === ants[2]
    @test kept.U == Set([u[2], u[3]])
    @test kept.V == Set([v[2], v[3]])

    pheromones = ColonyPheromones(fg, 1)
    deposit_solution_pheromone!(pheromones, kept, winner.species, 1.0)
    @test pheromones.species[1].U[u[1]] == 0
    @test pheromones.species[1].V[v[1]] == 0
    @test pheromones.species[1].U[u[2]] == 1
    @test pheromones.species[1].U[u[3]] == 1
    @test pheromones.species[1].V[v[2]] == 1
    @test pheromones.species[1].V[v[3]] == 1
end

@testset "best-only colony deposits the returned solution at full strength" begin
    g, _, _, _, _ = k33()
    Random.seed!(1)
    sols, _, _, pheromones, remapping = aco(deepcopy(g), 1, 4, 1, 0.5, 0, 1, 1;
        parallelize=false, elite_seed=false, prefer_smaller_side=false,
        neighbor_scope_limit=false, reduction=ReductionMode.none,
        best_only_pheromone=true)
    sol = sols[1]
    @test Subgraph.vertex_count(sol) > 0
    for (ids, τ, original) in (
        (sol.U, pheromones.species[1].U, remapping.u_original),
        (sol.V, pheromones.species[1].V, remapping.v_original),
    )
        for (cid, orig) in enumerate(original)
            expected = orig in ids ? 1.0 : 0.0
            @test τ[cid] == expected
        end
    end
end

@testset "method knobs" begin
    m = make_method("aco"; ants=1, iterations=1)
    @test m.eta_exp == Float64(ETA_EXP)
    @test m.best_only_pheromone == false
    m2 = make_method_from_dict("aco", Dict("eta_exp" => 2, "best_only_pheromone" => true,
        "ants" => 1, "iterations" => 1))
    @test m2.eta_exp == 2.0
    @test m2.best_only_pheromone
    @test_throws ArgumentError make_method("aco"; best_only_pheromone=true, elite_pheromone=true)
    @test_throws ArgumentError make_method("aco"; eta_exp=Inf)
end
