using Test

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

# L = {a, b, c} = {1, 2, 3}, R = {1, 2, 3, 4}.
# N(a) = {1, 2}, N(b) = {1, 2, 3}, N(c) = {4}. a ≼ b; seeds are b and c.
function example_graph()
    g = BipartiteGraph{Nothing}()
    add_edge!(g, 1, 1, nothing)
    add_edge!(g, 1, 2, nothing)
    add_edge!(g, 2, 1, nothing)
    add_edge!(g, 2, 2, nothing)
    add_edge!(g, 2, 3, nothing)
    add_edge!(g, 3, 4, nothing)
    return freeze(g)
end

@testset "maximal neighborhood seeds" begin
    fg = example_graph()
    # Degree order and an order that is not degree-sorted both drop a.
    @test maximal_neighborhood_seeds(fg, true, [2, 1, 3]) == [2, 3]
    @test maximal_neighborhood_seeds(fg, true, [1, 2, 3]) == [2, 3]
    @test maximal_neighborhood_seeds(fg, true, get_degree_order(fg, true, true)) == [2, 3]

    # Identical neighborhoods: keep the earliest vertex in the given order.
    g = BipartiteGraph{Nothing}()
    for u in (1, 2)
        add_edge!(g, u, 10, nothing)
        add_edge!(g, u, 20, nothing)
    end
    add_edge!(g, 3, 30, nothing)
    same = freeze(g)
    @test maximal_neighborhood_seeds(same, true, [1, 2, 3]) == [1, 3]
    @test maximal_neighborhood_seeds(same, true, [2, 1, 3]) == [2, 3]

    # Incomparable neighborhoods are both seeds.
    g = BipartiteGraph{Nothing}()
    add_edge!(g, 1, 10, nothing)
    add_edge!(g, 1, 20, nothing)
    add_edge!(g, 2, 20, nothing)
    add_edge!(g, 2, 30, nothing)
    inc = freeze(g)
    @test maximal_neighborhood_seeds(inc, true, [1, 2]) == [1, 2]

    # Empty neighborhoods are one seed; a non-empty neighborhood dominates them.
    g = BipartiteGraph{Nothing}()
    add_u!(g, 1)
    add_u!(g, 2)
    add_v!(g, 10)
    empty_side = freeze(g)
    @test maximal_neighborhood_seeds(empty_side, true, [1, 2]) == [1]
    add_edge!(g, 2, 10, nothing)
    mixed = freeze(g)
    @test maximal_neighborhood_seeds(mixed, true, [1, 2]) == [2]

    # Same inclusion rule on the V side.
    g = BipartiteGraph{Nothing}()
    add_edge!(g, 10, 1, nothing)
    add_edge!(g, 20, 1, nothing)
    add_edge!(g, 10, 2, nothing)
    add_edge!(g, 20, 2, nothing)
    add_edge!(g, 30, 2, nothing)
    add_edge!(g, 40, 3, nothing)
    vg = freeze(g)
    @test maximal_neighborhood_seeds(vg, false, [1, 2, 3]) == [2, 3]
end

@testset "split ACO uses reduced seeds" begin
    fg = example_graph()
    reduced = split_aco(fg, 0, 1, 1, 1, 1, 0.95, 1;
        local_reduction=false, reduce_seeds=true, parallelize=false)
    @test reduced.n_side == 3
    @test reduced.n_roots == 2
    @test reduced.root_is_u

    full = split_aco(fg, 0, 1, 1, 1, 1, 0.95, 1;
        local_reduction=false, reduce_seeds=false, parallelize=false)
    @test full.n_side == 3
    @test full.n_roots == 3

    m = make_method("split_aco"; reduce_seeds=true, ants=1, iterations=1)
    @test m.reduce_seeds
    g = BipartiteGraph{Nothing}()
    add_edge!(g, 1, 1, nothing)
    add_edge!(g, 1, 2, nothing)
    add_edge!(g, 2, 1, nothing)
    add_edge!(g, 2, 2, nothing)
    add_edge!(g, 2, 3, nothing)
    add_edge!(g, 3, 4, nothing)
    r = run_method!(m, g, 0, 1; reduction=ReductionMode.none)
    @test r.meta["roots"] == 2
    @test r.meta["root_candidates"] == 3
    @test r.meta["reduce_seeds"]
    @test r.meta["n_seeds"] == 50
end

# K_{6,6} on ids 1..6 plus pendant edges 7..16. Biclique edges have S_C = 1;
# pendants have S_C = 0. The smaller side is U (|U| = |V|).
function sc_graph()
    g = BipartiteGraph{Nothing}()
    for u in 1:6, v in 1:6
        add_edge!(g, u, v, nothing)
    end
    for i in 7:16
        add_edge!(g, i, i, nothing)
    end
    return freeze(g)
end

@testset "top S_C seeds" begin
    fg = sc_graph()
    order = get_degree_order(fg, true, true)
    # Identical biclique neighborhoods collapse to one maximal seed; pendants stay.
    maximal = maximal_neighborhood_seeds(fg, true, order)
    @test length(maximal) == 11
    @test count(id -> id <= 6, maximal) == 1

    scores = incident_sc(fg, true; θ=5)
    biclique_id = only(filter(id -> id <= 6, maximal))
    @test scores[fg.u_index[biclique_id]] > 0
    for id in maximal
        id <= 6 && continue
        @test scores[fg.u_index[id]] == 0
    end

    top = top_sc_seeds(fg, true, maximal, 1; θ=5)
    @test top == [biclique_id]

    # Without the inclusion filter, the six biclique vertices outrank the pendants.
    top3 = top_sc_seeds(fg, true, order, 3; θ=5)
    @test length(top3) == 3
    @test all(id -> id <= 6, top3)

    # A budget larger than the candidate list keeps every candidate, in order.
    @test top_sc_seeds(fg, true, maximal, 50; θ=5) == maximal
    @test top_sc_seeds(fg, true, maximal, 0; θ=5) == maximal

    capped = split_aco(fg, 0, 1, 1, 1, 1, 0.95, 1;
        local_reduction=false, reduce_seeds=true, n_seeds=1, parallelize=false)
    @test capped.n_roots == 1
    @test capped.best_root == biclique_id
    @test capped.n_side == 16
end

function complete_biclique(n::Int)
    g = BipartiteGraph{Nothing}()
    for u in 1:n, v in 1:n
        add_edge!(g, u, v, nothing)
    end
    return g
end

@testset "upper bound halts an ant and drops its pheromone" begin
    g = complete_biclique(3)
    fg, _ = compact_frozen(freeze(g))
    k, θ = 0, 1

    # Empty S on K_{3,3} has edge upper bound 9, so an incumbent of 9 prunes immediately.
    ants = new_ants(fg, 1, 1)
    target = BoundTarget(true, 9, 1.0)
    @test ant_bound_prunes(fg, ants[1], k, θ, target)
    pheromones = ColonyPheromones(fg, 1)
    additions, invalids = advance_ants!(fg, pheromones, 1, ants, k, θ, [1];
        prune_by_bound=true, bound_target=target, neighbor_scope_limit=false)
    @test invalids == [1]
    @test ants[1].bound_pruned
    @test isempty(ants[1].deposits)
    @test sum(additions.shared.U) + sum(additions.shared.V) == 0
    @test sum(additions.species[1].U) + sum(additions.species[1].V) == 0

    # A weaker incumbent lets the ant deposit once; raising the bar then erases that deposit.
    ants = new_ants(fg, 1, 1)
    pheromones = ColonyPheromones(fg, 1)
    weak = BoundTarget(false, 0, 0.0)
    additions, invalids = advance_ants!(fg, pheromones, 1, ants, k, θ, [1];
        prune_by_bound=true, bound_target=weak, neighbor_scope_limit=false)
    @test isempty(invalids)
    @test length(ants[1].deposits) == 1
    merge_pheromones!(pheromones, additions)
    @test sum(pheromones.species[1].U) + sum(pheromones.species[1].V) == 1
    @test sum(pheromones.shared.U) + sum(pheromones.shared.V) == 6 + 0.25

    strong = BoundTarget(true, 9, 1.0)
    additions, invalids = advance_ants!(fg, pheromones, 1, ants, k, θ, [1];
        prune_by_bound=true, bound_target=strong, neighbor_scope_limit=false)
    @test invalids == [1]
    @test ants[1].bound_pruned
    @test length(ants[1].deposits) == 1
    @test sum(additions.shared.U) + sum(additions.shared.V) == 0
    rollback_ant_pheromone!(pheromones, ants[1])
    @test isempty(ants[1].deposits)
    @test pheromones.species[1].U == zeros(3)
    @test pheromones.species[1].V == zeros(3)
    @test pheromones.shared.U == ones(3)
    @test pheromones.shared.V == ones(3)

    # Colony still returns the full biclique; the second iteration cannot beat it,
    # so that ant deposits nothing and evaporation is the only change to the trail.
    g2 = complete_biclique(4)
    sols, _, _, pheromones, _ = aco(g2, 1, 1, 2, 0.95, 0, 2, 1;
        prune_by_bound=true, parallelize=false, elite_seed=false,
        neighbor_scope_limit=false, prefer_smaller_side=false,
        reduction=ReductionMode.none)
    @test Subgraph.edge_count(freeze(g2), sols[1]) == 16
    kept = 0.95^2
    @test all(τ -> τ ≈ kept, pheromones.species[1].U)
    @test all(τ -> τ ≈ kept, pheromones.species[1].V)
    @test all(τ -> τ ≈ (1 + 0.25) * kept, pheromones.shared.U)
    @test all(τ -> τ ≈ (1 + 0.25) * kept, pheromones.shared.V)
end

# K_{5,5} on high ids, plus a pendant that CNN (θ − k = 3) must drop.
# The pendant sits in the root's 3-hop, so reduction has to run on that
# neighborhood — not on the parent id span — before ACO.
@testset "3-hop reduction before ACO" begin
    g = BipartiteGraph{Nothing}()
    for u in 10:14, v in 20:24
        add_edge!(g, u, v, nothing)
    end
    add_edge!(g, 100, 20, nothing)
    fg = freeze(g)

    same, opp = three_hop_neighborhood(fg, 20, false)
    @test 100 in opp
    lg, remap = compact_induced_bipartite(fg, opp, same)
    @test minimum(keys(lg.adjU)) == 1
    @test maximum(keys(lg.adjU)) == length(lg.adjU)
    @test maximum(keys(lg.adjV)) == length(lg.adjV)
    pendant_c = findfirst(==(100), remap.u_original)
    root_c = findfirst(==(20), remap.v_original)
    @test pendant_c !== nothing && root_c !== nothing
    lfg = apply_graph_reductions!(lg, 2, 5, nothing, nothing, true, ReductionMode.simple)
    @test !haskey(lfg.u_index, pendant_c)
    @test haskey(lfg.v_index, root_c)
    @test length(lfg.u_ids) == 5
    @test length(lfg.v_ids) == 5

    r = split_aco(fg, 2, 5, 1, 1, 1, 0.95, 1;
        local_reduction=true, reduce_seeds=true, n_seeds=0, parallelize=false,
        one_colony=false)
    @test 100 ∉ r.best.U
    @test !isempty(r.best.U)
    @test all(u -> 10 <= u <= 14, r.best.U)
    @test all(v -> 20 <= v <= 24, r.best.V)
    @test !r.one_colony
end

@testset "one colony round-robins seeds" begin
    m = make_method("split_aco")
    @test m.one_colony
    off = make_method("split_aco"; one_colony=false)
    @test !off.one_colony

    fg = example_graph()
    per_root = split_aco(fg, 0, 1, 1, 1, 1, 0.95, 1;
        local_reduction=false, reduce_seeds=true, parallelize=false, one_colony=false)
    @test per_root.n_roots == 2
    @test per_root.n_ran == 2
    @test !per_root.one_colony

    shared = split_aco(fg, 0, 1, 1, 1, 1, 0.95, 1;
        local_reduction=false, reduce_seeds=true, parallelize=false, one_colony=true)
    @test shared.one_colony
    @test shared.n_roots == 2
    @test shared.n_ran == 1
    @test shared.best_root !== nothing
    @test root_in_solution(shared.best_root, shared.root_is_u, shared.best)

    # Iteration 1 plants the front of the list; iteration 2 walks forward.
    # Each seed already occupies both sides, so k=0 cannot jump components.
    # Component B is the K_{2,2}, reached only on the second iteration.
    g = BipartiteGraph{Nothing}()
    add_edge!(g, 1, 1, nothing)
    for u in 2:3, v in 2:3
        add_edge!(g, u, v, nothing)
    end
    seeds = [
        SubGraph(Set([1]), Set([1])),
        SubGraph(Set([2]), Set([2])),
    ]
    sols, _, _, _, _ = aco(g, 1, 1, 2, 0.95, 0, 1, 1;
        ant_seeds=seeds, reduction=ReductionMode.none, parallelize=false,
        elite_seed=false, neighbor_scope_limit=false, prefer_smaller_side=false,
        prune_by_bound=false)
    @test 2 in sols[1].U
    @test 1 ∉ sols[1].U
    @test Subgraph.edge_count(freeze(g), sols[1]) == 4

    fg2 = freeze(g)
    ants = new_ants(fg2, 1, 1)
    seed_ants_round_robin!(ants, fg2, 0, seeds, 1)
    @test ants[1].explored.U == Set([1])
    seed_ants_round_robin!(ants, fg2, 0, seeds, 2)
    @test ants[1].explored.U == Set([2])
    @test 2 in ants[1].explored.V
end
