using Test
using Random

const SRC = joinpath(@__DIR__, "..", "src")
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))

"""The worked example: L = {a,b,c}, costs 0, 1, 2, best R = {x,v}, 5 edges."""
function user_example()
    g = BipartiteGraph{Nothing}()
    # a,b,c = 1,2,3 and x,v,w = 1,2,3.
    for (u, v) in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1))
        add_edge!(g, u, v, nothing)
    end
    return g
end

function complete_biclique(n::Int=6)
    g = BipartiteGraph{Nothing}()
    for u in 1:n, v in 1:n
        add_edge!(g, u, v, nothing)
    end
    return g
end

"""5×5 biclique plus two cheap defects and one isolated right vertex."""
function repair_graph()
    g = BipartiteGraph{Nothing}()
    for u in 1:5, v in 1:5
        add_edge!(g, u, v, nothing)
    end
    for u in 2:5
        add_edge!(g, u, 6, nothing)
    end
    for u in (1, 3, 4, 5)
        add_edge!(g, u, 7, nothing)
    end
    add_v!(g, 8)
    return g
end

function random_bipartite(nU::Int, nV::Int, p::Float64)
    g = BipartiteGraph{Nothing}()
    for u in 1:nU
        add_u!(g, u)
    end
    for v in 1:nV
        add_v!(g, v)
    end
    for u in 1:nU, v in 1:nV
        rand() < p && add_edge!(g, u, v, nothing)
    end
    return g
end

function id_map(originals::Vector{Int})
    return Dict(orig => i for (i, orig) in enumerate(originals))
end

@testset "exact oracle matches the worked example" begin
    g = user_example()
    fg, remap = compact_frozen(freeze(g))
    u_of = id_map(remap.u_original)
    v_of = id_map(remap.v_original)
    st = OracleState(fg, true)
    for orig in 1:3
        add_left!(st, fg, u_of[orig])
    end
    @test oracle_state_consistent(st)
    edges, nR = oracle_score(st, 2)
    @test edges == 5
    @test nR == 2
    sg = oracle_solution(fg, st, 2)
    @test Subgraph.edge_count(fg, sg) == 5
    @test Subgraph.missing_edges(fg, sg) <= 2
    @test v_of[1] in sg.V
    @test v_of[2] in sg.V
    @test !(v_of[3] in sg.V)
    @test oracle_upper_bound(st, 2) >= edges
end

@testset "incremental add and remove match a full recompute" begin
    Random.seed!(1)
    g = random_bipartite(8, 11, 0.35)
    fg, _ = compact_frozen(freeze(g))
    st = OracleState(fg, true)
    for _ in 1:60
        free = [u for u in 1:st.nSame if !st.inL[u]]
        if !isempty(free) && (st.nL == 0 || rand() < 0.65)
            add_left!(st, fg, rand(free))
        elseif st.nL > 0
            remove_left!(st, fg)
        end
        @test oracle_state_consistent(st)
        edges, nR = oracle_score(st, 2)
        @test oracle_upper_bound(st, 2) >= edges
        @test oracle_alive(st, 2) >= nR
        if st.nL > 0
            sg = oracle_solution(fg, st, 2)
            @test Subgraph.edge_count(fg, sg) == edges
            @test Subgraph.missing_edges(fg, sg) <= 2
            @test length(sg.V) == nR
            packed = pack_opposite(fg, true, st.members, 2)
            packed_sg = SubGraph(Set{Int}(st.members), Set{Int}(packed))
            @test Subgraph.edge_count(fg, packed_sg) == edges
        end
    end
    while st.nL > 0
        remove_left!(st, fg)
        @test oracle_state_consistent(st)
    end
    @test oracle_score(st, 2) == (0, 0)
end

@testset "adding a left vertex never revives a dead right vertex" begin
    Random.seed!(2)
    g = random_bipartite(7, 9, 0.4)
    fg, _ = compact_frozen(freeze(g))
    st = OracleState(fg, true)
    k = 2
    for u in 1:min(4, st.nSame)
        add_left!(st, fg, u)
    end
    dead = [v for v in 1:st.nR if st.nL - st.hits[v] > k]
    @test !isempty(dead)
    for u in (st.nSame - 1):st.nSame
        st.inL[u] && continue
        add_left!(st, fg, u)
        for v in dead
            @test st.nL - st.hits[v] > k
        end
    end
end

@testset "greedy oracle ACO recovers the repaired biclique" begin
    g = repair_graph()
    Random.seed!(1)
    r = oracle_aco(deepcopy(g), 1, 8, 3, 0.95, 2, 3;
        discrepancy=0, decompose=false, pool=8, seed_size=2,
        reduction=ReductionMode.none, verbose=false)
    @test r.search_is_u
    @test r.edges == 33
    sc = score_solution(freeze(g), r.best, 2, 3)
    @test sc.valid
    @test sc.edges == 33
    @test sc.nU == 5
    @test sc.nV == 7
end

@testset "limited discrepancy stays valid and matches the oracle" begin
    g = repair_graph()
    Random.seed!(1)
    r = oracle_aco(deepcopy(g), 1, 4, 2, 0.95, 2, 3;
        discrepancy=1, decompose=false, branch=3, pool=8, max_nodes=500,
        reduction=ReductionMode.none, verbose=false)
    sc = score_solution(freeze(g), r.best, 2, 3)
    @test sc.valid
    @test sc.edges == 33

    g6 = complete_biclique()
    Random.seed!(1)
    c = oracle_aco(deepcopy(g6), 1, 2, 1, 0.95, 1, 3;
        discrepancy=1, decompose=false, branch=2, pool=8,
        reduction=ReductionMode.none, verbose=false)
    sc6 = score_solution(freeze(g6), c.best, 1, 3)
    @test sc6.valid
    @test sc6.edges == 36
end

@testset "seed decomposition prunes later seeds after a strong find" begin
    g = complete_biclique()
    Random.seed!(1)
    r = oracle_aco(deepcopy(g), 1, 2, 1, 0.95, 1, 3;
        discrepancy=1, decompose=true, n_seeds=4, branch=2, pool=8,
        reduction=ReductionMode.none, verbose=false)
    sc = score_solution(freeze(g), r.best, 1, 3)
    @test sc.valid
    @test sc.edges == 36
    @test r.seeds_run >= 1
    @test r.seeds_pruned >= 1
    @test r.seeds_run + r.seeds_pruned == 4
end

@testset "method wrapper is θ-feasible and k-valid" begin
    g = complete_biclique()
    Random.seed!(1)
    result = run_method!(OracleACOMethod(; ants=3, iterations=1, discrepancy=0,
            decompose=false, parallelize=false),
        deepcopy(g), 1, 3; reduction=ReductionMode.none)
    @test result.name == "oracle_aco"
    @test result.meta["search_side"] == "U"
    @test result.meta["decompose"] == false
    sc = score_result(freeze(g), result, 1, 3)
    @test sc.valid
    @test sc.edges == 36
end
