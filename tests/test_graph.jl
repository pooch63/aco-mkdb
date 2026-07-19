using Test

include(joinpath(@__DIR__, "..", "graph.jl"))
using .Subgraph

# =============================================================================
# PHASE 1: BipartiteGraph mutation
# =============================================================================
@testset "BipartiteGraph mutation" begin
    g = BipartiteGraph{String}()

    # isolated vertices
    add_u!(g, 4)
    add_v!(g, 40)
    @test haskey(g.adjU, 4)
    @test haskey(g.adjV, 40)
    @test isempty(g.adjU[4])
    @test isempty(g.adjV[40])

    # edges (also implicitly adds endpoints)
    add_edge!(g, 1, 10, "a")
    add_edge!(g, 1, 20, "b")
    add_edge!(g, 2, 20, "c")
    add_edge!(g, 2, 30, "d")
    add_edge!(g, 3, 10, "e")
    add_edge!(g, 3, 30, "f")

    @test Set(keys(g.adjU)) == Set([1, 2, 3, 4])
    @test Set(keys(g.adjV)) == Set([10, 20, 30, 40])

    @test neighbors_u(g, 1) == Set([10, 20])
    @test neighbors_u(g, 2) == Set([20, 30])
    @test neighbors_u(g, 3) == Set([10, 30])
    @test neighbors_v(g, 10) == Set([1, 3])
    @test neighbors_v(g, 20) == Set([1, 2])
    @test neighbors_v(g, 30) == Set([2, 3])

    @test has_edge(g, 1, 10)
    @test !has_edge(g, 1, 30)
    @test edge_data(g, 1, 10) == "a"
    @test edge_data(g, 2, 30) == "d"

    # rem_edge!
    @test rem_edge!(g, 1, 10) == true
    @test !has_edge(g, 1, 10)
    @test !(10 in neighbors_u(g, 1))
    @test !(1 in neighbors_v(g, 10))
    @test rem_edge!(g, 1, 10) == false          # already gone
    @test rem_edge!(g, 999, 999) == false       # never existed

    # rem_v! cascades into adjU and edge_data
    @test rem_v!(g, 20) == true
    @test !haskey(g.adjV, 20)
    @test !(20 in neighbors_u(g, 1))
    @test !(20 in neighbors_u(g, 2))
    @test !haskey(g.edge_data, (1, 20))
    @test !haskey(g.edge_data, (2, 20))
    @test rem_v!(g, 20) == false                # already gone

    # rem_u! cascades into adjV and edge_data
    @test rem_u!(g, 3) == true
    @test !haskey(g.adjU, 3)
    @test !(3 in neighbors_v(g, 10))
    @test !(3 in neighbors_v(g, 30))
    @test !haskey(g.edge_data, (3, 10))
    @test !haskey(g.edge_data, (3, 30))
    @test rem_u!(g, 3) == false                 # already gone

    # remaining edges: (2,30,"d") only
    @test Set(keys(g.edge_data)) == Set([(2, 30)])
end

# =============================================================================
# FREEZE + point queries (whole graph, no SubGraph)
# =============================================================================
function build_base_graph()
    g = BipartiteGraph{String}()
    add_edge!(g, 1, 10, "a")
    add_edge!(g, 1, 20, "b")
    add_edge!(g, 2, 20, "c")
    add_edge!(g, 2, 30, "d")
    add_edge!(g, 3, 10, "e")
    add_edge!(g, 3, 30, "f")
    add_u!(g, 4)   # isolated U
    add_v!(g, 40)  # isolated V
    return g
end

@testset "freeze + whole-graph point queries" begin
    g = build_base_graph()
    fg = freeze(g)

    @test Set(fg.u_ids) == Set([1, 2, 3, 4])
    @test Set(fg.v_ids) == Set([10, 20, 30, 40])
    @test length(fg.u_offsets) == length(fg.u_ids) + 1
    @test length(fg.v_offsets) == length(fg.v_ids) + 1
    @test length(fg.v_adj) == 6
    @test length(fg.edge_data) == 6
    @test length(fg.u_adj) == 6
    @test length(fg.v_edge_data) == 6

    # forward direction is guaranteed sorted by neighbor id
    @test neighbors_u(fg, 1) == [10, 20]
    @test neighbors_u(fg, 2) == [20, 30]
    @test neighbors_u(fg, 3) == [10, 30]
    @test neighbors_u(fg, 4) == Int[]
    @test neighbors_u(fg, 999) == Int[]          # nonexistent id

    # reverse direction: compare as sets/sorted (order not doc-guaranteed)
    @test sort(neighbors_v(fg, 10)) == [1, 3]
    @test sort(neighbors_v(fg, 20)) == [1, 2]
    @test sort(neighbors_v(fg, 30)) == [2, 3]
    @test neighbors_v(fg, 40) == Int[]
    @test neighbors_v(fg, 999) == Int[]

    @test degree_u(fg, 1) == 2
    @test degree_u(fg, 2) == 2
    @test degree_u(fg, 3) == 2
    @test degree_u(fg, 4) == 0
    @test degree_u(fg, 999) == 0

    @test degree_v(fg, 10) == 2
    @test degree_v(fg, 20) == 2
    @test degree_v(fg, 30) == 2
    @test degree_v(fg, 40) == 0
    @test degree_v(fg, 999) == 0

    # edge_data consistency between forward and reverse CSR
    ui1 = fg.u_index[1]
    for k in neighbor_range_u(fg, ui1)
        vi = fg.v_adj[k]
        v_id = fg.v_ids[vi]
        d_fwd = fg.edge_data[k]
        # find the same edge from the reverse side and check the data matches
        vi_idx = fg.v_index[v_id]
        found = false
        for k2 in neighbor_range_v(fg, vi_idx)
            if fg.u_ids[fg.u_adj[k2]] == 1
                @test fg.v_edge_data[k2] == d_fwd
                found = true
            end
        end
        @test found
    end

    # backward-compat alias
    @test neighbor_range(fg, ui1) == neighbor_range_u(fg, ui1)
end

# =============================================================================
# SubGraph basics
# =============================================================================
@testset "SubGraph basics" begin
    sg = SubGraph(Set([1, 2, 3]), Set([10, 20]))
    @test subgraph_length(sg) == 5

    empty_sg = SubGraph(Set{Int}(), Set{Int}())
    @test subgraph_length(empty_sg) == 0
end

# =============================================================================
# Bigger graph for PHASE 2 / PHASE 3a (batch) tests
#
# Community A: U={1,2}, V={10,20}  -- complete K(2,2), 4 edges
# Community B: U={3,4}, V={30,40}  -- 3 edges, missing (4,40)
# Cross edge:  (2,30)              -- u in A, v in B -> must NOT be counted
#                                       in either subgraph
# Unassigned:  u=5 connects to v=10 (in A) and v=30 (in B), but u=5 itself
#              is not a member of either subgraph -- these edges must not be
#              counted in the batch sweep since u=5 has no subgraph.
# =============================================================================
function build_community_graph()
    g = BipartiteGraph{Float64}()
    add_edge!(g, 1, 10, 1.0)
    add_edge!(g, 1, 20, 2.0)
    add_edge!(g, 2, 10, 3.0)
    add_edge!(g, 2, 20, 4.0)
    add_edge!(g, 3, 30, 5.0)
    add_edge!(g, 3, 40, 6.0)
    add_edge!(g, 4, 30, 7.0)
    add_edge!(g, 2, 30, 8.0)   # cross edge
    add_edge!(g, 5, 10, 9.0)   # u=5 unassigned
    add_edge!(g, 5, 30, 10.0)  # u=5 unassigned
    return g
end

@testset "split_fg / subgraph_edge_counts / accumulate_edges!" begin
    g = build_community_graph()
    fg = freeze(g)

    sgA = SubGraph(Set([1, 2]), Set([10, 20]))
    sgB = SubGraph(Set([3, 4]), Set([30, 40]))
    subgraphs = [sgA, sgB]

    sub_edges = split_fg(fg, subgraphs)
    @test length(sub_edges) == 2
    @test Set(sub_edges[1]) == Set([(1, 10), (1, 20), (2, 10), (2, 20)])
    @test Set(sub_edges[2]) == Set([(3, 30), (3, 40), (4, 30)])

    counts = subgraph_edge_counts(fg, subgraphs)
    @test counts == [4, 3]

    # accumulate_edges!: sum of edge data per subgraph
    sums = zeros(Float64, length(subgraphs))
    accumulate_edges!(sums, fg, subgraphs, (dest, s, u, v, data) -> dest[s] += data)
    @test sums[1] == 1.0 + 2.0 + 3.0 + 4.0
    @test sums[2] == 5.0 + 6.0 + 7.0

    # accumulate_edges!: collect (u,v) pairs per subgraph, should match split_fg
    collected = [Tuple{Int,Int}[] for _ in subgraphs]
    accumulate_edges!(collected, fg, subgraphs, (dest, s, u, v, data) -> push!(dest[s], (u, v)))
    @test Set(collected[1]) == Set(sub_edges[1])
    @test Set(collected[2]) == Set(sub_edges[2])

    # vertex ids that don't exist in fg are silently ignored
    sg_missing = SubGraph(Set([1, 999]), Set([10, 888]))
    sub_edges2 = split_fg(fg, [sg_missing])
    @test Set(sub_edges2[1]) == Set([(1, 10)])

    # empty subgraphs list
    @test split_fg(fg, SubGraph[]) == []
    @test subgraph_edge_counts(fg, SubGraph[]) == Int[]
end

@testset "overlapping subgraphs: later subgraph wins for shared vertex" begin
    g = build_community_graph()
    fg = freeze(g)

    # v=10 and v=30 both appear in sg1 and sg2; sg2 comes later in the list
    # so _dense_assignment resolves the conflict in favor of sg2.
    sg1 = SubGraph(Set([1, 2]), Set([10, 20, 30]))
    sg2 = SubGraph(Set([3]), Set([10, 30]))

    counts = subgraph_edge_counts(fg, [sg1, sg2])
    # sg1: u=1(->1) nbrs 10(->2,no),20(->1,yes) => (1,20)
    #      u=2(->1) nbrs 10(->2,no),20(->1,yes),30(->2,no) => (2,20)
    # sg2: u=3(->2) nbrs 30(->2,yes),40(unassigned,no) => (3,30)
    @test counts == [2, 1]
end

# =============================================================================
# PHASE 3b: POINT queries restricted to a SubGraph
# =============================================================================
@testset "point queries in subgraph" begin
    g = build_community_graph()
    fg = freeze(g)

    sgA = SubGraph(Set([1, 2]), Set([10, 20]))
    sgB = SubGraph(Set([3, 4]), Set([30, 40]))

    # u=2 belongs to A but also has a cross-edge into B's v=30
    @test neighbors_u_in_subgraph(fg, 2, sgA) == [10, 20]
    @test neighbors_u_in_subgraph(fg, 2, sgB) == [30]
    @test neighbors_u_in_subgraph(fg, 999, sgA) == Int[]   # nonexistent u

    @test degree_in_subgraph_u(fg, 2, sgA) == 2
    @test degree_in_subgraph_u(fg, 2, sgB) == 1
    @test degree_in_subgraph_u(fg, 999, sgA) == 0

    @test nondegree_in_subgraph_u(fg, 2, sgA) == length(sgA.V) - 2   # 0
    @test nondegree_in_subgraph_u(fg, 2, sgB) == length(sgB.V) - 1   # 1
    @test nondegree_in_subgraph_u(fg, 999, sgA) == length(sgA.V)     # 2, no edges at all
    @test nondegree_in_subgraph(fg, true, 2, sgB) == nondegree_in_subgraph_u(fg, 2, sgB)

    # v=30 is adjacent to u's {2,3,4,5}; only 2 is in sgA.U, {3,4} in sgB.U
    @test Set(neighbors_v_in_subgraph(fg, 30, sgA)) == Set([2])
    @test Set(neighbors_v_in_subgraph(fg, 30, sgB)) == Set([3, 4])
    @test neighbors_v_in_subgraph(fg, 999, sgA) == Int[]

    @test degree_in_subgraph_v(fg, 30, sgA) == 1
    @test degree_in_subgraph_v(fg, 30, sgB) == 2
    @test degree_in_subgraph_v(fg, 999, sgA) == 0

    @test nondegree_in_subgraph_v(fg, 30, sgA) == length(sgA.U) - 1   # 1
    @test nondegree_in_subgraph_v(fg, 30, sgB) == length(sgB.U) - 2   # 0
    @test nondegree_in_subgraph(fg, false, 30, sgA) == nondegree_in_subgraph_v(fg, 30, sgA)
end

# =============================================================================
# Subgraph submodule: node add/remove, set algebra, edge_count, missing_edges,
# nonneighbors_in_subgraph
# =============================================================================
@testset "Subgraph module: node/set operations" begin
    sg = SubGraph(Set([1, 2]), Set([10]))

    Subgraph.add_node!(sg, true, 3)
    @test 3 in sg.U
    Subgraph.add_node!(sg, false, 20)
    @test 20 in sg.V

    Subgraph.remove_node!(sg, true, 3)
    @test !(3 in sg.U)
    Subgraph.remove_node!(sg, false, 20)
    @test !(20 in sg.V)

    a = SubGraph(Set([1, 2]), Set([10]))
    b = SubGraph(Set([2, 3]), Set([10, 20]))

    combined = Subgraph.add!(a, b)   # mutates a, returns a
    @test combined === a
    @test a.U == Set([1, 2, 3])
    @test a.V == Set([10, 20])

    c = SubGraph(Set([1, 2, 3]), Set([10, 20, 30]))
    d = SubGraph(Set([2, 3]), Set([20]))
    diff = Subgraph.minus(c, d)
    @test diff.U == Set([1])
    @test diff.V == Set([10, 30])
    # minus is non-mutating
    @test c.U == Set([1, 2, 3])
    @test c.V == Set([10, 20, 30])

    e = SubGraph(Set([1, 2, 3]), Set([10, 20, 30]))
    f = SubGraph(Set([2, 3]), Set([20]))
    ediff = Subgraph.minus!(e, f)
    @test ediff === e
    @test e.U == Set([1])
    @test e.V == Set([10, 30])
end

@testset "Subgraph module: edge_count / missing_edges" begin
    g = build_community_graph()
    fg = freeze(g)

    sgA = SubGraph(Set([1, 2]), Set([10, 20]))   # complete K(2,2) -> 4 edges
    @test Subgraph.edge_count(fg, sgA) == 4
    @test Subgraph.missing_edges(fg, sgA) == length(sgA.U) * length(sgA.V) - 4  # 0

    sgB = SubGraph(Set([3, 4]), Set([30, 40]))   # 3 of 4 possible edges
    @test Subgraph.edge_count(fg, sgB) == 3
    @test Subgraph.missing_edges(fg, sgB) == length(sgB.U) * length(sgB.V) - 3  # 1

    # edge_count matches subgraph_edge_counts for the same (disjoint) subgraphs
    @test [Subgraph.edge_count(fg, sgA), Subgraph.edge_count(fg, sgB)] ==
          subgraph_edge_counts(fg, [sgA, sgB])

    # empty subgraph
    empty_sg = SubGraph(Set{Int}(), Set{Int}())
    @test Subgraph.edge_count(fg, empty_sg) == 0
    @test Subgraph.missing_edges(fg, empty_sg) == 0

    # subgraph containing a nonexistent u id is ignored, not an error
    sg_missing_u = SubGraph(Set([1, 2, 999]), Set([10, 20]))
    @test Subgraph.edge_count(fg, sg_missing_u) == 4
end

@testset "Subgraph module: nonneighbors_in_subgraph" begin
    g = build_community_graph()
    fg = freeze(g)

    sgA = SubGraph(Set([1, 2]), Set([10, 20]))
    sgB = SubGraph(Set([3, 4]), Set([30, 40]))

    # u=2 (is_u=true): neighbors are {10,20,30}
    @test Subgraph.nonneighbors_in_subgraph(fg, true, 2, sgA) == Set{Int}()       # both 10,20 covered
    @test Subgraph.nonneighbors_in_subgraph(fg, true, 2, sgB) == Set([40])        # 30 covered, 40 not
    @test Subgraph.nonneighbors_in_subgraph(fg, true, 4, sgB) == Set([40])        # u=4 only touches 30

    # v=30 (is_u=false): neighbors (U side) are {2,3,4,5}
    @test Subgraph.nonneighbors_in_subgraph(fg, false, 30, sgA) == Set([1])       # only 2 covered, 1 not
    @test Subgraph.nonneighbors_in_subgraph(fg, false, 30, sgB) == Set{Int}()     # 3,4 both covered

    # node not present in fg at all -> everything on the opposite side is "non-adjacent"
    @test Subgraph.nonneighbors_in_subgraph(fg, true, 999, sgA) == sgA.V
    @test Subgraph.nonneighbors_in_subgraph(fg, false, 999, sgB) == sgB.U
end

println("All tests completed.")