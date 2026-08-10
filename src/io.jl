#=
=================================================================================
Graph CSV I/O
=================================================================================
=#

using CSV

const __IO_JL__ = true

"""
    load_bipartite_graph(filepath; max_lines=nothing) -> (BipartiteGraph{Nothing}, edge_count)

Reads a `u,v[,...]` CSV (with header) and builds a mutable bipartite graph
where U = u and V = v. Extra columns beyond `u` and `v` are ignored.

Uses `CSV.File`, which memory-maps local files (`buffer_in_memory=false`) and
applies multithreaded chunked parsing when the input is large enough.
"""
function load_bipartite_graph(filepath::String; max_lines::Union{Int,Nothing}=nothing)
    g = BipartiteGraph{Nothing}()

    rows = CSV.File(
        filepath;
        header = 1,
        select = [:u, :v],
        types = Dict(:u => Int, :v => Int),
        buffer_in_memory = false,
        limit = max_lines,
    )

    for row in rows
        add_edge!(g, row.u, row.v, nothing)
    end

    edge_count = length(rows)
    println("Graph edges: $(edge_count)")
    return g, edge_count
end
