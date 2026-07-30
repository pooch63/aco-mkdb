#=
=================================================================================
Graph CSV I/O
=================================================================================
=#

using CSV

const __IO_JL__ = true

"""
    load_bipartite_graph(filepath; max_lines=nothing) -> (BipartiteGraph{Int}, edge_count)

Reads a `user_id,item_id,timestamp` CSV (with header) and builds a mutable
bipartite graph where U = user_id, V = item_id, and edge data = timestamp.

Uses `CSV.File`, which memory-maps local files (`buffer_in_memory=false`) and
applies multithreaded chunked parsing when the input is large enough.
"""
function load_bipartite_graph(filepath::String; max_lines::Union{Int,Nothing}=nothing)
    g = BipartiteGraph{Int}()

    rows = CSV.File(
        filepath;
        header = 1,
        select = [:user_id, :item_id, :timestamp],
        types = Dict(:user_id => Int, :item_id => Int, :timestamp => Int),
        buffer_in_memory = false,
        limit = max_lines,
    )

    for row in rows
        add_edge!(g, row.user_id, row.item_id, row.timestamp)
    end

    edge_count = length(rows)
    println("Graph edges: $(edge_count)")
    return g, edge_count
end
