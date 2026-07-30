#=
=================================================================================
Graph CSV I/O
=================================================================================
=#

const __IO_JL__ = true

"""
    load_bipartite_graph(filepath; max_lines=nothing) -> (BipartiteGraph{Int}, edge_count)

Reads a `user_id,item_id,timestamp` CSV (with header) and builds a mutable
bipartite graph where U = user_id, V = item_id, and edge data = timestamp.
"""
function load_bipartite_graph(filepath::String; max_lines::Union{Int,Nothing}=nothing)
    g = BipartiteGraph{Int}()
    edge_count = 0
    open(filepath, "r") do io
        readline(io)  # skip header line
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ",")
            u  = parse(Int, parts[1])
            v  = parse(Int, parts[2])
            ts = parse(Int, parts[3])
            add_edge!(g, u, v, ts)
            edge_count += 1
            max_lines !== nothing && edge_count >= max_lines && break
        end
        println("Graph edges: $(edge_count)")
    end
    return g, edge_count
end
