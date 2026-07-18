include("graph.jl")
include("opponent.jl")

"""
    load_bipartite_graph(filepath::String) -> FrozenBipartite{Int}

Reads a `user_id,item_id,timestamp` CSV (with header) and builds a frozen
bipartite graph where U = user_id, V = item_id, and edge data = timestamp.
"""
function load_bipartite_graph(filepath::String; max_lines::Union{Int,Nothing}=nothing)
    g = BipartiteGraph{Int}()
    open(filepath, "r") do io
        readline(io)  # skip header line
        count = 0
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ",")
            u  = parse(Int, parts[1])
            v  = parse(Int, parts[2])
            ts = parse(Int, parts[3])
            add_edge!(g, u, v, ts)
            count += 1
            max_lines !== nothing && count >= max_lines && break
        end
    end
    return freeze(g)
end


# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

# Example: Run your deep recursive function with a 2 GB stack
with_stacksize(2_000_000_000) do
    g = load_bipartite_graph("data/indexed_interactions.csv")
    D = SubGraph(Set(), Set())
    # D = branch_binary(SubGraph(Set(), Set()), SubGraph(Set(u for u in g.u_ids), Set(v for v in g.v_ids)), g, D, 1, 3)

D = branch_pivot(SubGraph(Set(), Set()), SubGraph(Set(u for u in g.u_ids), Set(v for v in g.v_ids)), g, D, 1, 2)
end

@show D