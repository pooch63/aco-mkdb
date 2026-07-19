#=
=================================================================================
Graph Loader
=================================================================================
This script loads a saved indexed graph from disk and runs the graph analysis
workflow on it. It accepts either:

  1. No argument, in which case it loads data/indexed_interactions.csv
  2. A dataset name, in which case it looks for data/<dataset_name>/indexed_interactions.csv
  3. An explicit file path to a CSV graph file

Prerequisites:
  Ensure you have Julia and the required packages installed.

How to Run from the Command Line:
  Format:
    julia load.jl [dataset_name_or_path]

  Examples:
    julia load.jl
    julia load.jl Grocery_and_Gourmet_Food
    julia load.jl /path/to/indexed_interactions.csv
=================================================================================
=#

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

function resolve_graph_path(dataset_name::Union{String,Nothing}=nothing)
    if dataset_name === nothing || isempty(dataset_name)
        return "data/indexed_interactions.csv"
    end

    if isfile(dataset_name)
        return dataset_name
    end

    return joinpath("data", dataset_name, "indexed_interactions.csv")
end

# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

function main()
    dataset_name = length(ARGS) >= 1 ? ARGS[1] : nothing
    graph_path = resolve_graph_path(dataset_name)

    if !isfile(graph_path)
        println(stderr, "Error: Could not find a saved graph at '$graph_path'.")
        exit(1)
    end

    println("Loading graph from: $graph_path")

    with_stacksize(2_000_000_000) do
        g = load_bipartite_graph(graph_path)
        D = SubGraph(Set(), Set())

        D = branch_binary(SubGraph(Set(), Set()), SubGraph(Set(u for u in g.u_ids), Set(v for v in g.v_ids)), g, D, 4, 10)
        # D = branch_pivot(SubGraph(Set(), Set()), SubGraph(Set(u for u in g.u_ids), Set(v for v in g.v_ids)), g, D, 1, 2)

        @show D
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end