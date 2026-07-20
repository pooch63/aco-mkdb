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

using Profile
using ProfileCanvas

include("graph.jl")
include("opponent.jl")
include("reduction.jl")

global const DEBUG = true

"""
    load_bipartite_graph(filepath::String) -> BipartiteGraph{Int}

Reads a `user_id,item_id,timestamp` CSV (with header) and builds a mutable
bipartite graph where U = user_id, V = item_id, and edge data = timestamp.
The graph is reduced and then frozen inside the search pipeline.
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
    return g
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

function parse_dataset_and_mode()
    dataset_name = nothing
    mode = pivot
    profile = false

    for arg in ARGS
        if arg == "--pivot"
            mode = pivot
        elseif arg == "--profile"
            profile = true
        elseif arg == "--binary"
            mode = binary
        elseif dataset_name === nothing
            dataset_name = arg
        else
            throw(ArgumentError("Unexpected argument: $arg"))
        end
    end

    if dataset_name === nothing
        dataset_name = DEBUG ? "boxes" : nothing
    end

    return dataset_name, mode, profile
end

# Create a wrapper function that runs code with a custom stack size
function with_stacksize(f, bytes::Int)
    fetch(schedule(Task(f, bytes)))
end

function main()
    dataset_name, mode, profile = parse_dataset_and_mode()
    graph_path = resolve_graph_path(dataset_name)

    k, θ = 2, 5

    if !isfile(graph_path)
        println(stderr, "Error: Could not find a saved graph at '$graph_path'.")
        exit(1)
    end

    println("Loading graph from: $graph_path")
    println("Mode: $(mode == pivot ? "pivot" : "binary")")

    with_stacksize(2_000_000_000) do
        if profile
            println("Profile mode enabled: warming up compilation on a small graph...")
            # Warm up: run the search once on a very small slice to compile methods
            gw = load_bipartite_graph(graph_path; max_lines = 50)
            Dw = find_kmdb!(gw, true, mode)

            # Load full graph for the actual profiled run
            g = load_bipartite_graph(graph_path; max_lines = 20000)

            println("Starting profiling run (branch) — this may take a while...")
            Profile.clear()
            @profile begin
                D = find_kmdb(g, true, mode, k, θ)
            end

            # Display profile using ProfileCanvas
            try
                ProfileCanvas.canvas()
            catch e
                @warn "ProfileCanvas failed to display:" exception=(e, catch_backtrace())
            end
            @show D
        else
            g = load_bipartite_graph(graph_path; max_lines = 20000)
            D = find_kmdb!(g, true, mode, k, θ)

            @show D
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    a = @allocated begin main() end
    a > 0 && @show a
end