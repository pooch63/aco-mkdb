#=
=================================================================================
Same-neighbor classes on a reduced graph
================================================================================
Load one indexed graph, apply common-neighbor reduction (CNN), then group
vertices on each side by identical neighbor sets.

A class of size s is s vertices with the same neighbors. Coalescing each class
into one supernode removes `s - 1` vertices, so

    saved = (nodes combined) - (supernodes)
          = |V_side| - (number of distinct neighborhoods)

Singletons are already supernodes; they contribute 0 to `saved`. The same
total is `sum(s for s > 1) - (number of classes with s > 1)`.

Usage:
  julia bin/same-neighbors.jl <dataset_or_path> [--k=2] [--theta=5]

Examples:
  julia bin/same-neighbors.jl konect-small/moreno
  julia bin/same-neighbors.jl amazon/boxes --k=2 --theta=5
================================================================================
=#

const ROOT = dirname(@__DIR__)
const SRC = joinpath(ROOT, "src")
isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))
isdefined(@__MODULE__, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))

function parse_int_flag(flag::String, default::Int)
    prefix = "--$flag="
    for arg in ARGS
        startswith(arg, prefix) || continue
        return parse(Int, arg[length(prefix)+1:end])
    end
    return default
end

function positional_arg()
    for arg in ARGS
        startswith(arg, "--") && continue
        return arg
    end
    return nothing
end

"""Class sizes of vertices with identical neighbor lists in a CSR side."""
function neighborhood_class_sizes(offsets::Vector{Int}, adj::Vector{Int}, n::Int)
    n == 0 && return Int[]
    by_hash = Dict{UInt, Vector{Int}}()
    sizehint!(by_hash, n)
    for i in 1:n
        r = offsets[i]:(offsets[i + 1] - 1)
        h = hash(@view adj[r])
        push!(get!(by_hash, h, Int[]), i)
    end

    sizes = Int[]
    sizehint!(sizes, n)
    for idxs in values(by_hash)
        reps = Int[]
        counts = Int[]
        for i in idxs
            placed = false
            ri = offsets[i]:(offsets[i + 1] - 1)
            for (t, rep) in enumerate(reps)
                rr = offsets[rep]:(offsets[rep + 1] - 1)
                if length(ri) == length(rr) && @view(adj[ri]) == @view(adj[rr])
                    counts[t] += 1
                    placed = true
                    break
                end
            end
            if !placed
                push!(reps, i)
                push!(counts, 1)
            end
        end
        append!(sizes, counts)
    end
    return sizes
end

function report_side(name::String, sizes::Vector{Int})
    n = isempty(sizes) ? 0 : sum(sizes)
    n_classes = length(sizes)
    multi = filter(>(1), sizes)
    combined = isempty(multi) ? 0 : sum(multi)
    n_super_multi = length(multi)
    saved = n - n_classes

    println()
    println("$name side")
    println("  nodes: $n")
    println("  distinct neighborhoods (supernodes): $n_classes")
    if n == 0
        println("  class size min: —")
        println("  class size max: —")
        println("  nodes saved by coalescing: 0")
        return
    end

    min_s, max_s = minimum(sizes), maximum(sizes)
    min_classes = count(==(min_s), sizes)
    max_classes = count(==(max_s), sizes)
    println("  class size min: $min_s  ($min_classes class$(min_classes == 1 ? "" : "es"), $(min_s * min_classes) nodes)")
    println("  class size max: $max_s  ($max_classes class$(max_classes == 1 ? "" : "es"), $(max_s * max_classes) nodes)")
    println("  nodes in classes of size > 1: $combined")
    println("  supernodes from those classes: $n_super_multi")
    println("  nodes saved by coalescing: $saved  (= $combined - $n_super_multi)")

    hist = Dict{Int, Int}()
    for s in sizes
        hist[s] = get(hist, s, 0) + 1
    end
    println("  distribution (class size → number of classes, nodes covered):")
    for s in sort!(collect(keys(hist)))
        c = hist[s]
        label = c == 1 ? "class" : "classes"
        println("    $s → $c $label, $(s * c) nodes")
    end
end

function main()
    dataset = positional_arg()
    dataset === nothing && error("usage: julia bin/same-neighbors.jl <dataset_or_path> [--k=2] [--theta=5]")
    k = parse_int_flag("k", 2)
    θ = parse_int_flag("theta", 5)
    k < 0 && error("k must be >= 0, got $k")
    θ < 1 && error("theta must be >= 1, got $θ")

    graph_path = resolve_graph_path(dataset; data_root=joinpath(ROOT, "data"))
    isfile(graph_path) || error("graph file not found: $graph_path")

    g, n_edges = load_bipartite_graph(graph_path)
    nU0 = length(g.adjU)
    nV0 = length(g.adjV)
    num_U = isempty(g.adjU) ? 0 : maximum(keys(g.adjU))
    num_V = isempty(g.adjV) ? 0 : maximum(keys(g.adjV))

    reduce_graph!(g, k, θ, num_U, num_V)
    fg = freeze(g)
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    nE = length(fg.v_adj)

    println("graph: $dataset")
    println("file:  $graph_path")
    println("CNN reduction: k=$k  θ=$θ  threshold θ−k=$(θ - k)")
    println("before: |U|=$nU0  |V|=$nV0  |E|=$n_edges")
    println("after:  |U|=$nU  |V|=$nV  |E|=$nE")

    u_sizes = neighborhood_class_sizes(fg.u_offsets, fg.v_adj, nU)
    v_sizes = neighborhood_class_sizes(fg.v_offsets, fg.u_adj, nV)
    report_side("U", u_sizes)
    report_side("V", v_sizes)

    saved = (nU - length(u_sizes)) + (nV - length(v_sizes))
    println()
    println("total nodes saved by coalescing both sides: $saved")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
