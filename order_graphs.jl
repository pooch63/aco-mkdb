#=
=================================================================================
List indexed graphs under data/ ordered by edge count (ascending).

Usage:
  julia order_graphs.jl              # print "edges\tkey" lines
  julia order_graphs.jl --keys-only  # print dataset keys only (for bash loops)

Used by vary.bash so easy (small) graphs run first.
=================================================================================
=#

const SRC = joinpath(@__DIR__, "src")
isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))

function main()
    keys_only = "--keys-only" in ARGS
    data_root = joinpath(@__DIR__, "data")

    entries = order_graphs_by_edges(; data_root=data_root, ascending=true)
    if isempty(entries)
        println(stderr, "No indexed graphs found under $data_root")
        exit(1)
    end

    for e in entries
        if keys_only
            println(e.key)
        else
            println("$(e.edges)\t$(e.key)")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
