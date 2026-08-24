#=
=================================================================================
List indexed graphs under data/ ordered by edge count (ascending).

Usage:
  julia order_graphs.jl                         # print "edges\tkey" lines
  julia order_graphs.jl --keys-only             # print dataset keys only (for bash loops)
  julia order_graphs.jl --keys-only --prefix=konect-small

Used by vary.bash / compare-seeds.bash so easy (small) graphs run first.
`--prefix=` is slash-bounded (konect ≠ konect-small). Comma-OR multiple prefixes.
=================================================================================
=#

const SRC = joinpath(@__DIR__, "src")
isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))

function parse_prefix_flag()
    for arg in ARGS
        if arg == "--prefix" || arg == "--prefix="
            throw(ArgumentError("--prefix requires a value, e.g. --prefix=konect-small"))
        elseif startswith(arg, "--prefix=")
            return split(arg, "=", limit=2)[2]
        end
    end
    return nothing
end

function main()
    keys_only = "--keys-only" in ARGS
    prefix = parse_prefix_flag()
    data_root = joinpath(@__DIR__, "data")

    entries = order_graphs_by_edges(; data_root=data_root, ascending=true, prefix=prefix)
    if isempty(entries)
        if prefix !== nothing && !isempty(strip(prefix))
            println(stderr, "No indexed graphs matching --prefix=$prefix under $data_root")
            println(stderr, "Index them first, e.g. julia process.jl $prefix/<dataset> --source=...")
        else
            println(stderr, "No indexed graphs found under $data_root")
        end
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
