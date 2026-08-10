#=
=================================================================================
Dataset Processing Pipeline
=================================================================================
Dispatches to a per-provider adapter (see providers/) that knows how to
download and convert that source's raw format into the shared edge CSV, then
indexes IDs via index_dataset.jl.

Output layout (nested names supported):
  data/<provider>/<dataset>/
    raw/                      # downloaded archives / extracted files
    interactions.csv          # u,v[,...] (source IDs)
    indexed_interactions.csv  # integer-indexed u,v graph
    mappings/                 # reverse maps + metadata

How to Run:
  julia process.jl <provider>/<dataset> --download
  julia process.jl <provider>/<dataset> --source=/path/to/raw.file
  julia process.jl <provider> <dataset> --download

Examples:
  julia process.jl amazon/Appliances --download
  julia process.jl amazon Appliances --download
  julia process.jl amazon/boxes --source=data/subscription-boxes.jsonl
  julia process.jl tcga/TCGA-BRCA --download
  julia process.jl tcga/BRCA --download

To add a new source, create providers/<name>.jl implementing download + convert,
then include it from providers/registry.jl.
=================================================================================
=#

include(joinpath(@__DIR__, "providers", "registry.jl"))

function parse_process_args(args)
    download = false
    source = nothing
    positionals = String[]

    for arg in args
        if arg == "--download"
            download = true
        elseif startswith(arg, "--source=")
            source = split(arg, "=", limit=2)[2]
        elseif arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--")
            throw(ArgumentError("Unknown flag: $arg"))
        else
            push!(positionals, arg)
        end
    end

    provider = nothing
    dataset = nothing

    if length(positionals) == 1
        parts = split(replace(positionals[1], '\\' => '/'), '/'; keepempty=false)
        if length(parts) < 2
            throw(ArgumentError(
                "Pass a nested key 'provider/dataset' or two args: <provider> <dataset>"))
        end
        provider, dataset = split_provider_dataset(positionals[1])
    elseif length(positionals) == 2
        provider, dataset = positionals[1], positionals[2]
    else
        throw(ArgumentError("Expected <provider>/<dataset> or <provider> <dataset>"))
    end

    if !download && source === nothing
        throw(ArgumentError("Provide --download or --source=<path>"))
    end

    return (; provider, dataset, download, source)
end

function print_usage()
    println(stderr, "Usage:")
    println(stderr, "  julia process.jl <provider>/<dataset> --download")
    println(stderr, "  julia process.jl <provider>/<dataset> --source=/path/to/raw")
    println(stderr, "  julia process.jl <provider> <dataset> --download")
    println(stderr, "")
    println(stderr, "Registered providers: $(join(list_providers(), ", "))")
    println(stderr, "")
    println(stderr, "Examples:")
    println(stderr, "  julia process.jl amazon/Appliances --download")
    println(stderr, "  julia process.jl amazon boxes --source=data/subscription-boxes.jsonl")
    println(stderr, "  julia process.jl tcga/TCGA-BRCA --download")
end

function main()
    try
        parsed = parse_process_args(ARGS)
        if parsed === nothing
            print_usage()
            exit(0)
        end
        process_with_provider(parsed.provider, parsed.dataset;
            download=parsed.download, source=parsed.source)
    catch e
        if e isa ArgumentError
            println(stderr, "Error: ", e.msg)
            print_usage()
            exit(1)
        else
            rethrow()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
