#=
=================================================================================
Graph data provider adapters
=================================================================================
Each provider knows how to download (optional) and convert raw data into the
common `user_id,item_id,timestamp` interactions CSV. Indexing is shared.
=#

const __PROVIDERS_JL__ = true

using Downloads

isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(@__DIR__, "..", "src", "paths.jl"))

# index_dataset.jl has no guard constant; only include once.
if !isdefined(@__MODULE__, :index_dataset)
    include(joinpath(@__DIR__, "..", "index_dataset.jl"))
end

"""
Adapter for a graph data source.

Fields:
  - `name`           provider key (e.g. `"amazon"`)
  - `download_raw!`  `(dataset_name, dest_dir) -> raw_path` or `nothing` if unsupported
  - `convert!`       `(raw_path, interactions_csv) -> nothing`
"""
struct ProviderAdapter
    name::String
    download_raw!::Union{Nothing,Function}
    convert!::Function
end

const PROVIDER_REGISTRY = Dict{String,ProviderAdapter}()

function register_provider!(adapter::ProviderAdapter)
    PROVIDER_REGISTRY[lowercase(adapter.name)] = adapter
    return adapter
end

function get_provider(name::AbstractString)
    key = lowercase(String(name))
    haskey(PROVIDER_REGISTRY, key) || throw(ArgumentError(
        "Unknown provider '$name'. Registered: $(join(sort!(collect(keys(PROVIDER_REGISTRY))), ", "))"))
    return PROVIDER_REGISTRY[key]
end

list_providers() = sort!(collect(keys(PROVIDER_REGISTRY)))

"""
Gunzip `src.gz` to `dest` (defaults to path without `.gz`), then remove the archive.
"""
function gunzip_file!(gz_path::AbstractString; dest::Union{Nothing,AbstractString}=nothing)
    endswith(gz_path, ".gz") || throw(ArgumentError("Expected a .gz file, got $gz_path"))
    out = dest === nothing ? String(gz_path[1:end-3]) : String(dest)
    mkpath(dirname(out))
    if Sys.which("gunzip") === nothing
        error("gunzip not found on PATH; install gzip or provide an already-unzipped file via --source=")
    end
    open(out, "w") do output
        open(`gunzip -c $gz_path`) do input
            write(output, input)
        end
    end
    rm(gz_path; force=true)
    return out
end

"""
Run the full pipeline for `provider/dataset` into `data/<provider>/<dataset>/`.

Options:
  - `download=true`  fetch raw data via the provider adapter
  - `source=path`    use an existing raw file instead of downloading
"""
function process_with_provider(provider_name::AbstractString, dataset_name::AbstractString;
    download::Bool=false,
    source::Union{Nothing,AbstractString}=nothing,
    data_root::AbstractString="data")

    adapter = get_provider(provider_name)
    out_dir = dataset_dir(joinpath(provider_name, dataset_name); data_root=data_root)
    mkpath(out_dir)

    raw_dir = joinpath(out_dir, "raw")
    interactions_csv = joinpath(out_dir, "interactions.csv")
    indexed_csv = joinpath(out_dir, "indexed_interactions.csv")
    mapping_dir = joinpath(out_dir, "mappings")

    raw_path = if source !== nothing
        String(source)
    elseif download
        adapter.download_raw! === nothing && error("Provider '$(adapter.name)' does not support --download")
        mkpath(raw_dir)
        println("Downloading via provider '$(adapter.name)'…")
        adapter.download_raw!(dataset_name, raw_dir)
    else
        throw(ArgumentError(
            "Provide --download or --source=<path> so the pipeline has a raw input file."))
    end

    if !isfile(raw_path)
        error("Raw input file not found: $raw_path")
    end

    println("Converting raw data → interactions CSV…")
    adapter.convert!(raw_path, interactions_csv)

    println("Indexing dataset…")
    index_dataset(interactions_csv, indexed_csv, mapping_dir)

    println("Pipeline complete.")
    println("  Raw input:      $raw_path")
    println("  Interactions:   $interactions_csv")
    println("  Indexed graph:  $indexed_csv")
    println("  Mappings:       $mapping_dir")
    return (; out_dir, interactions_csv, indexed_csv, mapping_dir)
end
