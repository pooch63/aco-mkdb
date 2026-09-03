#=
=================================================================================
Graph data provider adapters
=================================================================================
Each provider knows how to download (optional) and convert raw data into the
common `u,v[,...]` interactions CSV. Indexing is shared.
=#

const __PROVIDERS_JL__ = true

using Downloads

isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(@__DIR__, "..", "src", "paths.jl"))

# index_dataset.jl has no guard constant; only include once.
if !isdefined(@__MODULE__, :index_dataset)
    include(joinpath(@__DIR__, "..", "bin", "index_dataset.jl"))
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

# ------------------------------------------------------------------------------
# Shared local-key → remote-name aliases (`data/aliases.txt`)
# ------------------------------------------------------------------------------

const _PROVIDER_ALIASES_LOADED = Ref(false)
"""
provider → (local_key → remote_name). Populated by `load_provider_aliases!`.
"""
const PROVIDER_ALIASES = Dict{String,Dict{String,String}}()

function _aliases_path()
    return joinpath(@__DIR__, "..", "data", "aliases.txt")
end

function _legacy_konect_aliases_path()
    return joinpath(@__DIR__, "..", "data", "konect_aliases.txt")
end

"""
Parse one aliases file into `PROVIDER_ALIASES`.

Lines are `provider/local=remote`. Blank lines and `#` comments are ignored.
Bare `local=remote` (no slash) is treated as `konect/local=remote` for legacy
`data/konect_aliases.txt` compatibility.
"""
function _merge_aliases_file!(path::AbstractString; default_provider::Union{Nothing,String}=nothing)
    isfile(path) || return
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, '#') && continue
        parts = split(s, '=', limit=2)
        length(parts) == 2 || continue
        left = strip(parts[1])
        remote = strip(parts[2])
        isempty(left) && continue
        isempty(remote) && continue

        provider = default_provider
        local_key = left
        if occursin('/', left)
            segs = split(replace(left, '\\' => '/'), '/'; keepempty=false)
            length(segs) >= 2 || continue
            provider = lowercase(segs[1])
            local_key = join(segs[2:end], '/')
        elseif provider === nothing
            continue
        end

        bucket = get!(PROVIDER_ALIASES, provider) do
            Dict{String,String}()
        end
        bucket[local_key] = remote
    end
    return nothing
end

"""
Load `data/aliases.txt` (and legacy `data/konect_aliases.txt` if present) once.
"""
function load_provider_aliases!()
    _PROVIDER_ALIASES_LOADED[] && return PROVIDER_ALIASES
    empty!(PROVIDER_ALIASES)
    _merge_aliases_file!(_aliases_path())
    # Legacy bare `local=remote` lines → konect namespace
    _merge_aliases_file!(_legacy_konect_aliases_path(); default_provider="konect")
    _PROVIDER_ALIASES_LOADED[] = true
    return PROVIDER_ALIASES
end

"""
Resolve `local_key` for `provider` using builtins then `data/aliases.txt` overrides.

`normalize` maps the lookup key (e.g. lowercase for Amazon). Returns `fallback`
when no alias matches.
"""
function resolve_alias(provider::AbstractString, local_key::AbstractString;
    builtins::AbstractDict{<:AbstractString,<:AbstractString}=Dict{String,String}(),
    normalize::Function=identity,
    fallback::AbstractString=String(local_key))

    load_provider_aliases!()
    key = String(normalize(String(local_key)))
    file_map = get(PROVIDER_ALIASES, lowercase(String(provider)), nothing)
    if file_map !== nothing
        for (k, v) in file_map
            if String(normalize(k)) == key
                return v
            end
        end
    end
    for (k, v) in builtins
        if String(normalize(String(k))) == key
            return String(v)
        end
    end
    return String(fallback)
end

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
