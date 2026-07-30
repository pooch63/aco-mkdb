#=
=================================================================================
Dataset path helpers
=================================================================================
Supports nested dataset names under data/, e.g. "boxes" or "amazon/boxes".
=#

const __PATHS_JL__ = true

"""
Normalize a dataset key so nested folders use the OS path separator.
Accepts `"amazon/boxes"`, `"amazon\\\\boxes"`, or already-joined paths.
"""
function normalize_dataset_name(name::AbstractString)
    parts = split(replace(String(name), '\\' => '/'), '/'; keepempty=false)
    isempty(parts) && throw(ArgumentError("Dataset name must be non-empty"))
    return joinpath(parts...)
end

"""
Directory for a dataset: `data/<dataset_name>/` (nested names allowed).
"""
function dataset_dir(dataset_name::AbstractString; data_root::AbstractString="data")
    return joinpath(data_root, normalize_dataset_name(dataset_name))
end

"""
    resolve_graph_path(dataset_name=nothing; data_root="data") -> String

Resolve a graph CSV path from:
  1. `nothing` / empty → `data/indexed_interactions.csv`
  2. an existing file path → that path
  3. a dataset key (possibly nested, e.g. `amazon/boxes`) →
     `data/<dataset>/indexed_interactions.csv`
"""
function resolve_graph_path(dataset_name::Union{AbstractString,Nothing}=nothing;
    data_root::AbstractString="data")
    if dataset_name === nothing || isempty(dataset_name)
        return joinpath(data_root, "indexed_interactions.csv")
    end

    name = String(dataset_name)
    if isfile(name)
        return name
    end

    return joinpath(dataset_dir(name; data_root=data_root), "indexed_interactions.csv")
end

"""
Split `"amazon/boxes"` into `("amazon", "boxes")`.
A flat name like `"twitter"` becomes `("twitter", "twitter")` when used as a
provider key with a single-segment dataset.
"""
function split_provider_dataset(dataset_name::AbstractString)
    parts = split(replace(String(dataset_name), '\\' => '/'), '/'; keepempty=false)
    isempty(parts) && throw(ArgumentError("Dataset name must be non-empty"))
    if length(parts) == 1
        return String(parts[1]), String(parts[1])
    end
    provider = String(parts[1])
    dataset = join(parts[2:end], "/")
    return provider, dataset
end
