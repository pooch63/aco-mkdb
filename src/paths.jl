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

"""
Count edges in an indexed CSV (`u,v` with a header row) without loading the graph.
"""
function count_indexed_edges(path::AbstractString)
    n = countlines(path)
    return max(0, n - 1)  # subtract header
end

"""
    load_data_ignore_patterns(data_root="data"; extra=()) -> Vector{String}

Read `data_root/.dataignore` (gitignore-style) plus `extra` patterns.
Blank lines and `#` comments are ignored. A pattern ending in `/` matches that
directory and everything under it. A pattern without `/` matches that path
segment anywhere (e.g. `example` skips `example` and `foo/example`). A pattern
containing `/` matches from the start of the dataset key. `*` matches within
one path segment.

Always includes `raw/` so download staging is never swept by default.
"""
function load_data_ignore_patterns(data_root::AbstractString="data";
    extra=())
    patterns = String["raw/"]
    path = joinpath(data_root, ".dataignore")
    if isfile(path)
        for line in eachline(path)
            s = strip(line)
            isempty(s) && continue
            startswith(s, '#') && continue
            push!(patterns, replace(s, '\\' => '/'))
        end
    end
    for s in extra
        p = replace(strip(String(s)), '\\' => '/')
        isempty(p) && continue
        # Legacy skip=("raw",) names → directory patterns.
        if !endswith(p, "/") && !occursin('/', p) && !occursin('*', p)
            p = p * "/"
        end
        push!(patterns, p)
    end
    return patterns
end

"""
    dataset_key_ignored(key, patterns) -> Bool

True if dataset key `amazon/boxes`-style matches any `.dataignore` pattern.
"""
function dataset_key_ignored(key::AbstractString, patterns)
    k = replace(String(key), '\\' => '/')
    parts = split(k, '/'; keepempty=false)
    for raw in patterns
        p = rstrip(String(raw), '/')
        isempty(p) && continue
        dir_only = endswith(String(raw), '/')

        if occursin('*', p)
            # Segment-wise glob, optionally requiring a full-key prefix match
            # when the pattern contains '/'.
            if occursin('/', p)
                _glob_match(k, p) && return true
            else
                any(_glob_match(seg, p) for seg in parts) && return true
                _glob_match(k, p) && return true
            end
            continue
        end

        if occursin('/', p) || dir_only
            (k == p || startswith(k, p * "/")) && return true
        else
            (k == p || startswith(k, p * "/") || p in parts) && return true
        end
    end
    return false
end

function _glob_match(text::AbstractString, pattern::AbstractString)
    buf = IOBuffer()
    print(buf, '^')
    for c in String(pattern)
        if c == '*'
            print(buf, "[^/]*")
        elseif c in ('.', '+', '^', '$', '(', ')', '|', '{', '}', '[', ']', '\\')
            print(buf, '\\', c)
        else
            print(buf, c)
        end
    end
    print(buf, '\$')
    return occursin(Regex(String(take!(buf))), String(text))
end

"""
    discover_indexed_graphs(; data_root="data", skip=()) -> Vector{NamedTuple}

Walk `data_root` for directories containing `indexed_interactions.csv`.
Returns unsorted entries `(key, path, edges)` where `key` is the dataset path
relative to `data_root` using `/` (e.g. `"amazon/boxes"`).

Skips keys matching `data_root/.dataignore` (see `load_data_ignore_patterns`).
`skip` adds extra patterns (legacy: bare names like `"raw"` mean that directory).
"""
function discover_indexed_graphs(; data_root::AbstractString="data",
    skip=())
    root = abspath(data_root)
    patterns = load_data_ignore_patterns(data_root; extra=skip)
    entries = NamedTuple{(:key, :path, :edges), Tuple{String,String,Int}}[]

    isdir(root) || return entries

    for (dir, _subdirs, files) in walkdir(root)
        "indexed_interactions.csv" in files || continue
        rel = relpath(dir, root)
        rel == "." && continue
        parts = split(replace(rel, '\\' => '/'), '/'; keepempty=false)
        isempty(parts) && continue
        key = join(parts, "/")
        dataset_key_ignored(key, patterns) && continue

        path = joinpath(dir, "indexed_interactions.csv")
        edges = count_indexed_edges(path)
        push!(entries, (key=key, path=path, edges=edges))
    end

    return entries
end

"""
    dataset_key_matches_prefix(key, prefix) -> Bool

True if `key` is `prefix` or a nested dataset under it (`prefix/...`).
`konect` matches `konect/bitcoin` but not `konect-small/foo`.
Empty `prefix` matches everything. Comma-separated prefixes are OR'd.
"""
function dataset_key_matches_prefix(key::AbstractString, prefix::AbstractString)
    k = replace(String(key), '\\' => '/')
    raw = String(prefix)
    isempty(strip(raw)) && return true
    for part in split(raw, ',')
        p = rstrip(replace(strip(part), '\\' => '/'), '/')
        isempty(p) && continue
        (k == p || startswith(k, p * "/")) && return true
    end
    return false
end

"""
    order_graphs_by_edges(; data_root="data", skip=(), ascending=true, prefix=nothing)

Discover indexed graphs under `data_root` and return them sorted by edge count
(ascending by default — smallest / easiest first).

Honors `data_root/.dataignore` (and always skips `raw/`). `skip` adds patterns.
`prefix` restricts to a provider or nested key (`"konect-small"`, `"amazon"`).
Comma-separated values are OR'd. Path-prefix matching is slash-bounded, so
`"konect"` does not match `"konect-small/..."`.
"""
function order_graphs_by_edges(; data_root::AbstractString="data",
    skip=(), ascending::Bool=true, prefix::Union{Nothing,AbstractString}=nothing)
    entries = discover_indexed_graphs(; data_root=data_root, skip=skip)
    if prefix !== nothing && !isempty(strip(String(prefix)))
        filter!(e -> dataset_key_matches_prefix(e.key, prefix), entries)
    end
    return sort!(entries; by=(e -> e.edges), rev=!ascending)
end
