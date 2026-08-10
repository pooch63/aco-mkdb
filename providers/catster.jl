#=
=================================================================================
catster (KONECT Petster / Catster) provider
=================================================================================
Downloads a KONECT TSV network archive and converts the `out.*` edgelist to the
shared interactions CSV. Sparse / non-dense endpoint IDs are remapped to 1..n by
the shared index_dataset step (same as Amazon / tnet / roadmap).

Line format (KONECT out.*):
  % comment lines are ignored
  <u> <v> [w] [t]
Only the first two columns are kept as `u,v`. Arbitrary whitespace is fine.

URL template:
  http://konect.cc/files/download.tsv.{NAME}.tar.bz2

Examples:
  julia process.jl catster/petster-friendships-cat --download
  julia process.jl catster/petster-friendships-cat --source=data/raw/download.tsv.petster-friendships-cat/petster-friendships-cat/out.petster-friendships-cat-uniq
=#

const __PROVIDER_CATSTER_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

const CATSTER_DATASETS_URL = "http://konect.cc/files"

"""
Normalize a dataset key to the KONECT internal name (no path/archive prefix).
"""
function catster_dataset_stem(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    name = replace(name, r"^download\.tsv\." => "")
    name = replace(name, r"\.tar\.bz2$" => "")
    name = replace(name, r"^out\." => "")
    name = replace(name, r"-uniq$" => "")
    isempty(name) && throw(ArgumentError("Empty catster dataset name from '$dataset_name'"))
    return name
end

"""
Locate the KONECT `out.*` edgelist under an extracted network directory.
"""
function catster_find_out_file(extract_dir::AbstractString)
    isdir(extract_dir) || error("Expected extracted network directory at '$extract_dir'")
    outs = filter(f -> startswith(basename(f), "out."), readdir(extract_dir; join=true))
    filter!(isfile, outs)
    isempty(outs) && error("No out.* edgelist found under '$extract_dir'")
    return first(sort!(outs))
end

"""
Download `download.tsv.{NAME}.tar.bz2`, extract under `dest_dir`, and return the
path to the `out.*` edgelist.
"""
function catster_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    name = catster_dataset_stem(dataset_name)
    url = "$(CATSTER_DATASETS_URL)/download.tsv.$(name).tar.bz2"
    archive = joinpath(dest_dir, "download.tsv.$(name).tar.bz2")
    extract_dir = joinpath(dest_dir, name)

    println("  URL: $url")
    println("  Saving archive to: $archive")
    mkpath(dest_dir)
    Downloads.download(url, archive)

    println("  Extracting…")
    if Sys.which("tar") === nothing
        error("tar not found on PATH; install tar or provide an already-extracted out.* file via --source=")
    end
    run(`tar -xjf $archive -C $dest_dir`)
    rm(archive; force=true)

    out_path = catster_find_out_file(extract_dir)
    println("  Raw edgelist: $out_path")
    return out_path
end

"""
Convert a KONECT `out.*` edgelist (`u v [w] [t]`, whitespace-separated; `%`
comments and blank lines ignored) to interactions CSV with columns `u,v`.
"""
function catster_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
    if !isfile(raw_path)
        error("Input file '$raw_path' does not exist.")
    end

    mkpath(dirname(String(interactions_csv)))

    println("Starting conversion...")
    println("Input source:  $raw_path")
    println("Output target: $interactions_csv")

    kept = 0
    skipped = 0

    open(interactions_csv, "w") do out
        println(out, "u,v")
        open(raw_path, "r") do input
            for (line_count, line) in enumerate(eachline(input))
                cleaned = strip(line)
                isempty(cleaned) && continue
                startswith(cleaned, '%') && continue

                parts = split(cleaned)
                if length(parts) < 2 || isempty(parts[1]) || isempty(parts[2])
                    skipped += 1
                    println(stderr, "Warning: Skipping malformed row at line $line_count")
                    continue
                end

                println(out, parts[1], ",", parts[2])
                kept += 1
            end
        end
    end

    println("Success! Successfully processed $kept entries" *
            (skipped > 0 ? " ($skipped skipped)." : "."))
    return nothing
end

register_provider!(ProviderAdapter(
    "catster",
    catster_download_raw!,
    catster_convert!,
))
