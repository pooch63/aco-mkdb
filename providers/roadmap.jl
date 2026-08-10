#=
=================================================================================
roadmap (SNAP road networks) provider
=================================================================================
Downloads a SNAP road-network edgelist and converts it to the shared
interactions CSV. Sparse / non-dense endpoint IDs are remapped to 1..n by the
shared index_dataset step (same as Amazon / tnet).

Line format (SNAP roadNet-*.txt):
  # comment lines are ignored
  <FromNodeId>	<ToNodeId>
Only the first two columns are kept as `u,v`. Arbitrary whitespace is fine.

URL template:
  https://snap.stanford.edu/data/{NAME}.txt.gz

Examples:
  julia process.jl roadmap/roadNet-PA --download
  julia process.jl roadmap/roadNet-PA --source=data/roadNet-PA.txt
=#

const __PROVIDER_ROADMAP_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

const ROADMAP_DATASETS_URL = "https://snap.stanford.edu/data"

"""
Normalize a dataset key to the leaf stem used on the SNAP mirror (no path/extension).
"""
function roadmap_dataset_stem(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    name = replace(name, r"\.txt(\.gz)?$" => "")
    isempty(name) && throw(ArgumentError("Empty roadmap dataset name from '$dataset_name'"))
    return name
end

"""
Download `{NAME}.txt.gz` under `dest_dir`, unzip to `{NAME}.txt`, and return the path.
"""
function roadmap_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    name = roadmap_dataset_stem(dataset_name)
    url = "$(ROADMAP_DATASETS_URL)/$(name).txt.gz"
    gz_path = joinpath(dest_dir, "$(name).txt.gz")
    txt_path = joinpath(dest_dir, "$(name).txt")

    println("  URL: $url")
    println("  Saving archive to: $gz_path")
    mkpath(dest_dir)
    Downloads.download(url, gz_path)

    println("  Unzipping…")
    gunzip_file!(gz_path; dest=txt_path)
    println("  Raw edgelist: $txt_path")
    return txt_path
end

"""
Convert a SNAP road-network edgelist (`u v`, whitespace-separated; `#` comments
and blank lines ignored) to interactions CSV with columns `u,v`.
"""
function roadmap_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
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
                startswith(cleaned, '#') && continue

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
    "roadmap",
    roadmap_download_raw!,
    roadmap_convert!,
))
