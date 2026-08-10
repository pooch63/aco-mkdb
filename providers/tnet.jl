#=
=================================================================================
tnet (Tore Opsahl) provider
=================================================================================
Downloads a space/tab-separated edgelist from the tnet datasets mirror and
converts it to the shared interactions CSV. Sparse / non-dense endpoint IDs
are remapped to 1..n by the shared index_dataset step (same as Amazon).

Line format (tnet edgelist):
  <u> <v> [w] [t]
  % and # comment lines are ignored
Only the first two columns are kept as `u,v`.

URL template:
  http://opsahl.co.uk/tnet/datasets/{NAME}.txt

Examples:
  julia process.jl tnet/Newman-Cond_mat_95-99-two_mode --download
  julia process.jl tnet/Newman-Cond_mat_95-99-two_mode --source=data/Newman-Cond_mat_95-99-two_mode.txt
=#

const __PROVIDER_TNET_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

const TNET_DATASETS_URL = "http://opsahl.co.uk/tnet/datasets"

"""
Normalize a dataset key to the leaf stem used on the tnet mirror (no path/extension).
"""
function tnet_dataset_stem(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    name = replace(name, r"\.txt$" => "")
    isempty(name) && throw(ArgumentError("Empty tnet dataset name from '$dataset_name'"))
    return name
end

"""
Download `{NAME}.txt` under `dest_dir` and return the path.
"""
function tnet_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    name = tnet_dataset_stem(dataset_name)
    url = "$(TNET_DATASETS_URL)/$(name).txt"
    txt_path = joinpath(dest_dir, "$(name).txt")

    println("  URL: $url")
    println("  Saving to: $txt_path")
    mkpath(dest_dir)
    Downloads.download(url, txt_path)
    println("  Raw edgelist: $txt_path")
    return txt_path
end

"""
Convert a tnet edgelist (`u v [w] [t]`, whitespace-separated; `%` and `#`
comments and blank lines ignored) to interactions CSV with columns `u,v`.
"""
function tnet_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
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
                (startswith(cleaned, '%') || startswith(cleaned, '#')) && continue

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
    "tnet",
    tnet_download_raw!,
    tnet_convert!,
))
