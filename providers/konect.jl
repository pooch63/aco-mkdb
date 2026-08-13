#=
=================================================================================
konect provider
=================================================================================
Downloads a space/tab-separated edgelist from the konect datasets mirror and
converts it to the shared interactions CSV. Sparse / non-dense endpoint IDs
are remapped to 1..n by the shared index_dataset step (same as Amazon).

Line format (konect edgelist):
  <u> <v> [w] [t]
  % and # comment lines are ignored
Only the first two columns are kept as `u,v`.

URL template:
  http://opsahl.co.uk/konect/datasets/{NAME}.txt

Examples:
  julia process.jl konect/Newman-Cond_mat_95-99-two_mode --download
  julia process.jl konect/Newman-Cond_mat_95-99-two_mode --source=data/Newman-Cond_mat_95-99-two_mode.txt
=#

const __PROVIDER_konect_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

const konect_DATASETS_URL = "http://opsahl.co.uk/konect/datasets"

"""
Normalize a dataset key to the leaf stem used on the konect mirror (no path/extension).
"""
function konect_dataset_stem(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    name = replace(name, r"\.txt$" => "")
    isempty(name) && throw(ArgumentError("Empty konect dataset name from '$dataset_name'"))
    return name
end

"""
Download `{NAME}.txt` under `dest_dir` and return the path.
"""
function konect_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    @error "Can't download for konect!"
end

"""
Convert a konect edgelist (`u v [w] [t]`, whitespace-separated; `%` and `#`
comments and blank lines ignored) to interactions CSV with columns `u,v`.
"""
function konect_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
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
    "konect",
    konect_download_raw!,
    konect_convert!,
))
