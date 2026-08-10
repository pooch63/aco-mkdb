#=
=================================================================================
Wikipedia dump index provider
=================================================================================
Converts a pages-articles multistream index (offset:page_id:title) into the
shared interactions CSV. Sparse / non-dense endpoint IDs are remapped to 1..n
by the shared index_dataset step (same as Amazon).

Index line format:
  <byte_offset>:<page_id>:<title…>

URL template:
  https://dumps.wikimedia.org/{wiki}/{date}/{wiki}-{date}-pages-articles-multistream-index.txt.bz2

Examples:
  julia process.jl wikipedia/jawiki-20260801 --download
  julia process.jl wikipedia/jawiki-20260801 --source=data/jawiki-20260801-pages-articles-multistream-index.txt
=#

const __PROVIDER_WIKIPEDIA_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

const WIKIPEDIA_DUMP_URL = "https://dumps.wikimedia.org"

"""
Parse a dump key like `jawiki-20260801` into `(wiki, date)`.
Also accepts a leaf stem that already includes the index suffix.
"""
function wikipedia_parse_dump_key(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    name = replace(name, r"\.txt(\.bz2)?$" => "")
    name = replace(name, r"-pages-articles-multistream-index$" => "")

    m = match(r"^([A-Za-z0-9_]+)-(\d{8})$", name)
    m === nothing && throw(ArgumentError(
        "Wikipedia dump key must look like 'jawiki-20260801', got '$dataset_name'"))
    return String(m.captures[1]), String(m.captures[2])
end

"""
Bunzip `src.bz2` to `dest` (defaults to path without `.bz2`), then remove the archive.
"""
function bunzip_file!(bz2_path::AbstractString; dest::Union{Nothing,AbstractString}=nothing)
    endswith(bz2_path, ".bz2") || throw(ArgumentError("Expected a .bz2 file, got $bz2_path"))
    out = dest === nothing ? String(bz2_path[1:end-4]) : String(dest)
    mkpath(dirname(out))
    if Sys.which("bunzip2") === nothing
        error("bunzip2 not found on PATH; install bzip2 or provide an already-unzipped file via --source=")
    end
    open(out, "w") do output
        open(`bunzip2 -c $bz2_path`) do input
            write(output, input)
        end
    end
    rm(bz2_path; force=true)
    return out
end

"""
Download `{wiki}-{date}-pages-articles-multistream-index.txt.bz2`, unzip under
`dest_dir`, and return the `.txt` path.
"""
function wikipedia_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    wiki, date = wikipedia_parse_dump_key(dataset_name)
    stem = "$(wiki)-$(date)-pages-articles-multistream-index"
    url = "$(WIKIPEDIA_DUMP_URL)/$(wiki)/$(date)/$(stem).txt.bz2"
    bz2_path = joinpath(dest_dir, "$(stem).txt.bz2")
    txt_path = joinpath(dest_dir, "$(stem).txt")

    println("  URL: $url")
    println("  Saving archive to: $bz2_path")
    Downloads.download(url, bz2_path)

    println("  Unzipping…")
    bunzip_file!(bz2_path; dest=txt_path)
    println("  Raw index: $txt_path")
    return txt_path
end

"""
Convert a multistream index (`u:v:title…`) to interactions CSV with columns `u,v`.
"""
function wikipedia_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
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

                # Titles may contain ':'; only split off the first two fields.
                parts = split(cleaned, ':'; limit=3)
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
    "wikipedia",
    wikipedia_download_raw!,
    wikipedia_convert!,
))
