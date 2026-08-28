#=
=================================================================================
Rebuild datasets from a manifest file
=================================================================================
Reads one dataset key per line (e.g. `data/datasets.txt`) and runs the provider
pipeline with `--download` for each entry.

Usage:
  julia bin/fetch_datasets.jl
  julia bin/fetch_datasets.jl data/datasets.txt
  julia bin/fetch_datasets.jl --skip-existing
  julia bin/fetch_datasets.jl --prefix=konect-small
  julia bin/fetch_datasets.jl --dry-run

Options:
  --skip-existing   skip keys that already have indexed_interactions.csv
  --prefix=NAME     only fetch keys equal to or under NAME/ (slash-bounded)
  --dry-run         print planned actions without downloading
  --data-root=PATH  data directory (default: data)
=#

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "providers", "registry.jl"))

function parse_fetch_args(args)
    manifest = joinpath(ROOT, "data", "datasets.txt")
    skip_existing = false
    dry_run = false
    prefix = nothing
    data_root = "data"

    for arg in args
        if arg == "--skip-existing"
            skip_existing = true
        elseif arg == "--dry-run"
            dry_run = true
        elseif startswith(arg, "--prefix=")
            prefix = strip(split(arg, "=", limit=2)[2])
            isempty(prefix) && throw(ArgumentError("--prefix requires a value"))
        elseif startswith(arg, "--data-root=")
            data_root = strip(split(arg, "=", limit=2)[2])
            isempty(data_root) && throw(ArgumentError("--data-root requires a value"))
        elseif arg == "--help" || arg == "-h"
            return nothing
        elseif startswith(arg, "--")
            throw(ArgumentError("Unknown flag: $arg"))
        else
            manifest = arg
        end
    end

    return (; manifest, skip_existing, dry_run, prefix, data_root)
end

function print_fetch_usage()
    println(stderr, "Usage:")
    println(stderr, "  julia bin/fetch_datasets.jl [manifest.txt] [options]")
    println(stderr, "")
    println(stderr, "Options:")
    println(stderr, "  --skip-existing       skip when indexed_interactions.csv exists")
    println(stderr, "  --prefix=NAME         only keys under NAME/ (slash-bounded)")
    println(stderr, "  --dry-run             list actions without downloading")
    println(stderr, "  --data-root=PATH      data root (default: data)")
    println(stderr, "")
    println(stderr, "Registered providers: $(join(list_providers(), ", "))")
end

"""
Return true when `key` matches `prefix` using slash-bounded prefix rules
(same as order_graphs.jl: konect ≠ konect-small).
"""
function dataset_key_matches_prefix(key::AbstractString, prefix::AbstractString)
    p = replace(String(prefix), '\\' => '/')
    k = replace(String(key), '\\' => '/')
    return k == p || startswith(k, p * "/")
end

"""
Read manifest lines: one `provider/dataset` key per line; `#` comments allowed.
"""
function read_dataset_manifest(path::AbstractString)
    isfile(path) || throw(ArgumentError("Manifest not found: $path"))
    keys = String[]
    for line in eachline(path)
        s = strip(line)
        isempty(s) && continue
        startswith(s, '#') && continue
        push!(keys, replace(s, '\\' => '/'))
    end
    return keys
end

function main()
    try
        parsed = parse_fetch_args(ARGS)
        if parsed === nothing
            print_fetch_usage()
            exit(0)
        end

        keys = read_dataset_manifest(parsed.manifest)
        if parsed.prefix !== nothing
            filter!(k -> dataset_key_matches_prefix(k, parsed.prefix), keys)
        end

        if isempty(keys)
            println("No datasets to fetch (manifest=$(parsed.manifest)).")
            exit(0)
        end

        println("Manifest: $(parsed.manifest)  ($(length(keys)) dataset(s))")
        parsed.dry_run && println("Dry run — no downloads.")

        failed = String[]
        skipped = 0
        ran = 0

        for (i, key) in enumerate(keys)
            provider, dataset = split_provider_dataset(key)
            out_csv = joinpath(dataset_dir(key; data_root=parsed.data_root), "indexed_interactions.csv")

            println()
            println("[$(i)/$(length(keys))] $key")

            if parsed.skip_existing && isfile(out_csv)
                println("  Skipping (exists): $out_csv")
                skipped += 1
                continue
            end

            if parsed.dry_run
                println("  Would run: process_with_provider($(provider), $(dataset); download=true)")
                ran += 1
                continue
            end

            try
                process_with_provider(provider, dataset;
                    download=true, data_root=parsed.data_root)
                ran += 1
            catch err
                println(stderr, "  FAILED: $key")
                showerror(stderr, err, catch_backtrace())
                println(stderr)
                push!(failed, key)
            end
        end

        println()
        println("Done. Processed=$ran  Skipped=$skipped  Failed=$(length(failed))")
        if !isempty(failed)
            println(stderr, "Failed datasets:")
            for k in failed
                println(stderr, "  - $k")
            end
            exit(1)
        end
    catch e
        if e isa ArgumentError
            println(stderr, "Error: ", e.msg)
            print_fetch_usage()
            exit(1)
        else
            rethrow()
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
