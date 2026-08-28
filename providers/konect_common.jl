#=
=================================================================================
Shared KONECT download / extract helpers
=================================================================================
KONECT hosts downloadable archives at:

  http://konect.cc/files/download.tsv.{INTERNAL_NAME}.tar.bz2

Each archive unpacks to `{INTERNAL_NAME}/out.*` (whitespace-separated edgelist).

Local dataset keys under data/konect* may differ from KONECT internal names; see
`konect_remote_name` and optional overrides in `data/konect_aliases.txt`.
=#

const __KONECT_COMMON_JL__ = true

const KONECT_FILES_URL = "http://konect.cc/files"
const KONECT_OPSAHL_URL = "http://opsahl.co.uk/konect/datasets"

"""
Built-in local-key → KONECT-internal-name overrides for this repo's datasets.
Optional `data/konect_aliases.txt` can add or replace entries (`local=remote`).
"""
const KONECT_BUILTIN_ALIASES = Dict{String,String}(
    "euroroad" => "subelj_euroroad",
    "facebook" => "ego-facebook",
    "urwiki" => "edit-urwiki",
    "nawiki" => "edit-nawiki",
    "jawiki" => "edit-jawiki",
    "eswiki" => "edit-eswiki",
    "dblp_cite" => "dblp-cite",
    "dimacs-polblogs" => "dimacs10-polblogs",
    "pgp-arenas" => "arenas-pgp",
    "ciaodvd" => "librec-ciaodvd-trust",
    "matter" => "opsahl-collaboration",
    "bitcoin" => "soc-sign-bitcoinalpha",
    "gnutella" => "p2p-Gnutella04",
)

const _KONECT_ALIASES_LOADED = Ref(false)
const KONECT_ALIASES = Dict{String,String}()

function _konect_aliases_path()
    return joinpath(@__DIR__, "..", "data", "konect_aliases.txt")
end

"""
Load `data/konect_aliases.txt` once (if present) into `KONECT_ALIASES`.
Lines are `local=remote`; blank lines and `#` comments are ignored.
"""
function load_konect_aliases!()
    _KONECT_ALIASES_LOADED[] && return KONECT_ALIASES
    empty!(KONECT_ALIASES)
    for (k, v) in KONECT_BUILTIN_ALIASES
        KONECT_ALIASES[k] = v
    end
    path = _konect_aliases_path()
    if isfile(path)
        for line in eachline(path)
            s = strip(line)
            isempty(s) && continue
            startswith(s, '#') && continue
            parts = split(s, '=', limit=2)
            length(parts) == 2 || continue
            local_name = strip(parts[1])
            remote_name = strip(parts[2])
            isempty(local_name) && continue
            isempty(remote_name) && continue
            KONECT_ALIASES[local_name] = remote_name
        end
    end
    _KONECT_ALIASES_LOADED[] = true
    return KONECT_ALIASES
end

"""
Normalize a dataset key to the leaf stem (no path / archive prefix).
"""
function konect_dataset_stem(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    name = replace(name, r"^download\.tsv\." => "")
    name = replace(name, r"\.tar\.bz2$" => "")
    name = replace(name, r"^out\." => "")
    name = replace(name, r"-uniq$" => "")
    name = replace(name, r"\.txt$" => "")
    isempty(name) && throw(ArgumentError("Empty KONECT dataset name from '$dataset_name'"))
    return name
end

"""
Map a local dataset stem to the KONECT internal archive name.

Uses `data/konect_aliases.txt` and built-in overrides when present; otherwise
returns the local stem unchanged. KONECT uses both hyphens and underscores
(e.g. `wiki_talk_nds`, `dblp-cite`), so we do not guess by rewriting `_` → `-`.
"""
function konect_remote_name(local_stem::AbstractString)
    load_konect_aliases!()
    stem = konect_dataset_stem(local_stem)
    return get(KONECT_ALIASES, stem, stem)
end

"""
Locate the KONECT `out.*` edgelist under an extracted network directory.
"""
function konect_find_out_file(extract_dir::AbstractString)
    isdir(extract_dir) || error("Expected extracted network directory at '$extract_dir'")
    outs = filter(f -> startswith(basename(f), "out."), readdir(extract_dir; join=true))
    filter!(isfile, outs)
    isempty(outs) && error("No out.* edgelist found under '$extract_dir'")
    return first(sort!(outs))
end

"""
Extract `archive` into `dest_dir` using `tar`.
"""
function konect_extract_archive!(archive::AbstractString, dest_dir::AbstractString)
    mkpath(dest_dir)
    if Sys.which("tar") === nothing
        error("tar not found on PATH; install tar or provide an already-extracted out.* file via --source=")
    end
    # -j handles .tar.bz2; plain -x also works on many systems.
    run(`tar -xjf $archive -C $dest_dir`)
    return nothing
end

"""
Download `download.tsv.{remote}.tar.bz2`, extract under `dest_dir`, return `out.*` path.
"""
function konect_download_tsv_archive!(remote_name::AbstractString, dest_dir::AbstractString)
    url = "$(KONECT_FILES_URL)/download.tsv.$(remote_name).tar.bz2"
    archive = joinpath(dest_dir, "download.tsv.$(remote_name).tar.bz2")
    extract_dir = joinpath(dest_dir, remote_name)

    println("  URL: $url")
    println("  Saving archive to: $archive")
    mkpath(dest_dir)
    Downloads.download(url, archive)

    println("  Extracting…")
    konect_extract_archive!(archive, dest_dir)
    rm(archive; force=true)

    out_path = konect_find_out_file(extract_dir)
    println("  Raw edgelist: $out_path")
    return out_path
end

"""
Fallback: plain `{stem}.txt` edgelist from the legacy Opsahl KONECT mirror.
"""
function konect_download_opsahl_txt!(stem::AbstractString, dest_dir::AbstractString)
    url = "$(KONECT_OPSAHL_URL)/$(stem).txt"
    txt_path = joinpath(dest_dir, "$(stem).txt")

    println("  URL (opsahl fallback): $url")
    println("  Saving to: $txt_path")
    mkpath(dest_dir)
    Downloads.download(url, txt_path)
    println("  Raw edgelist: $txt_path")
    return txt_path
end

"""
Download a KONECT network for local dataset `dataset_name` into `dest_dir`.

Tries konect.cc TSV archive first, then the Opsahl `.txt` mirror using the local stem.
"""
function konect_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    local_stem = konect_dataset_stem(dataset_name)
    remote = konect_remote_name(local_stem)
    remote == local_stem ||
        println("  KONECT remote name: $remote (local key: $local_stem)")

    try
        return konect_download_tsv_archive!(remote, dest_dir)
    catch err
        @warn "KONECT TSV download failed; trying Opsahl .txt mirror" local_stem remote exception=err
        return konect_download_opsahl_txt!(local_stem, dest_dir)
    end
end

"""
Convert a KONECT edgelist (`u v [w] [t]`, whitespace-separated; `%` / `#` comments and
blank lines ignored) to interactions CSV with columns `u,v`.
"""
function konect_convert_edgelist!(raw_path::AbstractString, interactions_csv::AbstractString)
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
