#=
=================================================================================
TCGA (The Cancer Genome Atlas) provider
=================================================================================
Pulls RNA-seq gene expression from the GDC REST API (no R / TCGAbiolinks),
thresholds per-gene z-scores into a bipartite sample–gene edge list, then
writes the shared interactions CSV (user_id=sample, item_id=gene, timestamp=0).

Required packages (install once):
  using Pkg
  Pkg.add(["HTTP", "JSON3", "CSV", "DataFrames", "CodecZlib", "Statistics"])

Example:
  julia process.jl tcga/TCGA-BRCA --download
  julia process.jl tcga/BRCA --download          # TCGA- prefix added automatically
  julia process.jl tcga/TCGA-BRCA --source=data/tcga/TCGA-BRCA/raw/manifest.csv
=#

const __PROVIDER_TCGA_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

using HTTP
using JSON3
using CSV
using DataFrames
using CodecZlib
using Statistics

const GDC_API = "https://api.gdc.cancer.gov"

# Tunables for the download → threshold pipeline.
const TCGA_MAX_FILES = 50
const TCGA_Z_THRESH = 2.0
const TCGA_VALUE_COL = :tpm_unstranded
const TCGA_WORKFLOW = "STAR - Counts"
const TCGA_DATA_TYPE = "Gene Expression Quantification"

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

"""
Normalize a dataset leaf name to a GDC project id (e.g. BRCA → TCGA-BRCA).
"""
function tcga_project_id(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    upper = uppercase(name)
    return startswith(upper, "TCGA-") ? upper : "TCGA-" * upper
end

# ---------------------------------------------------------------------
# GDC query / download / parse / threshold
# ---------------------------------------------------------------------

"""
Query the GDC `/files` endpoint for RNA-seq gene expression files for a project.
Returns a DataFrame: file_id, file_name, case_id, submitter_id.
"""
function tcga_query_case_files(project_id::String;
    workflow_type::String=TCGA_WORKFLOW,
    data_type::String=TCGA_DATA_TYPE,
    size::Int=2000)

    payload = Dict(
        "filters" => Dict(
            "op" => "and",
            "content" => [
                Dict("op" => "in", "content" => Dict(
                    "field" => "cases.project.project_id", "value" => [project_id])),
                Dict("op" => "in", "content" => Dict(
                    "field" => "data_type", "value" => [data_type])),
                Dict("op" => "in", "content" => Dict(
                    "field" => "analysis.workflow_type", "value" => [workflow_type])),
            ],
        ),
        "fields" => "file_id,file_name,cases.case_id,cases.submitter_id",
        "format" => "JSON",
        "size" => string(size),
    )

    resp = HTTP.post(
        GDC_API * "/files",
        ["Content-Type" => "application/json"],
        JSON3.write(payload),
    )
    result = JSON3.read(String(resp.body))

    rows = NamedTuple[]
    for hit in result.data.hits
        case = hit.cases[1]
        push!(rows, (
            file_id=String(hit.file_id),
            file_name=String(hit.file_name),
            case_id=String(case.case_id),
            submitter_id=String(case.submitter_id),
        ))
    end
    return DataFrame(rows)
end

"""
Download a single GDC file by UUID into dest_dir. Skips if already present.
"""
function tcga_download_file(file_id::String, dest_dir::String)
    mkpath(dest_dir)
    dest_path = joinpath(dest_dir, file_id * ".tsv.gz")
    if isfile(dest_path)
        return dest_path
    end
    r = HTTP.get(GDC_API * "/data/" * file_id)
    open(dest_path, "w") do io
        write(io, r.body)
    end
    return dest_path
end

"""
Download every file in files_df; adds a `local_path` column.
"""
function tcga_download_all(files_df::DataFrame, dest_dir::String; verbose::Bool=true)
    paths = String[]
    n = nrow(files_df)
    for (i, row) in enumerate(eachrow(files_df))
        verbose && println("  Downloading $(i)/$(n): $(row.file_name)")
        push!(paths, tcga_download_file(row.file_id, dest_dir))
    end
    files_df = copy(files_df)
    files_df.local_path = paths
    return files_df
end

"""
Parse a gzipped GDC STAR - Counts file into gene_id / value columns.
"""
function tcga_parse_star_counts(path::String; value_col::Symbol=TCGA_VALUE_COL)
    df = open(path) do io
        stream = GzipDecompressorStream(io)
        CSV.read(stream, DataFrame; delim='\t', header=1)
    end
    filter!(row -> startswith(String(row.gene_id), "ENSG"), df)
    return DataFrame(gene_id=df.gene_id, value=df[!, value_col])
end

"""
Build long-format sample/gene/value table from a downloaded files manifest.
"""
function tcga_build_long_expression(files_df::DataFrame; value_col::Symbol=TCGA_VALUE_COL)
    pieces = DataFrame[]
    for row in eachrow(files_df)
        gdf = tcga_parse_star_counts(row.local_path; value_col=value_col)
        gdf.u_id = fill(row.submitter_id, nrow(gdf))
        rename!(gdf, :gene_id => :v_id, :value => :weight)
        push!(pieces, gdf[:, [:u_id, :v_id, :weight]])
    end
    isempty(pieces) && return DataFrame(u_id=String[], v_id=String[], weight=Float64[])
    return vcat(pieces...)
end

"""
Keep edges where a gene is significantly over-expressed in a sample (z > z_thresh).
"""
function tcga_edges_from_zscore(long_df::DataFrame; z_thresh::Float64=TCGA_Z_THRESH)
    out = DataFrame(u_id=String[], v_id=String[], weight=Float64[])
    for gdf in groupby(long_df, :v_id)
        vals = gdf.weight
        mu = mean(vals)
        sigma = std(vals)
        sigma == 0 && continue
        z = (vals .- mu) ./ sigma
        keep = z .> z_thresh
        if any(keep)
            append!(out, DataFrame(
                u_id=gdf.u_id[keep],
                v_id=gdf.v_id[keep],
                weight=gdf.weight[keep],
            ))
        end
    end
    return out
end

# ---------------------------------------------------------------------
# Provider adapter hooks
# ---------------------------------------------------------------------

"""
Query GDC, download STAR counts into `dest_dir`, write `manifest.csv`, return its path.
"""
function tcga_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    project_id = tcga_project_id(dataset_name)
    mkpath(dest_dir)

    println("  Project: $project_id")
    println("  Querying GDC…")
    files_df = tcga_query_case_files(project_id)
    n_found = nrow(files_df)
    n_use = min(TCGA_MAX_FILES, n_found)
    println("  Found $n_found files; using first $n_use (TCGA_MAX_FILES=$TCGA_MAX_FILES).")
    files_df = first(files_df, n_use)

    files_dir = joinpath(dest_dir, "files")
    println("  Downloading expression files to $files_dir …")
    files_df = tcga_download_all(files_df, files_dir)

    manifest_path = joinpath(dest_dir, "manifest.csv")
    CSV.write(manifest_path, files_df)
    println("  Manifest: $manifest_path")
    return manifest_path
end

"""
Convert a TCGA download manifest into `user_id,item_id,timestamp` interactions CSV.
"""
function tcga_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
    path = String(raw_path)
    isfile(path) || error("TCGA raw input not found: $path")

    println("  Reading manifest: $path")
    files_df = CSV.read(path, DataFrame)
    required = ["submitter_id", "local_path"]
    missing_cols = setdiff(required, String.(names(files_df)))
    isempty(missing_cols) || error(
        "TCGA manifest must contain columns $(join(required, ", ")); missing: $(join(missing_cols, ", "))")

    # Re-resolve paths relative to the manifest if needed (portable re-runs).
    for i in 1:nrow(files_df)
        p = String(files_df.local_path[i])
        if !isfile(p)
            alt = joinpath(dirname(path), "files", basename(p))
            isfile(alt) || error("Expression file missing: $p")
            files_df.local_path[i] = alt
        end
    end

    println("  Parsing STAR counts…")
    long_df = tcga_build_long_expression(files_df; value_col=TCGA_VALUE_COL)
    println("  Long expression rows: $(nrow(long_df))")

    println("  Thresholding edges (z > $TCGA_Z_THRESH)…")
    edges = tcga_edges_from_zscore(long_df; z_thresh=TCGA_Z_THRESH)
    n_samples = length(unique(edges.u_id))
    n_genes = length(unique(edges.v_id))
    println("  Kept $(nrow(edges)) edges ($n_samples samples × $n_genes genes)")

    mkpath(dirname(String(interactions_csv)))
    out = DataFrame(
        user_id=edges.u_id,
        item_id=edges.v_id,
        timestamp=fill(0, nrow(edges)),
    )
    CSV.write(interactions_csv, out)
    return nothing
end

register_provider!(ProviderAdapter(
    "tcga",
    tcga_download_raw!,
    tcga_convert!,
))
