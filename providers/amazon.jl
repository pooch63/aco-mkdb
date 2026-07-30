#=
=================================================================================
Amazon Reviews 2023 provider
=================================================================================
Downloads category review JSONL from McAuley Lab and converts via edges.jl.

URL template:
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{NAME}.jsonl.gz

Example:
  julia process.jl amazon/Appliances --download
=#

const __PROVIDER_AMAZON_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

# Amazon JSONL → interactions CSV (user_id, asin→item_id, timestamp)
if !isdefined(@__MODULE__, :convert_jsonl_to_csv)
    include(joinpath(@__DIR__, "..", "src", "edges.jl"))
end

const AMAZON_REVIEW_URL =
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"

"""
Download `{NAME}.jsonl.gz`, unzip to `{NAME}.jsonl` under `dest_dir`, return the jsonl path.
"""
function amazon_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    name = String(dataset_name)
    # Allow nested leftover segments; Amazon category is the leaf name.
    name = split(replace(name, '\\' => '/'), '/')[end]

    url = "$(AMAZON_REVIEW_URL)/$(name).jsonl.gz"
    gz_path = joinpath(dest_dir, "$(name).jsonl.gz")
    jsonl_path = joinpath(dest_dir, "$(name).jsonl")

    println("  URL: $url")
    println("  Saving archive to: $gz_path")
    Downloads.download(url, gz_path)

    println("  Unzipping…")
    gunzip_file!(gz_path; dest=jsonl_path)
    println("  Raw JSONL: $jsonl_path")
    return jsonl_path
end

function amazon_convert!(raw_path::AbstractString, interactions_csv::AbstractString)
    convert_jsonl_to_csv(String(raw_path), String(interactions_csv))
    return nothing
end

register_provider!(ProviderAdapter(
    "amazon",
    amazon_download_raw!,
    amazon_convert!,
))
