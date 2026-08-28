#=
=================================================================================
Amazon Reviews 2023 provider
=================================================================================
Downloads category review JSONL from McAuley Lab and converts via edges.jl.

URL template:
  https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/{NAME}.jsonl.gz

Examples:
  julia process.jl amazon/boxes --download
  julia process.jl amazon/Appliances --download
=#

const __PROVIDER_AMAZON_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))

# Amazon JSONL → interactions CSV (user_id→u, asin→v)
if !isdefined(@__MODULE__, :convert_jsonl_to_csv)
    include(joinpath(@__DIR__, "..", "src", "edges.jl"))
end

const AMAZON_REVIEW_URL =
    "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories"

"""
Short local keys → Amazon Reviews 2023 category filenames.
"""
const AMAZON_CATEGORY_ALIASES = Dict{String,String}(
    "boxes" => "Subscription_Boxes",
    "appliances" => "Appliances",
    "music" => "Digital_Music",
    "grocery" => "Grocery_and_Gourmet_Food",
)

function amazon_category_name(dataset_name::AbstractString)
    name = String(dataset_name)
    name = split(replace(name, '\\' => '/'), '/')[end]
    key = lowercase(name)
    return get(AMAZON_CATEGORY_ALIASES, key, name)
end

"""
Download `{NAME}.jsonl.gz`, unzip to `{NAME}.jsonl` under `dest_dir`, return the jsonl path.
"""
function amazon_download_raw!(dataset_name::AbstractString, dest_dir::AbstractString)
    category = amazon_category_name(dataset_name)
    leaf = split(replace(String(dataset_name), '\\' => '/'), '/')[end]
    lowercase(leaf) != lowercase(category) && println("  Amazon category: $category")

    url = "$(AMAZON_REVIEW_URL)/$(category).jsonl.gz"
    gz_path = joinpath(dest_dir, "$(category).jsonl.gz")
    jsonl_path = joinpath(dest_dir, "$(category).jsonl")

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
