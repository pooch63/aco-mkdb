#=
=================================================================================
konect provider
=================================================================================
Downloads a KONECT TSV network archive from konect.cc (with Opsahl .txt fallback)
and converts the edgelist to the shared interactions CSV.

URL template (primary):
  http://konect.cc/files/download.tsv.{INTERNAL_NAME}.tar.bz2

URL template (fallback):
  http://opsahl.co.uk/konect/datasets/{LOCAL_NAME}.txt

Examples:
  julia bin/process.jl konect/bitcoin --download
  julia bin/process.jl konect/dblp_cite --download
  julia bin/process.jl konect/bitcoin --source=data/konect/bitcoin/raw/bitcoin.txt
=#

const __PROVIDER_konect_JL__ = true

isdefined(@__MODULE__, :__PROVIDERS_JL__) || include(joinpath(@__DIR__, "common.jl"))
isdefined(@__MODULE__, :__KONECT_COMMON_JL__) || include(joinpath(@__DIR__, "konect_common.jl"))

register_provider!(ProviderAdapter(
    "konect",
    konect_download_raw!,
    konect_convert_edgelist!,
))
