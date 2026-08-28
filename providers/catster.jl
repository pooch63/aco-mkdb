#=
=================================================================================
catster (KONECT Petster / Catster) provider
=================================================================================
Same KONECT download path as konect / konect-small; kept as a separate provider key
for historical dataset layout.

Examples:
  julia bin/process.jl catster/petster-friendships-cat --download
=#

const __PROVIDER_CATSTER_JL__ = true

isdefined(@__MODULE__, :__PROVIDER_konect_JL__) || include(joinpath(@__DIR__, "konect.jl"))

register_provider!(ProviderAdapter(
    "catster",
    konect_download_raw!,
    konect_convert_edgelist!,
))
