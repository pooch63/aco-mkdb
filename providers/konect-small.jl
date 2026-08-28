#=
=================================================================================
konect-small provider
=================================================================================
Same download/convert as `konect`, written under `data/konect-small/<dataset>/`.

Examples:
  julia bin/process.jl konect-small/euroroad --download
  julia bin/process.jl konect-small/facebook --download
=#

const __PROVIDER_konect_small_JL__ = true

isdefined(@__MODULE__, :__PROVIDER_konect_JL__) || include(joinpath(@__DIR__, "konect.jl"))

register_provider!(ProviderAdapter(
    "konect-small",
    konect_download_raw!,
    konect_convert_edgelist!,
))
