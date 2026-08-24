#=
=================================================================================
konect-small provider
=================================================================================
Same download/convert as `konect`, written under `data/konect-small/<dataset>/`.

Examples:
  julia process.jl konect-small/Newman-Cond_mat_95-99-two_mode --download
  julia process.jl konect-small/Newman-Cond_mat_95-99-two_mode --source=data/Newman-Cond_mat_95-99-two_mode.txt
=#

const __PROVIDER_konect_small_JL__ = true

isdefined(@__MODULE__, :__PROVIDER_konect_JL__) || include(joinpath(@__DIR__, "konect.jl"))

register_provider!(ProviderAdapter(
    "konect-small",
    konect_download_raw!,
    konect_convert!,
))
