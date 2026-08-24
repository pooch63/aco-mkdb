#=
=================================================================================
Load all registered graph providers
=================================================================================
=#

const __PROVIDER_REGISTRY_JL__ = true

include(joinpath(@__DIR__, "common.jl"))
include(joinpath(@__DIR__, "amazon.jl"))
include(joinpath(@__DIR__, "wikipedia.jl"))
include(joinpath(@__DIR__, "konect.jl"))
include(joinpath(@__DIR__, "konect-small.jl"))
include(joinpath(@__DIR__, "roadmap.jl"))
include(joinpath(@__DIR__, "catster.jl"))

# Add new providers here, e.g.:
# include(joinpath(@__DIR__, "twitter.jl"))
