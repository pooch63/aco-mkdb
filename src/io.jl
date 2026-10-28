# io.jl now lives in the shared GraphLogic package (../graph-logic/src/io.jl).
const __IO_JL__ = true
isdefined(@__MODULE__, :__GRAPHLOGIC_JL__) || include(joinpath(@__DIR__, "graphlogic.jl"))
