# graph.jl now lives in the shared GraphLogic package (../graph-logic/src/graph.jl).
const __GRAPH_JL__ = true
isdefined(@__MODULE__, :__GRAPHLOGIC_JL__) || include(joinpath(@__DIR__, "graphlogic.jl"))
