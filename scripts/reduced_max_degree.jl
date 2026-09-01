#!/usr/bin/env julia
# Print reduced_max_degree for one vary.jl graph setup.
# Args: dataset seed k theta inject(0|1) [inject_u inject_v]
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "bin", "load.jl"))

dataset = ARGS[1]
seed = parse(UInt64, ARGS[2])
k = parse(Int, ARGS[3])
θ = parse(Int, ARGS[4])
do_inject = ARGS[5] == "1"
inj_u = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 5
inj_v = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 5

Random.seed!(seed)
graph_path = resolve_graph_path(dataset)
isfile(graph_path) || error("missing graph: $graph_path")

inject = (; enabled=do_inject, nU=inj_u, nV=inj_v, attempts=20)
g, _edges, _plant = load_graph_maybe_inject(graph_path, inject, k, Random.default_rng())
g_red = deepcopy(g)
fg = apply_graph_reductions!(g_red, k, θ, nothing, nothing, true, ReductionMode.simple)
max_deg, _avg_deg = reduced_degree_stats(fg)
println(max_deg)
