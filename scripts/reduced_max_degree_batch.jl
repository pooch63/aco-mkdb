#!/usr/bin/env julia
# Batch-compute reduced_max_degree for vary.jl result JSON setups.
# Usage: julia scripts/reduced_max_degree_batch.jl jobs.json
# jobs.json: [{ "dataset", "seed", "k", "theta", "inject", "key" }, ...]
# Prints one JSON object: { "nU,nV,E": max_deg, ... }
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "bin", "load.jl"))
using JSON3

jobs = JSON3.read(read(ARGS[1], String), Vector{Dict{String,Any}})
out = Dict{String,Int}()

for job in jobs
    key = string(job["key"])
    haskey(out, key) && continue
    dataset = string(job["dataset"])
    seed = parse(UInt64, string(job["seed"]))
    k = Int(job["k"])
    θ = Int(job["theta"])
    do_inject = Bool(job["inject"])
    Random.seed!(seed)
    graph_path = resolve_graph_path(dataset)
    isfile(graph_path) || continue
    inject = (; enabled=do_inject, nU=5, nV=5, attempts=20)
    g, _edges, _plant = load_graph_maybe_inject(graph_path, inject, k, Random.default_rng())
    g_red = deepcopy(g)
    fg = apply_graph_reductions!(g_red, k, θ, nothing, nothing, true, ReductionMode.simple)
    max_deg, avg_deg = reduced_degree_stats(fg)
    nU = length(g.adjU)
    nV = length(g.adjV)
    inject_cfg = if do_inject
        (; enabled=true, nU=5, nV=5, attempts=20)
    else
        (; enabled=false, nU=0, nV=0, attempts=0)
    end
    structure_key = graph_structure_cache_key(k, θ, ReductionMode.simple, inject_cfg; seed=seed)
    store_graph_structure_cache!(dataset, structure_key, nU, nV, _edges, length(fg.u_ids),
        length(fg.v_ids), length(fg.v_adj), max_deg, avg_deg)
    out[key] = max_deg
end

print(JSON3.write(out))
