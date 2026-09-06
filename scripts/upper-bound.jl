#!/usr/bin/env julia
#=
Tight edge upper bound on the CNN-reduced graph for a vary.jl result JSON,
using `upper_bound` from src/search.jl (called from opponent.jl pruning).

Usage:
  julia scripts/upper-bound.jl results/vary_k2t5i_PN/magazine_ants.json
  julia scripts/upper-bound.jl path/to/*_ants.json --verbose

Default stdout is a single integer: upper_e for S=∅, C=V(G_R), S_missing=0.
With an empty incumbent every candidate has nondegree 0 into S, so
upper_e = |U_R|·|V_R| — still the root bound branch-and-bound uses before
branching.
=#

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "bin", "load.jl"))

using JSON3
using Random

function input_path()
    for arg in ARGS
        !startswith(arg, "--") && return arg
    end
    return nothing
end

function want_verbose()
    return any(a -> a in ("--verbose", "-v"), ARGS) ||
           lowercase(get(ENV, "VERBOSE", "0")) in ("1", "true", "yes", "on")
end

function folder_uses_inject(json_path::AbstractString)
    dir = basename(dirname(json_path))
    return occursin(r"k\d+t\d+i", dir)
end

function load_vary_meta(json_path)
    data = JSON3.read(read(json_path, String), Dict{String, Any})
    dataset = String(get(data, "dataset", ""))
    isempty(dataset) && error("JSON missing dataset: $json_path")
    k = Int(get(data, "k", 2))
    θ = Int(get(data, "theta", 5))
    seed = get(data, "base_seed", get(data, "seed", "1"))
    seed = parse(UInt64, string(seed))
    red_raw = lowercase(string(get(data, "reduction", "simple")))
    reduction = if red_raw in ("simple", "lo")
        ReductionMode.simple
    elseif red_raw == "none"
        ReductionMode.none
    else
        # vary ant-count always uses CNN/simple for the recorded reduced graph.
        ReductionMode.simple
    end
    return (; dataset, k, θ, seed, reduction)
end

function reduced_upper_bound(json_path::AbstractString; quiet::Bool=true)
    meta = load_vary_meta(json_path)
    do_inject = folder_uses_inject(json_path)
    inject = (; enabled=do_inject, nU=meta.θ, nV=meta.θ, attempts=20)

    Random.seed!(meta.seed)
    graph_path = resolve_graph_path(meta.dataset)
    isfile(graph_path) || error("missing graph CSV: $graph_path")

    # load_bipartite_graph / inject / reduction print progress; keep stdout clean
    # so the default mode can return a single integer.
    sink = quiet ? devnull : stdout
    g, _edges, _plant = redirect_stdout(sink) do
        load_graph_maybe_inject(graph_path, inject, meta.k, Random.default_rng())
    end
    g_red = deepcopy(g)
    fg = redirect_stdout(sink) do
        if meta.reduction == ReductionMode.none
            freeze(g_red)
        else
            apply_graph_reductions!(g_red, meta.k, meta.θ, nothing, nothing, true, meta.reduction)
        end
    end

    S = SubGraph(Set{Int}(), Set{Int}())
    C = SubGraph(Set(fg.u_ids), Set(fg.v_ids))
    bind_membership!(S, fg)
    bind_membership!(C, fg)
    upper_u, upper_v, upper_e = upper_bound(S, C, fg, meta.k, 0)
    return (; meta..., inject=do_inject,
            reduced_nU=length(fg.u_ids), reduced_nV=length(fg.v_ids),
            upper_u, upper_v, upper_e)
end

function main()
    path = input_path()
    path === nothing && error("usage: upper-bound.jl <vary_*_ants.json> [--verbose]")
    isfile(path) || error("file not found: $path")

    verbose = want_verbose()
    r = reduced_upper_bound(path; quiet=!verbose)
    if verbose
        println("dataset\t", r.dataset)
        println("k\t", r.k)
        println("theta\t", r.θ)
        println("inject\t", r.inject)
        println("reduced_nU\t", r.reduced_nU)
        println("reduced_nV\t", r.reduced_nV)
        println("upper_u\t", r.upper_u)
        println("upper_v\t", r.upper_v)
        println("upper_e\t", r.upper_e)
    else
        println(r.upper_e)
    end
end

main()
