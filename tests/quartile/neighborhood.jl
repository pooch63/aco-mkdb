"""
Neighborhood-Jaccard (S_C) edge-scoring test on real-world graphs.

For every edge e = (u, v) we compute:

    C_U(u, v) = mean of top-θ J(N(u), N(u')) over u' ∈ N(v) \ {u}
    C_V(u, v) = mean of top-θ J(N(v), N(v')) over v' ∈ N(u) \ {v}
    S_C(u, v) = C_U(u, v) · C_V(u, v)

where J(A, B) = |A ∩ B| / |A ∪ B| is the Jaccard similarity.

We inject a planted biclique, compute S_C on the full graph and the
CNN-reduced graph, then print a recall table at 1 / 5 / 10 / 20 / 30 %
of edges retained — mirroring the structure of butterfly.jl.

Usage:
  julia tests/quartile/neighborhood.jl              # first 5 graphs (ascending |E|)
  julia tests/quartile/neighborhood.jl amazon/boxes # one specific graph
  julia tests/quartile/neighborhood.jl --N=10
  julia tests/quartile/neighborhood.jl --prefix=konect-small --N=3
"""

using Random

include(joinpath(@__DIR__, "graph_selection.jl"))

isdefined(@__MODULE__, :__PATHS_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "pulse.jl"))   # pulls in graph.jl + neighborhood.jl
isdefined(@__MODULE__, :__LOAD_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "bin", "load.jl")) # inject_biclique! + apply_graph_reductions!

const K = 2
const θ = 5

# ── Recall thresholds ──────────────────────────────────────────────────────────
const THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30]

# ── S_C combination variants to benchmark ─────────────────────────────────────
# Each entry is (label, combine_fn) matching the `combine` kwarg of neighborhood_scores.
const COMBINE_VARIANTS = [
    ("product  [C_U · C_V]", (*)),
    ("min      [min(C_U, C_V)]", min),
]

# ── Recall computation (shared with butterfly.jl style) ───────────────────────

"""
    recall_at_thresholds(scores, plant_slots, thresholds) -> Vector{Float64}

Given S_C `scores` (parallel to fg.v_adj), the CSR slot indices of the planted
biclique's existing edges, and a vector of top-fraction thresholds, return the
recall (fraction of plant edges recovered) at each threshold.
"""
function recall_at_thresholds(scores::Vector{Float64},
                               plant_slots::Vector{Int},
                               thresholds::Vector{Float64})::Vector{Float64}
    nE      = length(scores)
    n_plant = length(plant_slots)
    n_plant == 0 && return fill(NaN, length(thresholds))

    order       = sortperm(scores, rev=true)
    plant_set   = Set(plant_slots)

    recalls      = Float64[]
    plant_found  = 0
    ptr          = 0

    for threshold in thresholds
        n_keep = max(1, round(Int, threshold * nE))
        while ptr < n_keep
            ptr += 1
            if order[ptr] in plant_set
                plant_found += 1
            end
        end
        push!(recalls, plant_found / n_plant)
    end
    return recalls
end

"""
    plant_slots_in(fg, chosen_U, chosen_V, missing_edges) -> Vector{Int}

Return the forward-CSR slot indices for planted biclique edges that exist in
`fg` (i.e. not in `missing_edges` and not removed by reduction).
"""
function plant_slots_in(fg::FrozenBipartite,
                         chosen_U::Vector{Int}, chosen_V::Vector{Int},
                         missing_edges::Set{Tuple{Int,Int}})::Vector{Int}
    slots = Int[]
    for u in chosen_U
        ui = get(fg.u_index, u, nothing)
        ui === nothing && continue
        r_u = neighbor_range_u(fg, ui)
        for v in chosen_V
            (u, v) in missing_edges && continue
            vi = get(fg.v_index, v, nothing)
            vi === nothing && continue
            for k in r_u
                if fg.v_adj[k] == vi
                    push!(slots, k)
                    break
                end
            end
        end
    end
    return slots
end

# ── Pretty table ───────────────────────────────────────────────────────────────

function print_recall_table(label::String,
                             thresholds::Vector{Float64},
                             recalls::Vector{Float64},
                             nE::Int, n_plant::Int)
    println("    ┌─ $label  (|E|=$nE, plant edges=$n_plant)")
    println("    │   Top-%  │  Edges kept  │  Recall")
    println("    │──────────┼──────────────┼──────────")
    for (th, rc) in zip(thresholds, recalls)
        kept   = max(1, round(Int, th * nE))
        pct_s  = lpad("$(round(Int, 100*th))%", 4)
        kept_s = lpad(string(kept), 10)
        rc_s   = isnan(rc) ? "    N/A" : lpad("$(round(100*rc; digits=1))%", 7)
        println("    │   $pct_s    │  $kept_s  │ $rc_s")
    end
    println("    └" * "─"^48)
end

# ── Per-graph test ─────────────────────────────────────────────────────────────

function test_neighborhood_on(dataset_key::AbstractString, path::AbstractString,
                               edge_count::Int)
    println("\n── $dataset_key ──")
    println("  Loading: $path  (|E|=$edge_count)")

    g, _ = load_bipartite_graph(path)
    println("  Graph: |U|=$(length(g.adjU))  |V|=$(length(g.adjV))  |E|=$edge_count")

    # Inject planted biclique (same RNG seed for all variants)
    rng = MersenneTwister(1)
    chosen_U, chosen_V, inserted, missing_edges, existing =
        inject_biclique!(g, θ, θ, K, rng)
    println("  Injected biclique: u=$θ v=$θ k=$K  " *
            "(existing=$existing inserted=$inserted missing=$(length(missing_edges)))")
    println("    planted U: $chosen_U")
    println("    planted V: $chosen_V")

    # Freeze full graph and compute plant slots once (shared across variants)
    fg_full = freeze(g)
    slots_full = plant_slots_in(fg_full, chosen_U, chosen_V, missing_edges)

    # Freeze reduced graph and compute plant slots once
    g_red  = deepcopy(g)
    nU_h   = length(g_red.adjU)
    nV_h   = length(g_red.adjV)
    fg_red    = apply_graph_reductions!(g_red, K, θ, nU_h, nV_h, false, ReductionMode.simple)
    slots_red = plant_slots_in(fg_red, chosen_U, chosen_V, missing_edges)
    n_red_E   = length(fg_red.v_adj)

    # ── Run each S_C variant ───────────────────────────────────────────────────
    for (var_label, combine_fn) in COMBINE_VARIANTS
        println("\n  ┄┄ combine: $var_label ┄┄")

        # Full graph
        t_full  = @elapsed scores_full = neighborhood_scores(fg_full; θ=θ, combine=combine_fn)
        recalls_full = recall_at_thresholds(scores_full, slots_full, THRESHOLDS)
        println("  [FULL GRAPH]  |E|=$(length(scores_full))  (scored in $(round(t_full; digits=3))s)")
        print_recall_table("S_C on full graph", THRESHOLDS, recalls_full,
                           length(scores_full), length(slots_full))

        # Reduced graph
        t_red   = @elapsed scores_red = neighborhood_scores(fg_red; θ=θ, combine=combine_fn)
        recalls_red = recall_at_thresholds(scores_red, slots_red, THRESHOLDS)
        println("  [REDUCED GRAPH]  |U|=$(length(fg_red.u_ids))  " *
                "|V|=$(length(fg_red.v_ids))  |E|=$n_red_E  " *
                "(scored in $(round(t_red; digits=3))s)")
        print_recall_table("S_C on reduced graph", THRESHOLDS, recalls_red,
                           n_red_E, length(slots_red))

        r30_full = recalls_full[end]
        r30_red  = recalls_red[end]
        println("  Summary ($var_label): recall@30%  " *
                "full=$(round(100*r30_full; digits=1))%  " *
                "reduced=$(round(100*r30_red; digits=1))%")
    end

    println("  ✓ done")
    return true
end

# ── Main ────────────────────────────────────────────────────────────────────────

graphs = selected_graphs()

if isempty(graphs)
    println("No graphs found. Check that data/ is populated and --prefix= is correct.")
    exit(1)
end

println("Neighborhood-Jaccard (S_C) edge-score recall test on $(length(graphs)) graph(s) (ascending |E|):")
println("Thresholds: $(join(string.(round.(Int, 100 .* THRESHOLDS)) .* "%", ", "))")

failed = String[]
for g in graphs
    try
        test_neighborhood_on(g.key, g.path, g.edges)
    catch e
        println("  ✗ FAILED: $(g.key): $e")
        push!(failed, g.key)
        rethrow()
    end
end

println()
if isempty(failed)
    println("All $(length(graphs)) neighborhood test(s) completed.")
else
    println("$(length(failed)) / $(length(graphs)) test(s) FAILED:")
    for key in failed
        println("  $key")
    end
    exit(1)
end
