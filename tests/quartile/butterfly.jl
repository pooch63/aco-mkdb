"""
Butterfly (degree-normalised 4-cycle) edge-scoring test on real-world graphs.

For every edge e=(u,v) we compute:

    S4(u,v) = #{4-cycles containing (u,v)} / ((deg(u)-1)(deg(v)-1))

i.e. the fraction of the (deg(u)-1)(deg(v)-1) local rectangles around the
edge that actually exist.  Because 4-cycles in a bipartite graph are exactly
the butterflies, this is equivalent to wedge-counting:

    #{4-cycles through (u,v)} = Σ_{u2 ∈ N(v)\\{u}} |N(u2) ∩ N(u)| - 1

We use the wedge accumulation trick: for every u, accumulate a counter cnt[v2]
= #{v ∈ N(u) : u is connected to v and v2 via u2} then credit each edge.
Cost: O(Σ_u deg(u) · d_max_V).

The test injects a planted biclique, computes S4 on the full graph and on the
CNN-reduced graph, then prints a recall table at 1 / 5 / 10 / 20 / 30 % of
edges.

Usage:
  julia tests/quartile/butterfly.jl              # first 5 graphs (ascending |E|)
  julia tests/quartile/butterfly.jl amazon/boxes # one specific graph
  julia tests/quartile/butterfly.jl --N=10
  julia tests/quartile/butterfly.jl --prefix=konect-small --N=3
"""

using Random

include(joinpath(@__DIR__, "graph_selection.jl"))

isdefined(@__MODULE__, :__PATHS_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "src", "paths.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "io.jl"))
include(joinpath(@__DIR__, "..", "..", "src", "pulse.jl"))   # pulls in graph.jl
isdefined(@__MODULE__, :__LOAD_JL__) ||
    include(joinpath(@__DIR__, "..", "..", "bin", "load.jl")) # inject_biclique! + apply_graph_reductions!

const K = 2
const θ = 5

# ── Recall thresholds ─────────────────────────────────────────────────────────
const THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30]

# ── Core scoring ──────────────────────────────────────────────────────────────

"""
    butterfly_s4_scores(fg) -> Vector{Float64}

Compute the degree-normalised 4-cycle score S4(u,v) for every edge in `fg`,
returned in CSR order (parallel to fg.v_adj).

A butterfly in a bipartite graph U∪V is a C4: u1-v1-u2-v2-u1.
The edge (u,v) participates in such a cycle for each pair (u2,v2) where
  u2 ∈ N(v)\\{u},  v2 ∈ N(u)\\{v},  (u2,v2) ∈ E.

Algorithm (wedge accumulation from the U side):
  For each U-vertex u (dense index ui):
    Phase A — accumulate cnt:
      For each v  ∈ N(u):
        For each u2 ∈ N(v)\\{u}:
          For each v2 ∈ N(u2):
            cnt[v2] += 1          # v2 reachable from u in 3 hops
    Phase B — credit edges:
      For each v2 ∈ N(u):
        raw[slot(u,v2)] += cnt[v2]   # cnt[v2] = # of butterflies at (u,v2) from u's side
    Reset cnt.

Normalise: S4(u,v) = raw[slot(u,v)] / max(1, (deg(u)-1)*(deg(v)-1))
"""
function butterfly_s4_scores(fg::FrozenBipartite)::Vector{Float64}
    nU = length(fg.u_ids)
    nV = length(fg.v_ids)
    nE = length(fg.v_adj)

    raw = zeros(Float64, nE)

    cnt   = zeros(Int, nV)
    dirty = Int[]
    sizehint!(dirty, 64)

    for ui in 1:nU
        r_u = neighbor_range_u(fg, ui)
        isempty(r_u) && continue

        empty!(dirty)

        # Phase A: walk u → v → u2 (≠u) → v2 and tally cnt[v2]
        for k in r_u
            vi = fg.v_adj[k]
            for l in neighbor_range_v(fg, vi)
                u2i = fg.u_adj[l]
                u2i == ui && continue          # skip the trivial u2 = u
                for m in neighbor_range_u(fg, u2i)
                    v2i = fg.v_adj[m]
                    if cnt[v2i] == 0
                        push!(dirty, v2i)
                    end
                    cnt[v2i] += 1
                end
            end
        end

        # Phase B: credit every edge (u, v2) with cnt[v2]
        for k in r_u
            v2i = fg.v_adj[k]
            if cnt[v2i] > 0
                raw[k] += cnt[v2i]
            end
        end

        # Reset
        for v2i in dirty
            cnt[v2i] = 0
        end
    end

    # Normalise
    scores = Vector{Float64}(undef, nE)
    for ui in 1:nU
        deg_u = length(neighbor_range_u(fg, ui))
        for k in neighbor_range_u(fg, ui)
            vi = fg.v_adj[k]
            deg_v = length(neighbor_range_v(fg, vi))
            denom = max(1, (deg_u - 1) * (deg_v - 1))
            scores[k] = raw[k] / denom
        end
    end
    return scores
end

# ── Recall computation ─────────────────────────────────────────────────────────

"""
    recall_at_thresholds(scores, plant_slots, thresholds, nE) -> Vector{Float64}

Given S4 `scores` (length nE, parallel to fg.v_adj), the CSR slot indices of
the planted biclique's existing edges, and a vector of top-fraction thresholds,
return the recall at each threshold.
"""
function recall_at_thresholds(scores::Vector{Float64},
                               plant_slots::Vector{Int},
                               thresholds::Vector{Float64})::Vector{Float64}
    nE = length(scores)
    n_plant = length(plant_slots)
    n_plant == 0 && return fill(NaN, length(thresholds))

    order = sortperm(scores, rev=true)
    plant_set = Set(plant_slots)

    recalls   = Float64[]
    plant_found = 0
    ptr = 0

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

Return the CSR u-side slot indices for existing planted biclique edges in `fg`.
Edges whose original (u,v) pair is in `missing_edges` are excluded.
Returns an empty vector if `fg` does not contain those vertices (reduced away).
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

# ── Pretty table ──────────────────────────────────────────────────────────────

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

function test_butterfly_on(dataset_key::AbstractString, path::AbstractString,
                            edge_count::Int)
    println("\n── $dataset_key ──")
    println("  Loading: $path  (|E|=$edge_count)")

    g, _ = load_bipartite_graph(path)
    println("  Graph: |U|=$(length(g.adjU))  |V|=$(length(g.adjV))  |E|=$edge_count")

    # Inject planted biclique
    rng = MersenneTwister(1)
    chosen_U, chosen_V, inserted, missing_edges, existing =
        inject_biclique!(g, θ, θ, K, rng)
    println("  Injected biclique: u=$θ v=$θ k=$K  " *
            "(existing=$existing inserted=$inserted missing=$(length(missing_edges)))")
    println("    planted U: $chosen_U")
    println("    planted V: $chosen_V")

    n_plant_possible = θ * θ - K   # edges that should exist after injection

    # ── Full graph ─────────────────────────────────────────────────────────────
    fg_full = freeze(g)
    t_full  = @elapsed scores_full = butterfly_s4_scores(fg_full)
    slots_full  = plant_slots_in(fg_full, chosen_U, chosen_V, missing_edges)
    recalls_full = recall_at_thresholds(scores_full, slots_full, THRESHOLDS)

    println("\n  [FULL GRAPH]  |E|=$(length(scores_full))  (scored in $(round(t_full; digits=3))s)")
    print_recall_table("S4 on full graph", THRESHOLDS, recalls_full,
                       length(scores_full), length(slots_full))

    # ── CNN-reduced graph ──────────────────────────────────────────────────────
    g_red  = deepcopy(g)
    nU_h   = length(g_red.adjU)
    nV_h   = length(g_red.adjV)
    fg_red = apply_graph_reductions!(g_red, K, θ, nU_h, nV_h, false, ReductionMode.simple)
    t_red  = @elapsed scores_red = butterfly_s4_scores(fg_red)
    slots_red  = plant_slots_in(fg_red, chosen_U, chosen_V, missing_edges)
    recalls_red = recall_at_thresholds(scores_red, slots_red, THRESHOLDS)

    n_red_E = length(scores_red)
    println("\n  [REDUCED GRAPH]  |U|=$(length(fg_red.u_ids))  " *
            "|V|=$(length(fg_red.v_ids))  |E|=$n_red_E  " *
            "(scored in $(round(t_red; digits=3))s)")
    print_recall_table("S4 on reduced graph", THRESHOLDS, recalls_red,
                       n_red_E, length(slots_red))

    r30_full = recalls_full[end]
    r30_red  = recalls_red[end]
    println("  Summary: recall@30%  full=$(round(100*r30_full; digits=1))%  " *
            "reduced=$(round(100*r30_red; digits=1))%")
    println("  ✓ done")
    return true
end

# ── Main ───────────────────────────────────────────────────────────────────────

graphs = selected_graphs()

if isempty(graphs)
    println("No graphs found. Check that data/ is populated and --prefix= is correct.")
    exit(1)
end

println("Butterfly (S4) edge-score recall test on $(length(graphs)) graph(s) (ascending |E|):")
println("Thresholds: $(join(string.(round.(Int, 100 .* THRESHOLDS)) .* "%", ", "))")

failed = String[]
for g in graphs
    try
        test_butterfly_on(g.key, g.path, g.edges)
    catch e
        println("  ✗ FAILED: $(g.key): $e")
        push!(failed, g.key)
        rethrow()
    end
end

println()
if isempty(failed)
    println("All $(length(graphs)) butterfly test(s) completed.")
else
    println("$(length(failed)) / $(length(graphs)) test(s) FAILED:")
    for key in failed
        println("  $key")
    end
    exit(1)
end
