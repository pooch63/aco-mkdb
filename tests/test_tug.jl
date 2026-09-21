#=
=================================================================================
Test & Benchmark Suite for Tug-of-War (Alternating Best-Response) Heuristic.

Covers:
1. Unit tests:
   - Degree peeling ((θ - k)-core)
   - Best response recruitment (cheapest-first price ordering)
   - Alternation & Convergence
   - Defect-based shaking
2. Real-world benchmark on indexed datasets:
   - Evaluates 3 versions of Tug-of-War (S_C, Degree, Random) + θ-heuristic baseline
   - Evaluates on all indexed graphs (or filtered by --prefix= / dataset arg)
   - Reports original & reduced graph sizes, injected biclique vertices, seeds
   - Shows nU, nV, edges, and runtime even for infeasible solutions
   - Dumps full structured results to JSON for reproducibility and comparison

Usage:
  julia --project=. tests/test_tug.jl --unit
  julia --project=. tests/test_tug.jl --N=5
  julia --project=. tests/test_tug.jl --prefix=konect-small
  julia --project=. tests/test_tug.jl amazon/boxes
  julia --project=. tests/test_tug.jl --save=results/tug_results.json
  julia --project=. tests/test_tug.jl
=================================================================================
=#

using Test
using Random
using Printf
using JSON3

const SRC = joinpath(@__DIR__, "..", "src")
const BIN = joinpath(@__DIR__, "..", "bin")

isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(SRC, "paths.jl"))
isdefined(@__MODULE__, :__IO_JL__) || include(joinpath(SRC, "io.jl"))
isdefined(@__MODULE__, :__GRAPH_JL__) || include(joinpath(SRC, "graph.jl"))
isdefined(@__MODULE__, :__REDUCTION_JL__) || include(joinpath(SRC, "reduction.jl"))
isdefined(@__MODULE__, :__THETA_HEURISTIC_JL__) || include(joinpath(SRC, "theta_heuristic.jl"))
isdefined(@__MODULE__, :__TUG_JL__) || include(joinpath(SRC, "tug.jl"))
isdefined(@__MODULE__, :__METHOD_JL__) || include(joinpath(SRC, "method.jl"))
isdefined(@__MODULE__, :__LOAD_JL__) || include(joinpath(BIN, "load.jl"))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit Tests
# ─────────────────────────────────────────────────────────────────────────────

function run_unit_tests()
    @testset "Tug-of-War Unit Tests" begin

        @testset "Degree peeling (θ - k core)" begin
            g = BipartiteGraph{Nothing}()
            # Core: 4x4 complete biclique
            for u in 1:4, v in 1:4
                add_edge!(g, u, v, nothing)
            end
            # Whisker: u5 connected to only v1 (degree 1)
            add_edge!(g, 5, 1, nothing)
            # Chain: u6 - v5 - u7 (degrees: u6=1, v5=2, u7=1)
            add_edge!(g, 6, 5, nothing)
            add_edge!(g, 7, 5, nothing)

            # Peeling with min_degree = 3
            peel_degrees!(g, 3)

            # Whisker and chain should be completely peeled
            @test haskey(g.adjU, 1) && haskey(g.adjU, 4)
            @test haskey(g.adjV, 1) && haskey(g.adjV, 4)
            @test !haskey(g.adjU, 5)
            @test !haskey(g.adjU, 6)
            @test !haskey(g.adjU, 7)
            @test !haskey(g.adjV, 5)
        end

        @testset "Best response cheapest-first recruitment" begin
            # Fixed U = {1, 2, 3, 4} (size 4)
            # Candidates in V:
            # v1 connected to all 4 (cost = 0)
            # v2 connected to 3 (cost = 1)
            # v3 connected to 3 (cost = 1)
            # v4 connected to 2 (cost = 2)
            # v5 connected to 0 (cost = 4)
            g = BipartiteGraph{Nothing}()
            for u in 1:4
                add_edge!(g, u, 1, nothing)
            end
            for u in 1:3
                add_edge!(g, u, 2, nothing)
                add_edge!(g, u, 3, nothing)
            end
            add_edge!(g, 1, 4, nothing)
            add_edge!(g, 2, 4, nothing)
            add_u!(g, 5) # v5 has no edges to U

            fg = freeze(g)
            U_dense = [fg.u_index[u] for u in 1:4]

            # k = 2, θ = 3
            # Cheapest 3: v1 (0), v2 (1), v3 (1). Total cost = 2 <= k.
            # Can we hire v4 (cost 2)? 2 + 2 = 4 > k, so v4 cannot be hired.
            V_hired = best_response_v_dense(fg, U_dense, 2, 3)
            @test V_hired !== nothing
            @test length(V_hired) == 3
            v_orig = Set(fg.v_ids[vi] for vi in V_hired)
            @test v_orig == Set([1, 2, 3])

            # If budget k = 1:
            # Cheapest 3 cost 2 > 1 => should return nothing
            V_fail = best_response_v_dense(fg, U_dense, 1, 3)
            @test V_fail === nothing
        end

        @testset "Alternation and Planted Biclique Recovery" begin
            g = BipartiteGraph{Nothing}()
            # 6x6 planted biclique with 2 defects (k=2, θ=5)
            # missing edges: (1, 1) and (2, 2)
            for u in 1:6, v in 1:6
                (u == 1 && v == 1) && continue
                (u == 2 && v == 2) && continue
                add_edge!(g, u, v, nothing)
            end
            # Add distractor sparse vertices
            for u in 7:12, v in 7:12
                if (u + v) % 3 == 0
                    add_edge!(g, u, v, nothing)
                end
            end

            fg = freeze(g)
            sol = tug_of_war_solve(fg, 2, 5; seed_source=:sc, num_seeds=5)

            @test length(sol.U) >= 5
            @test length(sol.V) >= 5
            edges = Subgraph.edge_count(fg, sol)
            missing_e = Subgraph.missing_edges(fg, sol)
            @test missing_e <= 2
            # Should recover all 6x6 vertices (34 edges)
            @test edges == 34
            @test sol.U == Set(1:6)
            @test sol.V == Set(1:6)
        end

        @testset "Defect Shaking Escapes Local Optima" begin
            g = BipartiteGraph{Nothing}()
            for u in 1:6, v in 1:6
                (u == 1 && v == 1) && continue
                add_edge!(g, u, v, nothing)
            end
            fg = freeze(g)
            sol = tug_of_war_solve(fg, 2, 5; seed_source=:sc, max_shakes=3)
            @test Subgraph.edge_count(fg, sol) == 35 # 6*6 - 1 = 35
        end

    end
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. CLI Argument Helpers
# ─────────────────────────────────────────────────────────────────────────────

function parse_string_flag(name::String, default::Union{String,Nothing}=nothing)
    prefix = "--$name="
    for arg in ARGS
        startswith(arg, prefix) && return split(arg, "=", limit=2)[2]
    end
    return default
end

function parse_int_flag(name::String, default::Int)
    prefix = "--$name="
    for arg in ARGS
        startswith(arg, prefix) && return parse(Int, split(arg, "=", limit=2)[2])
    end
    return default
end

function parse_seed_flag(default::UInt64=UInt64(1))
    for arg in ARGS
        startswith(arg, "--seed=") && return parse(UInt64, split(arg, "=", limit=2)[2])
    end
    return default
end

function parse_single_graph_arg()
    for arg in ARGS
        startswith(arg, "-") && continue
        return arg
    end
    return nothing
end

"""
    resolve_benchmark_graphs(; data_root="data")

Resolve target graphs from CLI arguments:
1. Positional single graph (e.g. `amazon/boxes`)
2. `--prefix=<str>` restriction (e.g. `--prefix=konect-small`)
3. If no prefix / single graph, returns all indexed graphs under `data/` sorted by |E|.
If `--N=<int>` is given, limits to the first N graphs; otherwise returns all.
"""
function resolve_benchmark_graphs(; data_root::AbstractString="data")
    single = parse_single_graph_arg()
    if single !== nothing
        path = resolve_graph_path(single; data_root=data_root)
        edges = isfile(path) ? count_indexed_edges(path) : 0
        return [(key=single, path=path, edges=edges)]
    end

    prefix = parse_string_flag("prefix", nothing)
    all_graphs = order_graphs_by_edges(; data_root=data_root, ascending=true, prefix=prefix)

    n_cap = parse_int_flag("N", -1)
    if n_cap > 0
        return all_graphs[1:min(n_cap, length(all_graphs))]
    end
    return all_graphs
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Real-World Benchmark & Seed Ablation
# ─────────────────────────────────────────────────────────────────────────────

function run_benchmark()
    data_root = joinpath(@__DIR__, "..", "data")
    graphs = resolve_benchmark_graphs(; data_root=data_root)
    if isempty(graphs)
        println("No graphs found.")
        return
    end

    k = parse_int_flag("k", 2)
    θ = parse_int_flag("theta", 5)
    do_inject = !("--no-inject" in ARGS)
    base_seed = parse_seed_flag(UInt64(1))
    save_path = parse_string_flag("save", joinpath(@__DIR__, "..", "results", "tug_benchmark.json"))

    reduction = ReductionMode.simple
    for arg in ARGS
        if startswith(arg, "--reduction=")
            val = lowercase(split(arg, "=", limit=2)[2])
            if val == "all"
                reduction = ReductionMode.all_reductions
            elseif val == "none"
                reduction = ReductionMode.none
            else
                reduction = ReductionMode.simple
            end
        end
    end

    single_graph = parse_single_graph_arg()

    println("="^120)
    println("TUG-OF-WAR vs. THETA-HEURISTIC BENCHMARK & SEED ABLATION")
    println("Parameters: k = $k, θ = $θ, Reduction = $(reduction), Inject = $(do_inject), Base Seed = $base_seed")
    println("Total Target Graphs: $(length(graphs))")
    println("Save Destination:    $(abspath(save_path))")
    println("="^120)

    # Accumulators for summary
    total_graphs = 0
    wins_sc = 0
    wins_deg = 0
    wins_rnd = 0
    wins_heur = 0
    ties = 0

    sc_greater_deg = 0
    sc_greater_rnd = 0

    total_time_sc = 0.0
    total_time_deg = 0.0
    total_time_rnd = 0.0
    total_time_heur = 0.0

    feas_sc_count = 0
    feas_deg_count = 0
    feas_rnd_count = 0
    feas_heur_count = 0

    graph_records = Any[]
    table_rows = Any[]

    for (idx, entry) in enumerate(graphs)
        total_graphs += 1
        key = entry.key
        path = entry.path

        if !isfile(path)
            println("Skipping $key: file not found at $path")
            continue
        end

        # Reproducible deterministic seed per graph
        graph_seed = (single_graph !== nothing) ? base_seed : (hash(base_seed, hash(key)) & 0x7fffffff)
        Random.seed!(graph_seed)

        println("\n[$idx/$(length(graphs))] Dataset: $key  (Seed: $graph_seed)")

        # Load graph
        g, _ = load_bipartite_graph(path)
        orig_nU = length(g.adjU)
        orig_nV = length(g.adjV)
        orig_edges = sum(length(nbrs) for (_, nbrs) in g.adjU)

        # Injected biclique
        plant_U = Int[]
        plant_V = Int[]
        inserted = 0
        existing_plant = 0
        missing_plant = Tuple{Int,Int}[]
        injected = false

        if do_inject
            rng = MersenneTwister(graph_seed)
            try
                chosen_U, chosen_V, ins, miss, ex = inject_biclique!(g, θ, θ, k, rng)
                plant_U = sort!(collect(chosen_U))
                plant_V = sort!(collect(chosen_V))
                inserted = ins
                existing_plant = ex
                missing_plant = sort!(collect(miss))
                injected = true
            catch e
                # Proceed with natural graph if injection is impossible (e.g. too few nodes)
            end
        end

        # Reduced graph properties
        g_for_red = deepcopy(g)
        fg_reduced = apply_graph_reductions!(g_for_red, k, θ, nothing, nothing, true, reduction)
        reduced_nU = length(fg_reduced.u_ids)
        reduced_nV = length(fg_reduced.v_ids)
        reduced_edges = length(fg_reduced.v_adj)

        println("  Original Graph: |U|=$orig_nU, |V|=$orig_nV, |E|=$orig_edges")
        println("  Reduced Graph:  |U|=$reduced_nU, |V|=$reduced_nV, |E|=$reduced_edges")
        if injected
            println("  Injected Biclique: |U|=$(length(plant_U)), |V|=$(length(plant_V)), inserted=$inserted, missing=$(length(missing_plant))")
            println("    plant_U: $plant_U")
            println("    plant_V: $plant_V")
        else
            println("  Injected Biclique: none")
        end

        # Evaluation freeze graph
        fg_eval = freeze(deepcopy(g))

        function evaluate_solution(sol::SubGraph, elapsed_time::Float64)
            u_nodes = sort!(collect(sol.U))
            v_nodes = sort!(collect(sol.V))
            u_cnt = length(u_nodes)
            v_cnt = length(v_nodes)
            e_cnt = Subgraph.edge_count(fg_eval, sol)
            m_cnt = Subgraph.missing_edges(fg_eval, sol)
            feas = (u_cnt >= θ && v_cnt >= θ && m_cnt <= k)
            return (
                nU = u_cnt,
                nV = v_cnt,
                edges = e_cnt,
                missing = m_cnt,
                wall_time_s = elapsed_time,
                feasible = feas,
                U = u_nodes,
                V = v_nodes,
            )
        end

        # 1. Tug-of-War with S_C seeds
        m_sc = TugOfWarMethod(; seed_source=:sc, num_seeds=10, max_shakes=3, peel=true)
        t0 = time()
        res_sc = run_method!(m_sc, deepcopy(g), k, θ; reduction=reduction)
        t_sc = (res_sc.wall_time_s > 0.0) ? res_sc.wall_time_s : (time() - t0)
        eval_sc = evaluate_solution(res_sc.sol, t_sc)

        # 2. Tug-of-War with Degree seeds
        m_deg = TugOfWarMethod(; seed_source=:degree, num_seeds=10, max_shakes=3, peel=true)
        t0 = time()
        res_deg = run_method!(m_deg, deepcopy(g), k, θ; reduction=reduction)
        t_deg = (res_deg.wall_time_s > 0.0) ? res_deg.wall_time_s : (time() - t0)
        eval_deg = evaluate_solution(res_deg.sol, t_deg)

        # 3. Tug-of-War with Random seeds
        m_rnd = TugOfWarMethod(; seed_source=:random, num_seeds=10, max_shakes=3, seed=Int(graph_seed & 0x7fffffff), peel=true)
        t0 = time()
        res_rnd = run_method!(m_rnd, deepcopy(g), k, θ; reduction=reduction)
        t_rnd = (res_rnd.wall_time_s > 0.0) ? res_rnd.wall_time_s : (time() - t0)
        eval_rnd = evaluate_solution(res_rnd.sol, t_rnd)

        # 4. θ-heuristic baseline
        m_heur = HeuristicMethod(; return_invalid=false)
        t0 = time()
        res_heur = run_method!(m_heur, deepcopy(g), k, θ; reduction=reduction)
        t_heur = (res_heur.wall_time_s > 0.0) ? res_heur.wall_time_s : (time() - t0)
        eval_heur = evaluate_solution(res_heur.sol, t_heur)

        # Track runtimes
        total_time_sc += eval_sc.wall_time_s
        total_time_deg += eval_deg.wall_time_s
        total_time_rnd += eval_rnd.wall_time_s
        total_time_heur += eval_heur.wall_time_s

        eval_sc.feasible && (feas_sc_count += 1)
        eval_deg.feasible && (feas_deg_count += 1)
        eval_rnd.feasible && (feas_rnd_count += 1)
        eval_heur.feasible && (feas_heur_count += 1)

        # Format cell strings: "edges (u×v) time" (+ "(inf)" if not feasible)
        function format_cell(data)
            tag = data.feasible ? "" : " (inf)"
            return @sprintf("%d (%d×%d) %.3fs%s", data.edges, data.nU, data.nV, data.wall_time_s, tag)
        end

        cell_sc = format_cell(eval_sc)
        cell_deg = format_cell(eval_deg)
        cell_rnd = format_cell(eval_rnd)
        cell_heur = format_cell(eval_heur)

        display_name = split(key, "/")[end]
        push!(table_rows, (
            dataset = display_name,
            cell_sc = cell_sc,
            cell_deg = cell_deg,
            cell_rnd = cell_rnd,
            cell_heur = cell_heur,
        ))

        println("  Results: TOW(S_C)=$cell_sc | TOW(Deg)=$cell_deg | TOW(Rnd)=$cell_rnd | θ-Heur=$cell_heur")

        # Quality scoring (feasible solutions strictly beat infeasible; higher edges wins)
        valid_sc = eval_sc.feasible ? eval_sc.edges : -1
        valid_deg = eval_deg.feasible ? eval_deg.edges : -1
        valid_rnd = eval_rnd.feasible ? eval_rnd.edges : -1
        valid_heur = eval_heur.feasible ? eval_heur.edges : -1

        # TOW(S_C) vs baseline
        if valid_sc > valid_heur
            wins_sc += 1
        elseif valid_sc == valid_heur && valid_sc > 0
            ties += 1
        else
            wins_heur += 1
        end

        if valid_sc > valid_deg
            sc_greater_deg += 1
        end
        if valid_sc > valid_rnd
            sc_greater_rnd += 1
        end

        # Record for JSON
        push!(graph_records, Dict{String,Any}(
            "dataset" => key,
            "seed" => Int(graph_seed),
            "original" => Dict{String,Int}(
                "nU" => orig_nU,
                "nV" => orig_nV,
                "edges" => orig_edges,
            ),
            "reduced" => Dict{String,Int}(
                "nU" => reduced_nU,
                "nV" => reduced_nV,
                "edges" => reduced_edges,
            ),
            "injected_biclique" => Dict{String,Any}(
                "injected" => injected,
                "plant_U" => plant_U,
                "plant_V" => plant_V,
                "inserted" => inserted,
                "existing" => existing_plant,
                "missing_edges" => [[e[1], e[2]] for e in missing_plant],
            ),
            "algorithms" => Dict{String,Any}(
                "tow_sc" => Dict{String,Any}(
                    "nU" => eval_sc.nU,
                    "nV" => eval_sc.nV,
                    "edges" => eval_sc.edges,
                    "missing" => eval_sc.missing,
                    "feasible" => eval_sc.feasible,
                    "wall_time_s" => round(eval_sc.wall_time_s; digits=6),
                    "U" => eval_sc.U,
                    "V" => eval_sc.V,
                ),
                "tow_degree" => Dict{String,Any}(
                    "nU" => eval_deg.nU,
                    "nV" => eval_deg.nV,
                    "edges" => eval_deg.edges,
                    "missing" => eval_deg.missing,
                    "feasible" => eval_deg.feasible,
                    "wall_time_s" => round(eval_deg.wall_time_s; digits=6),
                    "U" => eval_deg.U,
                    "V" => eval_deg.V,
                ),
                "tow_random" => Dict{String,Any}(
                    "nU" => eval_rnd.nU,
                    "nV" => eval_rnd.nV,
                    "edges" => eval_rnd.edges,
                    "missing" => eval_rnd.missing,
                    "feasible" => eval_rnd.feasible,
                    "wall_time_s" => round(eval_rnd.wall_time_s; digits=6),
                    "U" => eval_rnd.U,
                    "V" => eval_rnd.V,
                ),
                "theta_heuristic" => Dict{String,Any}(
                    "nU" => eval_heur.nU,
                    "nV" => eval_heur.nV,
                    "edges" => eval_heur.edges,
                    "missing" => eval_heur.missing,
                    "feasible" => eval_heur.feasible,
                    "wall_time_s" => round(eval_heur.wall_time_s; digits=6),
                    "U" => eval_heur.U,
                    "V" => eval_heur.V,
                ),
            ),
        ))
    end

    # Print Clean Final Table
    println()
    println("="^120)
    println("ALGORITHM PERFORMANCE COMPARISON TABLE (edges (u×v) time [inf=infeasible])")
    println("="^120)
    @printf("%-26s | %-22s | %-22s | %-22s | %-22s\n",
            "Dataset", "TOW (S_C)", "TOW (Degree)", "TOW (Random)", "θ-Heuristic")
    println("-"^120)
    for row in table_rows
        @printf("%-26s | %-22s | %-22s | %-22s | %-22s\n",
                row.dataset[1:min(26, length(row.dataset))], row.cell_sc, row.cell_deg, row.cell_rnd, row.cell_heur)
    end
    println("-"^120)

    # Summary calculations
    avg_t_sc = total_graphs > 0 ? (total_time_sc / total_graphs) : 0.0
    avg_t_deg = total_graphs > 0 ? (total_time_deg / total_graphs) : 0.0
    avg_t_rnd = total_graphs > 0 ? (total_time_rnd / total_graphs) : 0.0
    avg_t_heur = total_graphs > 0 ? (total_time_heur / total_graphs) : 0.0

    times = [
        ("TOW (S_C)", avg_t_sc),
        ("TOW (Degree)", avg_t_deg),
        ("TOW (Random)", avg_t_rnd),
        ("θ-Heuristic", avg_t_heur),
    ]
    quickest_algo, quickest_time = sort(times, by=x -> x[2])[1]

    println()
    println("="^120)
    println("SUMMARY RESULTS:")
    println("  Total graphs evaluated:              $total_graphs")
    @printf("  Quickest algorithm (avg runtime):   %s (%.4fs)\n", quickest_algo, quickest_time)
    println()
    println("  Average Wall-Clock Runtimes:")
    @printf("    TOW (S_C):                         %.4fs\n", avg_t_sc)
    @printf("    TOW (Degree):                      %.4fs\n", avg_t_deg)
    @printf("    TOW (Random):                      %.4fs\n", avg_t_rnd)
    @printf("    θ-Heuristic:                       %.4fs\n", avg_t_heur)
    println()
    println("  Feasibility Counts:")
    @printf("    TOW (S_C):                         %d / %d (%.1f%%)\n", feas_sc_count, total_graphs, 100.0 * feas_sc_count / max(1, total_graphs))
    @printf("    TOW (Degree):                      %d / %d (%.1f%%)\n", feas_deg_count, total_graphs, 100.0 * feas_deg_count / max(1, total_graphs))
    @printf("    TOW (Random):                      %d / %d (%.1f%%)\n", feas_rnd_count, total_graphs, 100.0 * feas_rnd_count / max(1, total_graphs))
    @printf("    θ-Heuristic:                       %d / %d (%.1f%%)\n", feas_heur_count, total_graphs, 100.0 * feas_heur_count / max(1, total_graphs))
    println()
    println("  Pairwise Quality Comparisons (edges among feasible):")
    println("    TOW (S_C) beat θ-heuristic:        $wins_sc / $total_graphs graphs")
    println("    TOW (S_C) tied θ-heuristic:        $ties / $total_graphs graphs")
    println("    θ-heuristic beat TOW (S_C):        $wins_heur / $total_graphs graphs")
    println("    Ablation: S_C beat Degree seeds:   $sc_greater_deg / $total_graphs graphs")
    println("    Ablation: S_C beat Random seeds:   $sc_greater_rnd / $total_graphs graphs")
    println("="^120)

    # Dump to JSON
    json_payload = Dict{String,Any}(
        "parameters" => Dict{String,Any}(
            "k" => k,
            "theta" => θ,
            "reduction" => string(reduction),
            "inject" => do_inject,
            "base_seed" => string(base_seed),
            "total_graphs" => total_graphs,
        ),
        "summary" => Dict{String,Any}(
            "total_graphs" => total_graphs,
            "quickest_algorithm" => quickest_algo,
            "avg_wall_time_s" => Dict{String,Float64}(
                "tow_sc" => round(avg_t_sc; digits=6),
                "tow_degree" => round(avg_t_deg; digits=6),
                "tow_random" => round(avg_t_rnd; digits=6),
                "theta_heuristic" => round(avg_t_heur; digits=6),
            ),
            "feasibility_counts" => Dict{String,Int}(
                "tow_sc" => feas_sc_count,
                "tow_degree" => feas_deg_count,
                "tow_random" => feas_rnd_count,
                "theta_heuristic" => feas_heur_count,
            ),
            "pairwise_comparisons" => Dict{String,Int}(
                "tow_sc_beat_heuristic" => wins_sc,
                "tow_sc_tied_heuristic" => ties,
                "heuristic_beat_tow_sc" => wins_heur,
                "tow_sc_beat_degree" => sc_greater_deg,
                "tow_sc_beat_random" => sc_greater_rnd,
            ),
        ),
        "graphs" => graph_records,
    )

    mkpath(dirname(abspath(save_path)))
    open(save_path, "w") do io
        JSON3.pretty(io, json_payload)
    end
    println("\n✓ Benchmark results successfully saved to: $(abspath(save_path))\n")
end

function main()
    run_units = "--unit" in ARGS
    run_bench = "--benchmark" in ARGS || !run_units

    if run_units
        run_unit_tests()
    end

    if run_bench
        run_benchmark()
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
