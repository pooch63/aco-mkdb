include(joinpath(@__DIR__, "..", "graph.jl"))
include(joinpath(@__DIR__, "..", "opponent.jl"))
include(joinpath(@__DIR__, "generate.jl"))

using Random

function parse_seed()
    for arg in ARGS
        if startswith(arg, "--seed=")
            return parse(UInt64, split(arg, "=", limit=2)[2])
        end
    end
    return UInt64(time_ns())
end

const SEED = parse_seed()
Random.seed!(SEED)

# =============================================================================
# Brute-force optimum k-MDB.
# Searches every possible biclique and returns the one with the maximum
# number of edges subject to the k-defective constraint and θ threshold.
# =============================================================================
function brute_force_mdb(g::FrozenBipartite, k::Int, θ::Int)

    Ulist = collect(g.u_ids)
    Vlist = collect(g.v_ids)

    best = SubGraph(Set{Int}(), Set{Int}())
    best_edges = 0

    nU = length(Ulist)
    nV = length(Vlist)

    for umask in 0:(2^nU - 1), vmask in 0:(2^nV - 1)

        U = Set(Ulist[i] for i in 1:nU if (umask >> (i - 1)) & 1 == 1)
        V = Set(Vlist[i] for i in 1:nV if (vmask >> (i - 1)) & 1 == 1)

        if length(U) < θ || length(V) < θ
            continue
        end

        D = SubGraph(U, V)

        if Subgraph.missing_edges(g, D) <= k
            e = Subgraph.edge_count(g, D)

            if e > best_edges
                best = D
                best_edges = e
            end
        end
    end

    return best, best_edges
end

# =============================================================================
# MAIN
# =============================================================================
function main()
    N = 10

    valid = 0
    optimal = 0

    total_ratio = 0.0
    worst_ratio = 1.0

    for trial in 1:N

        edges, nU, nV, k, θ = random_graph()
        g = build_frozen(edges, nU, nV)

        heuristic_sol = heuristic(g, k, θ)

        missing_edges = Subgraph.missing_edges(g, heuristic_sol)

        if missing_edges > k
            println("INVALID: exceeds k on trial $trial")
            println("Seed: --seed=$SEED")
            return
        end

        sufficient = length(heuristic_sol.U) ≥ θ && length(heuristic_sol.V) ≥ θ
        if sufficient
            valid += 1
        end

        _, opt_edges = brute_force_mdb(g, k, θ)

        heuristic_edges = Subgraph.edge_count(g, heuristic_sol)

        if heuristic_edges >= opt_edges && sufficient
            optimal += 1
        end

        ratio = opt_edges == 0 ? 1.0 : heuristic_edges / opt_edges

        total_ratio += ratio
        worst_ratio = min(worst_ratio, ratio)

        if ratio < 0.5
            println("Poor heuristic on trial $trial")
            println("Seed: --seed=$SEED")
            println("heuristic edges = $heuristic_edges")
            println("optimal edges   = $opt_edges")
            println("ratio           = $ratio")
        end
    end

    println()
    println("==================== RESULTS ====================")
    println("Trials               : $N")
    println("Seed                 : --seed=$SEED")
    println("Valid solutions      : $valid / $N")
    println("Optimal solutions    : $optimal / $N")
    println("Optimality rate      : $(100 * optimal / N)%")
    println("Average ratio        : $(total_ratio / N)")
    println("Worst ratio          : $worst_ratio")
    println("=================================================")

end

main()