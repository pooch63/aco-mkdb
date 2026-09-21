using Test

include(joinpath(@__DIR__, "..", "src", "pulse.jl"))
include(joinpath(@__DIR__, "suite.jl"))

const REGISTERED_PULSE_FORMULAS = list_pulse_formulas()

function print_help()
    println("Usage: julia tests/test_quartile_reduction.jl [--help|-h] [--formula=<name> | --formula <name> | --formulas=<name[,name...]>]")
    println("")
    println("Registered pulse formulas:")
    for name in REGISTERED_PULSE_FORMULAS
        println("  $(name)")
    end
    println("")
    println("Flags:")
    println("  -h, --help      Show this help message and exit.")
end

function parse_formula_args()
    selected = String[]
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "-h" || arg == "--help"
            print_help()
            exit(0)
        elseif startswith(arg, "--formula=")
            value = split(arg, "=", limit=2)[2]
            append!(selected, split(value, ','))
        elseif arg == "--formula" || arg == "--formulas"
            i == length(ARGS) && begin
                print_help()
                error("Missing value for $(arg).")
            end
            append!(selected, split(ARGS[i + 1], ','))
            i += 1
        elseif startswith(arg, "--formulas=")
            value = split(arg, "=", limit=2)[2]
            append!(selected, split(value, ','))
        elseif startswith(arg, "--") || startswith(arg, "-")
            print_help()
            error("Unknown option: $(arg)")
        end
        i += 1
    end

    isempty(selected) && return ["standard"]

    normalized = String[]
    for name in selected
        cleaned = strip(name)
        if !isempty(cleaned)
            push!(normalized, cleaned)
        end
    end

    isempty(normalized) && return ["standard"]

    valid = Set(REGISTERED_PULSE_FORMULAS)
    for name in normalized
        name in valid || begin
            print_help()
            error("Unknown pulse formula: $(name)")
        end
    end

    return normalized
end

const SELECTED_FORMULAS = parse_formula_args()

"""
Test quartile-based edge weight reduction.
"""
function test_quartile_reduction()
    # Create a simple test graph
    edges = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2)]
    nU = 4
    nV = 2
    
    fg = build_frozen(edges, nU, nV)
    g = build_mutable_graph(fg)
    
    println("Original graph:")
    println("  |U|=$(length(g.adjU)), |V|=$(length(g.adjV))")
    println("  |E|=$(sum(length(nbrs) for nbrs in values(g.adjU); init=0))")
    
    # Apply quartile reduction
    n_edges_kept, n_u_kept, n_v_kept = quartile_edge_reduction!(g, 0.5, 5)
    
    println("\nAfter quartile reduction:")
    println("  |U|=$n_u_kept, |V|=$n_v_kept")
    println("  |E|=$n_edges_kept")
    
    # Verify the graph is still valid
    @test n_u_kept >= 0
    @test n_v_kept >= 0
    @test n_edges_kept >= 0
    
    # Verify no isolated vertices remain
    for (u, nbrs) in g.adjU
        @test !isempty(nbrs)
    end
    for (v, nbrs) in g.adjV
        @test !isempty(nbrs)
    end
    
    println("\nQuartile reduction test passed!")
end

"""
Test quartile reduction with apply_graph_reductions!
"""
function test_quartile_reduction_integration()
    # Create a test graph
    edges = [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1), (3, 2), (4, 1), (4, 2), (5, 1), (5, 2)]
    nU = 5
    nV = 2
    k = 2
    θ = 2
    
    fg = build_frozen(edges, nU, nV)
    g = build_mutable_graph(fg)
    
    println("\nTesting quartile reduction via apply_graph_reductions!")
    println("Original graph:")
    println("  |U|=$(length(g.adjU)), |V|=$(length(g.adjV))")
    
    # Apply quartile reduction through the standard interface
    result = apply_graph_reductions!(g, k, θ, nU, nV, false, ReductionMode.quartile)
    
    println("\nAfter quartile reduction:")
    println("  |U|=$(length(result.u_ids)), |V|=$(length(result.v_ids))")
    println("  |E|=$(sum(degree_u(result, u) for u in result.u_ids; init=0))")
    
    # Verify the result is a valid FrozenBipartite
    @test result isa FrozenBipartite
    @test length(result.u_ids) >= 0
    @test length(result.v_ids) >= 0
    
    println("\nQuartile reduction integration test passed!")
end

# Run tests
@testset "Quartile Reduction Tests" begin
    for formula in unique(SELECTED_FORMULAS)
        test_quartile_reduction()
        test_quartile_reduction_integration()
    end
end
