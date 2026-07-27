#=
=================================================================================
Compare two suite JSON result files (from --save= on test_heuristic / test_ga / …).

Trials are matched by graph_seed so both runs must use the same --seed and --N
(and the same graph-generation flags).

When both files include an oracle optimum for a trial, optimality counts use that
reference. When neither has an optimum (`--no-optimum` / `opt_edges: null`), the
winner is the solution with more edges.

Usage:
  julia tests/compare.jl results_a.json results_b.json
  julia tests/compare.jl heuristic.json ga.json --label-a=heuristic --label-b=ga
=================================================================================
=#

using JSON3

function usage_and_exit()
    println(stderr, "Usage: julia tests/compare.jl <file_a.json> <file_b.json> [--label-a=A] [--label-b=B]")
    exit(1)
end

function parse_compare_args(args)
    length(args) >= 2 || usage_and_exit()
    file_a = args[1]
    file_b = args[2]
    label_a = nothing
    label_b = nothing
    for arg in args[3:end]
        if startswith(arg, "--label-a=")
            label_a = split(arg, "=", limit=2)[2]
        elseif startswith(arg, "--label-b=")
            label_b = split(arg, "=", limit=2)[2]
        else
            println(stderr, "Unknown argument: $arg")
            usage_and_exit()
        end
    end
    return file_a, file_b, label_a, label_b
end

function primary_solution(trial, preferred_label::Union{Nothing,String})
    sols = trial.solutions
    if preferred_label !== nothing && haskey(sols, Symbol(preferred_label))
        return preferred_label, sols[Symbol(preferred_label)]
    elseif preferred_label !== nothing && haskey(sols, preferred_label)
        return preferred_label, sols[preferred_label]
    end
    # JSON3 may use String or Symbol keys depending on version; normalize.
    keys_list = collect(keys(sols))
    isempty(keys_list) && error("Trial $(trial.trial) has no solutions")
    key = keys_list[1]
    return string(key), sols[key]
end

function solution_edges(sol)
    return Int(sol.edges)
end

function trial_opt_edges(trial)
    if !haskey(trial, :opt_edges) || trial.opt_edges === nothing
        return nothing
    end
    return Int(trial.opt_edges)
end

function solution_ratio(sol, opt_edges::Union{Int,Nothing})
    if opt_edges === nothing
        return nothing
    end
    if haskey(sol, :ratio) && sol.ratio !== nothing
        return Float64(sol.ratio)
    end
    return opt_edges == 0 ? 1.0 : solution_edges(sol) / opt_edges
end

function format_ratio(r::Union{Float64,Nothing})
    return r === nothing ? "n/a" : string(round(r; digits=3))
end

function load_suite(path::AbstractString)
    isfile(path) || error("File not found: $path")
    return JSON3.read(read(path, String))
end

function compare_suites(path_a::AbstractString, path_b::AbstractString;
    label_a=nothing, label_b=nothing)
    a = load_suite(path_a)
    b = load_suite(path_b)

    algo_a = label_a === nothing ? string(a.algorithm) : label_a
    algo_b = label_b === nothing ? string(b.algorithm) : label_b

    by_seed_a = Dict(string(t.graph_seed) => t for t in a.trials)
    by_seed_b = Dict(string(t.graph_seed) => t for t in b.trials)

    shared = sort!(collect(intersect(keys(by_seed_a), keys(by_seed_b))))
    only_a = sort!(collect(setdiff(keys(by_seed_a), keys(by_seed_b))))
    only_b = sort!(collect(setdiff(keys(by_seed_b), keys(by_seed_a))))

    a_better = NamedTuple[]
    b_better = NamedTuple[]
    ties = NamedTuple[]
    both_optimal = 0
    a_optimal = 0
    b_optimal = 0
    compared_with_opt = 0
    compared_by_score = 0

    for seed in shared
        ta = by_seed_a[seed]
        tb = by_seed_b[seed]
        _, sa = primary_solution(ta, label_a === nothing ? string(a.algorithm) : label_a)
        _, sb = primary_solution(tb, label_b === nothing ? string(b.algorithm) : label_b)

        opt_a = trial_opt_edges(ta)
        opt_b = trial_opt_edges(tb)
        # Need an optimum on both sides to score vs oracle; otherwise compare by edges only.
        has_opt = opt_a !== nothing && opt_b !== nothing

        ea = solution_edges(sa)
        eb = solution_edges(sb)

        if has_opt
            compared_with_opt += 1
            opt = opt_a
            if opt_b != opt
                @warn "opt_edges disagree for graph_seed=$seed" a=opt_a b=opt_b
            end
            ra = solution_ratio(sa, opt)
            rb = solution_ratio(sb, opt)

            a_is_opt = (opt == 0 && ea == 0) || (opt > 0 && ea >= opt)
            b_is_opt = (opt == 0 && eb == 0) || (opt > 0 && eb >= opt)
            a_is_opt && (a_optimal += 1)
            b_is_opt && (b_optimal += 1)
            a_is_opt && b_is_opt && (both_optimal += 1)
        else
            compared_by_score += 1
            opt = nothing
            ra = nothing
            rb = nothing
        end

        row = (
            graph_seed=seed,
            trial_a=Int(ta.trial),
            trial_b=Int(tb.trial),
            opt_edges=opt,
            edges_a=ea,
            edges_b=eb,
            ratio_a=ra,
            ratio_b=rb,
            nU=Int(ta.nU),
            nV=Int(ta.nV),
            k=Int(ta.k),
            theta=Int(ta.theta),
            by_score=!has_opt,
        )

        # Winner is always the higher edge count (score). When no optimum is
        # available on either side, that is the only comparison signal.
        if ea > eb
            push!(a_better, row)
        elseif eb > ea
            push!(b_better, row)
        else
            push!(ties, row)
        end
    end

    println("==================== COMPARE ====================")
    println("A: $algo_a  ($path_a)")
    println("B: $algo_b  ($path_b)")
    println("Shared trials        : $(length(shared))")
    println("Only in A            : $(length(only_a))")
    println("Only in B            : $(length(only_b))")
    println()
    if compared_with_opt > 0
        println("Optimal (vs oracle)  : A=$(a_optimal)  B=$(b_optimal)  both=$(both_optimal)  / $(compared_with_opt)")
    end
    if compared_by_score > 0
        println("Compared by score    : $(compared_by_score) trial(s) with no optimum on either side")
    end
    println("$algo_a better (more edges): $(length(a_better))")
    println("$algo_b better (more edges): $(length(b_better))")
    println("Ties (same edges)    : $(length(ties))")

    if !isempty(a_better)
        println()
        println("--- $algo_a better ---")
        for row in a_better
            opt_str = row.opt_edges === nothing ? "n/a" : string(row.opt_edges)
            println("  seed=$(row.graph_seed)  opt=$(opt_str)  A=$(row.edges_a) (r=$(format_ratio(row.ratio_a)))  B=$(row.edges_b) (r=$(format_ratio(row.ratio_b)))  nU=$(row.nU) nV=$(row.nV) k=$(row.k) θ=$(row.theta)")
        end
    end

    if !isempty(b_better)
        println()
        println("--- $algo_b better ---")
        for row in b_better
            opt_str = row.opt_edges === nothing ? "n/a" : string(row.opt_edges)
            println("  seed=$(row.graph_seed)  opt=$(opt_str)  A=$(row.edges_a) (r=$(format_ratio(row.ratio_a)))  B=$(row.edges_b) (r=$(format_ratio(row.ratio_b)))  nU=$(row.nU) nV=$(row.nV) k=$(row.k) θ=$(row.theta)")
        end
    end

    if !isempty(only_a) || !isempty(only_b)
        println()
        isempty(only_a) || println("Seeds only in A: $(join(only_a, ", "))")
        isempty(only_b) || println("Seeds only in B: $(join(only_b, ", "))")
    end

    println("=================================================")
    return (; shared, a_better, b_better, ties, a_optimal, b_optimal, both_optimal,
        compared_with_opt, compared_by_score)
end

if abspath(PROGRAM_FILE) == @__FILE__
    file_a, file_b, label_a, label_b = parse_compare_args(ARGS)
    compare_suites(file_a, file_b; label_a=label_a, label_b=label_b)
end
