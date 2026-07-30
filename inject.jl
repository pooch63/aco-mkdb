#=
=================================================================================
Biclique Injection Script
=================================================================================
This script injects a biclique of size p x q into a zero-based indexed CSV
edge file by choosing p existing users and q existing items and inserting
all edges between them except k missing edges.

Usage:
  julia inject.jl grocery --p=8 --q=7 --k=3
  julia inject.jl amazon/boxes --p=8 --q=7 --k=3
  julia inject.jl --file=data/grocery/indexed_interactions.csv --p=8 --q=7 --k=3

Arguments:
  dataset_name    Dataset key under data/ (flat or nested, e.g. grocery or amazon/boxes)

Flags:
  --file=PATH    Optional explicit CSV file path to modify instead of using a dataset name
  --p=N          Number of users in the biclique
  --q=M          Number of items in the biclique
  --k=K          Number of intentionally missing edges in the biclique
  --seed=S       Optional random seed for reproducible selection
  --attempts=N   Optional maximum random tries to find a valid biclique (default: 20)

The script appends new rows to the CSV file and preserves the original header.
=#

using Random

isdefined(@__MODULE__, :__PATHS_JL__) || include(joinpath(@__DIR__, "src", "paths.jl"))

function parse_flag(arg::String)
    if startswith(arg, "--") && occursin('=', arg)
        key, val = split(arg[3:end], '='; limit=2)
        return key, val
    end
    return nothing
end

function parse_args()
    args = Dict{String,String}()
    dataset_name = nothing

    for raw in ARGS
        if !startswith(raw, "--") && dataset_name === nothing
            dataset_name = raw
            continue
        end

        parsed = parse_flag(raw)
        if parsed === nothing
            println(stderr, "Error: Invalid argument format '$raw'. Expected --name=value or a single dataset name.")
            exit(1)
        end
        key, val = parsed
        args[key] = val
    end

    input_file = get(args, "file", nothing)
    if input_file === nothing
        if dataset_name === nothing
            input_file = resolve_graph_path("grocery")
        else
            input_file = resolve_graph_path(dataset_name)
        end
    end

    p = haskey(args, "p") ? parse(Int, args["p"]) : nothing
    q = haskey(args, "q") ? parse(Int, args["q"]) : nothing
    k = haskey(args, "k") ? parse(Int, args["k"]) : nothing
    seed = haskey(args, "seed") ? parse(Int, args["seed"]) : nothing
    attempts = haskey(args, "attempts") ? parse(Int, args["attempts"]) : 20

    if p === nothing || q === nothing || k === nothing
        println(stderr, "Error: Missing required flags. --p, --q, and --k are required.")
        println(stderr, "Usage: julia inject.jl [dataset_name] --p=N --q=M --k=K [--seed=S] [--attempts=N]")
        println(stderr, "Example: julia inject.jl amazon/boxes --p=8 --q=7 --k=3")
        exit(1)
    end

    if p <= 0 || q <= 0 || k < 0
        println(stderr, "Error: --p and --q must be positive integers and --k must be non-negative.")
        exit(1)
    end
    if k >= p * q
        println(stderr, "Error: --k must be smaller than p*q. Got p=$p, q=$q, k=$k.")
        exit(1)
    end
    if attempts <= 0
        println(stderr, "Error: --attempts must be a positive integer.")
        exit(1)
    end

    return input_file, p, q, k, seed, attempts
end

function collect_ids_and_timestamp(file_path::String)
    if !isfile(file_path)
        println(stderr, "Error: File not found: $file_path")
        exit(1)
    end

    user_set = Set{Int}()
    item_set = Set{Int}()
    max_timestamp = Int(typemin(Int))

    open(file_path, "r") do io
        header = readline(io)
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ',')
            if length(parts) < 3
                continue
            end
            u = parse(Int, parts[1])
            v = parse(Int, parts[2])
            ts = parse(Int, parts[3])
            push!(user_set, u)
            push!(item_set, v)
            if ts > max_timestamp
                max_timestamp = ts
            end
        end
    end

    if isempty(user_set) || isempty(item_set)
        println(stderr, "Error: No valid edges found in '$file_path'.")
        exit(1)
    end

    return collect(user_set), collect(item_set), max_timestamp
end

function sample_unique(rng::AbstractRNG, values::Vector{Int}, count::Int)
    if length(values) < count
        println(stderr, "Error: Cannot sample $count unique values from a pool of $(length(values)).")
        exit(1)
    end
    shuffled = copy(values)
    shuffle!(rng, shuffled)
    return shuffled[1:count]
end

function build_existing_pairs(file_path::String, users::Vector{Int}, items::Vector{Int})
    user_set = Set(users)
    item_set = Set(items)
    existing = Set{Tuple{Int,Int}}()

    open(file_path, "r") do io
        readline(io)
        for line in eachline(io)
            line = strip(line)
            isempty(line) && continue
            parts = split(line, ',')
            if length(parts) < 3
                continue
            end
            u = parse(Int, parts[1])
            v = parse(Int, parts[2])
            user_set_contains = u in user_set
            item_set_contains = v in item_set
            if user_set_contains && item_set_contains
                push!(existing, (u, v))
            end
        end
    end

    return existing
end

function choose_biclique(rng::AbstractRNG, users::Vector{Int}, items::Vector{Int}, p::Int, q::Int, k::Int, file_path::String, attempts::Int)
    all_users = users
    all_items = items

    for attempt in 1:attempts
        chosen_users = sample_unique(rng, all_users, p)
        chosen_items = sample_unique(rng, all_items, q)
        existing_pairs = build_existing_pairs(file_path, chosen_users, chosen_items)

        total_pairs = p * q
        absent_pairs = [(u, v) for u in chosen_users, v in chosen_items if (u, v) ∉ existing_pairs]
        absent_count = length(absent_pairs)

        if absent_count < k
            if attempt == attempts
                break
            end
            continue
        end

        shuffle!(rng, absent_pairs)
        missing_edges = Set(absent_pairs[1:k])
        to_insert = [(u, v) for u in chosen_users, v in chosen_items if (u, v) ∉ missing_edges && (u, v) ∉ existing_pairs]

        return chosen_users, chosen_items, existing_pairs, missing_edges, to_insert
    end

    println(stderr, "Error: Could not find a biclique with k=$k missing edges after $attempts attempts.")
    println(stderr, "Try a smaller k or use a larger input file with more absent edges.")
    exit(1)
end

function append_edges(file_path::String, edges::Vector{Tuple{Int,Int}}, timestamp::Int)
    if isempty(edges)
        return 0
    end

    open(file_path, "a") do io
        for (u, v) in edges
            println(io, "$(u),$(v),$(timestamp)")
        end
    end
    return length(edges)
end

function main()
    input_file, p, q, k, seed, attempts = parse_args()
    rng = seed === nothing ? MersenneTwister() : MersenneTwister(seed)

    println("Input file: $input_file")
    println("Biclique size: p=$p, q=$q, k=$k")
    println("Random seed: $(seed === nothing ? "none" : string(seed))")
    println("Max selection attempts: $attempts")

    users, items, max_timestamp = collect_ids_and_timestamp(input_file)
    println("Found $(length(users)) unique users and $(length(items)) unique items.")
    println("Current max timestamp: $max_timestamp")

    chosen_users, chosen_items, existing_pairs, missing_edges, to_insert = choose_biclique(rng, users, items, p, q, k, input_file, attempts)

    total_pairs = p * q
    inserted_count = append_edges(input_file, to_insert, max_timestamp + 1)

    println("Selected users: $(sort(chosen_users))")
    println("Selected items: $(sort(chosen_items))")
    println("Existing edges in the chosen biclique: $(length(existing_pairs))")
    println("Missing edges reserved: $(length(missing_edges))")
    println("New edges appended: $inserted_count")
    println("Final biclique edge count will be: $(total_pairs - k)")
    println("Injected edges were appended with timestamp $(max_timestamp + 1).")
    println("Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
