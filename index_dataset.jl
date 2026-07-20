#=
=================================================================================
CSV ID Indexer and Timestamp Zero-Baser Script
=================================================================================
This script takes the 3-column CSV output from the previous script (user_id, item_id,
timestamp) and converts it into a continuous, zero-based integer index format.

It outputs:
  1. A processed CSV file with columns: user_id, item_id, timestamp (all integers).
  2. A folder containing 'user_mapping.csv' and 'item_mapping.csv'.
  3. A 'metadata.txt' file capturing the minimum timestamp offset.

Prerequisites:
  Ensure you have Julia and the required packages installed:
    julia> using Pkg; Pkg.add(["DataFrames", "CSV"])

How to Run from the Command Line:
  Format:
    julia index_dataset.jl <input_csv> <output_csv> <output_mapping_dir>

  Example:
    julia index_dataset.jl data/interactions.csv data/indexed_interactions.csv mappings/
=================================================================================
=#

using DataFrames
using CSV

function index_dataset(input_path::String, output_csv_path::String, mapping_dir::String)
    if !isfile(input_path)
        error("Input file '$input_path' does not exist.")
    end

    println("Loading dataset...")
    df = CSV.read(input_path, DataFrame)

    required_cols = ["user_id", "item_id", "timestamp"]
    if !all(col -> col in names(df), required_cols)
        error("Input CSV must contain 'user_id', 'item_id', and 'timestamp' columns.")
    end

    println("Processing user mappings...")
    unique_users = unique(df.user_id)
    user_to_idx = Dict(user => i for (i, user) in enumerate(unique_users))

    println("Processing item mappings...")
    unique_items = unique(df.item_id)
    item_to_idx = Dict(item => i for (i, item) in enumerate(unique_items))

    println("Calculating timestamp baseline...")
    min_timestamp = minimum(df.timestamp)

    println("Transforming dataset...")
    transformed_df = DataFrame(
        user_id = [user_to_idx[u] for u in df.user_id],
        item_id = [item_to_idx[i] for i in df.item_id],
        timestamp = df.timestamp .- min_timestamp
    )

    if !isdir(mapping_dir)
        mkpath(mapping_dir)
    end

    println("Saving processed dataset...")
    CSV.write(output_csv_path, transformed_df)

    println("Saving reverse mapping files...")
    user_mapping_df = DataFrame(index = 1:length(unique_users), user_id = unique_users)
    item_mapping_df = DataFrame(index = 1:length(unique_items), item_id = unique_items)

    CSV.write(joinpath(mapping_dir, "user_mapping.csv"), user_mapping_df)
    CSV.write(joinpath(mapping_dir, "item_mapping.csv"), item_mapping_df)

    println("Saving metadata...")
    open(joinpath(mapping_dir, "metadata.txt"), "w") do f
        write(f, "Dataset Transformation Metadata\n")
        write(f, "=================================\n")
        write(f, "Timestamp Offset (min_timestamp subtracted): $min_timestamp\n")
        write(f, "Total Unique Users: $(length(unique_users))\n")
        write(f, "Total Unique Items: $(length(unique_items))\n")
    end

    println("\nSuccess! Transformation complete.")
    println("Indexed dataset written to: $output_csv_path")
    println("Mappings folder populated:  $mapping_dir")

    return transformed_df
end

function main()
    if length(ARGS) < 3
        println(stderr, "Error: Missing arguments.")
        println(stderr, "Usage: julia index_dataset.jl <input_csv> <output_csv> <output_mapping_dir>")
        exit(1)
    end

    input_path = ARGS[1]
    output_csv_path = ARGS[2]
    mapping_dir = ARGS[3]
    index_dataset(input_path, output_csv_path, mapping_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end