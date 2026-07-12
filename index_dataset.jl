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

function main()
    # Validate command line arguments
    if length(ARGS) < 3
        println(stderr, "Error: Missing arguments.")
        println(stderr, "Usage: julia index_dataset.jl <input_csv> <output_csv> <output_mapping_dir>")
        exit(1)
    end

    input_path = ARGS[1]
    output_csv_path = ARGS[2]
    mapping_dir = ARGS[3]

    # Verify input file exists
    if !isfile(input_path)
        println(stderr, "Error: Input file '$input_path' does not exist.")
        exit(1)
    end

    println("Loading dataset...")
    df = CSV.read(input_path, DataFrame)

    # Validate structure
    required_cols = ["user_id", "item_id", "timestamp"]
    if !all(col -> col in names(df), required_cols)
        println(stderr, "Error: Input CSV must contain 'user_id', 'item_id', and 'timestamp' columns.")
        exit(1)
    end

    println("Processing user mappings...")
    # Find unique users and build a mapping dictionary
    unique_users = unique(df.user_id)
    # Creating a 0-indexed integer mapping. Change `i - 1` to `i` if you prefer 1-based indexing.
    user_to_idx = Dict(user => i - 1 for (i, user) in enumerate(unique_users))
    
    println("Processing item mappings...")
    # Find unique items and build a mapping dictionary
    unique_items = unique(df.item_id)
    item_to_idx = Dict(item => i - 1 for (i, item) in enumerate(unique_items))

    println("Calculating timestamp baseline...")
    # Determine minimum timestamp
    min_timestamp = minimum(df.timestamp)

    println("Transforming dataset...")
    # Map raw strings to continuous integers and zero-base the timestamps
    transformed_df = DataFrame(
        user_id = [user_to_idx[u] for u in df.user_id],
        item_id = [item_to_idx[i] for i in df.item_id],
        timestamp = df.timestamp .- min_timestamp
    )

    # Create mapping directory if it doesn't exist
    if !isdir(mapping_dir)
        mkpath(mapping_dir)
    end

    println("Saving processed dataset...")
    CSV.write(output_csv_path, transformed_df)

    println("Saving reverse mapping files...")
    # Create DataFrames for exporting reverse lookups
    user_mapping_df = DataFrame(index = 0:(length(unique_users)-1), user_id = unique_users)
    item_mapping_df = DataFrame(index = 0:(length(unique_items)-1), item_id = unique_items)

    CSV.write(joinpath(mapping_dir, "user_mapping.csv"), user_mapping_df)
    CSV.write(joinpath(mapping_dir, "item_mapping.csv"), item_mapping_df)

    println("Saving metadata...")
    # Write metadata log detailing the translation metrics
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
end

main()