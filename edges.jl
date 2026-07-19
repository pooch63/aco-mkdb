#=
=================================================================================
JSONL to CSV Converter Script
=================================================================================
This script reads a JSONL file containing product review data and exports a 
processed CSV file containing only three columns: user_id, item_id, and timestamp.
The 'asin' field from the source JSONL is renamed to 'item_id' in the output.

Prerequisites:
  Ensure you have Julia and the required packages installed. You can install them
  by running Julia and entering the package manager:
    julia> r"using Pkg; Pkg.add([\"JSON3\", \"DataFrames\", \"CSV\"])"

How to Run from the Command Line:
  Format:
    julia edges.jl <path_to_input_jsonl> <path_to_output_csv>

  Example:
    julia edges.jl data/reviews.jsonl data/interactions.csv
=================================================================================
=#

using JSON3
using DataFrames
using CSV

function convert_jsonl_to_csv(input_path::String, output_path::String)
    if !isfile(input_path)
        error("Input file '$input_path' does not exist.")
    end

    mkpath(dirname(output_path))

    println("Starting conversion...")
    println("Input source:  $input_path")
    println("Output target: $output_path")

    user_ids = String[]
    item_ids = String[]
    timestamps = Int64[]

    try
        open(input_path, "r") do file
            line_count = 0
            for line in eachline(file)
                line_count += 1
                cleaned_line = strip(line)
                if !isempty(cleaned_line)
                    row = JSON3.read(cleaned_line)

                    if haskey(row, :user_id) && haskey(row, :asin) && haskey(row, :timestamp)
                        push!(user_ids, string(row[:user_id]))
                        push!(item_ids, string(row[:asin]))
                        push!(timestamps, Int64(row[:timestamp]))
                    else
                        println(stderr, "Warning: Skipping malformed row at line $line_count")
                    end
                end
            end
        end
    catch e
        error("An error occurred while reading the file: $e")
    end

    println("Assembling dataset...")
    df = DataFrame(
        user_id = user_ids,
        item_id = item_ids,
        timestamp = timestamps
    )

    try
        CSV.write(output_path, df)
        println("Success! Successfully processed $(nrow(df)) entries.")
    catch e
        error("An error occurred while writing the CSV file: $e")
    end

    return df
end

function main()
    if length(ARGS) < 2
        println(stderr, "Error: Missing arguments.")
        println(stderr, "Usage: julia edges.jl <input_jsonl_path> <output_csv_path>")
        exit(1)
    end

    input_path = ARGS[1]
    output_path = ARGS[2]
    convert_jsonl_to_csv(input_path, output_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end