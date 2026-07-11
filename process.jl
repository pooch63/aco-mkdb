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
    julia process.jl <path_to_input_jsonl> <path_to_output_csv>

  Example:
    julia process.jl data/reviews.jsonl data/interactions.csv
=================================================================================
=#

using JSON3
using DataFrames
using CSV

function main()
    # Validate command line arguments
    if length(ARGS) < 2
        println(stderr, "Error: Missing arguments.")
        println(stderr, "Usage: julia process_reviews.jl <input_jsonl_path> <output_csv_path>")
        exit(1)
    end

    input_path = ARGS[1]
    output_path = ARGS[2]

    # Verify input file exists
    if !isfile(input_path)
        println(stderr, "Error: Input file '$input_path' does not exist.")
        exit(1)
    end

    println("Starting conversion...")
    println("Input source:  $input_path")
    println("Output target: $output_path")

    # Initialize typed collections for high-performance array construction
    user_ids = String[]
    item_ids = String[]
    timestamps = Int64[]

    # Stream the JSONL file line-by-line to minimize memory footprint
    try
        open(input_path, "r") do file
            line_count = 0
            for line in eachline(file)
                line_count += 1
                cleaned_line = strip(line)
                if !isempty(cleaned_line)
                    # Parse using JSON3 for optimized performance
                    row = JSON3.read(cleaned_line)
                    
                    # Extract fields with fallback protection
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
        println(stderr, "An error occurred while reading the file: $e")
        exit(1)
    end

    # Build the target DataFrame
    println("Assembling dataset...")
    df = DataFrame(
        user_id = user_ids,
        item_id = item_ids,
        timestamp = timestamps
    )

    # Export to the specified CSV destination
    try
        CSV.write(output_path, df)
        println("Success! Successfully processed $(nrow(df)) entries.")
    catch e
        println(stderr, "An error occurred while writing the CSV file: $e")
        exit(1)
    end
end

# Trigger execution
main()