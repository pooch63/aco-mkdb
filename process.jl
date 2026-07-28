#=
=================================================================================
Dataset Processing Pipeline
=================================================================================
This script runs the full preprocessing pipeline for a temporal graph dataset.
It expects a path to a JSONL file and a dataset name, then:

  1. Converts the JSONL records into a raw edge CSV at data/<dataset_name>/interactions.csv
  2. Maps user/item IDs to one-based integers and saves the indexed graph at
     data/<dataset_name>/indexed_interactions.csv
  3. Writes the reverse mappings and metadata into data/<dataset_name>/mappings/

Prerequisites:
  Ensure you have Julia and the required packages installed:
    julia> using Pkg; Pkg.add(["JSON3", "DataFrames", "CSV"])

How to Run from the Command Line:
  Format:
    julia process.jl <path_to_input_jsonl> <dataset_name>

  Example:
    julia process.jl data/Grocery_and_Gourmet_Food.jsonl grocery
=================================================================================
=#

using DataFrames
using CSV

include(joinpath(@__DIR__, "src", "edges.jl"))
include("index_dataset.jl")

function process_dataset(jsonl_path::String, dataset_name::String)
    if !isfile(jsonl_path)
        error("Input file '$jsonl_path' does not exist.")
    end

    dataset_dir = joinpath("data", dataset_name)
    mkpath(dataset_dir)

    raw_csv_path = joinpath(dataset_dir, "interactions.csv")
    indexed_csv_path = joinpath(dataset_dir, "indexed_interactions.csv")
    mapping_dir = joinpath(dataset_dir, "mappings")

    println("Converting JSONL to CSV...")
    convert_jsonl_to_csv(jsonl_path, raw_csv_path)

    println("Indexing dataset...")
    index_dataset(raw_csv_path, indexed_csv_path, mapping_dir)

    println("Pipeline complete.")
    println("Raw edges: $raw_csv_path")
    println("Indexed graph: $indexed_csv_path")
    println("Mappings: $mapping_dir")
end

function main()
    if length(ARGS) < 2
        println(stderr, "Error: Missing arguments.")
        println(stderr, "Usage: julia process.jl <path_to_input_jsonl> <dataset_name>")
        exit(1)
    end

    jsonl_path = ARGS[1]
    dataset_name = ARGS[2]
    process_dataset(jsonl_path, dataset_name)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
