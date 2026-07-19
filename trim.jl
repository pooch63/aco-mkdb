#=
=================================================================================
Graph Trimmer
=================================================================================
This script trims a saved indexed graph CSV to a maximum number of data rows.
It accepts either:

  1. No dataset argument, in which case it trims data/indexed_interactions.csv
  2. A dataset name, in which case it reads data/<dataset_name>/indexed_interactions.csv
  3. An explicit file path to a CSV graph file

How to Run from the Command Line:
  Format:
    julia trim.jl <dataset_name_or_path> <max_lines> [--overwrite]

  Examples:
    julia trim.jl boxes 2000
    julia trim.jl /path/to/indexed_interactions.csv 100
    julia trim.jl boxes 2000 --overwrite

When --overwrite is provided, the script rewrites the original CSV file in place
with only the first max_lines data rows preserved.
=================================================================================
=#

function resolve_graph_path(dataset_name::Union{String,Nothing}=nothing)
    if dataset_name === nothing || isempty(dataset_name)
        return "data/indexed_interactions.csv"
    end

    if isfile(dataset_name)
        return dataset_name
    end

    return joinpath("data", dataset_name, "indexed_interactions.csv")
end

function build_output_path(input_path::String)
    dir_path = dirname(input_path)
    base_name = splitext(basename(input_path))[1]
    return joinpath(dir_path, "$(base_name)_trimmed.csv")
end

function trim_graph(input_path::String, max_lines::Int; overwrite::Bool=false)
    if max_lines <= 0
        error("max_lines must be a positive integer.")
    end

    if !isfile(input_path)
        error("Input graph file '$input_path' does not exist.")
    end

    lines = readlines(input_path)
    if isempty(lines)
        error("Input graph file '$input_path' is empty.")
    end

    header = lines[1]
    kept_lines = String[]
    count = 0

    for line in lines[2:end]
        line = strip(line)
        isempty(line) && continue

        push!(kept_lines, line)
        count += 1

        if count >= max_lines
            break
        end
    end

    output_path = overwrite ? input_path : build_output_path(input_path)
    mkpath(dirname(output_path))

    open(output_path, "w") do io_out
        write(io_out, header * "\n")
        for line in kept_lines
            write(io_out, line * "\n")
        end
    end

    println("Wrote trimmed graph to: $output_path")
    println("Kept $count data rows from '$input_path'.")
end

function main()
    if length(ARGS) < 2
        println(stderr, "Error: Missing arguments.")
        println(stderr, "Usage: julia trim.jl <dataset_name_or_path> <max_lines> [--overwrite]")
        exit(1)
    end

    overwrite = false
    args = ARGS[3:end]
    if any(arg -> arg == "--overwrite", args)
        overwrite = true
    end

    dataset_or_path = ARGS[1]
    max_lines = tryparse(Int, ARGS[2])
    if max_lines === nothing
        println(stderr, "Error: '$(ARGS[2])' is not a valid integer.")
        exit(1)
    end

    input_path = resolve_graph_path(dataset_or_path)
    trim_graph(input_path, max_lines; overwrite=overwrite)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
