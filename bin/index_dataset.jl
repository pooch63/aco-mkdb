#=
Remap a raw u,v CSV to 1-based integer ids (GraphLogic.index_dataset).

  julia bin/index_dataset.jl <input_csv> <output_csv> <output_mapping_dir>
=#

include(joinpath(dirname(@__DIR__), "src", "graphlogic.jl"))

if length(ARGS) < 3
    println(stderr, "Usage: julia bin/index_dataset.jl <input_csv> <output_csv> <output_mapping_dir>")
    exit(1)
end
index_dataset(ARGS[1], ARGS[2], ARGS[3])
