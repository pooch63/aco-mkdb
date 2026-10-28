#=
=================================================================================
Dataset Processing Pipeline
=================================================================================
Download + convert + index one dataset via the shared GraphLogic providers
(../graph-logic/src/providers/).

How to Run:
  julia bin/process.jl <provider>/<dataset> --download
  julia bin/process.jl <provider>/<dataset> --source=/path/to/raw.file
  julia bin/process.jl <provider> <dataset> --download

Examples:
  julia bin/process.jl amazon/Appliances --download
  julia bin/process.jl amazon boxes --source=data/subscription-boxes.jsonl
  julia bin/process.jl konect/bitcoin --download
=================================================================================
=#

include(joinpath(dirname(@__DIR__), "src", "graphlogic.jl"))
process_main(ARGS)
