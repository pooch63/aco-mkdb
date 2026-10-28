#=
=================================================================================
Rebuild datasets from a manifest file
=================================================================================
Reads one dataset key per line (e.g. `data/datasets.txt`) and runs the shared
GraphLogic provider pipeline with `--download` for each entry.

Usage:
  julia bin/fetch_datasets.jl
  julia bin/fetch_datasets.jl data/datasets.txt
  julia bin/fetch_datasets.jl --skip-existing
  julia bin/fetch_datasets.jl --prefix=konect-small
  julia bin/fetch_datasets.jl --dry-run

Options:
  --skip-existing   skip keys that already have indexed_interactions.csv
  --prefix=NAME     only fetch keys equal to or under NAME/ (slash-bounded)
  --dry-run         print planned actions without downloading
  --data-root=PATH  data directory (default: data)
=#

include(joinpath(dirname(@__DIR__), "src", "graphlogic.jl"))
fetch_main(ARGS)
