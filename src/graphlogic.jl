#=
Loads the shared GraphLogic package (graph structures, CSV I/O, dataset paths,
and download providers). Source lives in ../graph-logic.

One-time setup on a new machine:
  julia -e 'using Pkg; Pkg.develop(path="../graph-logic")'
=#

const __GRAPHLOGIC_JL__ = true

if Base.find_package("GraphLogic") === nothing
    error("GraphLogic is not installed. Run once from the repo root:\n" *
          "  julia -e 'using Pkg; Pkg.develop(path=\"../graph-logic\")'")
end

using GraphLogic
using GraphLogic: _dense_assignment
