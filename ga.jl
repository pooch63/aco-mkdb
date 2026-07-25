include("graph.jl")
include("search.jl")
include("opponent.jl")

# If the number of entries in g.adjU is not equal to the number of nodes or same for V,
# e.g., there are some gaps in node IDs, you'll need to pass the maximum node ID for each side
function heuristic(g::BipartiteGraph, use_heuristic::Bool,
    k::Int, θ::Int, n::Int, reduction::ReductionMode=progressive;
    num_U::Union{Int, Nothing}=nothing, num_V::Union{Int, Nothing}=nothing)

    @assert θ > k "θ must be greater than k"

    fg = freeze(g)

    if isempty(fg.u_ids) || isempty(fg.v_ids)
        return search(fg, k, θ, n, use_heuristic)
    end

    fg = apply_graph_reductions!(g, k, θ, num_U, num_V, use_heuristic, reduction)

    # If there's not enough nodes remaining on either side, then we
    # already know it's an invalid solution
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return SubGraph(Set(), Set())
    end

    return ga(
        fg,
        k,
        θ,
        n,
        use_heuristic
    )
end



# Implement heuristic that branches and bounds on only a subset of the graph,
# then branch and bound on the vertices that won
function search(g::BipartiteGraph, k::Int, θ::Int, n::Int, use_heuristic::Bool)
    D = use_heuristic ? initial_heuristic(freeze(g), k, θ) : SubGraph(Set(), Set())

    # Split the graph into evenly-distributed subgraphs


end
