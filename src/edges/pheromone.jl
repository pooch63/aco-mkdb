#=
Edge-trail pheromone for the edges-ACO variant.

Trails live on CSR U→V edge slots (`fg.v_adj` indices), not on vertices.
Shared / species / effective-τ arithmetic matches node ACO (`src/aco/pheromone.jl`).
=#

"""Pheromone parallel to `FrozenBipartite` U→V CSR (`length == length(fg.v_adj)`)."""
struct EdgePheromones
    τ::Vector{Float64}
end

EdgePheromones(n_edges::Int) = EdgePheromones(ones(n_edges))
EdgePheromones(fg::FrozenBipartite) = EdgePheromones(length(fg.v_adj))
zero_edge_pheromones(fg::FrozenBipartite) = EdgePheromones(zeros(length(fg.v_adj)))

"""
CSR slot into the U→V edge list for compact vertex ids `u_id`, `v_id`.
Returns `nothing` when the edge is absent.
"""
function edge_slot(fg::FrozenBipartite, u_id::Int, v_id::Int)
    ui = get(fg.u_index, u_id, nothing)
    ui === nothing && return nothing
    vi = get(fg.v_index, v_id, nothing)
    vi === nothing && return nothing
    @inbounds for k in neighbor_range_u(fg, ui)
        fg.v_adj[k] == vi && return k
    end
    return nothing
end

get_edge_pheromone(pheromones::EdgePheromones, slot::Int) = pheromones.τ[slot]
add_edge_pheromone!(pheromones::EdgePheromones, slot::Int, pheromone::Real) =
    (pheromones.τ[slot] += pheromone)

function evaporate_edge_pheromones!(pheromones::EdgePheromones, evaporation::Float64)
    pheromones.τ .*= evaporation
end

function default_edge_pheromone_bounds(deposit::Real, n_edges::Int, evaporation::Float64,
    best_edge_count::Int)
    n_edges >= 1 || throw(ArgumentError("n_edges must be >= 1, got $n_edges"))
    0.0 < evaporation <= 1.0 || throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    deposit > 0 || throw(ArgumentError("deposit must be > 0, got $deposit"))
    quality = max(1.0, log(max(best_edge_count, 1)))
    τ_max = 10 * Float64(deposit) * quality / max(1 - evaporation, eps(Float64))
    τ_min = τ_max / (2 * log(n_edges + 1))
    return τ_min, τ_max
end

function clamp_edge_pheromones!(pheromones::EdgePheromones, τ_min::Float64, τ_max::Float64)
    τ_min <= τ_max || throw(ArgumentError("pheromone_min ($τ_min) must be <= pheromone_max ($τ_max)"))
    clamp!(pheromones.τ, τ_min, τ_max)
    return pheromones
end

struct ColonyEdgePheromones
    shared::EdgePheromones
    species::Vector{EdgePheromones}
end

function ColonyEdgePheromones(fg::FrozenBipartite, num_subspecies::Int)
    ColonyEdgePheromones(EdgePheromones(fg), [zero_edge_pheromones(fg) for _ in 1:num_subspecies])
end

function zero_colony_edge_pheromones(fg::FrozenBipartite, num_subspecies::Int)
    ColonyEdgePheromones(zero_edge_pheromones(fg), [zero_edge_pheromones(fg) for _ in 1:num_subspecies])
end

function evaporate_edge_pheromones!(colony::ColonyEdgePheromones, evaporation::Float64)
    evaporate_edge_pheromones!(colony.shared, evaporation)
    for species_pheromones in colony.species
        evaporate_edge_pheromones!(species_pheromones, evaporation)
    end
end

function clamp_species_edge_pheromones!(colony::ColonyEdgePheromones, τ_min::Float64, τ_max::Float64)
    clamp_edge_pheromones!(colony.shared, τ_min, τ_max)
    for species_pheromones in colony.species
        clamp_edge_pheromones!(species_pheromones, τ_min, τ_max)
    end
    return colony
end

function clamp_species_edge_pheromones!(colony::ColonyEdgePheromones,
    τ_mins::AbstractVector{Float64}, τ_maxs::AbstractVector{Float64})
    length(τ_mins) == length(colony.species) ||
        throw(ArgumentError("τ_mins length $(length(τ_mins)) != num species $(length(colony.species))"))
    length(τ_maxs) == length(colony.species) ||
        throw(ArgumentError("τ_maxs length $(length(τ_maxs)) != num species $(length(colony.species))"))
    clamp_edge_pheromones!(colony.shared, minimum(τ_mins), maximum(τ_maxs))
    for s in eachindex(colony.species)
        clamp_edge_pheromones!(colony.species[s], τ_mins[s], τ_maxs[s])
    end
    return colony
end

function update_species_edge_pheromone_bounds!(τ_mins::Vector{Float64}, τ_maxs::Vector{Float64},
    deposit::Real, n_edges::Int, evaporation::Float64, best_subgraphs::Vector{SubGraph},
    fg::FrozenBipartite;
    pheromone_min::Union{Float64,Nothing}=nothing, pheromone_max::Union{Float64,Nothing}=nothing)
    for s in eachindex(best_subgraphs)
        best_e = max(1, Subgraph.edge_count(fg, best_subgraphs[s]))
        τ_lo, τ_hi = default_edge_pheromone_bounds(deposit, n_edges, evaporation, best_e)
        τ_mins[s] = pheromone_min === nothing ? τ_lo : pheromone_min
        τ_maxs[s] = pheromone_max === nothing ? τ_hi : pheromone_max
        τ_mins[s] <= τ_maxs[s] ||
            throw(ArgumentError("pheromone_min ($(τ_mins[s])) must be <= pheromone_max ($(τ_maxs[s])) for species $s"))
    end
    return τ_mins, τ_maxs
end

function merge_edge_pheromones!(colony::ColonyEdgePheromones, additions::ColonyEdgePheromones)
    colony.shared.τ .+= additions.shared.τ
    for s in eachindex(colony.species)
        colony.species[s].τ .+= additions.species[s].τ
    end
end

function effective_edge_pheromone(colony::ColonyEdgePheromones, slot::Int, species::Int)
    τ = get_edge_pheromone(colony.shared, slot) + get_edge_pheromone(colony.species[species], slot)
    for s in eachindex(colony.species)
        s == species && continue
        τ -= get_edge_pheromone(colony.species[s], slot)
    end
    return τ
end

"""
Mean effective edge-τ from candidate `node` into the opposite side of `sg`.

Returns 1.0 (neutral initial trail) when the opposite side is empty or the
candidate shares no graph edges with it (k-defective adds).
"""
function mean_edge_pheromone_into(colony::ColonyEdgePheromones, fg::FrozenBipartite,
    node::DegreeNode, species::Int, sg::SubGraph)
    sum_τ = 0.0
    n = 0
    if node.is_u
        for v in sg.V
            slot = edge_slot(fg, node.id, v)
            slot === nothing && continue
            sum_τ += effective_edge_pheromone(colony, slot, species)
            n += 1
        end
    else
        for u in sg.U
            slot = edge_slot(fg, u, node.id)
            slot === nothing && continue
            sum_τ += effective_edge_pheromone(colony, slot, species)
            n += 1
        end
    end
    n == 0 && return 1.0
    return max(sum_τ / n, eps(Float64))
end

"""Deposit `amount` on every existing edge between `node` and the opposite side of `sg`."""
function deposit_edges_into!(pheromones::EdgePheromones, fg::FrozenBipartite,
    node::Node, sg::SubGraph, amount::Real)
    if node.is_u
        for v in sg.V
            slot = edge_slot(fg, node.id, v)
            slot === nothing && continue
            add_edge_pheromone!(pheromones, slot, amount)
        end
    else
        for u in sg.U
            slot = edge_slot(fg, u, node.id)
            slot === nothing && continue
            add_edge_pheromone!(pheromones, slot, amount)
        end
    end
    return pheromones
end

"""Deposit on every edge of the induced bipartite subgraph `sg`."""
function deposit_induced_edges!(pheromones::EdgePheromones, fg::FrozenBipartite,
    sg::SubGraph, amount::Real)
    for u in sg.U
        for v in sg.V
            slot = edge_slot(fg, u, v)
            slot === nothing && continue
            add_edge_pheromone!(pheromones, slot, amount)
        end
    end
    return pheromones
end
