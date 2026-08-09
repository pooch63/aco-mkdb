struct Pheromones
    U::Vector{Float64}
    V::Vector{Float64}
end

Pheromones(nU::Int, nV::Int) = Pheromones(ones(nU), ones(nV))
Pheromones(fg::FrozenBipartite) = Pheromones(length(fg.u_ids), length(fg.v_ids))
zero_pheromones(fg::FrozenBipartite) = Pheromones(zeros(length(fg.u_ids)), zeros(length(fg.v_ids)))

get_pheromone(pheromones::Pheromones, node::Node) = (node.is_u ? pheromones.U : pheromones.V)[node.id]
add_pheromone!(pheromones::Pheromones, node::Node, pheromone::Real) = ((node.is_u ? pheromones.U : pheromones.V)[node.id] += pheromone)

function evaporate_pheromones!(pheromones::Pheromones, evaporation::Float64)
    pheromones.U .*= evaporation
    pheromones.V .*= evaporation
end

function default_species_pheromone_bounds(deposit::Real, n_nodes::Int, evaporation::Float64, best_node_count::Int)
    n_nodes >= 1 || throw(ArgumentError("n_nodes must be >= 1, got $n_nodes"))
    0.0 < evaporation <= 1.0 || throw(ArgumentError("evaporation must be in (0, 1], got $evaporation"))
    deposit > 0 || throw(ArgumentError("deposit must be > 0, got $deposit"))
    quality = max(1.0, log(max(best_node_count, 1)))
    τ_max = 10 * Float64(deposit) * quality / max(1 - evaporation, eps(Float64))
    τ_min = τ_max / (2 * n_nodes)
    return τ_min, τ_max
end

function clamp_pheromones!(pheromones::Pheromones, τ_min::Float64, τ_max::Float64)
    τ_min <= τ_max || throw(ArgumentError("pheromone_min ($τ_min) must be <= pheromone_max ($τ_max)"))
    clamp!(pheromones.U, τ_min, τ_max)
    clamp!(pheromones.V, τ_min, τ_max)
    return pheromones
end

struct ColonyPheromones
    shared::Pheromones
    species::Vector{Pheromones}
end

function ColonyPheromones(fg::FrozenBipartite, num_subspecies::Int)
    ColonyPheromones(Pheromones(fg), [zero_pheromones(fg) for _ in 1:num_subspecies])
end

function zero_colony_pheromones(fg::FrozenBipartite, num_subspecies::Int)
    ColonyPheromones(zero_pheromones(fg), [zero_pheromones(fg) for _ in 1:num_subspecies])
end

function evaporate_pheromones!(colony::ColonyPheromones, evaporation::Float64)
    evaporate_pheromones!(colony.shared, evaporation)
    for species_pheromones in colony.species
        evaporate_pheromones!(species_pheromones, evaporation)
    end
end

function clamp_species_pheromones!(colony::ColonyPheromones, τ_min::Float64, τ_max::Float64)
    for species_pheromones in colony.species
        clamp_pheromones!(species_pheromones, τ_min, τ_max)
    end
    return colony
end

function clamp_species_pheromones!(colony::ColonyPheromones,
    τ_mins::AbstractVector{Float64}, τ_maxs::AbstractVector{Float64})
    length(τ_mins) == length(colony.species) ||
        throw(ArgumentError("τ_mins length $(length(τ_mins)) != num species $(length(colony.species))"))
    length(τ_maxs) == length(colony.species) ||
        throw(ArgumentError("τ_maxs length $(length(τ_maxs)) != num species $(length(colony.species))"))
    for s in eachindex(colony.species)
        clamp_pheromones!(colony.species[s], τ_mins[s], τ_maxs[s])
    end
    return colony
end

function update_species_pheromone_bounds!(τ_mins::Vector{Float64}, τ_maxs::Vector{Float64},
    deposit::Real, n_nodes::Int, evaporation::Float64, best_subgraphs::Vector{SubGraph};
    pheromone_min::Union{Float64,Nothing}=nothing, pheromone_max::Union{Float64,Nothing}=nothing)
    for s in eachindex(best_subgraphs)
        best_n = max(2, Subgraph.vertex_count(best_subgraphs[s]))
        τ_lo, τ_hi = default_species_pheromone_bounds(deposit, n_nodes, evaporation, best_n)
        τ_mins[s] = pheromone_min === nothing ? τ_lo : pheromone_min
        τ_maxs[s] = pheromone_max === nothing ? τ_hi : pheromone_max
        τ_mins[s] <= τ_maxs[s] ||
            throw(ArgumentError("pheromone_min ($(τ_mins[s])) must be <= pheromone_max ($(τ_maxs[s])) for species $s"))
    end
    return τ_mins, τ_maxs
end

function merge_pheromones!(colony::ColonyPheromones, additions::ColonyPheromones)
    colony.shared.U .+= additions.shared.U
    colony.shared.V .+= additions.shared.V
    for s in eachindex(colony.species)
        colony.species[s].U .+= additions.species[s].U
        colony.species[s].V .+= additions.species[s].V
    end
end

function effective_pheromone(colony::ColonyPheromones, node::Node, species::Int)
    τ = get_pheromone(colony.shared, node) + get_pheromone(colony.species[species], node)
    for s in eachindex(colony.species)
        s == species && continue
        τ -= get_pheromone(colony.species[s], node)
    end
    return τ
end
