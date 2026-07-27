using Random

function random_graph(; nU_range=3:6, nV_range=3:6, edge_prob=0.5,
    θ_max::Union{Int,Nothing}=nothing, k_max::Int=4)
    isempty(nU_range) && throw(ArgumentError("nU_range is empty: $nU_range (need lo ≤ hi)"))
    isempty(nV_range) && throw(ArgumentError("nV_range is empty: $nV_range (need lo ≤ hi)"))
    nU = rand(nU_range)
    nV = rand(nV_range)

    # θ must be small enough that a valid biclique can exist at all.
    θ_hi = min(nU, nV)
    if θ_max !== nothing
        θ_hi = min(θ_hi, θ_max)
    end
    θ = rand(1:θ_hi)
    # k must be small enough to keep search meaningful (not "anything goes").
    k = rand(0:min(nU * nV, k_max, θ - 1))

    # Plant a θ×θ k-defective biclique so a feasible solution always exists.
    plant_U = randperm(nU)[1:θ]
    plant_V = randperm(nV)[1:θ]
    plant_pairs = [(u, v) for u in plant_U for v in plant_V]
    shuffle!(plant_pairs)
    # First k pairs are intentionally missing; the rest are present.
    edges = Set{Tuple{Int,Int}}(plant_pairs[k+1:end])
    plant_set = Set(plant_pairs)

    for u in 1:nU, v in 1:nV
        (u, v) in plant_set && continue
        if rand() < edge_prob
            push!(edges, (u, v))
        end
    end

    return edges, nU, nV, k, θ
end
