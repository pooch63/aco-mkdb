using Random

function random_graph(; nU_range=3:6, nV_range=3:6, edge_prob=0.5)
    nU = rand(nU_range)
    nV = rand(nV_range)

    edges = Set{Tuple{Int,Int}}()
    for u in 1:nU, v in 1:nV
        if rand() < edge_prob
            push!(edges, (u, v))
        end
    end

    # θ must be small enough that a valid biclique can exist at all.
    θ = rand(1:min(nU, nV))
    # k must be small enough to keep brute force meaningful (not "anything goes").
    k = rand(0:min(nU * nV, 4))

    return edges, nU, nV, k, θ
end
