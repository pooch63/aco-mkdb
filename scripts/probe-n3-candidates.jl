#!/usr/bin/env julia
#=
Temporary diagnostic: N²₊/N³₊ candidate sets per starting U (as in opponent.jl),
drop empty candidates, then pack the rest into exactly B buckets.

A bucket's load is |⋃_{u ∈ bucket} C(u)| — the union of post-reduction candidate
nodes (U-side and V-side tagged). Balancing those unions across B buckets is
NP-hard (identical-machine scheduling with submodular set-union costs), so we
use a greedy LPT-style heuristic: sort starts by |C| descending, assign each to
the bucket whose union grows the least (ties → fewest members).

Usage:
  julia scripts/probe-n3-candidates.jl <dataset> <inject_size> --buckets=B \
      [--k=2] [--theta=5] [--seed=1] [--list-members]

  --buckets=B     required; number of buckets
  --list-members  print every starting u in each bucket (default: summary only)
=#

const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "bin", "load.jl"))

using Random

struct StartCand
    i::Int          # 1-based rank in degree order
    u::Int          # original U id
    deg::Int
    nodes::Set{Int} # tagged candidates: +id for U, -id for V
end

Base.length(c::StartCand) = length(c.nodes)

function parse_args(args)
    positional = String[]
    k = 2
    θ = 5
    seed = UInt64(1)
    buckets = nothing
    list_members = false
    for arg in args
        if startswith(arg, "--k=")
            k = parse(Int, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--theta=")
            θ = parse(Int, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--seed=")
            seed = parse(UInt64, split(arg, "="; limit=2)[2])
        elseif startswith(arg, "--buckets=")
            buckets = parse(Int, split(arg, "="; limit=2)[2])
        elseif arg == "--list-members"
            list_members = true
        elseif startswith(arg, "--")
            error("unknown flag: $arg")
        else
            push!(positional, arg)
        end
    end
    length(positional) == 2 || error(
        "usage: probe-n3-candidates.jl <dataset> <inject_size> --buckets=B " *
        "[--k=2] [--theta=5] [--seed=1] [--list-members]")
    buckets === nothing && error("--buckets=B is required")
    buckets >= 1 || error("--buckets must be >= 1, got $buckets")
    dataset = positional[1]
    size = parse(Int, positional[2])
    size >= 1 || error("inject_size must be >= 1, got $size")
    return (; dataset, size, k, θ, seed, buckets, list_members)
end

"""Tagged candidate node id: U stays positive, V becomes -v (v≥1 ⇒ unique)."""
@inline tag_u(u::Int) = u
@inline tag_v(v::Int) = -v

"""
Mirror of the N²₊ / N³₊ construction + reduce_candidate_set in
`branch_from_all_u!` (opponent.jl), without branching.
Returns (n2_raw, n3_raw, tagged candidate Set).
"""
function candidate_after_n3(fg::FrozenBipartite, u::Int, i::Int, rank::Vector{Int},
    id_U::Int, id_V::Int, θ_U::Int, θ_V::Int, k::Int)

    _2_hop = falses(id_U)
    for v in neighbors_u(fg, u)
        for _2u in neighbors_v(fg, v)
            if rank[_2u] > i
                _2_hop[_2u] = true
            end
        end
    end

    _3_hop = falses(id_V)
    for uu in eachindex(_2_hop)
        if _2_hop[uu]
            for v in neighbors_u(fg, uu)
                _3_hop[v] = true
            end
        end
    end

    n2 = count(_2_hop)
    n3 = count(_3_hop)
    C = SubGraph(
        Set(uu for uu in eachindex(_2_hop) if _2_hop[uu]),
        Set(v for v in eachindex(_3_hop) if _3_hop[v]),
    )
    C = reduce_candidate_set(fg, C, u, θ_U, θ_V, k)

    nodes = Set{Int}()
    for uu in C.U
        push!(nodes, tag_u(uu))
    end
    for v in C.V
        push!(nodes, tag_v(v))
    end
    return n2, n3, nodes
end

"""
Greedy multiway partition minimizing max bucket union size.

Sort starts by |C| descending (LPT), assign each to the bucket where
|union ∪ C| is smallest; break ties by smaller current member count, then
smaller bucket index. Guarantees exactly `B` buckets (some may stay empty
only if there are fewer than B non-empty starts — then we use n buckets).
"""
function pack_buckets(cands::Vector{StartCand}, B::Int)
    n = length(cands)
    B_eff = min(B, n)
    order = sortperm(cands; by=length, rev=true)

    members = [Int[] for _ in 1:B_eff]          # indices into `cands`
    unions = [Set{Int}() for _ in 1:B_eff]

    for idx in order
        c = cands[idx]
        best_b = 1
        best_size = typemax(Int)
        best_members = typemax(Int)
        for b in 1:B_eff
            # |A ∪ C| = |A| + |C| - |A ∩ C|
            inter = 0
            for x in c.nodes
                inter += Int(x in unions[b])
            end
            new_size = length(unions[b]) + length(c.nodes) - inter
            nm = length(members[b])
            if new_size < best_size ||
               (new_size == best_size && nm < best_members) ||
               (new_size == best_size && nm == best_members && b < best_b)
                best_b = b
                best_size = new_size
                best_members = nm
            end
        end
        push!(members[best_b], idx)
        union!(unions[best_b], c.nodes)
    end

    return members, unions, B_eff
end

function bucket_intersection(cands::Vector{StartCand}, member_idxs::Vector{Int})
    isempty(member_idxs) && return Set{Int}()
    inter = copy(cands[member_idxs[1]].nodes)
    for j in 2:length(member_idxs)
        intersect!(inter, cands[member_idxs[j]].nodes)
    end
    return inter
end

function main()
    opts = parse_args(ARGS)
    Random.seed!(opts.seed)

    graph_path = resolve_graph_path(opts.dataset)
    isfile(graph_path) || error("missing graph: $graph_path")

    inject = (; enabled=true, nU=opts.size, nV=opts.size, attempts=20)
    g, _edges, plant = load_graph_maybe_inject(graph_path, inject, opts.k, Random.default_rng())

    g_red = deepcopy(g)
    fg = apply_graph_reductions!(g_red, opts.k, opts.θ, nothing, nothing, true, ReductionMode.simple)

    id_U = original_id_span(fg.u_ids, length(g.adjU))
    id_V = original_id_span(fg.v_ids, length(g.adjV))
    us = get_degree_order(fg, true, true)

    rank = zeros(Int, id_U)
    for (pos, u) in enumerate(us)
        rank[u] = pos
    end

    cands = StartCand[]
    dropped = 0
    for (i, u) in enumerate(us)
        _n2, _n3, nodes = candidate_after_n3(fg, u, i, rank, id_U, id_V, opts.θ, opts.θ, opts.k)
        if isempty(nodes)
            dropped += 1
            continue
        end
        push!(cands, StartCand(i, u, degree_u(fg, u), nodes))
    end

    members, unions, B_eff = pack_buckets(cands, opts.buckets)

    union_sizes = [length(unions[b]) for b in 1:B_eff]
    sum_sizes = [sum(length(cands[j]) for j in members[b]; init=0) for b in 1:B_eff]
    inter_sizes = [length(bucket_intersection(cands, members[b])) for b in 1:B_eff]
    counts = [length(members[b]) for b in 1:B_eff]

    println("dataset\t", opts.dataset)
    println("inject\t", opts.size, "x", opts.size, "\tk\t", opts.k, "\ttheta\t", opts.θ)
    println("plant_U\t", plant.U)
    println("plant_V\t", plant.V)
    println("reduced_nU\t", length(fg.u_ids), "\treduced_nV\t", length(fg.v_ids))
    println("u_total\t", length(us), "\tkept\t", length(cands), "\tdropped_empty\t", dropped)
    println("buckets_requested\t", opts.buckets, "\tbuckets_used\t", B_eff)
    println("heuristic\tgreedy-LPT-union (min max |⋃C|)")
    println()
    println("bucket\tn_starts\tunion\tsum_|C|\tall_intersect\timbalance_vs_mean_union")

    mean_u = isempty(union_sizes) ? 0.0 : sum(union_sizes) / length(union_sizes)
    for b in 1:B_eff
        imb = mean_u == 0 ? 0.0 : (union_sizes[b] - mean_u) / mean_u
        println(b, "\t", counts[b], "\t", union_sizes[b], "\t", sum_sizes[b], "\t",
                inter_sizes[b], "\t", round(imb; digits=4))
    end

    if !isempty(union_sizes)
        println()
        println("union_min\t", minimum(union_sizes),
                "\tunion_max\t", maximum(union_sizes),
                "\tunion_mean\t", round(mean_u; digits=2),
                "\tspread\t", maximum(union_sizes) - minimum(union_sizes))
        println("sum_min\t", minimum(sum_sizes),
                "\tsum_max\t", maximum(sum_sizes),
                "\tspread\t", maximum(sum_sizes) - minimum(sum_sizes))
    end

    if opts.list_members
        println()
        println("bucket\tstarts (degree-rank:u:|C|)")
        for b in 1:B_eff
            parts = String[]
            for j in sort(members[b]; by=j -> cands[j].i)
                c = cands[j]
                push!(parts, "$(c.i):$(c.u):$(length(c))")
            end
            println(b, "\t", join(parts, ","))
        end
    end
end

main()
