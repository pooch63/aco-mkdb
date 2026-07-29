const __UTILS_JL__ = true

using Base.Threads

function parallel_sort_by_keys(v::Vector{Int}, f, desc::Bool)
    keys = Vector{Int}(undef, length(v))

    @threads for i in eachindex(v)
        @inbounds keys[i] = f(v[i])
    end

    return v[sortperm(keys; rev = desc)]
end