#!/usr/bin/env julia
# Super-temporary helper: show ant runs that produced more edges than the reported optimum
using Printf
using Pkg
using JSON3

function show_beating(path::AbstractString)
    jsonmod = getfield(Main, :JSON3)
    s = read(path, String)
    data = jsonmod.read(s)

    # determine reported optimal edges (prefer pivot.optimal_edges)
    optimal = nothing
    if hasproperty(data, :pivot) && haskey(data.pivot, "optimal_edges")
        optimal = data.pivot["optimal_edges"]
    elseif hasproperty(data, :trials) && length(data.trials) > 0 && haskey(data.trials[1], "optimal_edges")
        optimal = data.trials[1]["optimal_edges"]
    end

    if optimal === nothing
        @printf("Could not determine pivot/optimal_edges in %s\n", path)
        return
    end

    @printf("Reported optimal edges: %d\n", Int(optimal))

    found = 0
    if hasproperty(data, :trials)
        for (i, t) in enumerate(data.trials)
            if haskey(t, "final_edges")
                fe = Int(t["final_edges"])
                if fe > Int(optimal)
                    found += 1
                    @printf("\nTrial %d: run=%s ants=%s seed=%s final_edges=%d optimal_edges=%d\n", i, get(t, "run", "?"), get(t, "ants", "?"), get(t, "seed", "?"), fe, Int(optimal))
                    # print the whole trial object compactly
                    try
                        println(jsonmod.write(t; indent=2))
                    catch
                        println(t)
                    end
                end
            end
        end
    else
        @printf("No trials array found in %s\n", path)
    end

    if found == 0
        @printf("No trials with final_edges > optimal found.\n")
    else
        @printf("\nFound %d trial(s) with more edges than the reported optimum.\n", found)
    end
end

function main()
    if length(ARGS) < 1
        println("Usage: show_beating_ants.jl <path-to-json>")
        return
    end
    show_beating(ARGS[1])
end

main()
