#=
=================================================================================
Unified solve-method contract.

Every algorithm (θ-heuristic, ACO, edges-ACO, diffusion-ACO, GA, pivot/opponent, tabu, VNS, retry, …) exposes the same
surface:

  run_method!(m::SolveMethod, g, k, θ; reduction=…) -> MethodResult

`MethodResult` always carries a best `SubGraph`, optional multi-solutions, wall
time / time-to-best / iterations-to-best when the method reports them, and a
`meta` dict for method-specific fields.

To add a new algorithm:

  1. `struct MyMethod <: SolveMethod … end`
  2. `method_id(::MyMethod) = "my-method"`
  3. `function run_method!(m::MyMethod, g::BipartiteGraph, k, θ; reduction, kwargs…) … end`
  4. `register_method!("my-method", (; kwargs…) -> MyMethod(; kwargs…))`

Then it is usable from `make_method`, `bin/load.jl`, `--benchmark=`, suite
adapters, and `bin/compare-methods.jl` without further harness changes.

Paper ant-count sweeps (`bin/vary.jl` / emit) remain ACO-specific for now.
=================================================================================
=#

const __METHOD_JL__ = true

isdefined(@__MODULE__, :__GRAPH_JL__) || include("graph.jl")
isdefined(@__MODULE__, :__SEARCH_JL__) || include("search.jl")
isdefined(@__MODULE__, :__OPPONENT_JL__) || include("opponent.jl")
isdefined(@__MODULE__, :__GA_JL__) || include("ga.jl")
isdefined(@__MODULE__, :__PARALLEL_TABU_JL__) || include("parallel_tabu.jl")
isdefined(@__MODULE__, :__VNS_JL__) || include("vns.jl")
isdefined(@__MODULE__, :__RETRY_JL__) || include("retry.jl")
isdefined(@__MODULE__, :__ACO_JL__) || include(joinpath("aco", "algorithm.jl"))
isdefined(@__MODULE__, :__EDGES_ACO_JL__) || include(joinpath("edges", "algorithm.jl"))
isdefined(@__MODULE__, :__DIFFUSION_JL__) || include("diffusion.jl")
isdefined(@__MODULE__, :__TUG_JL__) || include("tug.jl")

using EnumX

# ── Canonical result ──────────────────────────────────────────────────────────

"""
Unified output of any `SolveMethod`.

- `sol` — best subgraph (problem objective: edge count, then θ-feasibility)
- `all_sols` — every returned subgraph (ACO subspecies, pivot top-N, …);
  empty when the method only produces one
- `wall_time_s` — wall time of the call when measured inside `run_method!`
  (0.0 if the caller times externally via `measure_call`)
- `time_to_best_s` / `iterations_to_best` — optional discovery metrics
- `meta` — method-specific extras (ants, generations, pheromones, …)
"""
struct MethodResult
    name::String
    sol::SubGraph
    all_sols::Vector{SubGraph}
    wall_time_s::Float64
    time_to_best_s::Union{Nothing,Float64}
    iterations_to_best::Union{Nothing,Int}
    meta::Dict{String,Any}
end

function MethodResult(name::AbstractString, sol::SubGraph;
    all_sols::Vector{SubGraph}=SubGraph[],
    wall_time_s::Float64=0.0,
    time_to_best_s::Union{Nothing,Float64}=nothing,
    iterations_to_best::Union{Nothing,Int}=nothing,
    meta::Dict{String,Any}=Dict{String,Any}())
    sols = isempty(all_sols) ? SubGraph[sol] : all_sols
    return MethodResult(String(name), sol, sols, Float64(wall_time_s),
        time_to_best_s, iterations_to_best, meta)
end

# ── Abstract method + registry ────────────────────────────────────────────────

"""
Marker for drop-in solve algorithms. Implement `method_id` and `run_method!`.
"""
abstract type SolveMethod end

method_id(m::SolveMethod) = error("method_id not implemented for $(typeof(m))")

"""
Run `m` on mutable graph `g` (may be mutated / reduced in place depending on
`reduction`). Returns a `MethodResult`.
"""
function run_method! end

const METHOD_REGISTRY = Dict{String,Function}()

"""
Register a constructor `kwargs -> SolveMethod` under a lowercase name.
Overwrites an existing entry with the same name.
"""
function register_method!(name::AbstractString, constructor::Function)
    key = lowercase(strip(String(name)))
    isempty(key) && throw(ArgumentError("method name must be non-empty"))
    METHOD_REGISTRY[key] = constructor
    return key
end

"""
Build a registered method by name. Unknown names error with the registered list.
"""
function make_method(name::AbstractString; kwargs...)
    key = lowercase(strip(String(name)))
    ctor = get(METHOD_REGISTRY, key, nothing)
    if ctor === nothing
        known = join(sort!(collect(keys(METHOD_REGISTRY))), ", ")
        throw(ArgumentError("Unknown method '$name'. Registered: $known"))
    end
    return ctor(; kwargs...)
end

list_methods() = sort!(collect(keys(METHOD_REGISTRY)))

"""
Coerce a JSON / Dict value into a constructor kwarg for `make_method`.

Handles historical `pheremone` spelling, string→enum for GA `repair` and
pivot `mode`, and Int/Float/Bool widening from JSON numbers.
"""
const _FLOAT_METHOD_PARAMS = Set{Symbol}([
    :evaporation, :k_mutate, :beta, :β, :p,
])

function coerce_method_param(method::AbstractString, key::Symbol, val)
    if key === :repair && val isa AbstractString
        sym = Symbol(lowercase(strip(String(val))))
        return getproperty(RepairMode, sym)
    elseif key === :mode && val isa AbstractString
        sym = Symbol(lowercase(strip(String(val))))
        return getproperty(BranchMode, sym)
    elseif val isa Bool
        return val
    elseif key in _FLOAT_METHOD_PARAMS && val isa Real
        return Float64(val)
    elseif val isa Integer
        return Int(val)
    elseif val isa AbstractFloat
        return Float64(val)
    elseif val isa AbstractString
        return String(val)
    elseif val === nothing
        return nothing
    else
        throw(ArgumentError(
            "Unsupported param type for $method.$key: $(typeof(val)) = $val"))
    end
end

"""
Build a registered method from a name + string-keyed params dict (JSON-friendly).

Accepts either flat params or a side-spec shaped like
`{"method":"aco","params":{…},"label":"…"}` when `name` is omitted and the
dict carries `"method"`.
"""
function make_method_from_dict(name::AbstractString, params::AbstractDict=Dict{String,Any}())
    key = lowercase(strip(String(name)))
    kwargs = Dict{Symbol,Any}()
    for (raw_k, raw_v) in params
        kstr = String(raw_k)
        # Historical load.jl / suite typo.
        kstr == "pheremone" && (kstr = "pheromone")
        kstr in ("method", "label", "params") && continue
        sym = Symbol(kstr)
        kwargs[sym] = coerce_method_param(key, sym, raw_v)
    end
    return make_method(key; kwargs...)
end

function make_method_from_dict(spec::AbstractDict)
    haskey(spec, "method") || throw(ArgumentError(
        "method spec needs \"method\"; got keys=$(collect(keys(spec)))"))
    name = String(spec["method"])
    params = if haskey(spec, "params")
        p = spec["params"]
        p isa AbstractDict || throw(ArgumentError("\"params\" must be an object"))
        Dict{String,Any}(string(k) => v for (k, v) in p)
    else
        # Flat form: all keys except method/label are constructor kwargs.
        Dict{String,Any}(string(k) => v for (k, v) in spec
            if !(string(k) in ("method", "label")))
    end
    return make_method_from_dict(name, params)
end

# ── Scoring / comparison (paper semantics) ────────────────────────────────────

"""
Score a subgraph on a frozen graph: edges, missing, θ-feasibility, k-validity.
"""
function score_solution(fg::FrozenBipartite, sol::SubGraph, k::Int, θ::Int)
    edges = Subgraph.edge_count(fg, sol)
    missing = Subgraph.missing_edges(fg, sol)
    theta_feasible = length(sol.U) ≥ θ && length(sol.V) ≥ θ
    k_valid = missing <= k
    return (;
        edges,
        missing,
        nU = length(sol.U),
        nV = length(sol.V),
        theta_feasible,
        k_valid,
        valid = k_valid && theta_feasible,
    )
end

score_result(fg::FrozenBipartite, r::MethodResult, k::Int, θ::Int) =
    score_solution(fg, r.sol, k, θ)

"""
Paper-style win: challenger beats baseline iff it is θ-feasible, k-valid, and
has strictly more edges. Ties (equal edges) are not wins.
"""
function beats(challenger::MethodResult, baseline::MethodResult,
    fg::FrozenBipartite, k::Int, θ::Int)
    c = score_result(fg, challenger, k, θ)
    b = score_result(fg, baseline, k, θ)
    c.valid || return false
    return c.edges > b.edges
end

"""
Compare two results on the same graph.

Returns `:challenger`, `:baseline`, or `:tie`. Invalid (not θ-feasible or
over-k) solutions lose to valid ones; two invalids with equal edges tie.
"""
function compare_results(baseline::MethodResult, challenger::MethodResult,
    fg::FrozenBipartite, k::Int, θ::Int)
    b = score_result(fg, baseline, k, θ)
    c = score_result(fg, challenger, k, θ)
    if c.valid && !b.valid
        return :challenger
    elseif b.valid && !c.valid
        return :baseline
    elseif c.edges > b.edges
        return :challenger
    elseif b.edges > c.edges
        return :baseline
    else
        return :tie
    end
end

"""
Pick the best subgraph by edge count (primary) then instance fitness (tiebreak).
"""
function pick_best_subgraph(fg::FrozenBipartite, sols::Vector{SubGraph}, θ::Int)
    isempty(sols) && return SubGraph()
    return argmax(sols) do s
        (Subgraph.edge_count(fg, s), instance_fitness(fg, s, θ))
    end
end

"""
Build a mutable copy of a frozen bipartite graph (suite / compare adapters).
"""
function mutable_from_frozen(g::FrozenBipartite)
    mutable_graph = BipartiteGraph{Nothing}()
    for u in g.u_ids
        add_u!(mutable_graph, u)
    end
    for v in g.v_ids
        add_v!(mutable_graph, v)
    end
    for u_idx in eachindex(g.u_ids)
        u = g.u_ids[u_idx]
        for k in neighbor_range_u(g, u_idx)
            v = g.v_ids[g.v_adj[k]]
            add_edge!(mutable_graph, u, v, nothing)
        end
    end
    return mutable_graph
end

"""
Suite adapter: `(FrozenBipartite, k, θ) -> SubGraph` for `run_graph_suite`.

Uses `ReductionMode.none` because suite graphs are already the instance to solve.
"""
function as_suite_solver(m::SolveMethod; reduction::ReductionMode.T=ReductionMode.none)
    return function (g::FrozenBipartite, k::Int, θ::Int)
        mg = mutable_from_frozen(g)
        return run_method!(m, mg, k, θ; reduction=reduction).sol
    end
end

"""
Serialize a MethodResult (plus optional score) to a JSON-friendly Dict.

Drops live objects from `meta` (pheromones, remapping, …).
"""
function method_result_to_dict(r::MethodResult, fg::Union{Nothing,FrozenBipartite}=nothing;
    k::Int=0, θ::Int=0)
    meta = Dict{String,Any}()
    for (key, val) in r.meta
        if val isa Number || val isa AbstractString || val isa Bool || val === nothing
            meta[key] = val
        elseif val isa AbstractVector && all(x -> x isa Number || x isa AbstractString || x isa Bool, val)
            meta[key] = collect(val)
        elseif val isa AbstractDict
            meta[key] = val
        end
        # Skip live objects (pheromones, remapping, SubGraph, …).
    end
    d = Dict{String,Any}(
        "method" => r.name,
        "wall_time_s" => r.wall_time_s,
        "time_to_best_s" => r.time_to_best_s,
        "iterations_to_best" => r.iterations_to_best,
        "nU" => length(r.sol.U),
        "nV" => length(r.sol.V),
        "U" => sort!(collect(r.sol.U)),
        "V" => sort!(collect(r.sol.V)),
        "meta" => meta,
    )
    if fg !== nothing
        sc = score_result(fg, r, k, θ)
        d["final_edges"] = sc.edges
        d["missing"] = sc.missing
        d["theta_feasible"] = sc.theta_feasible
        d["k_valid"] = sc.k_valid
        d["beats_valid"] = sc.valid
    end
    return d
end

# ── Built-in methods ──────────────────────────────────────────────────────────

struct HeuristicMethod <: SolveMethod
    return_invalid::Bool
    incremental::Bool
end
HeuristicMethod(; return_invalid::Bool=true, incremental::Bool=false) =
    HeuristicMethod(return_invalid, incremental)
method_id(::HeuristicMethod) = "heuristic"

function run_method!(m::HeuristicMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    t0 = time()
    fg = if reduction == ReductionMode.none
        freeze(g)
    else
        apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    end
    sol = if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        SubGraph(Set(), Set())
    else
        theta_based_heuristic(fg, k, θ;
            incremental=m.incremental, return_invalid=m.return_invalid)
    end
    wall_time = time() - t0
    return MethodResult("heuristic", sol; wall_time_s=wall_time, meta=Dict{String,Any}(
        "incremental" => m.incremental,
        "return_invalid" => m.return_invalid,
    ))
end

struct PivotMethod <: SolveMethod
    mode::BranchMode.T
    use_heuristic::Bool
    num_solutions::Int
    initial_seed::SubGraph
end
function PivotMethod(; mode::BranchMode.T=BranchMode.pivot,
    use_heuristic::Bool=true, num_solutions::Int=1,
    initial_seed::SubGraph=SubGraph())
    return PivotMethod(mode, use_heuristic, num_solutions, initial_seed)
end
method_id(::PivotMethod) = "pivot"

function run_method!(m::PivotMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    sols = find_kmdb!(g, m.use_heuristic, m.mode, k, θ, reduction;
        num_solutions=m.num_solutions, initial_seed=m.initial_seed)
    fg = freeze(g)
    sol = isempty(sols) ? SubGraph() : first(sols)
    # Prefer highest edge count if multiple returned.
    if length(sols) > 1
        sol = pick_best_subgraph(fg, sols, θ)
    end
    return MethodResult("pivot", sol;
        all_sols=isempty(sols) ? SubGraph[sol] : sols,
        meta=Dict{String,Any}(
            "branch_mode" => string(m.mode),
            "num_solutions" => m.num_solutions,
            "use_heuristic" => m.use_heuristic,
        ))
end

struct GAMethod <: SolveMethod
    N::Int
    O::Int
    k_mutate::Float64
    generations::Int
    repair::RepairMode.T
    H::Int
    use_heuristic::Bool
end
function GAMethod(; N::Int=10, O::Int=2, k_mutate::Float64=0.02,
    generations::Int=500, repair::RepairMode.T=RepairMode.mixed,
    H::Int=2, use_heuristic::Bool=true)
    return GAMethod(N, O, k_mutate, generations, repair, H, use_heuristic)
end
method_id(::GAMethod) = "ga"

function run_method!(m::GAMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    # Reset GA globals that accumulate across generations/runs.
    global U = Set{Int}()
    global V = Set{Int}()
    sol = ga(g, k, θ, m.N, m.O, m.k_mutate, m.generations;
        H=m.H, use_heuristic=m.use_heuristic, reduction=reduction, repair=m.repair)
    return MethodResult("ga", sol; meta=Dict{String,Any}(
        "N" => m.N,
        "O" => m.O,
        "k_mutate" => m.k_mutate,
        "generations" => m.generations,
        "repair" => string(m.repair),
    ))
end

struct TabuMethod <: SolveMethod
    N::Int
    tt::Int
    tabu_patience::Int
    use_heuristic::Bool
end
function TabuMethod(; N::Int=10, tt::Int=3, tabu_patience::Int=10,
    use_heuristic::Bool=true)
    return TabuMethod(N, tt, tabu_patience, use_heuristic)
end
method_id(::TabuMethod) = "tabu"

function run_method!(m::TabuMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    result = parallel_tabu(g, k, θ, m.N;
        tt=m.tt, tabu_patience=m.tabu_patience,
        use_heuristic=m.use_heuristic, reduction=reduction)
    sol = result.best_fitness
    return MethodResult("tabu", sol;
        all_sols=SubGraph[result.best_fitness, result.most_vertices],
        meta=Dict{String,Any}(
            "N" => m.N,
            "tt" => m.tt,
            "tabu_patience" => m.tabu_patience,
        ))
end

struct VNSMethod <: SolveMethod
    kmax::Int
end
VNSMethod(; kmax::Int=10) = VNSMethod(kmax)
method_id(::VNSMethod) = "vns"

function run_method!(m::VNSMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    fg = if reduction == ReductionMode.none
        freeze(g)
    else
        apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    end
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return MethodResult("vns", SubGraph(); meta=Dict{String,Any}("kmax" => m.kmax))
    end
    sol, accepted = vns_search(fg, k, θ, m.kmax)
    return MethodResult("vns", sol;
        iterations_to_best=accepted,
        meta=Dict{String,Any}(
            "kmax" => m.kmax,
            "accepted_shakes" => accepted,
        ))
end

struct RetryMethod <: SolveMethod
    beta::Float64
    p::Float64
    max_steps::Int
end
function RetryMethod(; beta::Float64=0.04, β::Union{Nothing,Float64}=nothing,
    p::Float64=1.0, max_steps::Int=10_000)
    b = β === nothing ? beta : β
    (0.0 <= b < 1.0) || throw(ArgumentError("beta must be in [0, 1), got $b"))
    p > 0.0 || throw(ArgumentError("p must be > 0, got $p"))
    max_steps >= 1 || throw(ArgumentError("max_steps must be ≥ 1, got $max_steps"))
    return RetryMethod(b, p, max_steps)
end
method_id(::RetryMethod) = "retry"

function run_method!(m::RetryMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    fg = if reduction == ReductionMode.none
        freeze(g)
    else
        apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    end
    meta = Dict{String,Any}("beta" => m.beta, "p" => m.p, "max_steps" => m.max_steps)
    if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        return MethodResult("retry", SubGraph(); meta=meta)
    end
    sol, steps_to_best = retry_search(fg, k, θ, m.beta, m.p, m.max_steps)
    return MethodResult("retry", sol;
        iterations_to_best=steps_to_best,
        meta=meta)
end

struct ACOMethod <: SolveMethod
    pheromone::Int
    ants::Int
    iterations::Int
    evaporation::Float64
    subspecies::Int
    parallelize::Bool
    force_gc::Bool
    prefer_smaller_side::Bool
    neighbor_scope_limit::Bool
    elite_seed::Bool
    elite_seed_ants::Int
    elite_seed_remove::Int
    elite_pheromone::Bool
    aco_tabu::Bool
    mmas::Bool
end
function ACOMethod(; pheromone::Int=1, ants::Int=100, iterations::Int=5,
    evaporation::Float64=0.95, subspecies::Int=1,
    parallelize::Bool=false, force_gc::Bool=false,
    prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true,
    elite_seed::Bool=true, elite_seed_ants::Int=3, elite_seed_remove::Int=2,
    elite_pheromone::Bool=false, aco_tabu::Bool=false, mmas::Bool=false)
    return ACOMethod(pheromone, ants, iterations, evaporation, subspecies,
        parallelize, force_gc, prefer_smaller_side, neighbor_scope_limit,
        elite_seed, elite_seed_ants, elite_seed_remove,
        elite_pheromone, aco_tabu, mmas)
end
method_id(::ACOMethod) = "aco"

"""
Build an `ACOMethod` from the NamedTuple / positional-tuple shape used by
`parse_aco_options` in load.jl.
"""
function aco_method_from_options(aco_options; ants::Union{Nothing,Int}=nothing,
    iterations::Union{Nothing,Int}=nothing, parallelize::Bool=false,
    force_gc::Bool=false)
    # load.jl NamedTuple uses the historical typo `pheremone`.
    pheromone = hasproperty(aco_options, :pheremone) ?
        Int(aco_options.pheremone) : Int(get(aco_options, :pheromone, 1))
    num_ants = ants === nothing ? Int(aco_options.num_ants) : ants
    num_iterations = iterations === nothing ? Int(aco_options.num_iterations) : iterations
    return ACOMethod(;
        pheromone=pheromone,
        ants=num_ants,
        iterations=num_iterations,
        evaporation=Float64(aco_options.evaporation),
        subspecies=Int(aco_options.num_subspecies),
        parallelize=parallelize,
        force_gc=force_gc,
        prefer_smaller_side=get(aco_options, :prefer_smaller_side, true),
        neighbor_scope_limit=get(aco_options, :neighbor_scope_limit, true),
        elite_seed=get(aco_options, :elite_seed, true),
        elite_seed_ants=get(aco_options, :elite_seed_ants, 3),
        elite_seed_remove=get(aco_options, :elite_seed_remove, 2),
        elite_pheromone=get(aco_options, :elite_pheromone, false),
        aco_tabu=get(aco_options, :aco_tabu, false),
        mmas=get(aco_options, :mmas, false),
    )
end

function run_method!(m::ACOMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    iteration_callback=nothing, construction_stats=nothing, kwargs...)
    sols, best_iterations, best_times, pheromones, remapping = aco(
        g, m.pheromone, m.ants, m.iterations, m.evaporation, k, θ, m.subspecies;
        parallelize=m.parallelize,
        force_gc=m.force_gc,
        prefer_smaller_side=m.prefer_smaller_side,
        neighbor_scope_limit=m.neighbor_scope_limit,
        elite_seed=m.elite_seed,
        elite_seed_ants=m.elite_seed_ants,
        elite_seed_remove=m.elite_seed_remove,
        elite_pheromone=m.elite_pheromone,
        aco_tabu=m.aco_tabu,
        mmas=m.mmas,
        reduction=reduction,
        iteration_callback=iteration_callback,
        construction_stats=construction_stats)
    fg = freeze(g)
    if isempty(sols)
        return MethodResult("aco", SubGraph();
            meta=Dict{String,Any}(
                "ants" => m.ants,
                "iterations" => m.iterations,
                "pheromone" => m.pheromone,
                "evaporation" => m.evaporation,
                "subspecies" => m.subspecies,
            ))
    end
    sol = pick_best_subgraph(fg, sols, θ)
    best_idx = findfirst(s -> s === sol || (s.U == sol.U && s.V == sol.V), sols)
    best_idx = best_idx === nothing ? argmax(i -> Subgraph.edge_count(fg, sols[i]), eachindex(sols)) : best_idx
    itb = isempty(best_iterations) ? nothing : best_iterations[best_idx]
    ttb = isempty(best_times) ? nothing : best_times[best_idx]
    return MethodResult("aco", sol;
        all_sols=sols,
        time_to_best_s=ttb,
        iterations_to_best=itb,
        meta=Dict{String,Any}(
            "ants" => m.ants,
            "iterations" => m.iterations,
            "pheromone" => m.pheromone,
            "evaporation" => m.evaporation,
            "subspecies" => m.subspecies,
            "best_subspecies" => best_idx,
            "prefer_smaller_side" => m.prefer_smaller_side,
            "neighbor_scope_limit" => m.neighbor_scope_limit,
            # Keep raw ACO extras for specialized callers (aco-reduce, etc.).
            "pheromones" => pheromones,
            "remapping" => remapping,
            "best_iterations" => best_iterations,
            "best_times" => best_times,
        ))
end

# ── Edges-ACO (pheromone on CSR edges) ────────────────────────────────────────

struct EdgesMethod <: SolveMethod
    pheromone::Int
    ants::Int
    iterations::Int
    evaporation::Float64
    subspecies::Int
    parallelize::Bool
    force_gc::Bool
    prefer_smaller_side::Bool
    neighbor_scope_limit::Bool
    elite_seed::Bool
    elite_seed_ants::Int
    elite_seed_remove::Int
    elite_pheromone::Bool
    aco_tabu::Bool
    mmas::Bool
end
function EdgesMethod(; pheromone::Int=1, ants::Int=100, iterations::Int=5,
    evaporation::Float64=0.95, subspecies::Int=1,
    parallelize::Bool=false, force_gc::Bool=false,
    prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true,
    elite_seed::Bool=true, elite_seed_ants::Int=3, elite_seed_remove::Int=2,
    elite_pheromone::Bool=false, aco_tabu::Bool=false, mmas::Bool=false)
    return EdgesMethod(pheromone, ants, iterations, evaporation, subspecies,
        parallelize, force_gc, prefer_smaller_side, neighbor_scope_limit,
        elite_seed, elite_seed_ants, elite_seed_remove,
        elite_pheromone, aco_tabu, mmas)
end
method_id(::EdgesMethod) = "edges"

"""Build an `EdgesMethod` from the same NamedTuple shape as `aco_method_from_options`."""
function edges_method_from_options(aco_options; ants::Union{Nothing,Int}=nothing,
    iterations::Union{Nothing,Int}=nothing, parallelize::Bool=false,
    force_gc::Bool=false)
    pheromone = hasproperty(aco_options, :pheremone) ?
        Int(aco_options.pheremone) : Int(get(aco_options, :pheromone, 1))
    num_ants = ants === nothing ? Int(aco_options.num_ants) : ants
    num_iterations = iterations === nothing ? Int(aco_options.num_iterations) : iterations
    return EdgesMethod(;
        pheromone=pheromone,
        ants=num_ants,
        iterations=num_iterations,
        evaporation=Float64(aco_options.evaporation),
        subspecies=Int(aco_options.num_subspecies),
        parallelize=parallelize,
        force_gc=force_gc,
        prefer_smaller_side=get(aco_options, :prefer_smaller_side, true),
        neighbor_scope_limit=get(aco_options, :neighbor_scope_limit, true),
        elite_seed=get(aco_options, :elite_seed, true),
        elite_seed_ants=get(aco_options, :elite_seed_ants, 3),
        elite_seed_remove=get(aco_options, :elite_seed_remove, 2),
        elite_pheromone=get(aco_options, :elite_pheromone, false),
        aco_tabu=get(aco_options, :aco_tabu, false),
        mmas=get(aco_options, :mmas, false),
    )
end

function run_method!(m::EdgesMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    iteration_callback=nothing, construction_stats=nothing, kwargs...)
    sols, best_iterations, best_times, pheromones, remapping = edges_aco(
        g, m.pheromone, m.ants, m.iterations, m.evaporation, k, θ, m.subspecies;
        parallelize=m.parallelize,
        force_gc=m.force_gc,
        prefer_smaller_side=m.prefer_smaller_side,
        neighbor_scope_limit=m.neighbor_scope_limit,
        elite_seed=m.elite_seed,
        elite_seed_ants=m.elite_seed_ants,
        elite_seed_remove=m.elite_seed_remove,
        elite_pheromone=m.elite_pheromone,
        aco_tabu=m.aco_tabu,
        mmas=m.mmas,
        reduction=reduction,
        iteration_callback=iteration_callback,
        construction_stats=construction_stats)
    fg = freeze(g)
    if isempty(sols)
        return MethodResult("edges", SubGraph();
            meta=Dict{String,Any}(
                "ants" => m.ants,
                "iterations" => m.iterations,
                "pheromone" => m.pheromone,
                "evaporation" => m.evaporation,
                "subspecies" => m.subspecies,
            ))
    end
    sol = pick_best_subgraph(fg, sols, θ)
    best_idx = findfirst(s -> s === sol || (s.U == sol.U && s.V == sol.V), sols)
    best_idx = best_idx === nothing ? argmax(i -> Subgraph.edge_count(fg, sols[i]), eachindex(sols)) : best_idx
    itb = isempty(best_iterations) ? nothing : best_iterations[best_idx]
    ttb = isempty(best_times) ? nothing : best_times[best_idx]
    return MethodResult("edges", sol;
        all_sols=sols,
        time_to_best_s=ttb,
        iterations_to_best=itb,
        meta=Dict{String,Any}(
            "ants" => m.ants,
            "iterations" => m.iterations,
            "pheromone" => m.pheromone,
            "evaporation" => m.evaporation,
            "subspecies" => m.subspecies,
            "best_subspecies" => best_idx,
            "prefer_smaller_side" => m.prefer_smaller_side,
            "neighbor_scope_limit" => m.neighbor_scope_limit,
            "pheromones" => pheromones,
            "remapping" => remapping,
            "best_iterations" => best_iterations,
            "best_times" => best_times,
        ))
end

# ── Diffusion ACO (vertex pheromone + diffusion cohesion reward) ──────────────

struct DiffusionMethod <: SolveMethod
    pheromone::Int
    ants::Int
    iterations::Int
    evaporation::Float64
    subspecies::Int
    parallelize::Bool
    force_gc::Bool
    prefer_smaller_side::Bool
    neighbor_scope_limit::Bool
    elite_seed::Bool
    elite_seed_ants::Int
    elite_seed_remove::Int
    elite_pheromone::Bool
    aco_tabu::Bool
    mmas::Bool
    diffusion_iters::Int
end
function DiffusionMethod(; pheromone::Int=1, ants::Int=100, iterations::Int=5,
    evaporation::Float64=0.95, subspecies::Int=1,
    parallelize::Bool=false, force_gc::Bool=false,
    prefer_smaller_side::Bool=true, neighbor_scope_limit::Bool=true,
    elite_seed::Bool=true, elite_seed_ants::Int=3, elite_seed_remove::Int=2,
    elite_pheromone::Bool=false, aco_tabu::Bool=false, mmas::Bool=false,
    diffusion_iters::Int=20)
    diffusion_iters >= 0 || throw(ArgumentError(
        "diffusion_iters must be ≥ 0, got $diffusion_iters"))
    return DiffusionMethod(pheromone, ants, iterations, evaporation, subspecies,
        parallelize, force_gc, prefer_smaller_side, neighbor_scope_limit,
        elite_seed, elite_seed_ants, elite_seed_remove,
        elite_pheromone, aco_tabu, mmas, diffusion_iters)
end
method_id(::DiffusionMethod) = "diffusion"

"""Build a `DiffusionMethod` from the same NamedTuple shape as `aco_method_from_options`."""
function diffusion_method_from_options(aco_options; ants::Union{Nothing,Int}=nothing,
    iterations::Union{Nothing,Int}=nothing, parallelize::Bool=false,
    force_gc::Bool=false, diffusion_iters::Int=20)
    pheromone = hasproperty(aco_options, :pheremone) ?
        Int(aco_options.pheremone) : Int(get(aco_options, :pheromone, 1))
    num_ants = ants === nothing ? Int(aco_options.num_ants) : ants
    num_iterations = iterations === nothing ? Int(aco_options.num_iterations) : iterations
    return DiffusionMethod(;
        pheromone=pheromone,
        ants=num_ants,
        iterations=num_iterations,
        evaporation=Float64(aco_options.evaporation),
        subspecies=Int(aco_options.num_subspecies),
        parallelize=parallelize,
        force_gc=force_gc,
        prefer_smaller_side=get(aco_options, :prefer_smaller_side, true),
        neighbor_scope_limit=get(aco_options, :neighbor_scope_limit, true),
        elite_seed=get(aco_options, :elite_seed, true),
        elite_seed_ants=get(aco_options, :elite_seed_ants, 3),
        elite_seed_remove=get(aco_options, :elite_seed_remove, 2),
        elite_pheromone=get(aco_options, :elite_pheromone, false),
        aco_tabu=get(aco_options, :aco_tabu, false),
        mmas=get(aco_options, :mmas, false),
        diffusion_iters=diffusion_iters,
    )
end

function run_method!(m::DiffusionMethod, g::BipartiteGraph, k::Int, θ::Int;
    reduction::ReductionMode.T=ReductionMode.all_reductions,
    iteration_callback=nothing, construction_stats=nothing, kwargs...)
    sols, best_iterations, best_times, pheromones, remapping = diffusion_aco(
        g, m.pheromone, m.ants, m.iterations, m.evaporation, k, θ, m.subspecies;
        diffusion_iters=m.diffusion_iters,
        parallelize=m.parallelize,
        force_gc=m.force_gc,
        prefer_smaller_side=m.prefer_smaller_side,
        neighbor_scope_limit=m.neighbor_scope_limit,
        elite_seed=m.elite_seed,
        elite_seed_ants=m.elite_seed_ants,
        elite_seed_remove=m.elite_seed_remove,
        elite_pheromone=m.elite_pheromone,
        aco_tabu=m.aco_tabu,
        mmas=m.mmas,
        reduction=reduction,
        iteration_callback=iteration_callback,
        construction_stats=construction_stats,
        kwargs...)
    fg = freeze(g)
    meta = Dict{String,Any}(
        "ants" => m.ants,
        "iterations" => m.iterations,
        "pheromone" => m.pheromone,
        "evaporation" => m.evaporation,
        "subspecies" => m.subspecies,
        "diffusion_iters" => m.diffusion_iters,
    )
    if isempty(sols)
        return MethodResult("diffusion", SubGraph(); meta=meta)
    end
    sol = pick_best_subgraph(fg, sols, θ)
    best_idx = findfirst(s -> s === sol || (s.U == sol.U && s.V == sol.V), sols)
    best_idx = best_idx === nothing ? argmax(i -> Subgraph.edge_count(fg, sols[i]), eachindex(sols)) : best_idx
    itb = isempty(best_iterations) ? nothing : best_iterations[best_idx]
    ttb = isempty(best_times) ? nothing : best_times[best_idx]
    meta["best_subspecies"] = best_idx
    meta["prefer_smaller_side"] = m.prefer_smaller_side
    meta["neighbor_scope_limit"] = m.neighbor_scope_limit
    meta["pheromones"] = pheromones
    meta["remapping"] = remapping
    meta["best_iterations"] = best_iterations
    meta["best_times"] = best_times
    inj_b = get(kwargs, :injected_biclique, nothing)
    meta["heat_stats"] = get_diffusion_heat_stats(fg;
        iterations=m.diffusion_iters,
        injected_biclique=inj_b,
        solution_biclique=sol,
        remapping=remapping)
    return MethodResult("diffusion", sol;
        all_sols=sols,
        time_to_best_s=ttb,
        iterations_to_best=itb,
        meta=meta)
end

# ── Tug-of-War (Alternating Best-Response) ──────────────────────────────────

struct TugOfWarMethod <: SolveMethod
    seed_source::Symbol
    num_seeds::Int
    max_shakes::Int
    seed::Union{Nothing,Int}
    peel::Bool
end

function TugOfWarMethod(; seed_source::Union{Symbol,AbstractString}=:sc,
                        num_seeds::Int=10,
                        max_shakes::Int=3,
                        seed::Union{Nothing,Int}=nothing,
                        peel::Bool=true)
    src_sym = Symbol(lowercase(String(seed_source)))
    return TugOfWarMethod(src_sym, num_seeds, max_shakes, seed, peel)
end

method_id(::TugOfWarMethod) = "tug"

function run_method!(m::TugOfWarMethod, g::BipartiteGraph, k::Int, θ::Int;
                     reduction::ReductionMode.T=ReductionMode.all_reductions, kwargs...)
    t0 = time()

    if m.peel
        peel_degrees!(g, θ - k)
        if length(g.adjU) < θ || length(g.adjV) < θ
            return MethodResult("tug", SubGraph(Set(), Set()); wall_time_s=time() - t0)
        end
    end

    fg = if reduction == ReductionMode.none
        freeze(g)
    else
        apply_graph_reductions!(g, k, θ, nothing, nothing, true, reduction)
    end

    sol = if length(fg.u_ids) < θ || length(fg.v_ids) < θ
        SubGraph(Set(), Set())
    else
        tug_of_war_solve(fg, k, θ;
                         seed_source=m.seed_source,
                         num_seeds=m.num_seeds,
                         max_shakes=m.max_shakes,
                         seed=m.seed)
    end

    wall_time = time() - t0

    return MethodResult("tug", sol;
        wall_time_s=wall_time,
        meta=Dict{String,Any}(
            "seed_source" => string(m.seed_source),
            "num_seeds" => m.num_seeds,
            "max_shakes" => m.max_shakes,
            "peel" => m.peel,
        )
    )
end

# Aliases: "opponent" / "branch" map to pivot (exact search).
function _register_builtins!()
    register_method!("heuristic", (; kwargs...) -> HeuristicMethod(; kwargs...))
    register_method!("pivot", (; kwargs...) -> PivotMethod(; kwargs...))
    register_method!("opponent", (; kwargs...) -> PivotMethod(; kwargs...))
    register_method!("branch", (; kwargs...) -> PivotMethod(; kwargs...))
    register_method!("ga", (; kwargs...) -> GAMethod(; kwargs...))
    register_method!("tabu", (; kwargs...) -> TabuMethod(; kwargs...))
    register_method!("vns", (; kwargs...) -> VNSMethod(; kwargs...))
    register_method!("retry", (; kwargs...) -> RetryMethod(; kwargs...))
    register_method!("aco", (; kwargs...) -> ACOMethod(; kwargs...))
    register_method!("edges", (; kwargs...) -> EdgesMethod(; kwargs...))
    register_method!("diffusion", (; kwargs...) -> DiffusionMethod(; kwargs...))
    register_method!("tug", (; kwargs...) -> TugOfWarMethod(; kwargs...))
    register_method!("tow", (; kwargs...) -> TugOfWarMethod(; kwargs...))
    register_method!("alternation", (; kwargs...) -> TugOfWarMethod(; kwargs...))
end

_register_builtins!()
