# ACO-MKDB — Agent Guide

This repository implements and evaluates heuristics for the **maximum k-defective edge biclique (k-MDB)** problem on bipartite graphs. Read this before making changes or running experiments.

## Goal

Given a bipartite graph \(G = (U, V, E)\), a defect budget \(k\), and a minimum side size \(\theta\):

- A **k-defective biclique** is a subgraph \(S = (U_S, V_S)\) with at most \(k\) missing cross edges: \(|\bar{E}(S)| \leq k\).
- The **k-MDB** is the largest such subgraph by **edge count** \(|E(S)|\).
- This project seeks **heuristic solutions** that are **θ-feasible**: \(|U_S| \geq \theta\) **and** \(|V_S| \geq \theta\).

Without the \(\theta\) constraint, real graphs often admit dense subgraphs with lopsided sides (many vertices on one side, few on the other), which are not useful. \(\theta\) restricts the search to balanced, interpretable bicliques.

**Paper defaults:** \(k = 2\), \(\theta = 5\). Many result directories are named `vary_k2t5i_*` to encode these values.

The primary baseline is the **θ-heuristic** from Cui et al. [1]. The main contribution is an **Ant Colony Optimization (ACO)** heuristic that often finds larger k-MDBs and can seed exact branch-and-pivot search.

## ACO approach

We adapt **MAX-MIN Ant System (MMAS)**-style ACO to grow k-defective bicliques incrementally.

**High-level loop** (`src/aco/algorithm.jl`, `src/aco/advance.jl`):

1. Apply **common-neighbor graph reduction** (CNN / `ReductionMode.simple`) before search.
2. Each **epoch**, a colony of **ants** independently builds a subgraph by repeatedly adding vertices.
3. At each step, an ant may only add vertices that keep the defect budget: \(|\bar{E}(S \cup \{u\})| \leq k\).
4. The next vertex is sampled proportional to **desirability** = pheromone \(\times\) \(\eta^3\), where \(\eta\) combines subgraph degree, global degree (sigmoid-damped), and optional biases.
5. After all ants finish, the best subgraph by **instance fitness** updates the global best; pheromone evaporates and deposits along the best ant's path.

**Key design choices for θ-feasibility and quality:**

| Flag | Name | Effect |
|------|------|--------|
| **\(F_P\)** | prefer-smaller-side | When one side has \(\geq \theta\) vertices and the other does not, boost desirability of vertices on the smaller side (factor 2). |
| **\(F_N\)** | neighbor-scope limit | Prefer candidates in the intersection of the full candidate set and the last-added vertex's neighbors, narrowing search. |

**\(F_{PN}\)** = both \(F_P\) and \(F_N\) enabled — this is the **default** for main paper experiments.

**Edge-trail variant** (`src/edges/`, method id `edges`): same colony loop as node ACO, but pheromone is stored on CSR graph edges. Candidate desirability uses the mean edge-τ into the opposite side of \(S\); deposit reinforces those cross edges (elite deposit reinforces the induced edge set).

**Diffusion-guided variant** (`src/diffusion.jl`, method id `diffusion`): vertex-trail ACO with a precomputed diffusion field. Vertices start with random values in \([-0.5, 0.5]\) and iteratively average with their neighbors (`diffusion_iters`, default 20). Instance fitness and pheromone deposit are scaled by cohesion \(1/(1+\mathrm{std})\) of the subgraph's diffused values, so topologically similar vertices are rewarded.

**Instance fitness** rewards balanced growth once \(\theta\) is met on one side: if \(\min(|U_S|, |V_S|) \geq \theta\), fitness is \(a \times b^2\) where \(a, b = \min\max(|U_S|, |V_S|)\); otherwise \(a^2\).

**Paper hyperparameters** (see `README.md` for env-var mapping):

- 5 epochs (\(T = 5\)), evaporation \(\rho = 0.05\), deposit \(\Delta\tau = 1\)
- Ant-count sweep: \(\{1, 2, 5, 10, 20, 50, 100\}\)
- 6 stochastic replicates per ant count (`ACO_RUNS=6`, `SEED=1`); **run 1 is a
  Julia JIT warmup** on the real graph (`jit_warmup=true` in JSON) and is
  discarded by `emit/` / `best_trial` / discovery cost, leaving 5 counted runs
- θ-heuristic is solved twice per graph (first discarded for JIT; second timed)
- Optional planted biclique injection (\(u = v = 5\), \(k\) missing edges) for validation

**Entry points:** `bin/load.jl` (single-graph runs), `bin/vary.jl` (ant-count sweeps and `--vary=param` sweeps of ρ, Δτ, and the η exponent), `scripts/vary.bash` (batch over all datasets), `scripts/param-ablation.bash` (the three hyperparameter sweeps at A=100).

## θ-heuristic (baseline)

The **θ-heuristic** (`src/theta_heuristic.jl`) is the greedy construct-and-trim baseline from Cui et al.:

1. Start with empty \(U\) and all of \(V\).
2. Repeatedly add the \(u \in U_G \setminus U\) with largest degree into the current \(V\).
3. While missing edges exceed \(k\), remove the \(v \in V\) with largest nondegree (most missing edges to \(U\)).
4. Stop when \(|U| = \theta\); return \(U \cup V\).

Complexity is \(\mathcal{O}(\theta n + m)\). It is fast but often returns smaller bicliques than ACO. Cui et al. also use it to seed **branch-and-pivot** exact search (`src/search.jl`); we compare pivot wall time when seeded by θ alone vs. θ + best ACO subgraph (`bin/compare-seeds.jl`).

**θ-feasible** means \(|U_S| \geq \theta\) and \(|V_S| \geq \theta\). ACO trials are only counted as beating the heuristic when they have more edges **and** are θ-feasible.

## SolveMethod contract (`src/method.jl`)

All solution algorithms share one interface so they are drop-in replacements for
interactive runs, benchmarks, and A/B tests. Paper ant-count sweeps
(`bin/vary.jl` / emit) remain ACO-specific.

**Core types**

- `SolveMethod` — abstract; each algorithm is a concrete struct holding its knobs
- `MethodResult` — best `SubGraph`, optional `all_sols`, wall / time-to-best /
  iterations-to-best, plus a `meta` dict
- `run_method!(m, g, k, θ; reduction=…)` — unified entry point
- `register_method!("name", kwargs -> Method(…))` / `make_method("name"; …)` /
  `list_methods()` — string registry (aliases: `opponent` / `branch` → pivot)

**Built-ins:** `heuristic`, `aco`, `edges`, `diffusion`, `ga`, `tabu`, `vns`, `retry`, `pivot` (exact / opponent).

**Adding a new algorithm**

1. Implement `struct MyMethod <: SolveMethod`, `method_id`, and `run_method!`
   returning a `MethodResult` (put the code in `src/` and `include` it from
   `method.jl`, or register after include).
2. Call `register_method!("my-method", (; kwargs…) -> MyMethod(; kwargs…))`.
3. Use it anywhere methods are accepted — no further harness changes required.

**How to test whether a new version outperforms an old one**

```bash
# Real graph, paper win rule (θ-feasible + more edges):
julia bin/compare-methods.jl amazon/boxes --baseline=heuristic --challenger=aco --save=ab.json

# VNS vs ACO:
julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=vns --challenger-kmax=10 --save=vns_ab.json

# Edge-trail ACO vs vertex ACO:
julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=edges --save=edges_vs_aco.json

# Diffusion ACO vs vertex ACO:
julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=diffusion --save=diffusion_vs_aco.json

# Retry vs ACO:
julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=retry \
  --challenger-beta=0.04 --challenger-p=1 --challenger-max-steps=10000 --save=retry_ab.json

# Same method, different knobs (CLI ant overrides):
julia bin/compare-methods.jl amazon/boxes --baseline=aco --challenger=aco \
  --baseline-ants=10 --challenger-ants=50 --save=ants_ab.json

# JSON config: arbitrary params on both sides, graphs ascending by |E|,
# checkpoint after every graph (resume with skip_completed):
julia bin/compare-methods.jl --config=configs/aco_pheromone_ablation.json
julia bin/compare-methods.jl --config=configs/aco_ants_ab.json

# Prefix suite without a config file (still needs --save= for checkpoints).
# Omit --prefix= / dataset to run every indexed graph under data/:
julia bin/compare-methods.jl --prefix=konect-small --baseline=aco --challenger=vns \
 --save=vns_vs_aco.json
julia bin/compare-methods.jl --baseline=aco --challenger=vns --save=vns_vs_aco.json

# Synthetic suite + offline JSON compare (shared --seed / --N):
julia tests/test_ga.jl --seed=1 --N=5 --save=ga.json
julia tests/test_theta_heuristic.jl --seed=1 --N=5 --save=heuristic.json
julia tests/compare.jl heuristic.json ga.json
```

`beats(challenger, baseline, fg, k, θ)` encodes the paper win rule.
`compare_results` returns `:challenger` / `:baseline` / `:tie`.
Suite adapters: `as_suite_solver(make_method("aco"; …))` → `(g,k,θ)->SubGraph`.
`make_method_from_dict("aco", params)` / side-specs build methods from JSON.

**CLI**

- `bin/load.jl --ga|--aco|--edges|--diffusion|--heuristic|--tabu|--vns|--retry` → `solver_to_method` → `MethodResult`
  (`solve_method!`); legacy `solve!` still returns `SubGraph` / `Vector{SubGraph}`
- `--benchmark=aco,pivot,heuristic,ga,tabu,vns,retry,edges,diffusion` (any registered name)
- `bin/compare-methods.jl` — head-to-head; `--config=` for multi-graph A/B with
  full hyperparameter objects (see `configs/`)

## Paper build (`paper/`)

Prose lives in `paper/main.tex`. Figures and tables are **generated** from experiment JSON — never hand-edited in the final PDF path.

### Workflow

```
results/*.json  →  emit/  →  paper/generated/*.tex  →  build.py  →  build.pdf
```

1. **Experiments** write JSON under `results/` (see below).
2. **`python -m emit`** (or `make paper-emit`) reads JSON and writes LaTeX fragments to `paper/generated/`.
   Fragments rebuild when results JSON **or** the emit Python that produces them is newer (see `paper/build.py`).
3. **`paper/build.py`** substitutes `%%PLACEHOLDER%%` tokens in `main.tex` with generated fragments → `build.tex` → compiles PDF. `--tables` does the same for `tables.tex` → `tables.pdf`.

**Commands:**

```bash
make paper              # full rebuild from repo root
cd paper && make        # same, from paper/
python paper/build.py emit      # only regenerate fragments
python paper/build.py assemble  # only substitute placeholders
python paper/build.py pdf       # only compile
python paper/build.py --tables  # also compile paper/tables.tex → tables.pdf
```

**Configuration:** `paper/build.json` maps placeholders to emit modes, input directories, and options. Set `results_dir` once (default `../results`); fragment paths are bare names under that tree (e.g. `vary_k2t5i_PN`). Absolute / `../…` paths still resolve from `paper/`. Optional top-level `max_replicates` (or per-fragment override / `--max-replicates=R`) retrospectively prefixes counted non-JIT replicates per ant count for emit aggregates; omit/`null` keeps all. The `budget` and `pheromone-ablation` figures always use the full counted set.

| Placeholder | Emit mode | Purpose |
|-------------|-----------|---------|
| `%%QUALITY%%` | `quality` | One column: ant count, evaporation, deposit, and η-exponent, then epoch budget \(T\) and replicate count \(R\) (deviation on the shared scale, θ-feasibility on the bottom row). Median % deviation from the θ-heuristic (`ablation_dirs`, `budget_dir`). `fig:budget` is the same float |
| `%%SEED_COMPARE%%` | `seed-compare` | Full pivot table (appendix); `:highlights` = representative rows in §4.2 |
| `%%TABLE:k2t5i_PN%%` | `table` | Per-graph ACO vs. θ comparison (appendix); `:k2t5i_PN:highlights` = §4.3; `:k3t5i_PN` / `:k3t6i_PN` / `:k4t5i_PN` = other $(k,\theta)$ |
| `%%COMPARE:theta-time%%` | `compare` | θ-heuristic runtime vs. \(\theta n + m\) bound |
| `%%COMPARE:deg-size-time%%` | `compare` | ACO discovery time on log–log axes: vs.\ $n_R$ (empirical exponent) and vs.\ candidate bounds $n_R^2$ / ($T\cdot n_R+|E_R|$) with $T=5$ (slope≈1 ⇒ proportional) |
| `%%COMPARE:bound-time%%` | `compare` | 2-panel: normalized discovery time \(t/(n_S\cdot T)\) vs.\ naive $n_R^2$ (left) and practical $T\cdot n_R+|E_R|$ (right); shared y-axis for slope contrast |
| `%%COMPARE:k-sweep%%` | `compare` | Compact captionless pair: ACO win rate vs. \(k\) at fixed \(\theta\) and vs. \(\theta\) at fixed \(k\), in the text (`param_dirs`) |
| `%%COMPARE:theta-sweep%%` | `compare` | Comment only; the \(\theta\) panel is part of `k-sweep` |
| `%%COMPARE:density-wins%%` | `compare` | One column: ACO win rate versus \(k\) at fixed \(\theta\), and versus \(\theta\) at fixed \(k\), with equal-count reduced-density thirds as three bars |
| `%%COMPARE:param-runtime%%` | `compare` | Four instances, log time. A legend under each 2×2 names the sweep color (fixed \(\theta\) when \(k\) varies, fixed \(k\) when \(\theta\) varies) and the marker (circle ACO, triangle \(\theta\)-heuristic). Sweeps are `runtime_sweeps` in `build.json`: every completed value at that fixed parameter. Each panel's axis is the range that graph finished |
| `%%COMPARE:budget%%` | `compare` | Standalone epoch-trace 2×2. The paper draws these panels inside `%%QUALITY%%` instead |
| `%%COMPARE:pheromone-ablation%%` | `compare` | Anytime median with pheromone on vs \(\Delta\tau=0\) (\(\tau\equiv 1\)) and optional \(\rho\) sweep (`ablation_dirs`) |
| `%%STATISTICS:…%%` | `statistics` | Inline win/loss counts, Wilcoxon, median % edge increase (`median-edge-pct`), θ-feasibility rates, missing-at-size stats, pivot-tested vs excluded ACO-win size means (`compare_dir`) |
| `%%ABLATION%%` / `%%ABLATION:summary%%` | `ablation` | Settings table rows and the Section 4.3 sentence from `aco_settings_ablation.json` (pheromone off, best-only deposit, greedy fill at the end, GA) |

**Important:** `emit/` must **only read pre-recorded JSON** — it must not re-run Julia or re-simulate ACO. The build must tolerate incomplete data (warn, don't crash). See `paper/AGENTS.md` for emit-specific rules.

**Artifacts:** `paper/build.tex` and `paper/build.pdf` are generated (gitignored).

## Results collected

All experiment output is **JSON** (plus occasional `.txt` from `scripts/evaluate.bash`). Key directories:

| Directory pattern | Produced by | Contents |
|-------------------|-------------|----------|
| `results/vary_k2t5i_PN/` | `scripts/vary.bash` | Ant-count sweep per graph: `<graph>_ants.json` |
| `results/vary_k2t5i_<flags>/` | `scripts/vary.bash` / `scripts/tests.bash` | Same sweep for other \(F_P\)/\(F_N\) flag combinations |
| `results/compare_k2t5i_<subset>/` | `scripts/compare-seeds.bash` | Pivot timing per graph: `<graph>.json` |
| `results/k2t5i/` | `scripts/evaluate.bash` | Quick per-graph ACO vs. θ benchmark (`.txt`) |

Directory names encode hyperparameters: `vary_k{K}t{THETA}i_{P?}{N?}` (e.g. `vary_k2t5i_PN` = k=2, θ=5, injected plant, P+N flags).

### `vary.jl` JSON (`*_ants.json`)

One file per graph. Top-level metadata plus `trials[]` (one entry per ant-count × replicate) and a `heuristic` block.

**Per-trial fields agents care about:**

- `ants`, `final_edges`, `theta_feasible`, `beats_heuristic`
- `run`, `jit_warmup` — when `aco_runs > 1`, run 1 is JIT warmup (`jit_warmup=true`); emit ignores it
- `wall_time_s`, `time_to_best_s`, `iterations_to_best` — ETB is the epoch
  *within this replicate* when `final_edges` first appeared (`run` is the
  replicate index; each trial always ran the full `iterations_budget`)
- `U`, `V` — returned subgraph vertex ids (used to re-seed pivot)
- `construction.missing_at_size` — missing-edge count when subgraph first reaches each size
- `epoch_trace` — incumbent after each completed epoch: `t`, `edges`, `theta_feasible`, `elapsed_s`, `nU`, `nV`. Written by every `vary.jl` trial. `scripts/budget-trace.bash` runs the paper suite (\(T=100\), \(\ge 30\) counted replicates, optional \(\Delta\tau=0\) and \(\rho\) sweep) into `results/budget_k2t5i_PN`

**Timeout fields** (optional; set when `--aco-timeout=` / `ACO_TIMEOUT` is used):

- `aco_timeout_s` — shared wall-clock budget for all ACO trials on this graph
- `aco_timed_out` — `true` if the budget expired before the full ant×run sweep finished
- `aco_status` — `"ok"` | `"timeout"` | `"running"` (checkpoint after θ, before ACO finishes)

θ-heuristic always runs first. On timeout, JSON still has a full `heuristic` block;
emit skips timed-out files for ACO win/rate plots and clears ACO table columns.
`SKIP_EXISTING=1` re-runs a timed-out file only when the new `ACO_TIMEOUT` is
strictly larger than the previous `aco_timeout_s` (same upgrade pattern as pivot
`TIMEOUT` in compare-seeds).

**Graph metadata:** `nU`, `nV`, `reduced_nU`, `reduced_nV`, `reduced_edges`, `reduced_max_degree`

**Heuristic block:** `final_edges`, `theta_feasible`, `wall_time_s`, timing breakdowns (ITB, TTB, discovery)

### `compare-seeds.jl` JSON

Per graph where ACO beat θ (or a compact skip marker if not):

- `pivot_theta`, `pivot_aco_seed` — wall time, edges found, timeout flags
- `time_reduction_s`, `time_reduction_pct` — speedup from ACO seeding
- `heuristic_edges`, `aco_seed` metadata linking back to the winning vary trial

### What the paper measures

1. **Solution quality** — edge count of ACO vs. θ-heuristic across ant counts and ~35 benchmark graphs (KONECT, Amazon, Wikipedia).
2. **θ-feasibility** — fraction of ants whose final subgraph has \(\geq \theta\) vertices on each side.
3. **Runtime scaling** — ACO and θ-heuristic wall time vs. theoretical complexity bounds.
4. **Pivot seeding** — whether an ACO-found subgraph reduces branch-and-pivot search time vs. θ alone.

## Repository map (quick reference)

| Path | Role |
|------|------|
| `src/` | Graph types, reduction, ACO, edge-ACO, θ-heuristic, branch-and-pivot |
| `src/aco/` | Vertex-trail MAX-MIN ACO |
| `src/edges/` | Edge-trail ACO variant (pheromone on CSR edges) |
| `src/diffusion.jl` | Diffusion-guided vertex ACO (cohesion reward) |
| `src/method.jl` | Unified `SolveMethod` / `MethodResult` registry + adapters |
| `bin/` | CLI: `load.jl`, `vary.jl`, `compare-seeds.jl`, `compare-methods.jl` |
| `emit/` | JSON → LaTeX (Python, stdlib only) |
| `scripts/` | Batch experiment wrappers (`vary.bash`, `regenerate-paper-data.bash`, …) |
| `data/datasets.txt` | Graph manifest |
| `results/` | Experiment JSON output |
| `paper/` | LaTeX source, build config, generated fragments |
| `tests/` | Julia unit / integration tests |

For full replication steps (dataset download, env vars, hardware notes), see `README.md`.
