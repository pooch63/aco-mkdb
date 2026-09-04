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
| **P** | prefer-smaller-side | When one side has \(\geq \theta\) vertices and the other does not, boost desirability of vertices on the smaller side (factor \(P = 2\)). |
| **N** | neighbor-scope limit | Prefer candidates in the intersection of the full candidate set and the last-added vertex's neighbors, narrowing search. |

**ACO-PN** = both P and N enabled — this is the **default** for main paper experiments.

**Instance fitness** rewards balanced growth once \(\theta\) is met on one side: if \(\min(|U_S|, |V_S|) \geq \theta\), fitness is \(a \times b^2\) where \(a, b = \min\max(|U_S|, |V_S|)\); otherwise \(a^2\).

**Paper hyperparameters** (see `README.md` for env-var mapping):

- 5 epochs, evaporation \(\rho = 0.05\), deposit \(T = 1\)
- Ant-count sweep: \(\{1, 2, 5, 10, 20, 50, 100\}\)
- 6 stochastic replicates per ant count (`ACO_RUNS=6`, `SEED=1`); **run 1 is a
  Julia JIT warmup** on the real graph (`jit_warmup=true` in JSON) and is
  discarded by `emit/` / `best_trial` / discovery cost, leaving 5 counted runs
- θ-heuristic is solved twice per graph (first discarded for JIT; second timed)
- Optional planted biclique injection (\(u = v = 5\), \(k\) missing edges) for validation

**Entry points:** `bin/load.jl` (single-graph runs), `bin/vary.jl` (ant-count sweeps), `scripts/vary.bash` (batch over all datasets).

## θ-heuristic (baseline)

The **θ-heuristic** (`src/theta_heuristic.jl`) is the greedy construct-and-trim baseline from Cui et al.:

1. Start with empty \(U\) and all of \(V\).
2. Repeatedly add the \(u \in U_G \setminus U\) with largest degree into the current \(V\).
3. While missing edges exceed \(k\), remove the \(v \in V\) with largest nondegree (most missing edges to \(U\)).
4. Stop when \(|U| = \theta\); return \(U \cup V\).

Complexity is \(\mathcal{O}(\theta n + m)\). It is fast but often returns smaller bicliques than ACO. Cui et al. also use it to seed **branch-and-pivot** exact search (`src/search.jl`); we compare pivot wall time when seeded by θ alone vs. θ + best ACO subgraph (`bin/compare-seeds.jl`).

**θ-feasible** means \(|U_S| \geq \theta\) and \(|V_S| \geq \theta\). ACO trials are only counted as beating the heuristic when they have more edges **and** are θ-feasible.

## Paper build (`paper/`)

Prose lives in `paper/main.tex`. Figures and tables are **generated** from experiment JSON — never hand-edited in the final PDF path.

### Workflow

```
results/*.json  →  emit/  →  paper/generated/*.tex  →  build.py  →  build.pdf
```

1. **Experiments** write JSON under `results/` (see below).
2. **`python -m emit`** (or `make paper-emit`) reads JSON and writes LaTeX fragments to `paper/generated/`.
3. **`paper/build.py`** substitutes `%%PLACEHOLDER%%` tokens in `main.tex` with generated fragments → `build.tex` → compiles PDF.

**Commands:**

```bash
make paper              # full rebuild from repo root
cd paper && make        # same, from paper/
python paper/build.py emit      # only regenerate fragments
python paper/build.py assemble  # only substitute placeholders
python paper/build.py pdf       # only compile
```

**Configuration:** `paper/build.json` maps placeholders to emit modes, input directories, and options. Set `results_dir` once (default `../results`); fragment paths are bare names under that tree (e.g. `vary_k2t5i_PN`). Absolute / `../…` paths still resolve from `paper/`.

| Placeholder | Emit mode | Purpose |
|-------------|-----------|---------|
| `%%QUALITY%%` | `quality` | Ant-count sweep: quality vs. θ-heuristic, θ-feasibility rate, runtime |
| `%%SEED_COMPARE%%` | `seed-compare` | Full pivot table (appendix); `:highlights` = representative rows in §4.2 |
| `%%TABLE:k2t5i_PN%%` | `table` | Per-graph ACO vs. θ comparison (appendix); `:k2t5i_PN:highlights` = §4.3; `:k3t6i_PN` = $k{=}3,\theta{=}6$ |
| `%%COMPARE:theta-time%%` | `compare` | θ-heuristic runtime vs. \(\theta n + m\) bound |
| `%%COMPARE:deg-size-time%%` | `compare` | ACO runtime vs. complexity bounds |
| `%%STATISTICS:…%%` | `statistics` | Inline win/loss counts, Wilcoxon, missing-at-size stats |

**Important:** `emit/` must **only read pre-recorded JSON** — it must not re-run Julia or re-simulate ACO. The build must tolerate incomplete data (warn, don't crash). See `paper/AGENTS.md` for emit-specific rules.

**Artifacts:** `paper/build.tex` and `paper/build.pdf` are generated (gitignored).

## Results collected

All experiment output is **JSON** (plus occasional `.txt` from `scripts/evaluate.bash`). Key directories:

| Directory pattern | Produced by | Contents |
|-------------------|-------------|----------|
| `results/vary_k2t5i_PN/` | `scripts/vary.bash` | Ant-count sweep per graph: `<graph>_ants.json` |
| `results/vary_k2t5i_<flags>/` | `scripts/vary.bash` / `scripts/tests.bash` | Same sweep for other P/N flag combinations |
| `results/compare_k2t5i_<subset>/` | `scripts/compare-seeds.bash` | Pivot timing per graph: `<graph>.json` |
| `results/k2t5i/` | `scripts/evaluate.bash` | Quick per-graph ACO vs. θ benchmark (`.txt`) |

Directory names encode hyperparameters: `vary_k{K}t{THETA}i_{P?}{N?}` (e.g. `vary_k2t5i_PN` = k=2, θ=5, injected plant, P+N flags).

### `vary.jl` JSON (`*_ants.json`)

One file per graph. Top-level metadata plus `trials[]` (one entry per ant-count × replicate) and a `heuristic` block.

**Per-trial fields agents care about:**

- `ants`, `final_edges`, `theta_feasible`, `beats_heuristic`
- `run`, `jit_warmup` — when `aco_runs > 1`, run 1 is JIT warmup (`jit_warmup=true`); emit ignores it
- `wall_time_s`, `time_to_best_s`, `iterations_to_best`
- `U`, `V` — returned subgraph vertex ids (used to re-seed pivot)
- `construction.missing_at_size` — missing-edge count when subgraph first reaches each size

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
| `src/` | Graph types, reduction, ACO, θ-heuristic, branch-and-pivot |
| `bin/` | CLI: `load.jl`, `vary.jl`, `compare-seeds.jl` |
| `emit/` | JSON → LaTeX (Python, stdlib only) |
| `scripts/` | Batch experiment wrappers (`vary.bash`, `regenerate-paper-data.bash`, …) |
| `data/datasets.txt` | Graph manifest |
| `results/` | Experiment JSON output |
| `paper/` | LaTeX source, build config, generated fragments |
| `tests/` | Julia unit / integration tests |

For full replication steps (dataset download, env vars, hardware notes), see `README.md`.
