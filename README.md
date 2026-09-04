# ACO-MKDB

Ant Colony Optimization (ACO) for **k-defective maximum edge bicliques** on bipartite graphs, with benchmarks against the Cui et al. θ-heuristic and branch-and-pivot seeding experiments from [the paper](paper/build.pdf) (`paper/main.tex`).

This README describes how to reproduce the paper's datasets, experiments, and figures from scratch.

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Core graph, reduction, ACO, and branch-and-pivot code |
| `bin/` | CLI entry points (`load.jl`, `vary.jl`, `compare-seeds.jl`, …) |
| `providers/` | Dataset download/convert adapters (KONECT, Amazon, Wikipedia, …) |
| `data/datasets.txt` | Manifest of graphs used in the paper |
| `scripts/` | Bash wrappers that batch experiments across all datasets |
| `emit/` | Python tool that turns JSON results into LaTeX tables/figures |
| `results/` | Experiment output (JSON logs and text benchmarks) |
| `paper/` | LaTeX source and PDF |

Raw graph data under `data/` is **not** checked into git (see `data/.gitignore`). You must download it locally before running experiments.

## Requirements

- **Julia** 1.9+ (tested on 1.12)
- **Python** 3 (stdlib only; used by `emit/`)
- **Bash**, `tar`, and network access for dataset downloads
- **8 CPU threads** recommended (`JULIA_THREADS=8`, matching the paper)
- **Disk space**: large KONECT/Wikipedia graphs can require tens of GB per dataset
- **Time**: the full pipeline takes many hours on a laptop-class machine; pivot comparisons on small graphs are the slowest step

## Setup

From the repository root, install Julia packages:

```bash
julia -e 'using Pkg; Pkg.add([
  "CSV", "DataFrames", "Downloads", "EnumX", "JSON3",
  "ProfileCanvas", "StatsBase"
])'
```

No Python packages are required beyond the standard library.

## Replication workflow

The paper has three main experimental outputs:

1. **Ant-count sweep** — solution quality, θ-feasibility, and runtime vs. number of ants (Section 4 figures)
2. **ACO vs. θ-heuristic** — per-graph comparison on ~32 datasets (Appendix table)
3. **Pivot seed comparison** — branch-and-pivot runtime when seeded by ACO vs. θ-heuristic on graphs where ACO wins (Table 2)

All experiments use **k = 2**, **θ = 5**, a planted biclique injection (**u = v = 5**), **5 ACO epochs**, **5 stochastic replicates per ant count**, and ant counts **{1, 2, 5, 10, 20, 50, 100}** unless noted otherwise.

### Step 1 — Download datasets

The canonical list is `data/datasets.txt` (35 graphs from [KONECT](http://konect.cc/networks/), [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/), and Wikipedia edit networks).

```bash
# Download everything in the manifest (skips graphs already indexed)
./scripts/fetch-datasets.bash

# Or fetch only one provider subtree
PREFIX=konect-small ./scripts/fetch-datasets.bash

# Preview without downloading
DRY_RUN=1 ./scripts/fetch-datasets.bash
```

Each dataset is written to `data/<provider>/<name>/indexed_interactions.csv` with mappings under `mappings/`.

To fetch a single graph manually:

```bash
julia bin/process.jl konect/bitcoin --download
julia bin/process.jl amazon/boxes --download
```

**Notes:**

- Short dataset keys map to remote filenames via builtins and `data/aliases.txt`
  (`provider/local=Remote_Name`, e.g. `amazon/gift-cards=Gift_Cards`,
  `konect/moreno=moreno_propro`). Legacy `data/konect_aliases.txt` is still merged.
- Wikipedia dumps (`konect/jawiki`, `konect/eswiki`) depend on Wikimedia's dump servers and may fail intermittently; re-run or fetch them individually.
- Some very large graphs (`matter`, `appliances`, …) download slowly and use substantial RAM during indexing.

### Step 2 — Ant-count sweep (`vary.bash`)

Runs ACO across ant counts on every indexed graph (ordered by edge count, smallest first). Output: `results/vary_k2t5i_PN/<graph>_ants.json`.

Paper-matching settings:

```bash
export JULIA_THREADS=8
export ANTS_RANGE=1,2,5,10,20,50,100
export ITERATIONS=5
export ACO_RUNS=6
export SEED=1
export INJECT=1
export K=2
export THETA=5

# P = prefer smaller side, N = neighbor-scope candidate filter (paper defaults)
export PREFER_SMALLER_SIDE=true
export ENABLE_NEIGHBOR_SCOPE_LIMIT=true

./scripts/vary.bash
```

`ACO_RUNS=6` records six seeded replicates per ant count; **run 1 is Julia JIT
warmup** on the real instance and is omitted by `emit/` (and by `best_trial` /
discovery in the JSON), leaving five counted runs for quality, timing, TTB, and
ITB. The θ-heuristic is also run twice (first discarded, second timed).

To regenerate every paper result directory (flag ablation, (k,θ) PN tables,
quality sweep, compare-seeds, …):

```bash
./scripts/regenerate-paper-data.bash
# or a subset: PHASES=flags,sweep,kt ./scripts/regenerate-paper-data.bash
```

Useful options:

```bash
SKIP_EXISTING=1 ./scripts/vary.bash          # resume interrupted sweep
RESUME_FROM=13 ./scripts/vary.bash           # skip first 12 graphs
PREFIX=konect-small ./scripts/vary.bash        # restrict to one data subtree
RUN_PIVOT=1 ./scripts/vary.bash              # also run pivot for optimum (very slow)
```

The output directory name encodes hyperparameters: `vary_k{K}t{THETA}i_{P?}{N?}` → `vary_k2t5i_PN` with the defaults above.

`scripts/tests.bash` runs all four `(prefer-smaller-side × neighbor-scope-limit)` combinations; the paper results use **PN** (both enabled).

### Step 3 — Pivot seed comparison (`compare-seeds.bash`)

For graphs where ACO beat the θ-heuristic in Step 2, runs branch-and-pivot twice (θ seed vs. best ACO seed) and records the wall-time difference. The paper's Table 2 focuses on the **konect-small** subset.

```bash
export JULIA_THREADS=8
export TIMEOUT=2000          # seconds per pivot run (paper default)
export SKIP_EXISTING=1
export PREFIX=konect-small     # paper Table 2 uses the konect-small subset

./scripts/compare-seeds.bash results/vary_k2t5i_PN
```

Output: `results/compare_k2t5i_konect-small/<graph>.json`.

Omit `PREFIX` to run pivot comparisons on every graph in the vary output (much slower; several large graphs will time out).

If a pivot run times out, increase the limit and re-run; `SKIP_EXISTING=1` automatically retries timeout results when `TIMEOUT` is strictly larger than the previous limit:

```bash
TIMEOUT=4000 ./scripts/compare-seeds.bash results/vary_k2t5i_PN
```

### Step 4 (optional) — Direct ACO vs. θ-heuristic logs

`scripts/evaluate.bash` runs a single ACO + θ-heuristic benchmark per graph (100 ants, 5 iterations) without the full ant sweep. Useful for quick per-graph checks or the appendix-style discovery/ITB/TTB columns.

```bash
export JULIA_THREADS=8
export SEED=1

./scripts/evaluate.bash
# → results/k2t5i/<graph>.txt
```

For a secondary configuration (k = 5, θ = 6):

```bash
RUN_K5T6=1 ./scripts/evaluate.bash
```

### Step 5 — Build the paper

Placeholders in `paper/main.tex` (`%%QUALITY%%`, `%%SEED_COMPARE%%`, `%%TABLE%%`, `%%COMPARE:…%%`) are filled from experiment JSON by the `emit` package and compiled automatically:

```bash
# Full rebuild: regenerate LaTeX fragments → build.tex → build.pdf
make paper

# Or from paper/:
cd paper && make

# Individual steps
make paper-emit      # only run emit (writes paper/generated/*.tex)
make -C paper assemble   # substitute placeholders → build.tex
make -C paper pdf        # compile build.tex
```

Data paths and emit options live in `paper/build.json` (defaults point at `results/vary_k2t5i_PN`, `results/compare_k2t5i_konect-small`, etc.). Edit that file if your result directories differ.

To emit fragments manually:

```bash
python -m emit quality results/vary_k2t5i_PN -o paper/generated/quality.tex
python -m emit table results/vary_k2t5i_PN -o paper/generated/table.tex --ants=100
python -m emit seed-compare results/compare_k2t5i_konect-small \
  --vary-dir=results/vary_k2t5i_PN -o paper/generated/seed-compare.tex
python -m emit compare results/vary_k2t5i_ -o paper/generated/compare.tex
python -m emit compare results/vary_k2t5i_ --plots=theta-time,deg-size-time -o paper/generated/complexity.tex
python -m emit compare results/vary_k2t5i_ --plots=pn -o paper/generated/compare-pn.tex
```

The source of truth for prose is `paper/main.tex`; `paper/build.tex` and `paper/build.pdf` are generated artifacts (gitignored).

## Hyperparameters (paper ↔ code)

| Paper parameter | Value | Where set |
|-----------------|-------|-----------|
| k | 2 | `K=2` in scripts; `--k=2` in `bin/load.jl` |
| θ | 5 | `THETA=5` |
| Injected plant | u = v = 5, k missing edges | `INJECT=1`, `INJECT_U=5`, `INJECT_V=5` |
| ACO epochs (N_e) | 5 | `ITERATIONS=5` → `ACO_NUM_ITERATIONS` in `bin/load.jl` |
| Evaporation (ρ) | 0.05 | `ACO_EVAPORATION=0.95` in `bin/load.jl` (retain 95% per epoch) |
| Pheromone deposit (T) | 1 | default in `bin/load.jl` |
| Smaller-side factor (P) | 2 | `PREFER_SMALLER_SIDE=true` → `PREFER_SMALLER_SIDE_MULTIPLIER` in `src/aco/advance.jl` |
| Ant counts (N_S) | 1–100 sweep | `ANTS_RANGE=1,2,5,10,20,50,100` |
| ACO replicates | 6 (run 1 = JIT warmup; 5 counted) | `ACO_RUNS=6` |
| Random seed | 1 | `SEED=1` |
| Threads | 8 | `JULIA_THREADS=8` |

## Running a single graph

For debugging or one-off runs:

```bash
# ACO vs. θ-heuristic benchmark
julia -t 8 bin/load.jl konect-small/urwiki \
  --inject --u=5 --v=5 \
  --k=2 --theta=5 --seed=1 \
  --benchmark=aco,heuristic

# Ant-count sweep → JSON
julia -t 8 bin/load.jl konect-small/urwiki \
  --inject --u=5 --v=5 \
  --k=2 --theta=5 --seed=1 \
  --prefer-smaller-side=true \
  --neighbor-scope-limit=true \
  --reduce=lo \
  --vary=ant-count \
  --ants-range=1,2,5,10,20,50,100 \
  --iterations=5 --aco-runs=6 \
  --save=results/urwiki_ants.json
```

List datasets in edge-count order:

```bash
julia bin/order_graphs.jl --keys-only
julia bin/order_graphs.jl --keys-only --prefix=konect-small
```

## Tests

Unit tests live under `tests/`:

```bash
julia tests/suite.jl
julia tests/test_aco.jl
julia tests/test_mkdb.jl
```

## Expected runtime and hardware

The paper was run on **Ubuntu**, **8 threads**, an **Intel Core Ultra 7 258V**, and **32 GB RAM**. Exact wall times will vary, but expect:

- **vary sweep** on all 35 graphs: many hours (large graphs dominate)
- **compare-seeds** on konect-small: up to ~30 minutes per graph that beats θ (pivot is exponential)
- **branch-and-pivot** on large graphs (`matter`, `dimacs-polblogs`, …) may exceed practical timeouts — the paper excludes these from the seed comparison for that reason

Use `SKIP_EXISTING=1` and `RESUME_FROM` to restart long jobs without redoing finished graphs.

## Adding datasets

1. Implement or reuse a provider in `providers/`.
2. Add the key (`provider/name`) to `data/datasets.txt`.
3. Run `./scripts/fetch-datasets.bash` or `julia bin/process.jl provider/name --download`.
4. Re-run the experiment scripts; discovery order is automatic via `bin/order_graphs.jl`.

Graphs listed in `data/.dataignore` are skipped by the batch scripts.

## Citation

If you use this code, please cite the paper in `paper/main.tex` (author: Kiyaan Pillai).
