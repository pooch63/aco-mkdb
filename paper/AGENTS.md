# Paper build — rules for agents

Read this before changing `emit/` or `paper/build.py`.

## General guidelines

1. When I ask you to create a new graph, do NOT update the paper analysis. This way, I can make sure the trend is legitimate.

## Emit must not recompute experiment results

The `emit/` package turns **pre-recorded** JSON from `vary.jl` / `compare-seeds.jl` into LaTeX fragments. It is **not** a second experiment runner.

- **Never** re-simulate ACO, re-run Julia, or derive statistics that should have been written at experiment time.
- **Only read** fields already present in result JSON (and documented backfill scripts when fields were added later).
- If a value is missing from JSON, emit a placeholder or fallback sentence and **warn** on stderr — do not try to compute it another way.
- When `aco_runs > 1`, **omit run 1** (or any trial with `jit_warmup=true`) from quality, feasibility, timing, discovery, variance, seed selection, iteration-budget, and replicate-budget aggregates — that replicate is the real-graph Julia JIT warmup.
- Optional `max_replicates` in `build.json` (top-level or per fragment; or `--max-replicates=R`) retrospectively keeps only the first R counted replicates per ant count (ordered by `run`) for quality / table / statistics / seed-compare / compare aggregates. Omit or set `null` for all counted replicates. The epoch-trace plots (`iteration-budget`, `replicate-budget`, `budget-allocation`, `pheromone-ablation`) always use the full counted set.
- Each vary.jl trial is one independent **replicate** (`run` / seed). When `epoch_trace` is present, it is the incumbent after every completed epoch (`t`, `edges`, `theta_feasible`, `elapsed_s`). A prefix of that trace is the shorter epoch budget: Algorithm 1 has no schedule that depends on `T`. Do not invent traces that were not written. `iterations_to_best` is only the epoch at which the final incumbent first appeared; it is not a per-epoch edge history.
- `iteration-budget`, `replicate-budget`, `budget-allocation`, and `pheromone-ablation` read `budget_dir` (and `ablation_dirs` for the pheromone control). They require `epoch_trace`. Best-of-`R` is the expected maximum over subsets drawn without replacement, not the first `R` seeds. JIT warmup replicates are omitted. If the traces are missing, emit a placeholder caption and warn — do not fall back to an ETB credit rule.

## Emit must not fail the build on incomplete data

The paper build (`python paper/build.py`) **must complete** even when some result files are incomplete.

- **Warn** on stderr when JSON lacks optional or plot-specific fields.
- **Error** only when the build literally cannot proceed (e.g. unknown mode, missing `build.json` key for a required path, no output file written).
- Missing data points in plots are skipped; missing statistics get `"--"` or an explicit “not available” sentence.

## Workflow

1. Experiments write JSON under `results/` (or another tree named by `results_dir` in `build.json`).
2. `emit` reads JSON → `paper/generated/*.tex` (and optional `*.preamble.tex` sidecars).
3. `build.py` substitutes `%%PLACEHOLDER%%` in `main.tex` → `build.tex` → PDF.
   `python paper/build.py --tables` (or `make tables` from `paper/`) also
   substitutes placeholders in `tables.tex` → `tables-build.tex` → `tables.pdf`.
   Per-graph result tables live in `tables.tex` and are omitted from the default build.

Emit is incremental: a fragment rebuilds when results JSON, `build.json`, or the
**emit Python sources for that fragment** are newer than `paper/generated/*.tex`
(COMPARE plots also depend on their plot module, e.g. `param_sweep.py` for
`k-sweep`). Editing emit code therefore invalidates the matching figure without
`--force`. Use `make emit FORCE=1` only when you want every fragment rebuilt.

### `results_dir`

Set once in `paper/build.json` (default `../results`). Fragment paths
(`input`, `vary_dir`, `flag_dirs`, `param_dirs`, `missing_at_base`,
`compare_dir`, and `vary_base` + TABLE suffix) are bare names under that
directory, e.g. `"vary_k2t5i_PN"`. To rebuild from an archive tree, change
only `results_dir` (e.g. `../results_old`).

### `%%PREAMBLE%%`

Fragments that need preamble macros call `write_tex(..., preamble=...)`, which
writes a `*.preamble.tex` sidecar next to the fragment. Assemble concatenates
unique sidecars for placeholders used in `main.tex` into `%%PREAMBLE%%` (safe
when empty). No `build.json` entry is required for `PREAMBLE`.

Do not break step 2 or 3 because a subset of graphs is missing fields. Fix or backfill the JSON separately.
