# Paper build — rules for agents

Read this before changing `emit/` or `paper/build.py`.

## General guidelines

1. When I ask you to create a new graph, do NOT update the paper analysis. Wait to update the paper analysis until after a graph is generated. This way, I can make sure the trend is legitimate.

## Emit must not recompute experiment results

The `emit/` package turns **pre-recorded** JSON from `vary.jl` / `compare-seeds.jl` into LaTeX fragments. It is **not** a second experiment runner.

- **Never** re-simulate ACO, re-run Julia, or derive statistics that should have been written at experiment time.
- **Only read** fields already present in result JSON (and documented backfill scripts when fields were added later).
- If a value is missing from JSON, emit a placeholder or fallback sentence and **warn** on stderr — do not try to compute it another way.
- When `aco_runs > 1`, **omit run 1** (or any trial with `jit_warmup=true`) from quality, feasibility, timing, discovery, variance, seed selection, iteration-budget, and replicate-budget aggregates — that replicate is the real-graph Julia JIT warmup.
- Each vary.jl trial is one independent **replicate** (`run` / seed), always run for the full `iterations_budget` (= `n_E` epochs). `iterations_to_best` (ETB) is the epoch *inside that replicate* when its eventual `final_edges` first appeared — not a replicate index, and not evidence that shorter-budget experiments were run.
- The `iteration-budget` compare plot is a **retrospective truncation** of those full-budget replicates: for credited budget `E` it keeps only replicates with `ETB ≤ E` and uses their recorded `final_edges` / `heuristic.final_edges`. Left/middle panels aggregate per **graph** (best eligible replicate); the ETB CDF aggregates per **replicate**. Do not invent per-epoch edge histories that were never written to JSON.
- The `replicate-budget` compare plot prefixes the **ordered counted** (non-JIT) replicate list: for credited budget `R` it keeps only the first `R` counted seeds per graph. All three panels (wins/feas, % edge increase, replicates-to-best CDF) are per **graph**. Never put the JIT warmup replicate into the ordered prefix.

## Emit must not fail the build on incomplete data

The paper build (`python paper/build.py`) **must complete** even when some result files are incomplete.

- **Warn** on stderr when JSON lacks optional or plot-specific fields.
- **Error** only when the build literally cannot proceed (e.g. unknown mode, missing `build.json` key for a required path, no output file written).
- Missing data points in plots are skipped; missing statistics get `"--"` or an explicit “not available” sentence.

## Workflow

1. Experiments write JSON under `results/` (or another tree named by `results_dir` in `build.json`).
2. `emit` reads JSON → `paper/generated/*.tex` (and optional `*.preamble.tex` sidecars).
3. `build.py` substitutes `%%PLACEHOLDER%%` in `main.tex` → `build.tex` → PDF.

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
