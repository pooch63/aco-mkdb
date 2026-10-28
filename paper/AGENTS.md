# Paper build — rules for agents

Read this before changing `emit/` or `paper/build.py`.

## General guidelines

1. When I ask you to create a new graph, do NOT update the paper analysis. This way, I can make sure the trend is legitimate.
2. Every graph must be as simple as possible while still communicating its message. One marked line of the cross-graph median, a few ticks, a dashed parity line when the axis is a percent deviation, and a dotted line at the paper default. Do not add bands, faint per-graph curves, confidence intervals, dual axes, heatmaps, or contours unless the message cannot be read from that line.
3. Every graph fits in **one column** when that is still readable. Use a normal `figure` and size axes as a fraction of `\columnwidth` (`COLUMN_WIDTH` for one panel, `COL2_WIDTH` / `COL2_HEIGHT` for panels stacked in that column). A full-width `figure*` (`\textwidth`) is only for a plot that cannot be read in a single column, such as a wide multi-panel row. Do not size a one-column figure in fractions of `\textwidth`.

## Verification before you say the build is done

**An agent must run a real paper compile and confirm success before telling the user the paper build is fixed or complete.** Do not infer success from emit finishing, from a prior run in the conversation, or from assembling `build.tex` alone.

From the repo root or `paper/`:

```bash
cd paper && python3 build.py all
```

If you only changed `main.tex` / placeholders and did not touch emit or results, `python3 build.py assemble && python3 build.py pdf` is enough — but you still must run **`pdf`** and check the exit code.

Treat the build as **failed** unless the command exits 0 **and** `paper/build.pdf` exists with a fresh timestamp. On failure, read **`paper/build.log`** (use `python3 build.py pdf --keep-aux` or `--keep-tex` so the log is not deleted). `build.py` runs `latexmk -silent`, so TeX errors often do not appear in the terminal summary.

Do not mark paper/emit work complete without this compile step.

## Why `python3 build.py all` feels slow (typical ~15–25 s)

Rough breakdown on this project (order of magnitude):

| Step | Time | Cause |
|------|------|--------|
| **emit** | ~0 s if all fragments up to date; **~8–15 s** when compare/budget fragments rebuild | Each `%%COMPARE:…%%` placeholder is a **separate** `python -m emit compare` subprocess. Param-sweep plots rescan **every** `param_dirs` suite (~10 directories × tens of JSON files) per job. Budget figures read `epoch_trace` and emit one small quality-style groupplot. |
| **assemble** | &lt;1 s | Substituting placeholders into `build.tex` (~100 KB). |
| **pdf** | **~12 s almost every time** | `build.py` invokes `latexmk` with **`-g`** (force remake) and then **deletes `.aux` / `.log`** unless `--keep-aux`. So every full build is a **cold** multi-pass pdflatex run over many pgfplots/tikz figures—not an incremental compile (~0.1 s when aux is kept and `-g` is off). |

So slowness is expected for **`all`**, not a sign that emit is re-running Julia. To iterate on LaTeX only: keep aux (`--keep-aux`), assemble, then run `latexmk -pdf -jobname=build build.tex` without `-g`. To iterate on emit only: `python3 build.py emit` (incremental fragments). Avoid `emit --force` unless you mean to rebuild every fragment (~15 s+).

## Emit must not recompute experiment results

The `emit/` package turns **pre-recorded** JSON from `vary.jl` / `compare-seeds.jl` into LaTeX fragments. It is **not** a second experiment runner.

- **Never** re-simulate ACO, re-run Julia, or derive statistics that should have been written at experiment time.
- **Only read** fields already present in result JSON (and documented backfill scripts when fields were added later).
- If a value is missing from JSON, emit a placeholder or fallback sentence and **warn** on stderr — do not try to compute it another way.
- When `aco_runs > 1`, **omit run 1** (or any trial with `jit_warmup=true`) from quality, feasibility, timing, discovery, variance, seed selection, budget, and pheromone-ablation aggregates — that replicate is the real-graph Julia JIT warmup.
- Optional `max_replicates` in `build.json` (top-level or per fragment; or `--max-replicates=R`) retrospectively keeps only the first R counted replicates per ant count (ordered by `run`) for quality / table / statistics / seed-compare / compare aggregates. Omit or set `null` for all counted replicates. The epoch-trace plots (`budget`, `pheromone-ablation`) always use the full counted set.
- Each vary.jl trial is one independent **replicate** (`run` / seed). When `epoch_trace` is present, it is the incumbent after every completed epoch (`t`, `edges`, `theta_feasible`, `elapsed_s`). A prefix of that trace is the shorter epoch budget: Algorithm 1 has no schedule that depends on `T`. Do not invent traces that were not written. `iterations_to_best` is only the epoch at which the final incumbent first appeared; it is not a per-epoch edge history.
- `budget` and `pheromone-ablation` read `budget_dir` (and `ablation_dirs` for the pheromone control). They require `epoch_trace`. The paper draws the budget panels inside `%%QUALITY%%` (same float, also labeled `fig:budget`): median percent deviation from the θ-heuristic on the shared quality scale, then median θ-feasibility, against epoch budget \(T\) and replicate count \(R\). Evaporation, deposit, and the η exponent are a separate figure, `%%QUALITY:sweeps%%`. For \(R\), deviation is the expected best of \(R\) replicates drawn without replacement at the end of the recorded trace, and feasibility is the probability that subset contains a θ-feasible replicate. JIT warmup replicates are omitted, and these panels ignore `max_replicates`. If the traces are missing, leave the panels blank and warn — do not fall back to an ETB credit rule. `--plots=budget` still emits the standalone 2×2.

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
   Those landscape pages are opt-in. `main.tex` includes the same tables as
   short side-by-side blocks: `%%TABLE:…:compact%%`, `%%SEED_COMPARE:compact%%`,
   and `%%METHODS:dir:compact%%`. The appendix is one column, and each block
   row is repeated horizontally so the tables stay short.
   `%%FULL:…%%` … `%%ENDFULL%%` regions in `main.tex` are omitted from
   `build.pdf`. `python paper/build.py --full` (or `make full`) compiles
   `build.pdf` first, then the same source with those regions included,
   writing `full-paper.pdf`. There is no `full-paper.tex`. Placeholders that
   appear only inside a FULL region are emitted only for `--full`.

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
