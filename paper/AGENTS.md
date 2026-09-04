# Paper build — rules for agents

Read this before changing `emit/` or `paper/build.py`.

## Emit must not recompute experiment results

The `emit/` package turns **pre-recorded** JSON from `vary.jl` / `compare-seeds.jl` into LaTeX fragments. It is **not** a second experiment runner.

- **Never** re-simulate ACO, re-run Julia, or derive statistics that should have been written at experiment time.
- **Only read** fields already present in result JSON (and documented backfill scripts when fields were added later).
- If a value is missing from JSON, emit a placeholder or fallback sentence and **warn** on stderr — do not try to compute it another way.
- When `aco_runs > 1`, **omit run 1** (or any trial with `jit_warmup=true`) from quality, feasibility, timing, discovery, variance, and seed selection — that replicate is the real-graph Julia JIT warmup.

## Emit must not fail the build on incomplete data

The paper build (`python paper/build.py`) **must complete** even when some result files are incomplete.

- **Warn** on stderr when JSON lacks optional or plot-specific fields.
- **Error** only when the build literally cannot proceed (e.g. unknown mode, missing `build.json` key for a required path, no output file written).
- Missing data points in plots are skipped; missing statistics get `"--"` or an explicit “not available” sentence.

## Workflow

1. Experiments write JSON under `results/` (or another tree named by `results_dir` in `build.json`).
2. `emit` reads JSON → `paper/generated/*.tex` (and optional `*.preamble.tex` sidecars).
3. `build.py` substitutes `%%PLACEHOLDER%%` in `main.tex` → `build.tex` → PDF.

### `results_dir`

Set once in `paper/build.json` (default `../results`). Fragment paths
(`input`, `vary_dir`, `flag_dirs`, `missing_at_base`, and `vary_base` + TABLE
suffix) are bare names under that directory, e.g. `"vary_k2t5i_PN"`. To rebuild
from an archive tree, change only `results_dir` (e.g. `../results_old`).

### `%%PREAMBLE%%`

Fragments that need preamble macros call `write_tex(..., preamble=...)`, which
writes a `*.preamble.tex` sidecar next to the fragment. Assemble concatenates
unique sidecars for placeholders used in `main.tex` into `%%PREAMBLE%%` (safe
when empty). No `build.json` entry is required for `PREAMBLE`.

Do not break step 2 or 3 because a subset of graphs is missing fields. Fix or backfill the JSON separately.
