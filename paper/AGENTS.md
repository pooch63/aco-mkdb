# Paper build — rules for agents

Read this before changing `emit/` or `paper/build.py`.

## Emit must not recompute experiment results

The `emit/` package turns **pre-recorded** JSON from `vary.jl` / `compare-seeds.jl` into LaTeX fragments. It is **not** a second experiment runner.

- **Never** re-simulate ACO, re-run Julia, or derive statistics that should have been written at experiment time.
- **Only read** fields already present in result JSON (and documented backfill scripts when fields were added later).
- If a value is missing from JSON, emit a placeholder or fallback sentence and **warn** on stderr — do not try to compute it another way.

## Emit must not fail the build on incomplete data

The paper build (`python paper/build.py`) **must complete** even when some result files are incomplete.

- **Warn** on stderr when JSON lacks optional or plot-specific fields.
- **Error** only when the build literally cannot proceed (e.g. unknown mode, missing `build.json` key for a required path, no output file written).
- Missing data points in plots are skipped; missing statistics get `"--"` or an explicit “not available” sentence.

## Workflow

1. Experiments write JSON under `results/`.
2. `emit` reads JSON → `paper/generated/*.tex`.
3. `build.py` substitutes `%%PLACEHOLDER%%` in `main.tex` → `build.tex` → PDF.

Do not break step 2 or 3 because a subset of graphs is missing fields. Fix or backfill the JSON separately.
