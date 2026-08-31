#!/usr/bin/env python3
"""
Emit LaTeX figures and tables from ACO benchmark JSON.

Modes
-----
  quality
      Vary.jl ant-count format → 2-column groupplot:
        Top row: Cui θ-heuristic deviation | θ-feasibility rate
        Bottom row: mean wall-clock time (optimum-quality panel if present)

  seed-compare
      compare-seeds.jl (+ vary.jl) → table + figures:
        - table: every dataset with graph/reduced nodes & edges, pivot
          time saved ("--" if ACO did not beat θ), and percent reduction
        - bar charts: pivot time saved (s) and percent reduction per
          dataset (missing bar = ACO did not beat θ)
      Reads full pivot comparisons and beat_heuristic=false skip markers
      written by compare-seeds.jl. Pass --vary-dir (or rely on compare_* →
      vary_* name mapping / source_vary) for full graph sizes and datasets
      that still lack a compare JSON.

  table
      Vary.jl ant-count JSON → paper-style LaTeX table.

  compare
      Vary.jl ant-count JSON → complexity / flag-setting figures.
      Plot groups (pass comma-separated via --plots):
        theta-time, deg-size-time, density-size, max-deg-time, pn
      Pass a folder prefix such as vary_k2t5i_ to auto-load the four
      ACO flag settings: (none), N, P, and PN.

  statistics
      Vary.jl ant-count JSON → inline LaTeX for %%STATISTICS:field%%
      placeholders (win/loss counts, cross-run variance, Wilcoxon test,
      construction missing-at-size means). Fields: aco-wins, heur-wins,
      ties, n-graphs, variance, wilcoxon, aco-missing-at-5, aco-n-missing-at-5

Usage:
    python -m emit quality /path/to/json/dir -o plot.tex
    python -m emit seed-compare /path/to/json/dir -o seed.tex
    python -m emit seed-compare compare_k2t5i --vary-dir=vary_k2t5i -o seed.tex
    python -m emit table vary_k2t5i -o table.tex
    python -m emit table vary_k2t5i --ants=50
    python -m emit compare vary_k2t5i_ -o compare.tex
    python -m emit compare vary_k2t5i_ --plots=theta-time,deg-size-time -o complexity.tex
"""

from __future__ import annotations

import argparse

from . import MODES
from .common import list_json_paths


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m emit",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=MODES,
        help="quality | seed-compare | table | compare",
    )
    parser.add_argument(
        "directory",
        help="JSON directory, or (compare mode) vary folder prefix "
             "such as vary_k2t5i_ for ACO / N / P / PN",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Write LaTeX to this file instead of stdout",
    )
    parser.add_argument(
        "--vary-dir",
        default=None,
        help="vary.jl JSON directory for reduced graph sizes, ACO discovery "
             "costs, and no-beat datasets (seed-compare). Default: infer "
             "vary_* beside compare_*.",
    )
    parser.add_argument(
        "--ants",
        type=int,
        default=None,
        help="Only consider ACO trials with this ant count "
             "(table / compare modes)",
    )
    parser.add_argument(
        "--plots",
        default=None,
        help="compare mode: comma-separated plot groups "
             "(theta-time, deg-size-time, density-size, max-deg-time, pn)",
    )
    parser.add_argument(
        "--field",
        default=None,
        help="statistics mode: output field "
             "(aco-wins, heur-wins, ties, n-graphs, variance, wilcoxon, "
             "missing-at-5, aco-missing-at-5, aco-n-missing-at-5)",
    )
    parser.add_argument(
        "--vary-base",
        default=None,
        help="statistics mode: vary folder prefix for ACO flag directories "
             "(e.g. ../results/vary_k2t5i_)",
    )

    args = parser.parse_args(argv)

    if args.mode == "quality":
        json_paths = list_json_paths(args.directory)
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import quality
        quality.run(json_paths, args.output)

    elif args.mode == "seed-compare":
        json_paths = list_json_paths(args.directory)
        from . import seed_compare
        vary_dir = seed_compare.resolve_vary_dir(args.directory, args.vary_dir)
        if not json_paths and not vary_dir:
            raise SystemExit(f"No .json files found in {args.directory}")
        seed_compare.run(json_paths, args.output, vary_dir=vary_dir)

    elif args.mode == "table":
        json_paths = list_json_paths(args.directory)
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import table
        table.run(json_paths, args.output, ants=args.ants)

    elif args.mode == "compare":
        from . import compare
        compare.run(
            args.directory,
            args.output,
            ants=args.ants,
            plots=args.plots,
        )

    elif args.mode == "statistics":
        json_paths = list_json_paths(args.directory)
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import statistics
        statistics.run(
            json_paths,
            args.output,
            ants=args.ants,
            field=args.field,
            vary_base=args.vary_base,
        )

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
