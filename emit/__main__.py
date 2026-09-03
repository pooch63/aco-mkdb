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
      compare-seeds.jl (+ vary.jl) → table:
        graph/reduced nodes & edges, pivot time saved, and percent
        reduction. --subset=full (default) emits every dataset;
        --subset=highlights emits net-savings graphs plus the smallest
        and largest overhead cases.
      Reads full pivot comparisons and beat_heuristic=false skip markers
      written by compare-seeds.jl. Pass --vary-dir (or rely on compare_* →
      vary_* name mapping / source_vary) for full graph sizes and datasets
      that still lack a compare JSON.

  table
      Vary.jl ant-count JSON → paper-style LaTeX table.
      --subset=full (default) emits every dataset (appendix);
      --subset=highlights emits the largest relative ACO wins plus the
      closest θ-heuristic wins.

  compare
      Vary.jl ant-count JSON → complexity figures (ACO-PN).
      Plot groups (pass comma-separated via --plots):
        theta-time, deg-size-time, density-size, max-deg-time
      Pass a single results directory, e.g. vary_k2t5i_PN.

  statistics
      Vary.jl ant-count JSON → inline LaTeX for %%STATISTICS:field%%
      placeholders (win/loss counts, cross-run variance, Wilcoxon test,
      construction missing-at-size means). Fields: aco-wins, heur-wins,
      ties, n-graphs, variance, wilcoxon, aco-missing-at-5, aco-n-missing-at-5

      missing-at-5 fields read pre-recorded JSON only (no Julia at build time).

Usage:
    python -m emit quality /path/to/json/dir -o plot.tex
    python -m emit seed-compare /path/to/json/dir -o seed.tex
    python -m emit seed-compare compare_k2t5i --vary-dir=vary_k2t5i -o seed.tex
    python -m emit seed-compare compare_k2t5i --vary-dir=vary_k2t5i --subset=highlights -o seed.tex
    python -m emit table vary_k2t5i -o table.tex
    python -m emit table vary_k2t5i --ants=50
    python -m emit table vary_k2t5i_PN --ants=100 --subset=highlights -o highlights.tex
    python -m emit compare vary_k2t5i_PN -o compare.tex
    python -m emit compare vary_k2t5i_PN --plots=theta-time,deg-size-time -o complexity.tex
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
        help="JSON directory (compare / quality / table / statistics) "
             "or compare-seeds directory (seed-compare)",
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
        "--subset",
        default="full",
        choices=("full", "highlights"),
        help="seed-compare / table: full appendix table or representative "
             "highlights",
    )
    parser.add_argument(
        "--plots",
        default=None,
        help="compare mode: comma-separated plot groups "
             "(theta-time, deg-size-time, density-size, max-deg-time, "
             "flag-ablation)",
    )
    parser.add_argument(
        "--flag-dir",
        action="append",
        metavar="LABEL=DIR",
        default=None,
        help="compare flag-ablation: variant directory "
             "(repeat for ACO, ACO-P, ACO-N, ACO-PN)",
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
        seed_compare.run(
            json_paths, args.output, vary_dir=vary_dir, subset=args.subset
        )

    elif args.mode == "table":
        json_paths = list_json_paths(args.directory)
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import table
        table.run(
            json_paths, args.output, ants=args.ants, subset=args.subset
        )

    elif args.mode == "compare":
        json_paths = list_json_paths(args.directory)
        flag_dirs = None
        if args.flag_dir:
            flag_dirs = {}
            for item in args.flag_dir:
                label, sep, path = item.partition("=")
                if not sep or not label.strip() or not path.strip():
                    raise SystemExit(
                        f"Invalid --flag-dir entry {item!r}; use LABEL=DIR"
                    )
                flag_dirs[label.strip()] = path.strip()
        if not json_paths and not (
            args.plots and "flag-ablation" in args.plots.split(",")
        ):
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import compare
        compare.run(
            json_paths,
            args.output,
            ants=args.ants,
            plots=args.plots,
            flag_dirs=flag_dirs,
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
