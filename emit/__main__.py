#!/usr/bin/env python3
"""
Emit LaTeX figures and tables from ACO benchmark JSON.

Modes
-----
  quality
      Vary.jl ant-count format → 1xN groupplot (N = panels with data):
        - mean % deviation from optimum (feasible trials; omitted if none)
        - mean % deviation from Cui θ-heuristic (feasible trials)
        - % theta-feasible trials
        - mean wall-clock time vs ant count

  seed-compare
      compare-seeds.jl (+ vary.jl) → figures:
        - scatter: reduced vertices vs ACO edges saved (ACO beat θ only)
        - scatter: ACO edges saved vs % pivot time savings (beat θ only)
        - note: N graphs examined; M where ACO did not beat θ
        - scatter: reduced vertices vs ACO search cost (no-beat only)
      Reads full pivot comparisons and beat_heuristic=false skip markers
      written by compare-seeds.jl. Pass --vary-dir (or rely on compare_* →
      vary_* name mapping) for reduced graph sizes and datasets that still
      lack a compare JSON.

  table
      Vary.jl ant-count JSON → paper-style LaTeX table.

Usage:
    python -m emit quality /path/to/json/dir -o plot.tex
    python -m emit seed-compare /path/to/json/dir -o seed.tex
    python -m emit seed-compare compare_k2t5i --vary-dir=vary_k2t5i -o seed.tex
    python -m emit table vary_k2t5i -o table.tex
    python -m emit table vary_k2t5i --ants=50
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
        help="quality | seed-compare | table",
    )
    parser.add_argument(
        "directory",
        help="Directory containing the JSON result files",
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
        help="Only consider ACO trials with this ant count (table mode)",
    )

    args = parser.parse_args(argv)
    json_paths = list_json_paths(args.directory)

    if args.mode == "quality":
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import quality
        quality.run(json_paths, args.output)

    elif args.mode == "seed-compare":
        from . import seed_compare
        vary_dir = seed_compare.resolve_vary_dir(args.directory, args.vary_dir)
        if not json_paths and not vary_dir:
            raise SystemExit(f"No .json files found in {args.directory}")
        seed_compare.run(json_paths, args.output, vary_dir=vary_dir)

    elif args.mode == "table":
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import table
        table.run(json_paths, args.output, ants=args.ants)

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
