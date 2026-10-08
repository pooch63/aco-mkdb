#!/usr/bin/env python3
"""
Emit LaTeX figures and tables from ACO benchmark JSON.

Modes
-----
  quality
      Vary.jl ant-count format → cross-graph median percent deviation
      from the θ-heuristic vs ant count. Optional --ablation-dir=LABEL=DIR
      (rho, pheromone, eta-exp) adds those sweeps as compact panels in
      the same figure. Per-graph means, not pooled replicates;
      θ-infeasible outcomes count as −100%.

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
      --subset=highlights emits the largest relative ACO wins as a
      single-column table of $|E(D)|$ and discovery time.

  compare
      Vary.jl ant-count JSON → complexity figures ($F_N$).
      Plot groups (pass comma-separated via --plots):
        theta-time, deg-size-time, bound-time, density-size,
        max-deg-time, flag-ablation, flag-ablation-replicates,
        flag-feasibility, flag-feasibility-replicates, budget,
        pheromone-ablation,
        k-sweep, theta-sweep, density-wins, param-runtime
      Pass a single results directory, e.g. vary_k2t5i_PN.
      flag-ablation / flag-feasibility (and *-replicates) use
      --flag-dir=LABEL=DIR.
      k-sweep / theta-sweep / density-wins / param-runtime use
      --param-dir=LABEL=DIR (see build.json).
      budget and pheromone-ablation read per-epoch incumbent traces
      (``epoch_trace``). pheromone-ablation also takes
      ``--trace-dir=LABEL=DIR`` for the Δτ=0 run and any ρ sweep.
      These plots ignore ``--max-replicates`` and use every counted
      replicate.
      --max-replicates=R optionally prefixes counted replicates for
      all modes except budget and pheromone-ablation (which always
      use the full counted set).

  statistics
      Vary.jl ant-count JSON → inline LaTeX for %%STATISTICS:field%%
      placeholders (win/loss counts, cross-run variance, Wilcoxon test,
      θ-feasibility rates, construction missing-at-size means, pivot
      coverage among ACO wins). Fields include:
      aco-wins, heur-wins, ties, n-graphs, aco-nonwins, variance,
      wilcoxon, median-edge-pct, aco-theta-feasibility-rate,
      theta-heuristic-feasibility-rate, missing-at-5, missing-at-10,
      aco-missing-at-5, aco-n-missing-at-5, aco-missing-at-10,
      aco-n-missing-at-10, pivot-tested-wins, pivot-excluded-wins,
      pivot-*-mean-nR, pivot-*-mean-eR

      missing-at-* fields read pre-recorded JSON only (no Julia at build
      time). Pivot-coverage fields also need --compare-dir.

  methods
      load.jl --benchmark JSON (scripts/evaluate.bash with METHODS=…) →
      per-graph multi-method table. --methods=tow,aco,heuristic (default)
      lists method blocks in column order; the first is the focal method
      used to group rows and tally wins / ties / losses.

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
    python -m emit methods results/k2t5i_heuristic-aco-tow --methods=tow,aco,heuristic -o tow.tex
"""

from __future__ import annotations

import argparse

from . import MODES
from .common import list_json_paths, set_max_replicates


def _parse_runtime_sweeps(items):
    """``k:5:1,2,3,4`` or ``theta:3:5,6,7,8`` → sweep dicts."""
    if not items:
        return None
    sweeps = []
    for item in items:
        axis, sep, rest = item.partition(":")
        fixed, sep2, values = rest.partition(":")
        if axis not in ("k", "theta") or not sep or not sep2 or not values.strip():
            raise SystemExit(
                f"Invalid --runtime-sweep {item!r}; "
                "use k:THETA:k1,k2,... or theta:K:t1,t2,..."
            )
        try:
            sweeps.append(
                {
                    "axis": axis,
                    "fixed": int(fixed),
                    "values": [int(v) for v in values.split(",") if v.strip()],
                }
            )
        except ValueError as exc:
            raise SystemExit(f"Invalid --runtime-sweep {item!r}: {exc}") from exc
    return sweeps


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
        "--max-replicates",
        type=int,
        default=None,
        metavar="R",
        help="Retrospectively keep only the first R counted (non-JIT) "
             "replicates per ant count, ordered by run (default: all). "
             "Affects quality / table / statistics / seed-compare / "
             "compare except budget and pheromone-ablation, which always "
             "use the full counted set.",
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
             "(theta-time, deg-size-time, bound-time, density-size, "
             "max-deg-time, "
             "flag-ablation, flag-ablation-replicates, "
             "flag-feasibility, flag-feasibility-replicates, "
             "budget, pheromone-ablation, "
             "k-sweep, theta-sweep, density-wins, param-runtime)",
    )
    parser.add_argument(
        "--flag-dir",
        action="append",
        metavar="LABEL=DIR",
        default=None,
        help="compare flag-ablation / flag-feasibility (and *-replicates): "
             "variant directory "
             "(repeat for ACO, $F_P$, $F_N$, $F_{PN}$)",
    )
    parser.add_argument(
        "--param-dir",
        action="append",
        metavar="LABEL=DIR",
        default=None,
        help="compare k-sweep / theta-sweep / density-wins / "
             "param-runtime: (k, θ) suite directory (repeat; labels are "
             "free-form, k/θ read from JSON)",
    )
    parser.add_argument(
        "--runtime-sweep",
        action="append",
        metavar="AXIS:FIXED:V1,V2,...",
        default=None,
        help="param-runtime: k:THETA:k1,k2,... varies k at that θ; "
             "theta:K:t1,t2,... varies θ at that k. From build.json "
             "runtime_sweeps.",
    )
    parser.add_argument(
        "--trace-dir",
        action="append",
        metavar="LABEL=DIR",
        default=None,
        help="compare pheromone-ablation: extra epoch-trace directory "
             "(repeat; labels such as dt0 or rho0.2)",
    )
    parser.add_argument(
        "--ablation-dir",
        action="append",
        metavar="LABEL=DIR",
        default=None,
        help="quality mode: hyperparameter sweep directory "
             "(repeat; labels rho, pheromone, eta-exp)",
    )
    parser.add_argument(
        "--field",
        default=None,
        help="statistics mode: output field "
             "(aco-wins, heur-wins, ties, n-graphs, aco-nonwins, variance, "
             "wilcoxon, aco-theta-feasibility-rate, "
             "theta-heuristic-feasibility-rate, missing-at-5, "
             "missing-at-10, aco-missing-at-5, aco-n-missing-at-5, "
             "aco-missing-at-10, aco-n-missing-at-10, pivot-tested-wins, "
             "pivot-excluded-wins, pivot-*-mean-nR, pivot-*-mean-eR)",
    )
    parser.add_argument(
        "--vary-base",
        default=None,
        help="statistics mode: vary folder prefix for ACO flag directories "
             "(e.g. ../results/vary_k2t5i_)",
    )
    parser.add_argument(
        "--compare-dir",
        default=None,
        help="statistics mode: compare-seeds JSON directory for "
             "pivot-tested / pivot-excluded win coverage fields",
    )

    parser.add_argument(
        "--methods",
        default=None,
        help="methods mode: comma-separated benchmark blocks in column "
             "order; first is the focal method (default: tow,aco,heuristic)",
    )

    args = parser.parse_args(argv)

    if args.max_replicates is not None:
        try:
            set_max_replicates(args.max_replicates)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    if args.mode == "quality":
        json_paths = list_json_paths(args.directory)
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        ablation_dirs = None
        if args.ablation_dir:
            ablation_dirs = {}
            for item in args.ablation_dir:
                label, sep, path = item.partition("=")
                if not sep or not label.strip() or not path.strip():
                    raise SystemExit(
                        f"Invalid --ablation-dir entry {item!r}; use LABEL=DIR"
                    )
                ablation_dirs[label.strip()] = path.strip()
        from . import quality
        quality.run(json_paths, args.output, ablation_dirs=ablation_dirs)

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
        param_dirs = None
        if args.param_dir:
            param_dirs = {}
            for item in args.param_dir:
                label, sep, path = item.partition("=")
                if not sep or not label.strip() or not path.strip():
                    raise SystemExit(
                        f"Invalid --param-dir entry {item!r}; use LABEL=DIR"
                    )
                param_dirs[label.strip()] = path.strip()
        plot_names = (
            [p.strip() for p in args.plots.split(",") if p.strip()]
            if args.plots
            else []
        )
        multi_only = plot_names and all(
            p in (
                "flag-ablation",
                "flag-ablation-replicates",
                "flag-feasibility",
                "flag-feasibility-replicates",
                "k-sweep",
                "theta-sweep",
                "density-wins",
                "param-runtime",
                "budget",
                "pheromone-ablation",
            )
            for p in plot_names
        )
        trace_dirs = None
        if args.trace_dir:
            trace_dirs = {}
            for item in args.trace_dir:
                label, sep, path = item.partition("=")
                if not sep or not label.strip() or not path.strip():
                    raise SystemExit(
                        f"Invalid --trace-dir entry {item!r}; use LABEL=DIR"
                    )
                trace_dirs[label.strip()] = path.strip()
        if not json_paths and not multi_only:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import compare
        compare.run(
            json_paths,
            args.output,
            ants=args.ants,
            plots=args.plots,
            flag_dirs=flag_dirs,
            param_dirs=param_dirs,
            trace_dirs=trace_dirs,
            runtime_sweeps=_parse_runtime_sweeps(args.runtime_sweep),
        )

    elif args.mode == "ablation":
        from . import ablation
        ablation.run(args.directory, args.output, field=args.field)

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
            compare_dir=args.compare_dir,
        )

    elif args.mode == "methods":
        json_paths = list_json_paths(args.directory)
        if not json_paths:
            raise SystemExit(f"No .json files found in {args.directory}")
        from . import methods
        names = (
            [m.strip() for m in args.methods.split(",") if m.strip()]
            if args.methods
            else None
        )
        methods.run(json_paths, args.output, methods=names)

    else:
        raise SystemExit(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
