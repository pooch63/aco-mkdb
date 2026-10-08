"""
compare mode — vary.jl ant-count JSON → complexity figures.

Pass a single results directory (e.g. ``vary_k2t5i_PN``), like table or
statistics mode. Select plot groups with ``--plots`` (comma-separated):

  theta-time
    - θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    - 2-panel log--log discovery-time scaling (matched graphs);
      left: raw t vs n; right: t vs candidate bounds
      (naive n^2; practical T·n+m with T=5)

  bound-time
    - 2-panel log--log normalized discovery time t/(n_S·T):
      left vs naive n^2; right vs practical T·n+m

  density-size
    - edge density vs |E(D_{best})|

  max-deg-time
    - max reduced degree vs $F_N$ discovery time

  flag-ablation
    - 2-panel groupplot (quality | discovery time) across ACO, $F_P$,
      $F_N$, and $F_{PN}$ at a fixed ant count. Requires --flag-dir for
      each variant (see build.json flag_dirs). Quality averages scored
      edges (θ-infeasible trials count as 0). Ordered/colored by N.

  flag-ablation-replicates
    - Same layout as flag-ablation, but each boxplot observation is a
      counted replicate (scored |E| and wall_time_s) rather than a
      per-graph mean / discovery sum.

  flag-feasibility
    - 1-panel bar chart: % of matched graphs with ≥1 θ-feasible counted
      replicate, across the same four variants (ordered ACO, $F_P$,
      $F_N$, $F_{PN}$; grouped by $F_N$).

  flag-feasibility-replicates
    - Same layout, but the rate is % of counted replicates that are
      θ-feasible (pooled over matched graphs).

  budget
    - Quality-style 2×2. Median percent deviation from the θ-heuristic
      (θ-infeasible = −100%) and median θ-feasibility, versus epoch
      budget T and replicate count R.

  pheromone-ablation
    - Anytime median with pheromone on, against Δτ=0 and any ρ
      sweep passed via --trace-dir. Same ant count.

  k-sweep
    - Compact captionless pair: ACO win rate versus k at fixed θ, and
      versus θ at fixed k, side by side in the text (not a float).
      Requires --param-dir for each (k, θ) suite (build.json param_dirs).

  theta-sweep
    - Included as the right panel of k-sweep. Emits a comment only.

  density-wins
    - One column. ACO win rate versus k at fixed θ, and versus θ at
      fixed k. At each value, three bars are equal-count thirds of
      reduced density (light to dark).

  param-runtime
    - Log-scale runtime on four instances. Sweeps come from
      runtime_sweeps in build.json: each ``k`` entry is time versus
      k at one fixed θ, and each ``theta`` entry is time versus θ at
      one fixed k. The axis uses every completed suite at that fixed
      value, and each panel spans only the range that graph has.
      A legend under each 2×2 names the sweep color and the marker.
      Timed-out graphs are excluded.

ACO points use the same best-trial / discovery rules as table mode.

Layout
------
  helpers.py            shared metrics and pgfplots primitives
  complexity.py         time / complexity scaling figures
  flag_ablation.py      F_P/F_N flag ablation figure
  budget_curves.py      epoch/replicate budget and pheromone control
  param_sweep.py        k / θ sweep figures
"""

from __future__ import annotations

import sys

from ..common import (
    display_name,
    list_json_paths,
    load_json,
    report_skipped,
    write_tex,
)
from ..result_fields import validate_compare_directory
from ..table import summarize_file
from . import (
    budget_curves,
    complexity,
    flag_ablation,
    param_sweep,
)

PLOT_GROUPS = (
    complexity.PLOT_GROUPS
    + flag_ablation.PLOT_GROUPS
    + budget_curves.PLOT_GROUPS
    + param_sweep.PLOT_GROUPS
)
PARAM_SWEEP_PLOTS = frozenset(param_sweep.PLOT_GROUPS)
FLAG_ABLATION_PLOTS = frozenset(flag_ablation.PLOT_GROUPS)
MULTI_DIR_PLOTS = FLAG_ABLATION_PLOTS | PARAM_SWEEP_PLOTS
BUDGET_PLOTS = frozenset(budget_curves.PLOT_GROUPS)


def _parse_plots(plots):
    """Validate and normalize a comma-separated plot-group list."""
    if plots is None:
        return list(PLOT_GROUPS)
    selected = []
    for raw in plots.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in PLOT_GROUPS:
            raise SystemExit(
                f"Unknown compare plot group {name!r}; "
                f"choose from: {', '.join(PLOT_GROUPS)}"
            )
        if name not in selected:
            selected.append(name)
    if not selected:
        raise SystemExit("No compare plot groups selected.")
    return selected


def build_compare_plots(
    named_rows,
    plots=None,
    flag_ablation_matched=None,
    param_by_label=None,
    param_meta=None,
    budget_graphs=None,
    ablation_suites=None,
    runtime_sweeps=None,
):
    """
    Build selected compare figures from [(name, row), ...].

    Plot groups are listed in ``PLOT_GROUPS``; pass ``plots`` as a
    comma-separated subset (e.g. ``theta-time,deg-size-time``).
    """
    selected = _parse_plots(plots)
    builders = {
        **complexity.BUILDERS,
        **flag_ablation.BUILDERS,
        **budget_curves.BUILDERS,
        **param_sweep.BUILDERS,
    }

    parts = []
    for name in selected:
        if name in complexity.BUILDERS:
            block = builders[name](named_rows)
        elif name in flag_ablation.BUILDERS:
            block = builders[name](flag_ablation_matched or [])
        elif name in budget_curves.BUILDERS:
            block = builders[name](budget_graphs or [], ablation_suites or {})
        elif name == "param-runtime":
            block = builders[name](
                param_by_label or {},
                param_meta or {},
                sweeps=runtime_sweeps,
            )
        else:
            block = builders[name](param_by_label or {}, param_meta or {})
        if not block:
            continue
        if parts:
            parts.append("")
        parts += block

    return "\n".join(parts)


def run(json_paths, output, ants=None, plots=None, flag_dirs=None, param_dirs=None,
        trace_dirs=None, runtime_sweeps=None):
    selected = _parse_plots(plots) if plots else list(PLOT_GROUPS)
    matched_ablation = None
    param_by_label = None
    param_meta = None
    budget_graphs = None
    ablation_suites = None
    skipped = []

    if FLAG_ABLATION_PLOTS.intersection(selected):
        if not flag_dirs:
            raise SystemExit(
                "flag-ablation / flag-feasibility (and *-replicates) "
                "plots require "
                "--flag-dir=LABEL=DIR for ACO, $F_P$, $F_N$, and $F_{PN}$"
            )
        matched_ablation, flag_skipped = (
            flag_ablation.load_flag_ablation_matched(flag_dirs, ants=ants)
        )
        skipped.extend(flag_skipped)
        if not matched_ablation:
            print(
                "Warning: flag-ablation / flag-feasibility: no graphs "
                "matched across all four variant directories.",
                file=sys.stderr,
            )

    if PARAM_SWEEP_PLOTS.intersection(selected):
        param_by_label, param_meta, param_skipped = param_sweep.load_param_dirs(
            param_dirs or {}, ants=ants
        )
        skipped.extend(param_skipped)
        if not param_meta:
            print(
                "Warning: param-sweep plots: no usable param directories.",
                file=sys.stderr,
            )

    if BUDGET_PLOTS.intersection(selected):
        budget_ants = ants if ants is not None else budget_curves.DEFAULT_ANTS
        budget_graphs, budget_skipped = budget_curves.load_suite(
            json_paths, ants=budget_ants
        )
        skipped.extend(budget_skipped)
        if not budget_graphs:
            print(
                "Warning: budget curves: no epoch_trace "
                f"(ants={budget_ants}).",
                file=sys.stderr,
            )
        else:
            print(
                "# budget curves: "
                + budget_curves.summarize_for_log(budget_graphs)
                + f", ants={budget_ants}",
                file=sys.stderr,
            )
        ablation_suites = {}
        if "pheromone-ablation" in selected:
            for label, path in (trace_dirs or {}).items():
                suite, trace_skipped = budget_curves.load_suite(
                    list_json_paths(path), ants=budget_ants
                )
                skipped.extend(trace_skipped)
                if suite:
                    ablation_suites[label] = suite
                    print(
                        f"# pheromone-ablation {label}: "
                        + budget_curves.summarize_for_log(suite),
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Warning: pheromone-ablation: no epoch_trace in {path}.",
                        file=sys.stderr,
                    )

    named_rows = []
    needs_single_dir = any(
        p not in MULTI_DIR_PLOTS and p not in BUDGET_PLOTS
        for p in selected
    )
    if needs_single_dir and json_paths:
        for path in json_paths:
            data = load_json(path)
            if data is None:
                skipped.append((path, "unreadable"))
                continue
            row = summarize_file(data, ants=ants)
            if row is None:
                skipped.append((path, "not a vary.jl ant-count result"))
                continue
            named_rows.append((display_name(path, data), row))

        if not named_rows and not any(p in BUDGET_PLOTS for p in selected):
            raise SystemExit("No usable vary JSON files -- nothing to plot.")

        if named_rows:
            validate_compare_directory(json_paths, selected)
    elif any(p in BUDGET_PLOTS for p in selected) and json_paths:
        validate_compare_directory(json_paths, selected)

    print(
        f"# compare: {len(named_rows)} dataset(s)"
        + (f"; ants={ants}" if ants is not None else "")
        + (
            f"; flag-ablation matched={len(matched_ablation or [])}"
            if FLAG_ABLATION_PLOTS.intersection(selected)
            else ""
        )
        + (
            f"; param-dirs={len(param_meta or {})}"
            if PARAM_SWEEP_PLOTS.intersection(selected)
            else ""
        )
        + (
            f"; budget-curves graphs={len(budget_graphs)}"
            if budget_graphs
            else ""
        ),
        file=sys.stderr,
    )

    write_tex(
        build_compare_plots(
            named_rows,
            plots=plots,
            flag_ablation_matched=matched_ablation,
            param_by_label=param_by_label,
            param_meta=param_meta,
            budget_graphs=budget_graphs,
            ablation_suites=ablation_suites,
            runtime_sweeps=runtime_sweeps,
        ),
        output,
    )
    report_skipped(skipped)
