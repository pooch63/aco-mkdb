"""
compare mode — vary.jl ant-count JSON → complexity figures.

Pass a single results directory (e.g. ``vary_k2t5i_PN``), like table or
statistics mode. Select plot groups with ``--plots`` (comma-separated):

  theta-time
    - θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    - ratio of discovery time to each bound vs |U_R|+|V_R| (matched graphs)

  density-size
    - edge density vs |E(D*)|

  max-deg-time
    - max reduced degree vs ACO-PN discovery time

  flag-ablation
    - 2-panel groupplot (quality | discovery time) across ACO, ACO-P,
      ACO-N, and ACO-PN at a fixed ant count. Requires --flag-dir for
      each variant (see build.json flag_dirs).

  iteration-budget
    - 2-panel groupplot of ACO-PN quality vs. θ-heuristic (as a
      percentage of graphs) and the iterations-to-best CDF as the
      epoch budget I varies from 1 to n_E (uses recorded ITB on
      counted non-JIT replicates).

  k-sweep
    - Edge ratio (log) and ACO win rate across defect budgets k at
      fixed θ. Requires --param-dir for each (k, θ) suite (build.json
      param_dirs).

  theta-sweep
    - Same layout across minimum side sizes θ at fixed k.

  param-runtime
    - Discovery-time boxplots: vary k at fixed θ | vary θ at fixed k.

ACO points use the same best-trial / discovery rules as table mode.

Layout
------
  helpers.py            shared metrics and pgfplots primitives
  complexity.py         time / complexity scaling figures
  flag_ablation.py      P/N flag ablation figure
  iteration_budget.py   epoch-budget / ITB figure
  param_sweep.py        k / θ sweep figures
"""

from __future__ import annotations

import sys

from ..common import display_name, load_json, report_skipped, write_tex
from ..result_fields import validate_compare_directory
from ..table import summarize_file
from . import complexity, flag_ablation, iteration_budget, param_sweep

PLOT_GROUPS = (
    complexity.PLOT_GROUPS
    + flag_ablation.PLOT_GROUPS
    + iteration_budget.PLOT_GROUPS
    + param_sweep.PLOT_GROUPS
)
PARAM_SWEEP_PLOTS = frozenset(param_sweep.PLOT_GROUPS)
MULTI_DIR_PLOTS = frozenset({"flag-ablation"}) | PARAM_SWEEP_PLOTS
ITERATION_BUDGET_PLOTS = frozenset(iteration_budget.PLOT_GROUPS)


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
    iteration_budget_summary=None,
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
        **iteration_budget.BUILDERS,
        **param_sweep.BUILDERS,
    }

    parts = []
    for name in selected:
        if name in complexity.BUILDERS:
            block = builders[name](named_rows)
        elif name in flag_ablation.BUILDERS:
            block = builders[name](flag_ablation_matched or [])
        elif name in iteration_budget.BUILDERS:
            block = builders[name](iteration_budget_summary)
        else:
            block = builders[name](param_by_label or {}, param_meta or {})
        if not block:
            continue
        if parts:
            parts.append("")
        parts += block

    return "\n".join(parts)


def run(json_paths, output, ants=None, plots=None, flag_dirs=None, param_dirs=None):
    selected = _parse_plots(plots) if plots else list(PLOT_GROUPS)
    matched_ablation = None
    param_by_label = None
    param_meta = None
    iter_summary = None
    skipped = []

    if "flag-ablation" in selected:
        if not flag_dirs:
            raise SystemExit(
                "flag-ablation plot requires --flag-dir=LABEL=DIR for "
                "ACO, ACO-P, ACO-N, and ACO-PN"
            )
        matched_ablation, flag_skipped = (
            flag_ablation.load_flag_ablation_matched(flag_dirs, ants=ants)
        )
        skipped.extend(flag_skipped)
        if not matched_ablation:
            print(
                "Warning: flag-ablation: no graphs matched across all four "
                "variant directories.",
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

    if ITERATION_BUDGET_PLOTS.intersection(selected):
        iter_ants = ants if ants is not None else iteration_budget.DEFAULT_ANTS
        iter_summary, iter_skipped = iteration_budget.summarize_iteration_budget(
            json_paths, ants=iter_ants
        )
        skipped.extend(iter_skipped)
        if iter_summary is None:
            print(
                "Warning: iteration-budget: no usable graphs.",
                file=sys.stderr,
            )
        else:
            print(
                f"# iteration-budget: {iter_summary['n_graphs']} graph(s), "
                f"{iter_summary['n_trials']} trial(s), ants={iter_ants}; "
                f"wins@I={dict(zip(iter_summary['budgets'], iter_summary['wins']))}",
                file=sys.stderr,
            )

    named_rows = []
    needs_single_dir = any(
        p not in MULTI_DIR_PLOTS and p not in ITERATION_BUDGET_PLOTS
        for p in selected
    )
    # iteration-budget still needs the vary directory JSON paths
    if any(p in ITERATION_BUDGET_PLOTS for p in selected) and not json_paths:
        raise SystemExit(
            "iteration-budget requires a vary.jl JSON directory "
            "(e.g. vary_k2t5i_PN)"
        )
    if needs_single_dir:
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

        if not named_rows:
            raise SystemExit("No usable vary JSON files -- nothing to plot.")

        validate_compare_directory(json_paths, selected)
    elif any(p in ITERATION_BUDGET_PLOTS for p in selected):
        # Still warn on incomplete graph fields when only this plot is requested.
        validate_compare_directory(json_paths, selected)

    print(
        f"# compare: {len(named_rows)} dataset(s)"
        + (f"; ants={ants}" if ants is not None else "")
        + (
            f"; flag-ablation matched={len(matched_ablation or [])}"
            if "flag-ablation" in selected
            else ""
        )
        + (
            f"; param-dirs={len(param_meta or {})}"
            if PARAM_SWEEP_PLOTS.intersection(selected)
            else ""
        )
        + (
            f"; iteration-budget graphs={iter_summary['n_graphs']}"
            if iter_summary is not None
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
            iteration_budget_summary=iter_summary,
        ),
        output,
    )
    report_skipped(skipped)
