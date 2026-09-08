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

  iteration-budget
    - 3-panel groupplot of $F_N$ quality vs. θ-heuristic (as a
      percentage of graphs), log-scaled percent edge increase vs.
      the θ-heuristic, and the epochs-to-best CDF as the credited
      epoch budget T varies from 1 to T_max. Each JSON trial is a
      full-budget replicate; T is a retrospective ETB ≤ T credit
      rule (not a separately measured shorter run). All three
      panels are per-graph; the CDF uses each graph's earliest ETB
      among counted replicates that match its best final_edges.

  replicate-budget
    - Companion 3-panel groupplot as the credited replicate budget R
      varies from 1 to R_max (first R counted non-JIT seeds, ordered
      by run). Left/middle/CDF are all per-graph; CDF is
      replicates-to-best. JIT warmup (run 1) is never included.

  k-sweep
    - Edge ratio (log) and ACO win rate across defect budgets k at
      fixed θ. Requires --param-dir for each (k, θ) suite (build.json
      param_dirs).

  theta-sweep
    - Same layout across minimum side sizes θ at fixed k.

  density-wins
    - Rolling ACO win rate vs. reduced density, pooled over all
      graph×suite observations across $(k,\theta)$.

  param-density
    - Reduced-density boxplots: vary k at fixed θ | vary θ at fixed k.

  param-runtime
    - 2×2 groupplot: ACO/θ discovery-time ratio and absolute ACO
      discovery time vs k (fixed θ) and vs θ (fixed k). Timed-out
      graphs are excluded from the matched boxes.

ACO points use the same best-trial / discovery rules as table mode.

Layout
------
  helpers.py            shared metrics and pgfplots primitives
  complexity.py         time / complexity scaling figures
  flag_ablation.py      F_P/F_N flag ablation figure
  iteration_budget.py   epoch-budget / ETB figure
  replicate_budget.py   replicate-budget / RTB figure
  param_sweep.py        k / θ sweep figures
"""

from __future__ import annotations

import sys

from ..common import display_name, load_json, report_skipped, write_tex
from ..result_fields import validate_compare_directory
from ..table import summarize_file
from . import (
    complexity,
    flag_ablation,
    iteration_budget,
    param_sweep,
    replicate_budget,
)

PLOT_GROUPS = (
    complexity.PLOT_GROUPS
    + flag_ablation.PLOT_GROUPS
    + iteration_budget.PLOT_GROUPS
    + replicate_budget.PLOT_GROUPS
    + param_sweep.PLOT_GROUPS
)
PARAM_SWEEP_PLOTS = frozenset(param_sweep.PLOT_GROUPS)
FLAG_ABLATION_PLOTS = frozenset(flag_ablation.PLOT_GROUPS)
MULTI_DIR_PLOTS = FLAG_ABLATION_PLOTS | PARAM_SWEEP_PLOTS
BUDGET_PLOTS = frozenset(
    iteration_budget.PLOT_GROUPS + replicate_budget.PLOT_GROUPS
)


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
    replicate_budget_summary=None,
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
        **replicate_budget.BUILDERS,
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
        elif name in replicate_budget.BUILDERS:
            block = builders[name](replicate_budget_summary)
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
    repl_summary = None
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

    if "iteration-budget" in selected:
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
                f"wins@T={dict(zip(iter_summary['budgets'], iter_summary['wins']))}",
                file=sys.stderr,
            )

    if "replicate-budget" in selected:
        repl_ants = ants if ants is not None else replicate_budget.DEFAULT_ANTS
        repl_summary, repl_skipped = replicate_budget.summarize_replicate_budget(
            json_paths, ants=repl_ants
        )
        skipped.extend(repl_skipped)
        if repl_summary is None:
            print(
                "Warning: replicate-budget: no usable graphs.",
                file=sys.stderr,
            )
        else:
            print(
                f"# replicate-budget: {repl_summary['n_graphs']} graph(s), "
                f"{repl_summary['n_replicates']} counted replicate(s), "
                f"ants={repl_ants}; "
                f"wins@R={dict(zip(repl_summary['budgets'], repl_summary['wins']))}",
                file=sys.stderr,
            )

    named_rows = []
    needs_single_dir = any(
        p not in MULTI_DIR_PLOTS and p not in BUDGET_PLOTS
        for p in selected
    )
    if any(p in BUDGET_PLOTS for p in selected) and not json_paths:
        raise SystemExit(
            "iteration-budget / replicate-budget require a vary.jl "
            "JSON directory (e.g. vary_k2t5i_PN)"
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
    elif any(p in BUDGET_PLOTS for p in selected):
        # Still warn on incomplete graph fields when only budget plots
        # are requested.
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
            f"; iteration-budget graphs={iter_summary['n_graphs']}"
            if iter_summary is not None
            else ""
        )
        + (
            f"; replicate-budget graphs={repl_summary['n_graphs']}"
            if repl_summary is not None
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
            replicate_budget_summary=repl_summary,
        ),
        output,
    )
    report_skipped(skipped)
