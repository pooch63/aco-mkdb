"""
quality mode — vary.jl ant-count format → groupplot:

  Top row (side by side):
  - % deviation from θ-heuristic (all graphs; θ-infeasible → −100%):
    Q1 / median / Q3
  - % theta-feasible graphs: Q1 / median / Q3

  Bottom row:
  - wall-clock time vs ant count: Q1 / median / Q3

  Optional: % deviation from optimum is included when present
  (placed before the heuristic panel), also as IQR summary lines.
  Optimum panel still uses θ-feasible trials only.

  Each panel is per-graph: one value per graph at each ant count
  (mean over that graph's counted replicates), then Q1 / median / Q3
  across graphs. Not a pool of individual replicate trials.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict

from .common import (
    aco_timed_out,
    counted_trials,
    load_json,
    report_skipped,
    series_name,
    write_tex,
)

# Display order and pgfplots style for the three IQR summary curves.
IQR_SERIES = (
    ("1st quartile", "dashed, mark=triangle*, blue!55!black"),
    ("Median", "solid, thick, mark=*, blue!80!black"),
    ("3rd quartile", "dashed, mark=square*, blue!55!black"),
)


def pct_deviation(final_edges, baseline_edges):
    """Percent change of final_edges relative to baseline_edges.

    Positive means more edges than the baseline. Returns None when the
    baseline is missing or zero (division undefined).
    """
    if baseline_edges is None or baseline_edges == 0:
        return None
    return 100.0 * (final_edges - baseline_edges) / baseline_edges


def file_level_optimal(data):
    """Resolve optimum edge count from the vary.jl JSON shape."""
    if data.get("optimal_edges") is not None:
        return data["optimal_edges"]

    pivot = data.get("pivot") or {}
    if pivot.get("optimal_edges") is not None:
        return pivot["optimal_edges"]

    heuristic = data.get("heuristic") or {}
    return heuristic.get("optimal_edges")


def heuristic_baseline(data):
    """θ-heuristic edge count from the top-level heuristic block."""
    heuristic = data.get("heuristic") or {}
    return heuristic.get("final_edges")


def summarize_file(data):
    """
    Returns
        (pct_by_ants, heur_pct_by_ants, feasible_pct_by_ants, time_by_ants)
    or None if this file has neither a usable optimum nor a heuristic baseline.

    pct_by_ants: {ants: mean % deviation from optimum}, feasible trials only.

    heur_pct_by_ants: {ants: mean % deviation from θ-heuristic}. θ-feasible
                      counted replicates use the usual relative edge change;
                      θ-infeasible ones contribute −100%. Empty when the
                      file has no heuristic block. One value per graph.

    feasible_pct_by_ants: {ants: % of this graph's counted replicates that
                           were theta_feasible == True}. One value per
                           graph (0 or 100 when a single replicate is kept).

    time_by_ants: {ants: mean wall_time_s}, over counted replicates with
                  usable quality baselines. One value per graph.

    When aco_runs > 1, run 1 at each ant count is omitted from every panel
    (Julia JIT on the first measured replicate).
    """
    trials = counted_trials(data.get("trials") or [], data)
    file_level_opt = file_level_optimal(data)
    heur_edges = heuristic_baseline(data)

    pct_vals = defaultdict(list)
    heur_pct_vals = defaultdict(list)
    feasible_flags = defaultdict(list)
    time_vals = defaultdict(list)

    any_usable = False

    for t in trials:
        optimal = t.get("optimal_edges", file_level_opt)
        final = t.get("final_edges")
        ants = t.get("ants")

        if final is None or ants is None:
            continue

        pct = pct_deviation(final, optimal)
        feasible = bool(t.get("theta_feasible"))
        if heur_edges is None:
            heur_pct = None
        elif not feasible:
            # Keep infeasible runs in the θ-heuristic panel as a full miss.
            heur_pct = -100.0
        else:
            heur_pct = pct_deviation(final, heur_edges)

        # Need a non-zero baseline against which we can compute a % deviation.
        if pct is None and heur_pct is None:
            continue

        any_usable = True
        feasible_flags[ants].append(feasible)

        if feasible and pct is not None:
            pct_vals[ants].append(pct)

        if heur_pct is not None:
            heur_pct_vals[ants].append(heur_pct)

        wall_time = t.get("wall_time_s")
        if wall_time is not None:
            time_vals[ants].append(wall_time)

    if not any_usable:
        return None

    pct_by_ants = {
        ants: statistics.mean(vals)
        for ants, vals in pct_vals.items()
        if vals
    }

    heur_pct_by_ants = {
        ants: statistics.mean(vals)
        for ants, vals in heur_pct_vals.items()
        if vals
    }

    feasible_pct_by_ants = {
        ants: 100.0 * sum(flags) / len(flags)
        for ants, flags in feasible_flags.items()
    }

    time_by_ants = {
        ants: statistics.mean(vals)
        for ants, vals in time_vals.items()
        if vals
    }

    return pct_by_ants, heur_pct_by_ants, feasible_pct_by_ants, time_by_ants


def _quartile_triple(values):
    """Return (Q1, median, Q3) for a list of numbers."""
    if not values:
        return None
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    q1, _, q3 = statistics.quantiles(values, n=4, method="exclusive")
    return float(q1), float(statistics.median(values)), float(q3)


def aggregate_iqr_series(per_graph_series):
    """
    Collapse [(name, {ants: value}), ...] into three IQR curves.

    At each ant count, take the cross-graph distribution of per-graph
    means and return Q1 / median / Q3 as named series matching IQR_SERIES.
    """
    by_ants = defaultdict(list)
    for _, xy in per_graph_series:
        for ants, val in xy.items():
            if val is None:
                continue
            by_ants[ants].append(float(val))

    q1_xy = {}
    med_xy = {}
    q3_xy = {}
    for ants, vals in by_ants.items():
        triple = _quartile_triple(vals)
        if triple is None:
            continue
        q1, med, q3 = triple
        q1_xy[ants] = q1
        med_xy[ants] = med
        q3_xy[ants] = q3

    if not q1_xy:
        return []

    return [
        (IQR_SERIES[0][0], q1_xy),
        (IQR_SERIES[1][0], med_xy),
        (IQR_SERIES[2][0], q3_xy),
    ]


def _series_lookup(series_dicts):
    return {name: xy for name, xy in series_dicts}


def _addplots(ordered_names, series_lookup, *, with_legend=False):
    """
    Emit one \\addplot per IQR curve. Styles come from IQR_SERIES so
    Q1/Q3 stay dashed and the median is solid.
    """
    style_by_name = {name: style for name, style in IQR_SERIES}
    lines = []

    for name in ordered_names:
        xy = series_lookup.get(name) or {}
        coords = " ".join(
            f"({x},{y:.4f})"
            for x, y in sorted(xy.items())
        )
        style = style_by_name.get(name, "")
        opt = f"[{style}]" if style else ""
        lines.append(rf"\addplot{opt} coordinates {{{coords}}};")
        if with_legend:
            lines.append(rf"\addlegendentry{{{name}}}")

    return lines


def build_combined_latex(
    quality_series,
    heuristic_series,
    feasible_series,
    time_series,
):
    """
    Build a 2-column groupplot from whichever panels have data:

      Top row (side by side):
      - Output compared to θ-heuristic (IQR across graphs)
      - Theta-feasibility rate (IQR across graphs)

      Bottom row:
      - Run time (IQR across graphs)

      Optional optimum-quality panel is included when present and shares
      the bottom row with runtime. All panels use one value per graph.
    """

    ordered_names = [name for name, _ in IQR_SERIES]

    quality_lookup = _series_lookup(quality_series)
    heuristic_lookup = _series_lookup(heuristic_series)
    feasible_lookup = _series_lookup(feasible_series)
    time_lookup = _series_lookup(time_series)

    # Top row: heuristic quality | feasibility. Bottom row: runtime.
    # Optimum quality (rare) shares the bottom row with runtime when present.
    top_panels = []
    bottom_panels = []

    if heuristic_series:
        top_panels.append(
            {
                "lookup": heuristic_lookup,
                "opts": [
                    r"    xlabel={Number of ants},",
                    r"    ylabel={Dev. from $\theta$-heuristic (\%)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={Output compared to $\theta$-heuristic},",
                    r"    title style={font=\small},",
                    r"    legend pos=north west,",
                    r"    legend style={font=\footnotesize, cells={anchor=west}},",
                ],
                "legend": True,
            }
        )

    if feasible_series:
        top_panels.append(
            {
                "lookup": feasible_lookup,
                "opts": [
                    r"    xlabel={Number of ants},",
                    r"    ylabel={Feasible graphs (\%)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={$\theta$-feasibility rate},",
                    r"    title style={font=\small},",
                    r"    ymin=0,",
                    r"    ymax=100,",
                ],
                "legend": False,
            }
        )

    if quality_series:
        bottom_panels.append(
            {
                "lookup": quality_lookup,
                "opts": [
                    r"    xlabel={Number of ants},",
                    r"    ylabel={Dev. from optimum (\%)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={Solution quality (feasible graphs only)},",
                    r"    title style={font=\small},",
                ],
                "legend": False,
            }
        )

    if time_series:
        bottom_panels.append(
            {
                "lookup": time_lookup,
                "opts": [
                    r"    xlabel={Number of ants},",
                    r"    ylabel={WCT (s)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={Run time},",
                    r"    title style={font=\small},",
                    r"    ymode=log,",
                ],
                "legend": False,
            }
        )

    panels = top_panels + bottom_panels
    if not panels:
        raise ValueError("No panels to plot")

    # Side-by-side top row whenever both heuristic and feasibility exist.
    # Keep the multi-row height modest so [!tp] can sit at the top of a
    # page with following Results text underneath (avoids a float-only page).
    n_cols = 2 if len(top_panels) >= 2 else 1
    n_rows = (len(panels) + n_cols - 1) // n_cols
    # Slightly narrower panels leave room for larger horizontal sep.
    width = 0.44 if n_cols == 2 else 0.75
    height = 0.28 if n_rows >= 2 else 0.44
    hsep = 48 if n_cols == 2 else 28
    vsep = 52 if n_rows >= 2 else 48

    lines = [
        # Prefer top of the next page so leftover space can hold §4.2+ text
        # instead of a centered float-only page behind \FloatBarrier.
        r"\begin{figure}[!tp]",
        r"  \centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        (
            rf"    group style={{group size={n_cols} by {n_rows}, "
            rf"horizontal sep={hsep}pt, vertical sep={vsep}pt}},"
        ),
        r"    title style={yshift=-3pt},",
        rf"    width={width:.2f}\textwidth,",
        rf"    height={height:.2f}\textwidth,",
        r"    xmode=log,",
        r"    log basis x=2,",
        r"    grid=major,",
        r"]",
    ]

    for panel in panels:
        lines.append(r"\nextgroupplot[")
        lines.extend(panel["opts"])
        lines.append(r"]")
        lines += _addplots(
            ordered_names,
            panel["lookup"],
            with_legend=panel.get("legend", False),
        )

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"  \caption{$F_N$ solution quality vs.\ the $\theta$-heuristic, "
        r"$\theta$-feasibility rate, and wall-clock time vs.\ ant count. "
        r"Each panel shows the cross-graph first quartile, median, and "
        r"third quartile of per-graph values (not pooled replicate "
        r"trials; JIT warmup omitted). $\theta$-infeasible outcomes count "
        r"as $-100\%$ deviation from the $\theta$-heuristic. Quality and "
        r"feasibility generally rise with colony size but often plateau "
        r"before the largest $n_S$; runtime scales roughly linearly.}",
        r"  \label{fig:quality-groupplot}",
        r"\end{figure}",
    ]

    return "\n".join(lines)


def run(json_paths, output):
    quality_series = []
    heuristic_series = []
    feasible_series = []
    time_series = []
    skipped = []

    for path in json_paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue
        if aco_timed_out(data):
            limit = data.get("aco_timeout_s")
            reason = "aco timed out"
            if limit is not None:
                reason = f"aco timed out ({limit}s)"
            skipped.append((path, reason))
            continue

        result = summarize_file(data)

        if result is None:
            skipped.append((path, "no usable optimum or heuristic"))
            continue

        (
            pct_by_ants,
            heur_pct_by_ants,
            feasible_pct_by_ants,
            time_by_ants,
        ) = result
        name = series_name(path, data)

        if pct_by_ants:
            quality_series.append((name, pct_by_ants))

        if heur_pct_by_ants:
            heuristic_series.append((name, heur_pct_by_ants))

        if feasible_pct_by_ants:
            feasible_series.append((name, feasible_pct_by_ants))

        if time_by_ants:
            time_series.append((name, time_by_ants))

    if (
        not quality_series
        and not heuristic_series
        and not feasible_series
        and not time_series
    ):
        raise SystemExit(
            "No files had a usable optimum or heuristic -- nothing to plot."
        )

    quality_iqr = aggregate_iqr_series(quality_series)
    heuristic_iqr = aggregate_iqr_series(heuristic_series)
    feasible_iqr = aggregate_iqr_series(feasible_series)
    time_iqr = aggregate_iqr_series(time_series)

    combined_tex = build_combined_latex(
        quality_iqr,
        heuristic_iqr,
        feasible_iqr,
        time_iqr,
    )

    write_tex(combined_tex, output)
    n_graphs = max(
        len(quality_series),
        len(heuristic_series),
        len(feasible_series),
        len(time_series),
    )
    print(
        f"# quality: IQR summary over {n_graphs} graph(s) "
        f"(Q1 / median / Q3 per panel)",
        file=sys.stderr,
    )
    report_skipped(skipped)
