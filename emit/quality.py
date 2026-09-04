"""
quality mode — vary.jl ant-count format → groupplot:

  Top row (side by side):
  - mean % deviation from Cui θ-heuristic (feasible trials)
  - % theta-feasible trials

  Bottom row:
  - mean wall-clock time vs ant count

  Optional: mean % deviation from optimum is included when present
  (placed before the heuristic panel).
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict

from .common import (
    counted_trials,
    load_json,
    report_skipped,
    series_name,
    write_tex,
)

# Tukey fence multiplier for heuristic-panel outlier detection.
HEURISTIC_OUTLIER_IQR_K = 1.5


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
    """Cui θ-heuristic edge count from the top-level heuristic block."""
    heuristic = data.get("heuristic") or {}
    return heuristic.get("final_edges")


def summarize_file(data):
    """
    Returns
        (pct_by_ants, heur_pct_by_ants, feasible_pct_by_ants, time_by_ants)
    or None if this file has neither a usable optimum nor a heuristic baseline.

    pct_by_ants: {ants: mean % deviation from optimum}, feasible trials only.

    heur_pct_by_ants: {ants: mean % deviation from Cui heuristic}, feasible
                      trials only. Empty when the file has no heuristic block.

    feasible_pct_by_ants: {ants: % of trials at that ant count that were
                           theta_feasible == True}, over trials that had
                           either a usable optimum or a heuristic baseline.

    time_by_ants: {ants: mean wall_time_s}, over trials with usable quality
                  baselines.

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
        heur_pct = pct_deviation(final, heur_edges)

        # Need a non-zero baseline against which we can compute a % deviation.
        if pct is None and heur_pct is None:
            continue

        any_usable = True

        feasible = bool(t.get("theta_feasible"))
        feasible_flags[ants].append(feasible)

        if feasible and pct is not None:
            pct_vals[ants].append(pct)

        if feasible and heur_pct is not None:
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


def _series_lookup(series_dicts):
    return {name: xy for name, xy in series_dicts}


def _heuristic_series_max_pct(name_xy):
    """Largest mean % deviation from the Cui heuristic for one graph."""
    _, xy = name_xy
    if not xy:
        return None
    return max(xy.values())


def detect_heuristic_outliers(heuristic_series, *, iqr_k=HEURISTIC_OUTLIER_IQR_K):
    """
    Flag graphs whose peak Cui-heuristic deviation is a Tukey upper outlier.

    Uses the maximum mean deviation (over ant counts) per graph. Returns
    [(name, max_pct), ...] sorted by descending max_pct.
    """
    scored = []
    for name_xy in heuristic_series:
        peak = _heuristic_series_max_pct(name_xy)
        if peak is not None:
            scored.append((name_xy[0], peak))

    if len(scored) < 4:
        return []

    peaks = sorted(p for _, p in scored)
    q1, _, q3 = statistics.quantiles(peaks, n=4, method="exclusive")
    upper = q3 + iqr_k * (q3 - q1)

    outliers = [(name, peak) for name, peak in scored if peak > upper]
    outliers.sort(key=lambda item: item[1], reverse=True)
    return outliers


def filter_outlier_series(series, outlier_names):
    """Drop named series from a (name, data) list."""
    if not outlier_names:
        return series
    return [(name, data) for name, data in series if name not in outlier_names]


def _tex_escape_name(name):
    return (
        str(name)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _table_style_name(name):
    """Match emit.common.display_name / table Dataset column style."""
    return _tex_escape_name(
        str(name).replace("_", " ").replace("-", " ").title()
    )


def fmt_outlier_pct(value):
    """Format a deviation percentage for LaTeX prose."""
    if abs(value) >= 100:
        return f"{value:.0f}\\%"
    return f"{value:.1f}\\%"


def build_outlier_note(outliers):
    """
    LaTeX note listing graphs omitted from the quality groupplot.

    outliers: [(name, max_pct), ...] from detect_heuristic_outliers.
    Names use the same title-case style as table Dataset columns.
    """
    if not outliers:
        return None

    if len(outliers) == 1:
        name, peak = outliers[0]
        names_part = _table_style_name(name)
        values_part = fmt_outlier_pct(peak)
    else:
        parts = [
            rf"{_table_style_name(name)} ({fmt_outlier_pct(peak)})"
            for name, peak in outliers
        ]
        names_part = ", ".join(parts[:-1]) + ", and " + parts[-1]
        values_part = None

    if values_part is not None:
        body = (
            rf"{names_part} was omitted from all panels because its deviation "
            rf"from the Cui $\theta$-heuristic ({values_part}) was a statistical "
            rf"outlier (Tukey upper fence, $k={HEURISTIC_OUTLIER_IQR_K:g}$)."
        )
    else:
        body = (
            rf"{names_part} were omitted from all panels because their "
            rf"deviations from the Cui $\theta$-heuristic were statistical "
            rf"outliers (Tukey upper fence, $k={HEURISTIC_OUTLIER_IQR_K:g}$)."
        )

    return rf"\small\textit{{Note: {body}}}"


def _addplots(ordered_names, series_lookup):
    """
    Emit one \\addplot per name in ordered_names so pgfplots cycle lists
    stay aligned across panels even when a series is missing from a panel.
    Missing series get empty coordinates (no marks drawn).
    """
    lines = []

    for name in ordered_names:
        xy = series_lookup.get(name) or {}
        coords = " ".join(
            f"({x},{y:.4f})"
            for x, y in sorted(xy.items())
        )

        lines.append(r"\addplot coordinates {" + coords + "};")

    return lines


def build_combined_latex(
    quality_series,
    heuristic_series,
    feasible_series,
    time_series,
    *,
    outlier_note=None,
):
    """
    Build a 2-column groupplot from whichever panels have data:

      Top row (side by side):
      - Output compared to Cui Heuristic
      - Theta-feasibility rate

      Bottom row:
      - Run time (alone when present)

      Optional optimum-quality panel is included when present and shares
      the bottom row with runtime.

    Series order is the union of all panels so colors/markers stay
    consistent across panels.
    """

    ordered_names = []
    seen = set()
    for series in (
        quality_series,
        heuristic_series,
        feasible_series,
        time_series,
    ):
        for name, _ in series:
            if name not in seen:
                seen.add(name)
                ordered_names.append(name)

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
                    r"    ylabel={Dev. from Cui Heuristic (\%)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={Output compared to Cui Heuristic},",
                    r"    title style={font=\small},",
                ],
            }
        )

    if feasible_series:
        top_panels.append(
            {
                "lookup": feasible_lookup,
                "opts": [
                    r"    xlabel={Number of ants},",
                    r"    ylabel={Feasible trials (\%)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={$\theta$-feasibility rate},",
                    r"    title style={font=\small},",
                    r"    ymin=0,",
                    r"    ymax=100,",
                ],
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
                    r"    title={Solution quality (feasible trials only)},",
                    r"    title style={font=\small},",
                ],
            }
        )

    if time_series:
        bottom_panels.append(
            {
                "lookup": time_lookup,
                "opts": [
                    r"    xlabel={Number of ants},",
                    r"    ylabel={Mean WCT (s)},",
                    r"    ylabel style={align=center, font=\small},",
                    r"    title={Run time},",
                    r"    title style={font=\small},",
                    r"    ymode=log,",
                ],
            }
        )

    panels = top_panels + bottom_panels
    if not panels:
        raise ValueError("No panels to plot")

    # Side-by-side top row whenever both heuristic and feasibility exist.
    n_cols = 2 if len(top_panels) >= 2 else 1
    n_rows = (len(panels) + n_cols - 1) // n_cols
    width = 0.48 if n_cols == 2 else 0.75
    height = 0.36 if n_rows >= 2 else 0.44

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        (
            rf"    group style={{group size={n_cols} by {n_rows}, "
            r"horizontal sep=24pt, vertical sep=40pt},"
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
        lines += _addplots(ordered_names, panel["lookup"])

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"  \caption{ACO solution quality vs.\ the Cui $\theta$-heuristic, "
        r"$\theta$-feasibility rate, and mean wall-clock time vs.\ ant count "
        r"(first replicate per ant count omitted when multiple runs were "
        r"recorded, to exclude Julia JIT warmup).}",
        r"  \label{fig:quality-groupplot}",
    ]
    if outlier_note:
        lines += ["  \\medskip", f"  {outlier_note}"]
    lines.append(r"\end{figure}")

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

    outliers = detect_heuristic_outliers(heuristic_series)
    outlier_names = {name for name, _ in outliers}
    if outlier_names:
        quality_series = filter_outlier_series(quality_series, outlier_names)
        heuristic_series = filter_outlier_series(heuristic_series, outlier_names)
        feasible_series = filter_outlier_series(feasible_series, outlier_names)
        time_series = filter_outlier_series(time_series, outlier_names)

    combined_tex = build_combined_latex(
        quality_series,
        heuristic_series,
        feasible_series,
        time_series,
        outlier_note=build_outlier_note(outliers),
    )

    write_tex(combined_tex, output)
    if outliers:
        removed = ", ".join(
            f"{name} ({peak:.1f}%)" for name, peak in outliers
        )
        print(
            f"# quality: omitted {len(outliers)} heuristic outlier(s): {removed}",
            file=sys.stderr,
        )
    report_skipped(skipped)
