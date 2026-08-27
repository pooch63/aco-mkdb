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
from collections import defaultdict

from .common import load_json, report_skipped, series_name, write_tex


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

    time_by_ants: {ants: mean wall_time_s}, over the same trial set as
                  feasible_pct_by_ants.
    """
    trials = data.get("trials", [])
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

    combined_tex = build_combined_latex(
        quality_series,
        heuristic_series,
        feasible_series,
        time_series,
    )

    write_tex(combined_tex, output)
    report_skipped(skipped)
