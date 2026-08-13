#!/usr/bin/env python3
"""
Scan a directory of ACO benchmark JSON files and emit pgfplots LaTeX:

  A 1x3 groupplot:
       - top:    mean % deviation from optimum, computed ONLY over trials
                 that were theta-feasible.
       - middle: % of trials at that ant count that were infeasible.
       - bottom: mean wall-clock time vs. ant count.

Files with no optimum specified anywhere (per-trial or file-level
"optimal_edges") are skipped entirely.

Usage:
    python make_pct_optimum_plot.py /path/to/json/dir -o plot.tex
        -> writes plot.tex

    python make_pct_optimum_plot.py /path/to/json/dir
        -> prints LaTeX to stdout
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def pct_from_optimum(final_edges, optimal_edges):
    if not optimal_edges:
        return None
    return 100.0 * (final_edges - optimal_edges) / optimal_edges


def summarize_file(data):
    """
    Returns (pct_by_ants, infeasible_pct_by_ants, time_by_ants) or None if
    no trial in this file has a usable optimum.

    pct_by_ants: {ants: mean % deviation from optimum}, feasible trials only.

    infeasible_pct_by_ants: {ants: % of trials at that ant count that were
                             theta_feasible == False}, computed over ALL
                             trials with a usable optimum.

    time_by_ants: {ants: mean wall_time_s}, over all trials with a usable
                  optimum, regardless of feasibility.
    """
    trials = data.get("trials", [])
    file_level_optimal = data.get("optimal_edges")

    pct_vals = defaultdict(list)
    feasible_flags = defaultdict(list)
    time_vals = defaultdict(list)

    any_usable = False

    for t in trials:
        optimal = t.get("optimal_edges", file_level_optimal)
        final = t.get("final_edges")
        ants = t.get("ants")

        if optimal is None or final is None or ants is None:
            continue

        pct = pct_from_optimum(final, optimal)
        if pct is None:
            continue

        any_usable = True

        feasible = bool(t.get("theta_feasible"))
        feasible_flags[ants].append(feasible)

        if feasible:
            pct_vals[ants].append(pct)

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

    infeasible_pct_by_ants = {
        ants: 100.0 * (1 - sum(flags) / len(flags))
        for ants, flags in feasible_flags.items()
    }

    time_by_ants = {
        ants: statistics.mean(vals)
        for ants, vals in time_vals.items()
        if vals
    }

    return pct_by_ants, infeasible_pct_by_ants, time_by_ants


def series_name(path, data):
    label = data.get("dataset") or os.path.splitext(os.path.basename(path))[0]
    return str(label).replace("_", "-")


def _addplots(series_dicts, with_legend):
    """
    series_dicts: list of (name, {x: y}).

    If with_legend is False, emit only the addplot calls. This keeps
    colors/markers aligned with the panel that owns the legend.
    """
    lines = []

    for name, xy in series_dicts:
        coords = " ".join(
            f"({x},{y:.4f})"
            for x, y in sorted(xy.items())
        )

        lines.append(r"\addplot coordinates {" + coords + "};")

        if with_legend:
            lines.append(r"\addlegendentry{" + name + "}")

    return lines


LEGEND_COLUMNS = 3


def build_combined_latex(quality_series, infeasible_series, time_series):
    """
    Build a single 1x3 groupplot:

      1. Solution quality
      2. Theta-infeasibility rate
      3. Run time

    The legend is emitted only once by the top panel and placed below
    the complete figure using legend to name.

    The panels are deliberately taller and more widely separated than
    the original version. The original 0.28\textwidth height and 10pt
    vertical separation caused the titles and y-axis labels to crowd
    the neighboring panels.
    """

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        # More vertical space between panels prevents titles from colliding
        # with the x-axis/ticks of the panel above.
        r"    group style={group size=1 by 3, vertical sep=30pt, x descriptions at=edge bottom},",
        r"    title style={yshift=-3pt},",

        # Taller panels make the y-axis labels and titles much easier to read.
        # 0.31\textwidth is a good compromise between readability and total
        # figure height for a paper.
        r"    width=0.75\textwidth,",
        r"    height=0.33\textwidth,",

        r"    xmode=log,",
        r"    log basis x=2,",
        r"    grid=major,",
        r"]",

        r"\nextgroupplot[",
        r"    ylabel={Dev. from optimum (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Solution quality (feasible trials only)},",
        r"    title style={font=\small},",
        r"    legend to name=sharedlegend,",
        rf"    legend columns={LEGEND_COLUMNS},",
        r"    legend style={draw=none, fill=white, font=\small},",
        r"]",
    ]

    lines += _addplots(quality_series, with_legend=True)

    lines += [
        r"\nextgroupplot[",
        r"    ylabel={Infeasible trials (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Theta-infeasibility rate},",
        r"    title style={font=\small},",
        r"    ymin=0,",
        r"]",
    ]

    lines += _addplots(infeasible_series, with_legend=False)

    lines += [
        r"\nextgroupplot[",
        r"    xlabel={Number of ants},",
        r"    ylabel={Mean WCT (s)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Run time},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        r"]",
    ]

    lines += _addplots(time_series, with_legend=False)

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"",
        r"\begin{center}",
        r"\ref{sharedlegend}",
        r"\end{center}",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "directory",
        help="Directory containing the JSON result files",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Write the combined LaTeX to this file instead of stdout",
    )

    args = parser.parse_args()

    json_paths = sorted(
        glob.glob(os.path.join(args.directory, "*.json"))
    )

    if not json_paths:
        raise SystemExit(
            f"No .json files found in {args.directory}"
        )

    quality_series = []
    infeasible_series = []
    time_series = []

    skipped = []

    for path in json_paths:
        try:
            with open(path) as f:
                data = json.load(f)

        except (json.JSONDecodeError, OSError) as e:
            skipped.append((path, f"unreadable: {e}"))
            continue

        result = summarize_file(data)

        if result is None:
            skipped.append((path, "optimum not specified"))
            continue

        pct_by_ants, infeasible_pct_by_ants, time_by_ants = result
        name = series_name(path, data)

        if pct_by_ants:
            quality_series.append((name, pct_by_ants))

        if infeasible_pct_by_ants:
            infeasible_series.append(
                (name, infeasible_pct_by_ants)
            )

        if time_by_ants:
            time_series.append((name, time_by_ants))

    if not quality_series and not time_series:
        raise SystemExit(
            "No files had a specified optimum -- nothing to plot."
        )

    combined_tex = build_combined_latex(
        quality_series,
        infeasible_series,
        time_series,
    )

    if args.output:
        with open(args.output, "w") as f:
            f.write(combined_tex + "\n")
    else:
        print(combined_tex)

    if skipped:
        print(
            f"# Skipped {len(skipped)} file(s):",
            file=sys.stderr,
        )

        for path, reason in skipped:
            print(
                f"#   {os.path.basename(path)}: {reason}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
