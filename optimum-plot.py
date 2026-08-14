#!/usr/bin/env python3
"""
Scan a directory of ACO benchmark JSON files and emit pgfplots LaTeX.

Modes
-----
  quality (default)
      Vary.jl ant-count format → 1x4 groupplot:
        - mean % deviation from optimum (feasible trials)
        - mean % deviation from Cui θ-heuristic (feasible trials)
        - % theta-infeasible trials
        - mean wall-clock time vs ant count

  seed-compare
      compare-seeds.jl output → bar / coordinate plot of branch-and-pivot
      wall-time reduction when seeding with the best ACO subgraph (vs θ only).
      On each file, ACO trial selection already preferred max edges then min
      time_to_best_s; this mode plots time_reduction_pct (and optional absolute
      seconds).

Usage:
    python optimum-plot.py /path/to/json/dir -o plot.tex
    python optimum-plot.py /path/to/json/dir --mode=seed-compare -o seed.tex
"""

import argparse
import glob
import json
import os
import statistics
import sys
from collections import defaultdict


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
        (pct_by_ants, heur_pct_by_ants, infeasible_pct_by_ants, time_by_ants)
    or None if this file has neither a usable optimum nor a heuristic baseline.

    pct_by_ants: {ants: mean % deviation from optimum}, feasible trials only.

    heur_pct_by_ants: {ants: mean % deviation from Cui heuristic}, feasible
                      trials only. Empty when the file has no heuristic block.

    infeasible_pct_by_ants: {ants: % of trials at that ant count that were
                             theta_feasible == False}, over trials that had
                             either a usable optimum or a heuristic baseline.

    time_by_ants: {ants: mean wall_time_s}, over the same trial set as
                  infeasible_pct_by_ants.
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

    infeasible_pct_by_ants = {
        ants: 100.0 * (1 - sum(flags) / len(flags))
        for ants, flags in feasible_flags.items()
    }

    time_by_ants = {
        ants: statistics.mean(vals)
        for ants, vals in time_vals.items()
        if vals
    }

    return pct_by_ants, heur_pct_by_ants, infeasible_pct_by_ants, time_by_ants


def series_name(path, data):
    label = data.get("dataset") or os.path.splitext(os.path.basename(path))[0]
    return str(label).replace("_", "-")


def _series_lookup(series_dicts):
    return {name: xy for name, xy in series_dicts}


def _addplots(ordered_names, series_lookup, with_legend):
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

        if with_legend:
            lines.append(r"\addlegendentry{" + name + "}")

    return lines


LEGEND_COLUMNS = 3


def build_combined_latex(
    quality_series,
    heuristic_series,
    infeasible_series,
    time_series,
):
    """
    Build a single 1x4 groupplot:

      1. Solution quality vs optimum
      2. Output compared to Cui Heuristic
      3. Theta-infeasibility rate
      4. Run time

    The legend is emitted only once by the top panel and placed below
    the complete figure using legend to name. Series order is the union
    of all panels so colors/markers stay consistent.
    """

    ordered_names = []
    seen = set()
    for series in (
        quality_series,
        heuristic_series,
        infeasible_series,
        time_series,
    ):
        for name, _ in series:
            if name not in seen:
                seen.add(name)
                ordered_names.append(name)

    quality_lookup = _series_lookup(quality_series)
    heuristic_lookup = _series_lookup(heuristic_series)
    infeasible_lookup = _series_lookup(infeasible_series)
    time_lookup = _series_lookup(time_series)

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={group size=1 by 4, vertical sep=30pt, x descriptions at=edge bottom},",
        r"    title style={yshift=-3pt},",
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

    lines += _addplots(ordered_names, quality_lookup, with_legend=True)

    lines += [
        r"\nextgroupplot[",
        r"    ylabel={Dev. from Cui Heuristic (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Output compared to Cui Heuristic},",
        r"    title style={font=\small},",
        r"]",
    ]

    lines += _addplots(ordered_names, heuristic_lookup, with_legend=False)

    lines += [
        r"\nextgroupplot[",
        r"    ylabel={Infeasible trials (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Theta-infeasibility rate},",
        r"    title style={font=\small},",
        r"    ymin=0,",
        r"]",
    ]

    lines += _addplots(ordered_names, infeasible_lookup, with_legend=False)

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

    lines += _addplots(ordered_names, time_lookup, with_legend=False)

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"",
        r"\begin{center}",
        r"\ref{sharedlegend}",
        r"\end{center}",
    ]

    return "\n".join(lines)


def summarize_seed_compare(data):
    """
    Extract pivot timing fields from a compare-seeds.jl JSON.

    Returns dict with dataset label fields, or None if not a seed-compare file.
    """
    if data.get("compare") != "seeds":
        return None

    theta = data.get("pivot_theta") or {}
    aco = data.get("pivot_aco_seed") or {}
    trial = data.get("aco_trial") or {}

    theta_t = theta.get("wall_time_s")
    aco_t = aco.get("wall_time_s")
    if theta_t is None or aco_t is None:
        return None

    reduction_s = data.get("time_reduction_s")
    if reduction_s is None:
        reduction_s = float(theta_t) - float(aco_t)

    reduction_pct = data.get("time_reduction_pct")
    if reduction_pct is None:
        reduction_pct = (
            100.0 * float(reduction_s) / float(theta_t) if float(theta_t) > 0 else 0.0
        )

    return {
        "theta_wall_time_s": float(theta_t),
        "aco_seed_wall_time_s": float(aco_t),
        "time_reduction_s": float(reduction_s),
        "time_reduction_pct": float(reduction_pct),
        "aco_time_to_best_s": trial.get("time_to_best_s"),
        "aco_final_edges": trial.get("final_edges"),
        "heuristic_edges": trial.get("heuristic_edges"),
        "ants": trial.get("ants"),
    }


def build_seed_compare_latex(rows):
    """
    Build a 1x2 groupplot:

      1. % wall-time reduction of pivot when seeded with ACO vs θ only
      2. Absolute pivot wall times (θ seed vs θ+ACO seed)

    X positions are categorical symbolic coordinates (dataset names).
    """
    if not rows:
        raise ValueError("No seed-compare rows to plot")

    # Stable alphabetical order by display name.
    rows = sorted(rows, key=lambda r: r["name"])
    symbols = ",".join(r["name"] for r in rows)

    red_coords = " ".join(
        f"({r['name']},{r['time_reduction_pct']:.4f})" for r in rows
    )
    theta_coords = " ".join(
        f"({r['name']},{r['theta_wall_time_s']:.6f})" for r in rows
    )
    aco_coords = " ".join(
        f"({r['name']},{r['aco_seed_wall_time_s']:.6f})" for r in rows
    )

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=40pt, x descriptions at=edge bottom},",
        r"    title style={yshift=-3pt},",
        r"    width=0.85\textwidth,",
        r"    height=0.38\textwidth,",
        r"    grid=major,",
        r"    symbolic x coords={" + symbols + r"},",
        r"    xtick=data,",
        r"    x tick label style={rotate=45, anchor=east, font=\small},",
        r"    yticklabel style={font=\small},",
        r"]",

        r"\nextgroupplot[",
        r"    ylabel={Pivot time reduction (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Branch-and-pivot speedup from ACO seed (vs $\theta$-heuristic only)},",
        r"    title style={font=\small},",
        r"]",
        r"\addplot+[ybar, bar width=8pt, fill=black!40] coordinates {" + red_coords + r"};",

        r"\nextgroupplot[",
        r"    ylabel={Pivot WCT (s)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Absolute pivot wall time},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        r"    legend to name=seedcomparelegend,",
        r"    legend columns=2,",
        r"    legend style={draw=none, fill=white, font=\small},",
        r"]",
        r"\addplot+[mark=*, thick] coordinates {" + theta_coords + r"};",
        r"\addlegendentry{$\theta$ seed}",
        r"\addplot+[mark=square*, thick] coordinates {" + aco_coords + r"};",
        r"\addlegendentry{$\theta$ + ACO seed}",

        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"",
        r"\begin{center}",
        r"\ref{seedcomparelegend}",
        r"\end{center}",
    ]

    return "\n".join(lines)


def run_quality_mode(json_paths, output):
    quality_series = []
    heuristic_series = []
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
            skipped.append((path, "no usable optimum or heuristic"))
            continue

        (
            pct_by_ants,
            heur_pct_by_ants,
            infeasible_pct_by_ants,
            time_by_ants,
        ) = result
        name = series_name(path, data)

        if pct_by_ants:
            quality_series.append((name, pct_by_ants))

        if heur_pct_by_ants:
            heuristic_series.append((name, heur_pct_by_ants))

        if infeasible_pct_by_ants:
            infeasible_series.append(
                (name, infeasible_pct_by_ants)
            )

        if time_by_ants:
            time_series.append((name, time_by_ants))

    if not quality_series and not heuristic_series and not time_series:
        raise SystemExit(
            "No files had a usable optimum or heuristic -- nothing to plot."
        )

    combined_tex = build_combined_latex(
        quality_series,
        heuristic_series,
        infeasible_series,
        time_series,
    )

    if output:
        with open(output, "w") as f:
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


def run_seed_compare_mode(json_paths, output):
    rows = []
    skipped = []

    for path in json_paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            skipped.append((path, f"unreadable: {e}"))
            continue

        summary = summarize_seed_compare(data)
        if summary is None:
            skipped.append((path, "not a compare-seeds result (or missing timings)"))
            continue

        name = series_name(path, data)
        rows.append({"name": name, **summary})

    if not rows:
        raise SystemExit(
            "No compare-seeds JSON files with pivot timings -- nothing to plot."
        )

    tex = build_seed_compare_latex(rows)

    if output:
        with open(output, "w") as f:
            f.write(tex + "\n")
    else:
        print(tex)

    # Compact numeric summary for the terminal / paper notes.
    print(
        f"# seed-compare: {len(rows)} dataset(s); "
        f"mean time reduction = "
        f"{statistics.mean(r['time_reduction_pct'] for r in rows):.2f}%",
        file=sys.stderr,
    )

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

    parser.add_argument(
        "--mode",
        choices=("quality", "seed-compare"),
        default="quality",
        help="quality = vary.jl ant-count plots (default); "
             "seed-compare = pivot time reduction from ACO seeding",
    )

    args = parser.parse_args()

    json_paths = sorted(
        glob.glob(os.path.join(args.directory, "*.json"))
    )

    if not json_paths:
        raise SystemExit(
            f"No .json files found in {args.directory}"
        )

    if args.mode == "seed-compare":
        run_seed_compare_mode(json_paths, args.output)
    else:
        run_quality_mode(json_paths, args.output)


if __name__ == "__main__":
    main()
