"""
quality mode — vary.jl ant-count format → median percent deviation from
the θ-heuristic versus ant count. With ``--ablation-dir`` for ``rho``,
``pheromone``, and ``eta-exp``, the same median is drawn with the
ant-count curve in a one-column 2×2 (A=100 held fixed on the other
three panels). θ-infeasible outcomes count as −100%. One value per
graph at each axis point (mean over that graph's counted replicates),
then the median across graphs.
"""

from __future__ import annotations

import math
import os
import statistics
import sys
from collections import defaultdict

from .common import (
    COLUMN_HEIGHT,
    COLUMN_WIDTH,
    aco_timed_out,
    begin_float,
    begin_tikz,
    counted_trials,
    end_float,
    end_tikz,
    list_json_paths,
    load_json,
    report_skipped,
    series_name,
    write_tex,
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


def summarize_file(data, axis="ants"):
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
        ants = t.get(axis)

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


def aggregate_median_series(per_graph_series):
    """
    Collapse [(name, {ants: value}), ...] to the cross-graph median.

    At each ant count, take the median of the per-graph values.
    """
    by_ants = defaultdict(list)
    for _, xy in per_graph_series:
        for ants, val in xy.items():
            if val is None:
                continue
            by_ants[ants].append(float(val))

    median_xy = {}
    for ants, vals in by_ants.items():
        if not vals:
            continue
        median_xy[ants] = float(statistics.median(vals))
    return median_xy


def build_quality_latex(median_xy):
    """Single-axis plot of median % deviation from the θ-heuristic."""
    if not median_xy:
        raise ValueError("No median series to plot")

    coords = " ".join(
        f"({x},{y:.4f})" for x, y in sorted(median_xy.items())
    )
    xs = sorted(median_xy)
    xmin = xs[0] / 2
    xmax = xs[-1] * 2

    env, lines = begin_float("figure", star=False)
    lines += begin_tikz()
    lines += [
        r"\begin{axis}[",
        rf"    width={COLUMN_WIDTH},",
        rf"    height={COLUMN_HEIGHT},",
        r"    xlabel={Number of ants},",
        r"    ylabel={Dev. from $\theta$-heuristic (\%)},",
        r"    ylabel style={align=center},",
        r"    xmode=log,",
        r"    log basis x=2,",
        r"    grid=major,",
        rf"    xmin={xmin:g},",
        rf"    xmax={xmax:g},",
        r"]",
        (
            r"\addplot[solid, thick, mark=*, blue!80!black] "
            rf"coordinates {{{coords}}};"
        ),
        (
            rf"\draw[dashed, gray] (axis cs:{xmin:g},0) -- "
            rf"(axis cs:{xmax:g},0);"
        ),
        r"\end{axis}",
    ]
    lines += end_tikz()
    lines += [
        r"  \caption{Cross-graph median percent deviation of ACO from "
        r"the $\theta$-heuristic versus ant count. Each point is the "
        r"median, over graphs, of that graph's mean over counted "
        r"replicates (JIT warmup omitted). $\theta$-infeasible outcomes "
        r"count as $-100\%$. The dashed line is parity with the "
        r"$\theta$-heuristic.}",
        r"  \label{fig:quality-groupplot}",
    ]
    lines += end_float(env)
    return "\n".join(lines)


# Paper defaults marked with a dotted line. Ant count 100 is the colony
# size held fixed while the other three axes vary.
_PANEL_DEFAULTS = {
    "ants": 100.0,
    "rho": 0.05,
    "pheromone": 1.0,
    "eta-exp": 3.0,
}

_ROW_PANELS = (
    ("ants", "ants", r"Number of ants", "log2"),
    ("rho", "param_value", r"Evaporation $\rho$", "log10"),
    ("pheromone", "param_value", r"Deposit $\Delta\tau$", "linear"),
    ("eta-exp", "param_value", r"$\eta$ exponent", "linear"),
)


def _axis_limits(xs, scale):
    """Pad a sorted numeric axis. Log scales require a positive minimum."""
    lo, hi = float(xs[0]), float(xs[-1])
    if scale != "linear" and lo <= 0:
        scale = "linear"
    if scale == "linear":
        span = hi - lo
        if span == 0:
            span = 1.0 if hi == 0 else abs(hi)
        pad = 0.12 * span
        return lo - pad, hi + pad, "linear"
    return lo / 2.0, hi * 2.0, scale


def _tick_values(xs, scale):
    """Endpoints plus one interior label. More than three collide here.

    The interior label is the swept value nearest the middle of the
    axis (geometric middle on a log axis). The dotted line marks the
    paper default, so that default is not forced onto a tick.
    """
    xs = [float(x) for x in xs]
    if len(xs) <= 3:
        return xs
    if scale in ("log2", "log10") and xs[0] > 0 and xs[-1] > 0:
        target = math.exp(0.5 * (math.log(xs[0]) + math.log(xs[-1])))
    else:
        target = 0.5 * (xs[0] + xs[-1])
    mid = min(xs[1:-1], key=lambda x: abs(x - target))
    return [xs[0], mid, xs[-1]]


def _coords(xy):
    return " ".join(
        f"({x:g},{y:.4f})" for x, y in sorted(xy.items(), key=lambda item: float(item[0]))
    )


def _shared_ylim(series):
    vals = [float(y) for xy in series if xy for y in xy.values()]
    if not vals:
        return -10.0, 10.0
    lo, hi = min(vals), max(vals)
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    span = hi - lo
    if span == 0:
        span = 1.0
    pad = 0.08 * span
    return lo - pad, hi + pad


def _panel_plots(xy, xmin, xmax, ymin, ymax, default):
    """Coordinate plot, parity line, and the paper-default marker."""
    lines = []
    if xy:
        lines.append(
            r"\addplot[solid, thick, mark=*, mark size=1.1pt, blue!80!black] "
            rf"coordinates {{{_coords(xy)}}};"
        )
    else:
        lines.append(
            r"\node[font=\scriptsize, align=center, text=black!55] "
            r"at (rel axis cs:0.5,0.55) {not recorded};"
        )
    lines.append(
        rf"\draw[dashed, gray] (axis cs:{xmin:g},0) -- (axis cs:{xmax:g},0);"
    )
    if default is not None and xmin < float(default) < xmax:
        lines.append(
            rf"\draw[dotted, black!55] (axis cs:{default:g},{ymin:.4f}) -- "
            rf"(axis cs:{default:g},{ymax:.4f});"
        )
    return lines


def build_quality_row_latex(panels):
    """
    Four panels in a 2×2 that fits one column: ant count and evaporation
    on the first row, deposit and the η exponent on the second. Shared
    vertical scale, one ylabel for the left column. Missing sweeps stay
    blank.

    Axis boxes are a fraction of \\columnwidth and the picture is not
    resized, so a short grid stays small instead of stretching to the
    column. The shared ylabel sits left of the tick labels.
    """
    ymin, ymax = _shared_ylim([p["xy"] for p in panels])
    env, lines = begin_float("figure", star=False)
    lines += begin_tikz(resize=False)
    lines += [
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 2,",
        r"    horizontal sep=0.48cm, vertical sep=0.82cm},",
        r"  scale only axis,",
        r"  width=0.34\columnwidth,",
        r"  height=0.26\columnwidth,",
        r"  tick label style={font=\scriptsize, inner sep=1pt},",
        r"  label style={font=\scriptsize, inner sep=1pt},",
        r"  xlabel style={align=center},",
        r"  grid=major,",
        r"  grid style={gray!25},",
        r"  tick align=outside,",
        rf"  ymin={ymin:.4f},",
        rf"  ymax={ymax:.4f},",
        r"]",
    ]
    for i, panel in enumerate(panels):
        xy = panel["xy"] or {}
        if xy:
            xs = sorted(float(x) for x in xy)
            xmin, xmax, scale = _axis_limits(xs, panel["scale"])
            xtick = "{" + ",".join(f"{x:g}" for x in _tick_values(xs, scale)) + "}"
        else:
            xmin, xmax, scale = panel["empty"]
            xtick = None
        opts = [
            r"\nextgroupplot[",
            rf"  xlabel={{{panel['xlabel']}}},",
        ]
        # Right column shares the left column's scale.
        if i % 2 == 1:
            opts.append(r"  yticklabels={},")
        if scale == "log2":
            opts += [
                r"  xmode=log,",
                r"  log basis x=2,",
                r"  log ticks with fixed point,",
            ]
        elif scale == "log10":
            opts += [
                r"  xmode=log,",
                r"  log basis x=10,",
                r"  log ticks with fixed point,",
            ]
        opts.append(rf"  xmin={xmin:g},")
        opts.append(rf"  xmax={xmax:g},")
        if xtick is not None:
            opts.append(rf"  xtick={xtick},")
        opts.append(r"]")
        lines += opts
        lines += _panel_plots(xy, xmin, xmax, ymin, ymax, panel["default"])
    lines += [
        r"\end{groupplot}",
        r"\path (group c1r1.west) -- (group c1r2.west)",
        r"  node[midway, xshift=-32pt, rotate=90, anchor=center,",
        r"       font=\scriptsize, inner sep=0pt]",
        r"  {Dev.\ from $\theta$-heuristic (\%)};",
    ]
    lines += end_tikz(resize=False)
    lines += [
        r"  \caption{Cross-graph median percent deviation of ACO from "
        r"the $\theta$-heuristic. Each point is the median, over graphs, "
        r"of that graph's mean over counted replicates (JIT warmup omitted). "
        r"$\theta$-infeasible outcomes count as $-100\%$. "
        r"The ant-count panel varies $A$; the other panels hold $A=100$ and "
        r"vary evaporation $\rho$, pheromone deposit $\Delta\tau$, or the "
        r"exponent on $\eta$. The dashed line is parity with the "
        r"$\theta$-heuristic. The dotted line is the value used in the "
        r"other experiments ($A=100$, $\rho=0.05$, $\Delta\tau=1$, "
        r"exponent $3$). A panel marked ``not recorded'' has no sweep yet.}",
        r"  \label{fig:quality-groupplot}",
    ]
    lines += end_float(env)
    return "\n".join(lines)


def median_from_paths(paths, axis, *, expect_param=None):
    """Median % deviation series for one directory. Returns (xy or None, skipped)."""
    series = []
    skipped = []
    for path in paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue
        if expect_param is not None:
            found = data.get("param")
            if found not in (None, expect_param):
                skipped.append((path, f"param is {found}, expected {expect_param}"))
                continue
        if aco_timed_out(data):
            limit = data.get("aco_timeout_s")
            reason = "aco timed out"
            if limit is not None:
                reason = f"aco timed out ({limit}s)"
            skipped.append((path, reason))
            continue
        result = summarize_file(data, axis=axis)
        if result is None:
            skipped.append((path, "no usable optimum or heuristic"))
            continue
        _, heur_pct, _, _ = result
        if heur_pct:
            series.append((series_name(path, data), heur_pct))
    if not series:
        return None, skipped
    return aggregate_median_series(series), skipped


def run(json_paths, output, ablation_dirs=None):
    if not ablation_dirs:
        heuristic_series = []
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
            _, heur_pct_by_ants, _, _ = result
            if heur_pct_by_ants:
                heuristic_series.append((series_name(path, data), heur_pct_by_ants))
        if not heuristic_series:
            raise SystemExit(
                "No files had a usable heuristic baseline -- nothing to plot."
            )
        median_xy = aggregate_median_series(heuristic_series)
        write_tex(build_quality_latex(median_xy), output)
        print(
            f"# quality: median deviation over {len(heuristic_series)} graph(s)",
            file=sys.stderr,
        )
        report_skipped(skipped)
        return

    panels = []
    skipped = []
    for key, axis, xlabel, scale in _ROW_PANELS:
        if key == "ants":
            paths = json_paths
            directory = None
        else:
            directory = ablation_dirs.get(key)
            if not directory or not os.path.isdir(directory):
                print(
                    f"Warning: quality: no {key} sweep"
                    + (f" at {directory}" if directory else "")
                    + ".",
                    file=sys.stderr,
                )
                paths = []
            else:
                paths = list_json_paths(directory)
                if not paths:
                    print(
                        f"Warning: quality: no JSON in {directory}.",
                        file=sys.stderr,
                    )
        xy, panel_skipped = median_from_paths(
            paths, axis, expect_param=None if key == "ants" else key,
        )
        skipped.extend(panel_skipped)
        if key == "ants" and not xy:
            raise SystemExit(
                "No files had a usable heuristic baseline -- nothing to plot."
            )
        empty = {
            "ants": (1.0, 400.0, "log2"),
            "rho": (0.005, 1.0, "log10"),
            "pheromone": (-1.0, 12.0, "linear"),
            "eta-exp": (0.5, 5.5, "linear"),
        }[key]
        panels.append({
            "xy": xy,
            "xlabel": xlabel,
            "scale": scale,
            "default": _PANEL_DEFAULTS[key],
            "empty": empty,
        })
        status = "not recorded" if not xy else f"{len(xy)} point(s)"
        print(f"# quality: {key}: {status}", file=sys.stderr)

    write_tex(build_quality_row_latex(panels), output)
    report_skipped(skipped)
