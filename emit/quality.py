"""
quality mode — vary.jl ant-count format → median percent deviation from
the θ-heuristic versus ant count. ``--budget-dir`` adds the epoch-budget
and replicate-count sweeps (deviation, then θ-feasibility) from the
recorded epoch traces, in the same one-column figure. ``--plots=sweeps``
with ``--ablation-dir`` for ``rho``, ``pheromone``, and ``eta-exp``
writes those three panels as a separate figure. θ-infeasible outcomes
count as −100%. One value per graph at each axis point (mean over that
graph's counted replicates), then the median across graphs. The
epoch-trace panels ignore ``--max-replicates`` and use every counted
replicate.
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


# Paper defaults marked with a dotted line. Ant count 100 is the colony
# size held fixed while the other three axes vary.
_PANEL_DEFAULTS = {
    "ants": 100.0,
    "rho": 0.05,
    "pheromone": 1.0,
    "eta-exp": 3.0,
}

_MAIN_PANELS = (
    ("ants", "ants", r"Number of ants", "log2"),
)
_SWEEP_PANELS = (
    ("rho", "param_value", r"Evaporation $\rho$", "log10"),
    ("pheromone", "param_value", r"Deposit $\Delta\tau$", "linear"),
    ("eta-exp", "param_value", r"$\eta$ exponent", "linear"),
)
_ROW_PANELS = _MAIN_PANELS + _SWEEP_PANELS


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


def _xscale_opts(scale):
    """Log-axis keys for a panel. Linear scales add nothing."""
    basis = {"log2": 2, "log10": 10}.get(scale)
    if basis is None:
        return []
    return [
        r"  xmode=log,",
        rf"  log basis x={basis},",
        r"  log ticks with fixed point,",
    ]


def _panel_plots(xy, xmin, xmax, ymin, ymax, default, *, zero_line=True):
    """Coordinate plot, optional parity line, and the paper-default marker."""
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
    if zero_line:
        lines.append(
            rf"\draw[dashed, gray] (axis cs:{xmin:g},0) -- (axis cs:{xmax:g},0);"
        )
    if default is not None and xmin < float(default) < xmax:
        lines.append(
            rf"\draw[dotted, black!55] (axis cs:{default:g},{ymin:.4f}) -- "
            rf"(axis cs:{default:g},{ymax:.4f});"
        )
    return lines


_CAPTION = r"Median deviation from the $\theta$-heuristic."
_CAPTION_WITH_BUDGET = (
    r"Median deviation from the $\theta$-heuristic versus ant count, "
    r"epoch budget, and replicate count. "
    r"Bottom row: median $\theta$-feasibility."
)
_CAPTION_SWEEPS = (
    r"Median deviation from the $\theta$-heuristic versus evaporation "
    r"$\rho$, deposit $\Delta\tau$, and the $\eta$ exponent."
)

_METRIC_LABEL = {
    "dev": r"Dev.\ from $\theta$-heuristic (\%)",
    "feas": r"$\theta$-feasible (\%)",
}

# Epoch-trace panels appended when --budget-dir is set. Deviation shares
# the quality y-scale; feasibility keeps the 0–100 scale.
_BUDGET_ROW = (
    ("dev_t", r"Epochs $T$", "log2", 5.0, (0.5, 200.0, "log2"), "dev", True, None),
    ("dev_r", r"Replicates $R$", "log2", 5.0, (0.5, 60.0, "log2"), "dev", True, None),
    ("feas_t", r"Epochs $T$", "log2", 5.0, (0.5, 200.0, "log2"), "feas", False, "0,50,100"),
    ("feas_r", r"Replicates $R$", "log2", 5.0, (0.5, 60.0, "log2"), "feas", False, "0,50,100"),
)


def _quality_figure(
    body,
    *,
    bodies=None,
    resize=True,
    caption=_CAPTION,
    extra_labels=(),
    label="fig:quality-groupplot",
):
    """One column float around one or more tikz bodies."""
    chunks = bodies if bodies is not None else [body]
    env, lines = begin_float("figure", star=False)
    for i, chunk in enumerate(chunks):
        if i:
            lines.append(r"  \vspace{0.45em}")
        lines += begin_tikz(resize=resize)
        lines += chunk
        lines += end_tikz(resize=resize)
    lines += [
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
    ]
    for extra in extra_labels:
        lines.append(rf"  \label{{{extra}}}")
    lines += end_float(env)
    return "\n".join(lines)


def _ylabel_node(row_a, row_b, text):
    """Rotated ylabel centered on one row, or between two rows."""
    if row_a == row_b:
        anchor = rf"\path (group c1r{row_a}.west)"
        place = ""
    else:
        anchor = rf"\path (group c1r{row_a}.west) -- (group c1r{row_b}.west)"
        place = "midway, "
    return [
        anchor,
        rf"  node[{place}xshift=-32pt, rotate=90, anchor=center,"
        r" font=\scriptsize, inner sep=0pt]",
        rf"  {{{text}}};",
    ]


def _row_ylabels(panels, ncols):
    """One ylabel per run of rows that share a metric."""
    metrics = [
        panels[i].get("metric", "dev") for i in range(0, len(panels), ncols)
    ]
    lines = []
    start = 0
    for i, metric in enumerate(metrics + [None]):
        if i == start or metric == metrics[start]:
            continue
        lines += _ylabel_node(start + 1, i, _METRIC_LABEL[metrics[start]])
        start = i
    return lines


def build_quality_latex(median_xy):
    """Single-axis plot of median % deviation from the θ-heuristic."""
    if not median_xy:
        raise ValueError("No median series to plot")

    xs = sorted(float(x) for x in median_xy)
    xmin, xmax, _scale = _axis_limits(xs, "log2")
    ymin, ymax = _shared_ylim([median_xy])
    body = [
        r"\begin{axis}[",
        rf"    width={COLUMN_WIDTH},",
        rf"    height={COLUMN_HEIGHT},",
        r"    xlabel={Number of ants},",
        r"    ylabel={Dev.\ from $\theta$-heuristic (\%)},",
        r"    ylabel style={align=center},",
        r"    grid=major,",
        rf"    xmin={xmin:g},",
        rf"    xmax={xmax:g},",
        rf"    ymin={ymin:.4f},",
        rf"    ymax={ymax:.4f},",
    ]
    body += _xscale_opts("log2")
    body.append(r"]")
    body += _panel_plots(
        median_xy, xmin, xmax, ymin, ymax, _PANEL_DEFAULTS["ants"],
    )
    body.append(r"\end{axis}")
    return _quality_figure(body)


def _panel_bounds(panel, ymin, ymax):
    """Axis limits and tick list for one panel."""
    xy = panel["xy"] or {}
    if xy:
        xs = sorted(float(x) for x in xy)
        xmin, xmax, scale = _axis_limits(xs, panel["scale"])
        xtick = "{" + ",".join(f"{x:g}" for x in _tick_values(xs, scale)) + "}"
    else:
        xmin, xmax, scale = panel["empty"]
        xtick = None
    if panel.get("ymin") is not None:
        ymin = panel["ymin"]
    if panel.get("ymax") is not None:
        ymax = panel["ymax"]
    return xmin, xmax, scale, xtick, ymin, ymax


def _standalone_axis(panel, ymin, ymax, *, width, height):
    """One axis, used for the ant-count panel above the budget grid."""
    xmin, xmax, scale, xtick, ymin, ymax = _panel_bounds(panel, ymin, ymax)
    metric = panel.get("metric", "dev")
    lines = [
        r"\begin{axis}[",
        rf"  width={width},",
        rf"  height={height},",
        r"  scale only axis,",
        r"  tick label style={font=\scriptsize, inner sep=1pt},",
        r"  label style={font=\scriptsize, inner sep=1pt},",
        r"  xlabel style={align=center},",
        r"  ylabel style={font=\scriptsize},",
        rf"  xlabel={{{panel['xlabel']}}},",
        rf"  ylabel={{{_METRIC_LABEL[metric]}}},",
        r"  grid=major,",
        r"  grid style={gray!25},",
        r"  tick align=outside,",
        rf"  ymin={ymin:.4f},",
        rf"  ymax={ymax:.4f},",
    ]
    lines += _xscale_opts(scale)
    lines.append(rf"  xmin={xmin:g},")
    lines.append(rf"  xmax={xmax:g},")
    if xtick is not None:
        lines.append(rf"  xtick={xtick},")
    if panel.get("ytick"):
        lines.append(rf"  ytick={{{panel['ytick']}}},")
    lines.append(r"]")
    lines += _panel_plots(
        panel["xy"] or {}, xmin, xmax, ymin, ymax, panel["default"],
        zero_line=panel.get("zero_line", True),
    )
    lines.append(r"\end{axis}")
    return lines


def _groupplot_lines(
    panels,
    *,
    ncols=2,
    width=r"0.34\columnwidth",
    height=r"0.26\columnwidth",
):
    """
    Groupplot body for ``panels``. Missing sweeps stay blank.

    Axis boxes are a fraction of \\columnwidth and the picture is not
    resized, so a short grid stays small instead of stretching to the
    column. Each ylabel sits left of the tick labels.
    """
    nrows = (len(panels) + ncols - 1) // ncols
    shared_lo, shared_hi = _shared_ylim(
        [p["xy"] for p in panels if p.get("ymin") is None]
    )
    body = [
        r"\begin{groupplot}[",
        rf"  group style={{group size={ncols} by {nrows},",
        r"    horizontal sep=0.48cm, vertical sep=0.82cm},",
        r"  scale only axis,",
        rf"  width={width},",
        rf"  height={height},",
        r"  tick label style={font=\scriptsize, inner sep=1pt},",
        r"  label style={font=\scriptsize, inner sep=1pt},",
        r"  xlabel style={align=center},",
        r"  grid=major,",
        r"  grid style={gray!25},",
        r"  tick align=outside,",
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
        ymin = shared_lo if panel.get("ymin") is None else panel["ymin"]
        ymax = shared_hi if panel.get("ymax") is None else panel["ymax"]
        opts = [
            r"\nextgroupplot[",
            rf"  xlabel={{{panel['xlabel']}}},",
            rf"  ymin={ymin:.4f},",
            rf"  ymax={ymax:.4f},",
        ]
        # Right column shares the left column's scale.
        if i % ncols == 1:
            opts.append(r"  yticklabels={},")
        opts += _xscale_opts(scale)
        opts.append(rf"  xmin={xmin:g},")
        opts.append(rf"  xmax={xmax:g},")
        if xtick is not None:
            opts.append(rf"  xtick={xtick},")
        if panel.get("ytick"):
            opts.append(rf"  ytick={{{panel['ytick']}}},")
        opts.append(r"]")
        body += opts
        body += _panel_plots(
            xy, xmin, xmax, ymin, ymax, panel["default"],
            zero_line=panel.get("zero_line", True),
        )
    body.append(r"\end{groupplot}")
    body += _row_ylabels(panels, ncols)
    return body


def build_quality_row_latex(
    panels,
    *,
    ncols=2,
    width=r"0.34\columnwidth",
    height=r"0.26\columnwidth",
    caption=None,
    label="fig:quality-groupplot",
    extra_labels=(),
):
    """One-column groupplot. Caption follows the panels unless overridden."""
    has_budget = any(p.get("metric") == "feas" for p in panels)
    if caption is None:
        caption = _CAPTION_WITH_BUDGET if has_budget else _CAPTION
    if has_budget and not extra_labels:
        extra_labels = ("fig:budget",)
    return _quality_figure(
        _groupplot_lines(panels, ncols=ncols, width=width, height=height),
        resize=False,
        caption=caption,
        extra_labels=extra_labels,
        label=label,
    )


def build_main_quality_latex(ants_panel, budget):
    """
    Ant count above the epoch and replicate grid.

    Deviation panels share one vertical scale. θ-feasibility stays on
    the 0–100 scale in the bottom row of the grid.
    """
    shared_lo, shared_hi = _shared_ylim(
        [ants_panel["xy"]]
        + [p["xy"] for p in budget if p.get("ymin") is None]
    )
    locked = []
    for panel in budget:
        if panel.get("ymin") is None:
            panel = dict(panel)
            panel["ymin"] = shared_lo
            panel["ymax"] = shared_hi
        locked.append(panel)
    return _quality_figure(
        None,
        bodies=[
            _standalone_axis(
                ants_panel,
                shared_lo,
                shared_hi,
                width=r"0.74\columnwidth",
                height=r"0.26\columnwidth",
            ),
            _groupplot_lines(locked),
        ],
        resize=False,
        caption=_CAPTION_WITH_BUDGET,
        extra_labels=("fig:budget",),
    )


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


def budget_panels(budget_dir, skipped):
    """
    Deviation and θ-feasibility versus T and R, from recorded epoch traces.

    Returns four panel dicts. Missing traces stay blank and are warned
    about; they are not filled in from another rule.
    """
    from .compare.budget_curves import budget_series, load_suite

    series = {}
    if not budget_dir or not os.path.isdir(budget_dir):
        print(
            "Warning: quality: no epoch-trace directory"
            + (f" at {budget_dir}" if budget_dir else "")
            + ".",
            file=sys.stderr,
        )
    else:
        paths = list_json_paths(budget_dir)
        if not paths:
            print(
                f"Warning: quality: no JSON in {budget_dir}.",
                file=sys.stderr,
            )
        else:
            graphs, suite_skipped = load_suite(paths)
            skipped.extend(suite_skipped)
            if graphs:
                series = budget_series(graphs)
            else:
                print(
                    f"Warning: quality: no epoch traces in {budget_dir}.",
                    file=sys.stderr,
                )

    panels = []
    for key, xlabel, scale, default, empty, metric, zero_line, ytick in _BUDGET_ROW:
        xy = series.get(key) or None
        panel = {
            "xy": xy,
            "xlabel": xlabel,
            "scale": scale,
            "default": default,
            "empty": empty,
            "metric": metric,
            "zero_line": zero_line,
        }
        if ytick:
            panel["ytick"] = ytick
        if metric == "feas":
            panel["ymin"] = -8.0
            panel["ymax"] = 108.0
        panels.append(panel)
        status = "not recorded" if not xy else f"{len(xy)} point(s)"
        print(f"# quality: {key}: {status}", file=sys.stderr)
    return panels


_EMPTY_LIMITS = {
    "ants": (1.0, 400.0, "log2"),
    "rho": (0.005, 1.0, "log10"),
    "pheromone": (-1.0, 12.0, "linear"),
    "eta-exp": (0.5, 5.5, "linear"),
}


def _load_row_panels(specs, json_paths, ablation_dirs, skipped):
    """One panel dict per spec. Missing sweeps stay blank and are warned about."""
    panels = []
    for key, axis, xlabel, scale in specs:
        if key == "ants":
            paths = json_paths
        else:
            directory = (ablation_dirs or {}).get(key)
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
        panels.append({
            "xy": xy,
            "xlabel": xlabel,
            "scale": scale,
            "default": _PANEL_DEFAULTS[key],
            "empty": _EMPTY_LIMITS[key],
            "metric": "dev",
        })
        status = "not recorded" if not xy else f"{len(xy)} point(s)"
        print(f"# quality: {key}: {status}", file=sys.stderr)
    return panels


def run(json_paths, output, ablation_dirs=None, budget_dir=None, subset="main"):
    """
    ``subset="main"`` is ant count, plus epoch and replicate panels when
    ``budget_dir`` is set. ``subset="sweeps"`` is evaporation, deposit,
    and the η exponent.
    """
    if subset not in ("main", "sweeps"):
        raise SystemExit(f"quality: unknown --plots={subset}; use sweeps")

    skipped = []
    if subset == "sweeps":
        panels = _load_row_panels(
            _SWEEP_PANELS, json_paths, ablation_dirs, skipped,
        )
        write_tex(
            build_quality_row_latex(
                panels,
                ncols=1,
                width=r"0.72\columnwidth",
                height=r"0.24\columnwidth",
                caption=_CAPTION_SWEEPS,
                label="fig:quality-sweeps",
                extra_labels=(),
            ),
            output,
        )
        report_skipped(skipped)
        return

    panels = _load_row_panels(_MAIN_PANELS, json_paths, None, skipped)
    ants = panels[0]
    if not ants["xy"]:
        raise SystemExit(
            "No files had a usable heuristic baseline -- nothing to plot."
        )
    if budget_dir is not None:
        budget = budget_panels(budget_dir, skipped)
        write_tex(build_main_quality_latex(ants, budget), output)
    else:
        write_tex(build_quality_latex(ants["xy"]), output)
        print(
            f"# quality: median deviation, {len(ants['xy'])} ant-count point(s)",
            file=sys.stderr,
        )
    report_skipped(skipped)
