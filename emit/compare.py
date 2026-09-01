"""
compare mode — vary.jl ant-count JSON → complexity figures.

Pass a single results directory (e.g. ``vary_k2t5i_PN``), like table or
statistics mode. Select plot groups with ``--plots`` (comma-separated):

  theta-time
    - θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    - ACO-PN discovery time vs (|U_R|+|V_R|)^2
    - same vs (max degree + |U_R|+|V_R|)·(|U_D|+|V_D|)

  density-size
    - edge density vs |E(D*)|

  max-deg-time
    - max reduced degree vs ACO-PN discovery time

ACO points use the same best-trial / discovery rules as table mode.
"""

from __future__ import annotations

import math
import os
import sys

from .common import (
    display_name,
    load_json,
    report_skipped,
    write_tex,
)
from .result_fields import validate_compare_directory
from .table import summarize_file

PLOT_GROUPS = (
    "theta-time",
    "deg-size-time",
    "density-size",
    "max-deg-time",
)

ACO_LABEL = "ACO-PN"
HEUR_LABEL = r"$\theta$-Heuristic"


def edge_density(row):
    """Bipartite edge density |E| / (|U|·|V|) of the original graph."""
    nU, nV, edges = row.get("nU"), row.get("nV"), row.get("edge_count")
    if nU is None or nV is None or edges is None:
        return None
    denom = int(nU) * int(nV)
    if denom <= 0:
        return None
    return float(edges) / denom


def reduced_node_count(row):
    """Reduced graph |U_R| + |V_R|."""
    nU, nV = row.get("reduced_nU"), row.get("reduced_nV")
    if nU is None or nV is None:
        return None
    return int(nU) + int(nV)


def reduced_nodes_squared(row):
    """(|U_R| + |V_R|)^2."""
    nodes = reduced_node_count(row)
    if nodes is None:
        return None
    n = float(nodes)
    return n * n


def reduced_edge_density(row):
    """Bipartite edge density of the reduced graph."""
    nU, nV, edges = (
        row.get("reduced_nU"),
        row.get("reduced_nV"),
        row.get("reduced_edges"),
    )
    if nU is None or nV is None or edges is None:
        return None
    denom = int(nU) * int(nV)
    if denom <= 0:
        return None
    return float(edges) / denom


def theta_n_plus_m(row):
    """θ·n + m with n = |U_R|+|V_R|, m = |E_R| (θ-heuristic complexity proxy)."""
    theta = row.get("theta")
    nodes = reduced_node_count(row)
    edges = row.get("reduced_edges")
    if theta is None or nodes is None or edges is None:
        return None
    return float(theta) * float(nodes) + float(edges)


def reduced_max_degree(row):
    """Maximum degree over vertices in the reduced graph."""
    d = row.get("reduced_max_degree")
    if d is None:
        return None
    return float(d)


def reduced_avg_degree(row):
    """Mean degree over vertices in the reduced graph."""
    d = row.get("reduced_avg_degree")
    if d is not None:
        return float(d)
    # Back-compat: 2|E_R| / (|U_R|+|V_R|) when the JSON predates the field.
    nodes = reduced_node_count(row)
    edges = row.get("reduced_edges")
    if nodes is None or edges is None or nodes <= 0:
        return None
    return (2.0 * float(edges)) / float(nodes)


def solution_node_count(row):
    """|U_D| + |V_D| from the ACO best trial."""
    nU, nV = row.get("aco_nU"), row.get("aco_nV")
    if nU is None or nV is None:
        return None
    return int(nU) + int(nV)


def _deg_plus_nodes_times(deg_fn, size_fn):
    """(degree + |U_R|+|V_R|) · size_fn(row) for reduced-graph degree proxies."""

    def x_fn(row):
        deg = deg_fn(row)
        nodes = reduced_node_count(row)
        size = size_fn(row)
        if deg is None or nodes is None or size is None:
            return None
        return (float(deg) + float(nodes)) * float(size)

    return x_fn


def _scatter_coords(points):
    """points: iterable of (x, y) floats → pgfplots coordinates body."""
    parts = []
    for x, y in points:
        if x is None or y is None:
            continue
        parts.append(f"({float(x):.8g},{float(y):.8g})")
    return " ".join(parts)


_SERIES_COLORS = (
    "blue",
    "red",
    "green!60!black",
    "orange",
    "purple",
    "brown",
)


def _log_log_regression(points):
    """Fit log(y) = slope * log(x) + intercept; return (slope, intercept)."""
    valid = [(x, y) for x, y in points if x and y and x > 0 and y > 0]
    if len(valid) < 2:
        return None
    xs = [math.log(x) for x, _ in valid]
    ys = [math.log(y) for _, y in valid]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    slope = num / den
    intercept = mean_y - slope * mean_x
    return slope, intercept


def _trend_line_coords(points):
    """Endpoints for a log--log least-squares trend line."""
    reg = _log_log_regression(points)
    if reg is None:
        return None
    slope, intercept = reg
    xs = [x for x, y in points if x and y and x > 0 and y > 0]
    if not xs:
        return None
    xmin, xmax = min(xs), max(xs)

    def y_at(x):
        return math.exp(intercept) * (x**slope)

    return [(xmin, y_at(xmin)), (xmax, y_at(xmax))]


def _addplot_scatter(label, points, *, with_legend=True, alpha=None, color=None):
    coords = _scatter_coords(points)
    opts = ["only marks"]
    if alpha is not None:
        opts.append(f"opacity={alpha}")
        opts.append(f"mark options={{opacity={alpha}}}")
    if color:
        opts.append(color)
    lines = [rf"\addplot+[{', '.join(opts)}] coordinates {{{coords}}};"]
    if with_legend:
        lines.append(r"\addlegendentry{" + label + "}")
    return lines


def _addplot_trendline(points, *, color=None, alpha=0.85):
    coords = _trend_line_coords(points)
    if coords is None:
        return []
    coord_str = _scatter_coords(coords)
    opts = ["mark=none", "thick", "dashed", "forget plot"]
    if alpha is not None:
        opts.append(f"opacity={alpha}")
    if color:
        opts.append(color)
    return [rf"\addplot+[{', '.join(opts)}] coordinates {{{coord_str}}};"]



def _series_for_x(variant_rows, aco_order, x_fn, *, size_y_fn=None):
    """
    Build size/time point lists keyed by series label.

    size_y_fn(row, edges) → y for size scatters (default: raw |E(D*)|).
    Returns (aco_size, aco_time, heur_size, heur_time) where each ACO dict
    maps label → [(x, y), ...] and heur_* are flat lists.
    """
    if size_y_fn is None:
        def size_y_fn(_row, edges):
            return float(edges)

    heur_size = []
    heur_time = []
    seen_datasets = set()
    for label in aco_order:
        for name, row in variant_rows[label]:
            if name in seen_datasets:
                continue
            x = x_fn(row)
            if x is None:
                continue
            seen_datasets.add(name)
            if row.get("heur_edges") is not None:
                y = size_y_fn(row, row["heur_edges"])
                if y is not None:
                    heur_size.append((x, y))
            ht = row.get("heur_time")
            if ht is not None and float(ht) > 0:
                heur_time.append((x, float(ht)))

    aco_size = {}
    aco_time = {}
    for label in aco_order:
        size_pts = []
        time_pts = []
        for _name, row in variant_rows[label]:
            x = x_fn(row)
            if x is None:
                continue
            if row.get("aco_edges") is not None:
                y = size_y_fn(row, row["aco_edges"])
                if y is not None:
                    size_pts.append((x, y))
            at = row.get("aco_time")
            if at is not None and float(at) > 0:
                time_pts.append((x, float(at)))
        aco_size[label] = size_pts
        aco_time[label] = time_pts

    return aco_size, aco_time, heur_size, heur_time


def _size_figure(
    *,
    aco_order,
    aco_size,
    heur_size,
    xlabel,
    caption,
    label,
    legend_name=None,
    ylabel=r"Solution size $|E(D^*)|$",
    width="0.88\\textwidth",
    height="0.52\\textwidth",
    log_axes=True,
    log_y=None,
    y_equals_one=False,
    xmin=None,
    xmax=None,
    mark_alpha=None,
    trend_lines=False,
):
    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{axis}[",
        rf"    width={width},",
        rf"    height={height},",
        rf"    xlabel={{{xlabel}}},",
        rf"    ylabel={{{ylabel}}},",
    ]
    use_log_y = log_axes if log_y is None else log_y
    if log_axes:
        lines.append(r"    xmode=log,")
    if use_log_y:
        lines.append(r"    ymode=log,")
    if xmin is not None:
        lines.append(rf"    xmin={xmin},")
    if xmax is not None:
        lines.append(rf"    xmax={xmax},")
    lines.append(r"    grid=major,")
    if legend_name:
        lines += [
            rf"    legend to name={legend_name},",
            r"    legend columns=3,",
            r"    legend style={draw=none, fill=white, font=\small},",
            r"    legend cell align=left,",
        ]
    lines.append(r"  ]")
    if heur_size:
        lines += _addplot_scatter(
            HEUR_LABEL, heur_size, with_legend=bool(legend_name)
        )
    for i, lab in enumerate(aco_order):
        color = (
            _SERIES_COLORS[i % len(_SERIES_COLORS)]
            if mark_alpha is not None or trend_lines
            else None
        )
        pts = aco_size[lab]
        lines += _addplot_scatter(
            lab,
            pts,
            with_legend=bool(legend_name),
            alpha=mark_alpha,
            color=color,
        )
        if trend_lines:
            lines += _addplot_trendline(pts, color=color)
    if y_equals_one:
        lines.append(
            r"\addplot[black, densely dashed, forget plot] coordinates {(0,1) (1,1)};"
        )
    lines += [
        r"  \end{axis}",
        r"  \end{tikzpicture}",
        r"",
    ]
    if legend_name:
        lines += [
            r"  \begin{center}",
            rf"  \pgfplotslegendfromname{{{legend_name}}}",
            r"  \end{center}",
        ]
    lines += [
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
    ]
    return lines


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


def _deg_size_time_figures(
    *,
    aco_order,
    ctx,
    reduced_nodes_sq_x,
    max_deg_nodes_x,
    disc_y,
):
    """Discovery time vs reduced-node count and max-degree complexity proxies."""
    fig_nodes_sq = _size_figure(
        aco_order=aco_order,
        aco_size=ctx["reduced_nodes_sq_time"],
        heur_size=[],
        xlabel=reduced_nodes_sq_x,
        ylabel=disc_y,
        legend_name="compareReducedNodesSqTimeLegend",
        caption=(
            r"ACO-PN discovery time vs.\ "
            r"$(|U_R|+|V_R|)^2$ "
            r"(dashed line shows a log--log trend)."
        ),
        label="fig:compare-reduced-nodes-sq-time",
        mark_alpha=0.5,
        trend_lines=True,
    )
    fig_nodes = _size_figure(
        aco_order=aco_order,
        aco_size=ctx["max_deg_nodes_time"],
        heur_size=[],
        xlabel=max_deg_nodes_x,
        ylabel=disc_y,
        legend_name="compareMaxDegNodesTimeLegend",
        caption=(
            r"ACO-PN discovery time vs.\ "
            r"$(\Delta(G_R)+|U_R|+|V_R|)\cdot(|U_D|+|V_D|)$ "
            r"(dashed line shows a log--log trend)."
        ),
        label="fig:compare-max-deg-nodes-time",
        mark_alpha=0.5,
        trend_lines=True,
    )
    return fig_nodes_sq + [""] + fig_nodes


def build_compare_plots(named_rows, plots=None):
    """
    Build selected compare figures from [(name, row), ...].

    Plot groups are listed in ``PLOT_GROUPS``; pass ``plots`` as a
    comma-separated subset (e.g. ``theta-time,deg-size-time``).
    """
    selected = _parse_plots(plots)
    variant_rows = {ACO_LABEL: named_rows}
    aco_order = [ACO_LABEL]

    ctx = {}
    need_density = "density-size" in selected
    need_theta = "theta-time" in selected
    need_max_deg = "max-deg-time" in selected
    need_deg_size = "deg-size-time" in selected

    if need_density:
        full_size, _full_time, _full_heur_size, _full_heur_time = (
            _series_for_x(variant_rows, aco_order, edge_density)
        )
        ctx["full_size"] = full_size

    if need_theta:
        _unused_size2, _unused_time2, _unused_heur_size3, theta_heur_time = (
            _series_for_x(variant_rows, aco_order, theta_n_plus_m)
        )
        ctx["theta_heur_time"] = theta_heur_time

    if need_max_deg:
        _u3, max_deg_time, _u4, _u5 = _series_for_x(
            variant_rows, aco_order, reduced_max_degree
        )
        ctx["max_deg_time"] = max_deg_time

    if need_deg_size:
        max_deg_nodes_x = _deg_plus_nodes_times(
            reduced_max_degree, solution_node_count
        )
        _u9, reduced_nodes_sq_time, _u10, _u11 = _series_for_x(
            variant_rows, aco_order, reduced_nodes_squared
        )
        _u12, max_deg_nodes_time, _u13, _u14 = _series_for_x(
            variant_rows, aco_order, max_deg_nodes_x
        )
        ctx["reduced_nodes_sq_time"] = reduced_nodes_sq_time
        ctx["max_deg_nodes_time"] = max_deg_nodes_time

    dens_x = r"Edge density $|E|/(|U|\,|V|)$"
    theta_nm_x = r"$\theta(|U_R|+|V_R|)+|E_R|$"
    max_deg_x = r"Maximum reduced degree $\Delta(G_R)$"
    reduced_nodes_sq_x = r"$(|U_R|+|V_R|)^2$"
    max_deg_nodes_x = r"$(\Delta(G_R)+|U_R|+|V_R|)\cdot(|U_D|+|V_D|)$"
    disc_y = r"Discovery time (s)"

    group_builders = {
        "density-size": lambda: _size_figure(
            aco_order=aco_order,
            aco_size=ctx["full_size"],
            heur_size=[],
            xlabel=dens_x,
            legend_name="compareSizeLegend",
            caption=(
                r"ACO-PN solution size vs.\ graph edge density."
            ),
            label="fig:compare-density-size",
        ),
        "theta-time": lambda: _size_figure(
            aco_order=[],
            aco_size={},
            heur_size=ctx["theta_heur_time"],
            xlabel=theta_nm_x,
            ylabel=r"Time (s)",
            caption=(
                r"$\theta$-heuristic wall time vs.\ "
                r"$\theta(|U_R|+|V_R|)+|E_R|$."
            ),
            label="fig:compare-theta-nm-time",
            width="0.72\\textwidth",
            height="0.48\\textwidth",
        ),
        "max-deg-time": lambda: _size_figure(
            aco_order=aco_order,
            aco_size=ctx["max_deg_time"],
            heur_size=[],
            xlabel=max_deg_x,
            ylabel=disc_y,
            legend_name="compareMaxDegTimeLegend",
            caption=(
                r"ACO-PN discovery time vs.\ maximum degree in the reduced graph."
            ),
            label="fig:compare-max-deg-time",
        ),
        "deg-size-time": lambda: _deg_size_time_figures(
            aco_order=aco_order,
            ctx=ctx,
            reduced_nodes_sq_x=reduced_nodes_sq_x,
            max_deg_nodes_x=max_deg_nodes_x,
            disc_y=disc_y,
        ),
    }

    parts = []
    for name in selected:
        block = group_builders[name]()
        if not block:
            continue
        if parts:
            parts.append("")
        parts += block

    return "\n".join(parts)


def run(json_paths, output, ants=None, plots=None):
    named_rows = []
    skipped = []

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

    selected = _parse_plots(plots) if plots else list(PLOT_GROUPS)
    validate_compare_directory(json_paths, selected)

    print(
        f"# compare: {len(named_rows)} dataset(s)"
        + (f"; ants={ants}" if ants is not None else ""),
        file=sys.stderr,
    )

    write_tex(build_compare_plots(named_rows, plots=plots), output)
    report_skipped(skipped)
