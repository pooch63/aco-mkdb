"""Shared metrics and pgfplots helpers for compare-mode figures."""

from __future__ import annotations

import math

ACO_LABEL = "ACO-PN"
HEUR_LABEL = r"$\theta$-Heuristic"

_SERIES_COLORS = (
    "blue",
    "red",
    "green!60!black",
    "orange",
    "purple",
    "brown",
)


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


def deg_plus_nodes_times(deg_fn, size_fn):
    """(degree + |U_R|+|V_R|) · size_fn(row) for reduced-graph degree proxies."""

    def x_fn(row):
        deg = deg_fn(row)
        nodes = reduced_node_count(row)
        size = size_fn(row)
        if deg is None or nodes is None or size is None:
            return None
        return (float(deg) + float(nodes)) * float(size)

    return x_fn


def scatter_coords(points):
    """points: iterable of (x, y) floats → pgfplots coordinates body."""
    parts = []
    for x, y in points:
        if x is None or y is None:
            continue
        parts.append(f"({float(x):.8g},{float(y):.8g})")
    return " ".join(parts)


def log_log_regression_stats(points):
    """Fit log(y) = slope * log(x) + intercept; return (slope, intercept, r2)."""
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
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def log_log_regression(points):
    """Fit log(y) = slope * log(x) + intercept; return (slope, intercept)."""
    stats = log_log_regression_stats(points)
    if stats is None:
        return None
    slope, intercept, _r2 = stats
    return slope, intercept


def trend_line_coords(points):
    """Endpoints for a log--log least-squares trend line."""
    reg = log_log_regression(points)
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


def addplot_scatter(
    label, points, *, with_legend=True, alpha=None, color=None, mark=None
):
    coords = scatter_coords(points)
    opts = ["only marks"]
    if mark:
        opts.append(f"mark={mark}")
    if alpha is not None:
        opts.append(f"opacity={alpha}")
        opts.append(f"mark options={{opacity={alpha}}}")
    if color:
        opts.append(color)
    lines = [rf"\addplot+[{', '.join(opts)}] coordinates {{{coords}}};"]
    if with_legend:
        lines.append(r"\addlegendentry{" + label + "}")
    return lines


def addplot_trendline(points, *, color=None, alpha=0.85):
    coords = trend_line_coords(points)
    if coords is None:
        return []
    coord_str = scatter_coords(coords)
    opts = ["mark=none", "thick", "dashed", "forget plot"]
    if alpha is not None:
        opts.append(f"opacity={alpha}")
    if color:
        opts.append(color)
    return [rf"\addplot+[{', '.join(opts)}] coordinates {{{coord_str}}};"]


def series_for_x(variant_rows, aco_order, x_fn, *, size_y_fn=None):
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


def size_figure(
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
    placement="H",
    series_marks=None,
):
    lines = [
        rf"\begin{{figure}}[{placement}]",
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
        lines += addplot_scatter(
            HEUR_LABEL, heur_size, with_legend=bool(legend_name)
        )
    for i, lab in enumerate(aco_order):
        color = (
            _SERIES_COLORS[i % len(_SERIES_COLORS)]
            if mark_alpha is not None or trend_lines
            else None
        )
        pts = aco_size[lab]
        mark = series_marks.get(lab) if series_marks else None
        lines += addplot_scatter(
            lab,
            pts,
            with_legend=bool(legend_name),
            alpha=mark_alpha,
            color=color,
            mark=mark,
        )
        if trend_lines:
            lines += addplot_trendline(pts, color=color)
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
