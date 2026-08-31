"""
compare mode — vary.jl ant-count JSON → complexity / flag-setting figures.

Pass a folder *prefix* such as ``vary_k2t5i_`` to load ACO settings
``""``, ``N``, ``P``, and ``PN`` (prefer-smaller-side / neighbor-limit
flags). Select plot groups with ``--plots`` (comma-separated):

  theta-time
    - θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    - ACO discovery time vs (avg degree + |U_R|+|V_R|)·|E(D*)|

  density-size
    - edge density vs |E(D*)|

  max-deg-time
    - max reduced degree vs ACO discovery time

  pn
    - stacked bar: % instances where ACO-PN is worse / equal / better
    - summary table of size ratios vs ACO-PN

ACO points use the same best-trial / discovery rules as table mode.
"""

from __future__ import annotations

import math
import os
import statistics
import sys

from .common import (
    display_name,
    list_json_paths,
    load_json,
    report_skipped,
    write_tex,
)
from .table import summarize_file

PLOT_GROUPS = (
    "theta-time",
    "deg-size-time",
    "density-size",
    "max-deg-time",
    "pn",
)

# Flag suffixes written by scripts/vary.bash (P = prefer, N = neighbor).
ACO_SUFFIXES = ("", "N", "P", "PN")
ACO_LABELS = {
    "": "ACO",
    "N": "ACO-N",
    "P": "ACO-P",
    "PN": "ACO-PN",
}
HEUR_LABEL = r"$\theta$-Heuristic"


def _symbolic_coord(label):
    """Brace-wrap pgfplots symbolic coords that contain hyphens."""
    if "-" in label:
        return "{" + label + "}"
    return label


def resolve_aco_dirs(directory):
    """
    Expand a vary folder prefix into ACO flag directories.

    ``vary_k2t5i_`` → ``vary_k2t5i_``, ``vary_k2t5i_N``, ``vary_k2t5i_P``,
    ``vary_k2t5i_PN`` (whichever exist). A bare existing directory also
    works as a single-setting input.
    """
    prefix = os.path.abspath(directory.rstrip(os.sep))
    found = []
    for suffix in ACO_SUFFIXES:
        path = prefix + suffix
        if os.path.isdir(path):
            found.append((ACO_LABELS[suffix], path))
    if found:
        return found
    raise SystemExit(
        f"No ACO result directories found for prefix {directory!r} "
        f"(tried suffixes {list(ACO_SUFFIXES)})"
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
    return None if d is None else float(d)


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


def _deg_plus_nodes_times_size(deg_fn):
    """(degree + |U_R|+|V_R|) · |E(D*)| using ACO best-trial size."""

    def x_fn(row):
        deg = deg_fn(row)
        nodes = reduced_node_count(row)
        size = row.get("aco_edges")
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


def collect_variant_rows(directory, ants=None):
    """Load named summarize_file rows from one vary directory."""
    named_rows = []
    skipped = []
    for path in list_json_paths(directory):
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue
        row = summarize_file(data, ants=ants)
        if row is None:
            skipped.append((path, "not a vary.jl ant-count result"))
            continue
        named_rows.append((display_name(path, data), row))
    return named_rows, skipped


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


def _rows_by_name(named_rows):
    """Map display name → summarize_file row (last wins on duplicates)."""
    return {name: row for name, row in named_rows}


def _compare_order(variant_rows):
    """ACO / ACO-P / ACO-N labels present alongside ACO-PN."""
    pn_label = ACO_LABELS["PN"]
    order = [
        ACO_LABELS[s]
        for s in ("", "P", "N")
        if ACO_LABELS[s] in variant_rows
    ]
    if pn_label not in variant_rows or not order:
        return []
    return order


def _matched_rows_by_name(variant_rows):
    """
    Per-dataset rows for ACO-PN vs ACO / ACO-P / ACO-N.

    Returns (compare_order, {dataset_name: {label: row, ...}}).
    """
    compare_order = _compare_order(variant_rows)
    if not compare_order:
        return compare_order, {}

    pn_label = ACO_LABELS["PN"]
    pn_by_name = _rows_by_name(variant_rows[pn_label])
    matched = {}
    for name, pn_row in pn_by_name.items():
        entry = {pn_label: pn_row}
        for lab in compare_order:
            other = _rows_by_name(variant_rows[lab]).get(name)
            if other is None:
                break
            entry[lab] = other
        else:
            matched[name] = entry
    return compare_order, matched


def _size_ratio_other_over_pn(pn_row, other_row):
    """|E(D*)|(other) / |E(D*)|(ACO-PN); None if PN size is missing or zero."""
    e_pn = pn_row.get("aco_edges")
    e_other = other_row.get("aco_edges")
    if e_pn is None or e_other is None:
        return None
    e_pn, e_other = float(e_pn), float(e_other)
    if e_pn <= 0:
        return None
    return e_other / e_pn


def _quality_outcome(pn_row, other_row):
    """
    Compare ACO-PN vs another variant on solution size.

    Returns ``"worse"`` (PN smaller), ``"equal"``, or ``"better"`` (PN larger).
    """
    e_pn = pn_row.get("aco_edges")
    e_other = other_row.get("aco_edges")
    if e_pn is None or e_other is None:
        return None
    e_pn, e_other = int(e_pn), int(e_other)
    if e_pn > e_other:
        return "better"
    if e_pn < e_other:
        return "worse"
    return "equal"


def _collect_quality_stats(matched, compare_order):
    """
    Per compare-label quality breakdown and size-ratio summaries vs ACO-PN.
    """
    pn_label = ACO_LABELS["PN"]
    stats = {}
    for lab in compare_order:
        outcomes = {"worse": 0, "equal": 0, "better": 0}
        ratios = []
        for rows in matched.values():
            outcome = _quality_outcome(rows[pn_label], rows[lab])
            if outcome is None:
                continue
            outcomes[outcome] += 1
            ratio = _size_ratio_other_over_pn(rows[pn_label], rows[lab])
            if ratio is not None:
                ratios.append(ratio)
        total = sum(outcomes.values())
        if total == 0:
            continue
        stats[lab] = {
            "outcomes": outcomes,
            "total": total,
            "ratios": ratios,
            "pct_worse": 100.0 * outcomes["worse"] / total,
            "pct_equal": 100.0 * outcomes["equal"] / total,
            "pct_better": 100.0 * outcomes["better"] / total,
        }
    return stats


def _quality_stacked_bar_figure(*, compare_order, quality_stats):
    """Stacked bar: percent worse / equal / better for ACO-PN vs each other variant."""
    labels = [lab for lab in compare_order if lab in quality_stats]
    if not labels:
        return []

    xpos = list(range(1, len(labels) + 1))
    tick = ",".join(str(i) for i in xpos)
    tick_labels = ",".join(_symbolic_coord(label) for label in labels)

    def _coords(key):
        return " ".join(
            f"({xi},{quality_stats[lab][key]:.6g})"
            for xi, lab in zip(xpos, labels)
        )

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{axis}[",
        r"    width=0.72\textwidth,",
        r"    height=0.48\textwidth,",
        r"    ybar stacked,",
        r"    bar width=0.55,",
        r"    ymin=0, ymax=100,",
        r"    ylabel={Instances (\%)},",
        r"    xtick={" + tick + "},",
        r"    xticklabels={" + tick_labels + "},",
        r"    legend style={draw=none, fill=white, font=\small},",
        r"    legend cell align=left,",
        r"    legend columns=3,",
        r"  ]",
        r"\addplot+[fill=red!60] coordinates {" + _coords("pct_worse") + "};",
        r"\addlegendentry{ACO-PN worse}",
        r"\addplot+[fill=gray!45] coordinates {" + _coords("pct_equal") + "};",
        r"\addlegendentry{Equal}",
        r"\addplot+[fill=green!50!black] coordinates {"
        + _coords("pct_better")
        + "};",
        r"\addlegendentry{ACO-PN better}",
        r"  \end{axis}",
        r"  \end{tikzpicture}",
        r"  \caption{Solution quality of ACO-PN vs.\ other flag settings "
        r"(by $|E(D^*)|$ on matched instances).}",
        r"  \label{fig:compare-pn-quality-bar}",
        r"\end{figure}",
    ]
    return lines


def _fmt_pct(value):
    return f"{value:.1f}\\%"


def _fmt_ratio(value):
    return f"{value:.3f}"


def _quality_summary_table(*, compare_order, quality_stats):
    """LaTeX table: size-ratio mean/median and outcome percentages."""
    rows = []
    for lab in compare_order:
        st = quality_stats.get(lab)
        if not st:
            continue
        ratios = st["ratios"]
        mean_r = statistics.mean(ratios) if ratios else None
        med_r = statistics.median(ratios) if ratios else None
        rows.append(
            "    "
            + " & ".join(
                [
                    lab,
                    "--" if mean_r is None else _fmt_ratio(mean_r),
                    "--" if med_r is None else _fmt_ratio(med_r),
                    _fmt_pct(st["pct_equal"]),
                    _fmt_pct(st["pct_better"]),
                    _fmt_pct(st["pct_worse"]),
                ]
            )
            + r" \\"
        )
    if not rows:
        return []

    return [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Solution-quality comparison vs.\ ACO-PN "
        r"(Other $|E(D^*)|$ / ACO-PN $|E(D^*)|$; outcome percentages).}",
        r"  \label{tab:compare-pn-quality}",
        r"  \begin{tabular}{lrrrrr}",
        r"    \toprule",
        r"    Method & Mean ratio & Median ratio & \% equal & \% PN better & \% PN worse \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]


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


def build_compare_plots(variant_rows, plots=None):
    """
    Build selected compare figures from {aco_label: [(name, row), ...]}.

    Plot groups are listed in ``PLOT_GROUPS``; pass ``plots`` as a
    comma-separated subset (e.g. ``theta-time,deg-size-time``).
    """
    selected = _parse_plots(plots)
    aco_order = [
        ACO_LABELS[s] for s in ACO_SUFFIXES if ACO_LABELS[s] in variant_rows
    ]

    ctx = {}
    need_pn = "pn" in selected
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
        _u9, avg_deg_size_time, _u10, _u11 = _series_for_x(
            variant_rows, aco_order, _deg_plus_nodes_times_size(reduced_avg_degree)
        )
        ctx["avg_deg_size_time"] = avg_deg_size_time

    if need_pn:
        compare_order, matched = _matched_rows_by_name(variant_rows)
        quality_stats = _collect_quality_stats(matched, compare_order)
        ctx.update(
            {
                "aco_order": aco_order,
                "compare_order": compare_order,
                "quality_stats": quality_stats,
            }
        )

    dens_x = r"Edge density $|E|/(|U|\,|V|)$"
    theta_nm_x = r"$\theta(|U_R|+|V_R|)+|E_R|$"
    max_deg_x = r"Maximum reduced degree $\Delta(G_R)$"
    avg_deg_size_x = r"$(\bar{d}(G_R)+|U_R|+|V_R|)\cdot|E(D^*)|$"
    disc_y = r"Discovery time (s)"

    group_builders = {
        "density-size": lambda: _size_figure(
            aco_order=aco_order,
            aco_size=ctx["full_size"],
            heur_size=[],
            xlabel=dens_x,
            legend_name="compareSizeLegend",
            caption=(
                r"ACO solution size vs.\ graph edge density "
                r"(flag settings)."
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
                r"ACO discovery time vs.\ maximum degree in the reduced graph "
                r"(flag settings)."
            ),
            label="fig:compare-max-deg-time",
        ),
        "deg-size-time": lambda: _size_figure(
            aco_order=aco_order,
            aco_size=ctx["avg_deg_size_time"],
            heur_size=[],
            xlabel=avg_deg_size_x,
            ylabel=disc_y,
            legend_name="compareAvgDegSizeTimeLegend",
            caption=(
                r"ACO discovery time vs.\ "
                r"$(\bar{d}(G_R)+|U_R|+|V_R|)\cdot|E(D^*)|$ "
                r"(flag settings; dashed lines show log--log trends)."
            ),
            label="fig:compare-avg-deg-size-time",
            mark_alpha=0.5,
            trend_lines=True,
        ),
        "pn": lambda: _pn_figures(ctx),
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


def _pn_figures(ctx):
    """ACO-PN comparison figures and quality summary table."""
    compare_order = ctx["compare_order"]
    quality_stats = ctx["quality_stats"]

    pn_figures = [
        _quality_stacked_bar_figure(
            compare_order=compare_order,
            quality_stats=quality_stats,
        ),
        _quality_summary_table(
            compare_order=compare_order,
            quality_stats=quality_stats,
        ),
    ]
    parts = []
    for fig in pn_figures:
        if fig:
            if parts:
                parts.append("")
            parts += fig
    return parts


def run(directory, output, ants=None, plots=None):
    dirs = resolve_aco_dirs(directory)
    variant_rows = {}
    skipped = []

    for label, path in dirs:
        named_rows, skip = collect_variant_rows(path, ants=ants)
        skipped.extend(skip)
        if not named_rows:
            print(f"# warning: no usable rows in {path}", file=sys.stderr)
            continue
        variant_rows[label] = named_rows
        print(
            f"# compare [{label}]: {len(named_rows)} dataset(s)"
            + (f"; ants={ants}" if ants is not None else ""),
            file=sys.stderr,
        )

    if not variant_rows:
        raise SystemExit("No usable vary JSON files -- nothing to plot.")

    write_tex(build_compare_plots(variant_rows, plots=plots), output)
    report_skipped(skipped)
