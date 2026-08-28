"""
density mode — vary.jl ant-count JSON → edge-density / degree scatters.

Pass a folder *prefix* such as ``vary_k2t5i_`` to load ACO settings
``""``, ``N``, ``P``, and ``PN`` (prefer-smaller-side / neighbor-limit
flags). Output:

  Full graph
    - edge density vs |E(D*)|

  Reduced graph
    - θ·n + m vs θ-heuristic time (n = |U_R|+|V_R|, m = |E_R|)
    - max reduced degree vs ACO discovery time
    - (max reduced degree + n) · |E(D*)| vs ACO discovery time
    - (avg reduced degree + n) · |E(D*)| vs ACO discovery time
  ACO-PN comparisons (linear axes)
    - reduced edge density vs t(ACO|ACO-P|ACO-N) / t(ACO-PN)

ACO points use the same best-trial / discovery rules as table mode.
"""

from __future__ import annotations

import os
import sys

from .common import (
    display_name,
    list_json_paths,
    load_json,
    report_skipped,
    write_tex,
)
from .table import summarize_file

# Flag suffixes written by scripts/vary.bash (P = prefer, N = neighbor).
ACO_SUFFIXES = ("", "N", "P", "PN")
ACO_LABELS = {
    "": "ACO",
    "N": "ACO-N",
    "P": "ACO-P",
    "PN": "ACO-PN",
}
HEUR_LABEL = r"$\theta$-Heuristic"


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


def _addplot_scatter(label, points, *, with_legend=True):
    coords = _scatter_coords(points)
    lines = [r"\addplot+[only marks] coordinates {" + coords + "};"]
    if with_legend:
        lines.append(r"\addlegendentry{" + label + "}")
    return lines


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
    for lab in aco_order:
        lines += _addplot_scatter(
            lab, aco_size[lab], with_legend=bool(legend_name)
        )
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
            rf"  \ref{{{legend_name}}}",
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


def _pn_ratio_series(variant_rows, x_fn, *, y_pn_over_other):
    """
    Per-dataset ratios of ACO-PN vs ACO / ACO-P / ACO-N.

    y_pn_over_other(pn_row, other_row) → float ratio, or None to skip.
    Returns (compare_order, series) with series[label] = [(x, y), ...].
    """
    pn_label = ACO_LABELS["PN"]
    compare_order = [
        ACO_LABELS[s]
        for s in ("", "P", "N")
        if ACO_LABELS[s] in variant_rows
    ]
    if pn_label not in variant_rows or not compare_order:
        return compare_order, {}

    pn_by_name = _rows_by_name(variant_rows[pn_label])
    series = {lab: [] for lab in compare_order}
    for lab in compare_order:
        for name, other in variant_rows[lab]:
            pn = pn_by_name.get(name)
            if pn is None:
                continue
            x = x_fn(pn)
            if x is None:
                x = x_fn(other)
            if x is None:
                continue
            y = y_pn_over_other(pn, other)
            if y is None:
                continue
            series[lab].append((x, y))
    return compare_order, series


def _time_ratio_other_over_pn(pn_row, other_row):
    """t(other) / t(ACO-PN); None if either time is missing or non-positive."""
    t_pn = pn_row.get("aco_time")
    t_other = other_row.get("aco_time")
    if t_pn is None or t_other is None:
        return None
    t_pn, t_other = float(t_pn), float(t_other)
    if t_pn <= 0 or t_other <= 0:
        return None
    return t_other / t_pn


def build_density_plots(variant_rows):
    """
    Build density scatters from {aco_label: [(name, row), ...]}.

    Full-graph density vs size, θ-heuristic time vs θ·n+m,
    reduced max/avg degree composite metrics vs ACO time,
    and linear ACO-PN time-ratio comparisons.
    """
    aco_order = [
        ACO_LABELS[s] for s in ACO_SUFFIXES if ACO_LABELS[s] in variant_rows
    ]

    full_size, _full_time, _full_heur_size, _full_heur_time = _series_for_x(
        variant_rows, aco_order, edge_density
    )
    # θ-heuristic wall time vs θ·n + m.
    _unused_size2, _unused_time2, _unused_heur_size3, theta_heur_time = (
        _series_for_x(variant_rows, aco_order, theta_n_plus_m)
    )
    # ACO discovery time vs max reduced degree (and degree+nodes·size).
    _u3, max_deg_time, _u4, _u5 = _series_for_x(
        variant_rows, aco_order, reduced_max_degree
    )
    _u6, max_deg_size_time, _u7, _u8 = _series_for_x(
        variant_rows, aco_order, _deg_plus_nodes_times_size(reduced_max_degree)
    )
    _u9, avg_deg_size_time, _u10, _u11 = _series_for_x(
        variant_rows, aco_order, _deg_plus_nodes_times_size(reduced_avg_degree)
    )
    # Linear: t(other) / t(ACO-PN).
    time_cmp_order, time_ratio = _pn_ratio_series(
        variant_rows,
        reduced_edge_density,
        y_pn_over_other=_time_ratio_other_over_pn,
    )

    dens_x = r"Edge density $|E|/(|U|\,|V|)$"
    red_dens_x = r"Reduced edge density $|E_R|/(|U_R|\,|V_R|)$"
    theta_nm_x = r"$\theta(|U_R|+|V_R|)+|E_R|$"
    max_deg_x = r"Maximum reduced degree $\Delta(G_R)$"
    max_deg_size_x = (
        r"$(\Delta(G_R)+|U_R|+|V_R|)\cdot|E(D^*)|$"
    )
    avg_deg_size_x = (
        r"$(\bar{d}(G_R)+|U_R|+|V_R|)\cdot|E(D^*)|$"
    )
    disc_y = r"Discovery time (s)"

    parts = []
    parts += _size_figure(
        aco_order=aco_order,
        aco_size=full_size,
        heur_size=[],  # ACO-only: θ clutters the density–size comparison
        xlabel=dens_x,
        legend_name="densitySizeLegend",
        caption=(
            r"ACO solution size vs.\ graph edge density "
            r"(flag settings)."
        ),
        label="fig:density-size",
    )
    parts.append("")
    parts += _size_figure(
        aco_order=[],
        aco_size={},
        heur_size=theta_heur_time,
        xlabel=theta_nm_x,
        ylabel=r"Time (s)",
        caption=(
            r"$\theta$-heuristic wall time vs.\ "
            r"$\theta(|U_R|+|V_R|)+|E_R|$."
        ),
        label="fig:density-theta-nm-time",
        width="0.72\\textwidth",
        height="0.48\\textwidth",
    )
    parts.append("")
    parts += _size_figure(
        aco_order=aco_order,
        aco_size=max_deg_time,
        heur_size=[],
        xlabel=max_deg_x,
        ylabel=disc_y,
        legend_name="densityMaxDegTimeLegend",
        caption=(
            r"ACO discovery time vs.\ maximum degree in the reduced graph "
            r"(flag settings)."
        ),
        label="fig:density-max-deg-time",
    )
    parts.append("")
    parts += _size_figure(
        aco_order=aco_order,
        aco_size=max_deg_size_time,
        heur_size=[],
        xlabel=max_deg_size_x,
        ylabel=disc_y,
        legend_name="densityMaxDegSizeTimeLegend",
        caption=(
            r"ACO discovery time vs.\ "
            r"$(\Delta(G_R)+|U_R|+|V_R|)\cdot|E(D^*)|$ "
            r"(flag settings)."
        ),
        label="fig:density-max-deg-size-time",
    )
    parts.append("")
    parts += _size_figure(
        aco_order=aco_order,
        aco_size=avg_deg_size_time,
        heur_size=[],
        xlabel=avg_deg_size_x,
        ylabel=disc_y,
        legend_name="densityAvgDegSizeTimeLegend",
        caption=(
            r"ACO discovery time vs.\ "
            r"$(\bar{d}(G_R)+|U_R|+|V_R|)\cdot|E(D^*)|$ "
            r"(flag settings)."
        ),
        label="fig:density-avg-deg-size-time",
    )

    if time_ratio and any(time_ratio[lab] for lab in time_cmp_order):
        parts.append("")
        parts += _size_figure(
            aco_order=time_cmp_order,
            aco_size=time_ratio,
            heur_size=[],
            xlabel=red_dens_x,
            ylabel=r"Other time / ACO-PN time",
            legend_name="densityPnTimeRatioLegend",
            caption=(
                r"ACO / ACO-P / ACO-N discovery time divided by ACO-PN "
                r"discovery time vs.\ reduced edge density "
                r"(values $>1$ mean the other variant is slower)."
            ),
            label="fig:density-pn-time-ratio",
            log_axes=False,
            log_y=True,
            y_equals_one=True,
            xmin=0,
            xmax=1,
        )

    return "\n".join(parts)


def run(directory, output, ants=None):
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
            f"# density [{label}]: {len(named_rows)} dataset(s)"
            + (f"; ants={ants}" if ants is not None else ""),
            file=sys.stderr,
        )

    if not variant_rows:
        raise SystemExit("No usable vary JSON files -- nothing to plot.")

    write_tex(build_density_plots(variant_rows), output)
    report_skipped(skipped)
