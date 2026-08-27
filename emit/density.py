"""
density mode — vary.jl ant-count JSON → edge-density scatters.

Pass a folder *prefix* such as ``vary_k2t5i_`` to load ACO settings
``""``, ``N``, ``P``, and ``PN`` (prefer-smaller-side / neighbor-limit
flags). Output:

  Full graph
    - edge density vs |E(D*)|

  Reduced graph
    - reduced edge density vs |E(D*)| / (|U_R|+|V_R|)
    - reduced edge density vs ACO discovery time
    - θ·n + m vs θ-heuristic time (n = |U_R|+|V_R|, m = |E_R|)

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


def _size_per_reduced_node(row, edges):
    """|E(D*)| / (|U_R|+|V_R|); None if reduced sizes missing."""
    nodes = reduced_node_count(row)
    if nodes is None or nodes <= 0:
        return None
    return float(edges) / float(nodes)


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
        r"    xmode=log,",
        r"    ymode=log,",
        r"    grid=major,",
    ]
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


def build_density_plots(variant_rows):
    """
    Build density scatters from {aco_label: [(name, row), ...]}.

    Full-graph density vs size, then reduced-graph density vs size/nodes,
    ACO discovery time vs reduced density, and θ-heuristic time vs θ·n+m.
    """
    aco_order = [
        ACO_LABELS[s] for s in ACO_SUFFIXES if ACO_LABELS[s] in variant_rows
    ]

    full_size, _full_time, _full_heur_size, _full_heur_time = _series_for_x(
        variant_rows, aco_order, edge_density
    )
    # Size: reduced density vs |E(D*)| / nodes.
    red_nd_size, _unused_time, red_nd_heur_size, _unused_heur_time = _series_for_x(
        variant_rows,
        aco_order,
        reduced_edge_density,
        size_y_fn=_size_per_reduced_node,
    )
    # ACO discovery time vs reduced edge density (ACO only).
    _unused_size, red_dens_time, _unused_heur_size2, _unused_heur_time2 = (
        _series_for_x(variant_rows, aco_order, reduced_edge_density)
    )
    # θ-heuristic wall time vs θ·n + m.
    _unused_size2, _unused_time2, _unused_heur_size3, theta_heur_time = (
        _series_for_x(variant_rows, aco_order, theta_n_plus_m)
    )

    dens_x = r"Edge density $|E|/(|U|\,|V|)$"
    red_dens_x = r"Reduced edge density $|E_R|/(|U_R|\,|V_R|)$"
    theta_nm_x = r"$\theta(|U_R|+|V_R|)+|E_R|$"

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
        aco_order=aco_order,
        aco_size=red_nd_size,
        heur_size=red_nd_heur_size,
        xlabel=red_dens_x,
        ylabel=r"Solution size / reduced nodes $|E(D^*)|/(|U_R|+|V_R|)$",
        legend_name="densityRedNdSizeLegend",
        caption=(
            r"Solution size per reduced node vs.\ reduced edge density "
            r"($\theta$-heuristic and ACO flag settings)."
        ),
        label="fig:density-red-nd-size",
    )
    parts.append("")
    parts += _size_figure(
        aco_order=aco_order,
        aco_size=red_dens_time,
        heur_size=[],  # ACO-only
        xlabel=red_dens_x,
        ylabel=r"Discovery time (s)",
        legend_name="densityRedDensTimeLegend",
        caption=(
            r"ACO discovery time vs.\ reduced edge density "
            r"(flag settings)."
        ),
        label="fig:density-red-dens-time",
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
