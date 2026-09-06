"""
Complexity / scaling compare figures.

Plot groups
-----------
  theta-time
    θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    2-panel ratio of discovery time to each bound on matched graphs;
    naive bound n_R^2; practical bound n_E·n_R+|E_R| (n_E=5);
    panels vs n_R and vs |E_R|

  density-size
    edge density vs |E(D*)|

  max-deg-time
    max reduced degree vs ACO-PN discovery time
"""

from __future__ import annotations

from .helpers import (
    ACO_LABEL,
    addplot_scatter,
    addplot_trendline,
    edge_density,
    log_log_regression_stats,
    reduced_max_degree,
    reduced_node_count,
    series_for_x,
    size_figure,
    theta_n_plus_m,
)

_RATIO_COLORS = ("blue", "red")

PLOT_GROUPS = (
    "theta-time",
    "deg-size-time",
    "density-size",
    "max-deg-time",
)

# Paper epoch budget used for the n_E·n_R+|E_R| complexity proxy.
N_E = 5

_NAIVE_LABEL = r"$T / n_R^2$"
_PRACTICAL_LABEL = rf"$T / (n_E\cdot n_R+|E_R|)$ ($n_E{{=}}{N_E}$)"
_SERIES_ORDER = (_NAIVE_LABEL, _PRACTICAL_LABEL)
_SERIES_MARKS = {
    _NAIVE_LABEL: "*",
    _PRACTICAL_LABEL: "o",
}


def discovery_bound_row(row):
    """
    Per-graph discovery time and complexity proxies, or None if incomplete.

    naive_bound = (|U_R|+|V_R|)^2
    practical_bound = n_E · (|U_R|+|V_R|) + |E_R| with n_E = N_E
    """
    n_r = reduced_node_count(row)
    time_s = row.get("aco_time")
    edges = row.get("reduced_edges")
    if n_r is None or time_s is None or float(time_s) <= 0:
        return None
    if edges is None:
        return None
    n_r = float(n_r)
    m_r = float(edges)
    if n_r <= 0 or m_r < 0:
        return None
    practical_bound = float(N_E) * n_r + m_r
    if practical_bound <= 0:
        return None
    return {
        "n_R": n_r,
        "m_R": m_r,
        "time": float(time_s),
        "naive_bound": n_r * n_r,
        "practical_bound": practical_bound,
    }


def matched_discovery_bound_points(named_rows):
    """Graphs with discovery time and all candidate complexity bounds."""
    points = []
    for _name, row in named_rows:
        entry = discovery_bound_row(row)
        if entry is not None:
            points.append(entry)
    return points


def _fmt_log_log_regression_caption(stats):
    if stats is None:
        return "regression unavailable"
    slope, _intercept, r2 = stats
    return rf"slope $= {slope:.2f}$, $R^2 = {r2:.2f}$"


def _ratio_series(matched, x_key):
    """Build (x, T/bound) points and regression stats for each series."""
    naive_pts = [(p[x_key], p["time"] / p["naive_bound"]) for p in matched]
    practical_pts = [
        (p[x_key], p["time"] / p["practical_bound"]) for p in matched
    ]
    return {
        _NAIVE_LABEL: naive_pts,
        _PRACTICAL_LABEL: practical_pts,
    }, {
        _NAIVE_LABEL: log_log_regression_stats(naive_pts),
        _PRACTICAL_LABEL: log_log_regression_stats(practical_pts),
    }


def _emit_ratio_panel(*, title, xlabel, series, legend_name=None):
    """One groupplot panel of Time/bound scatter + log--log trends."""
    lines = [
        r"\nextgroupplot[",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        rf"    xlabel={{{xlabel}}},",
        r"    ylabel={Time / bound},",
        r"    xmode=log,",
        r"    ymode=log,",
    ]
    if legend_name:
        lines += [
            rf"    legend to name={legend_name},",
            r"    legend columns=-1,",
            r"    legend style={draw=none, fill=none, font=\normalsize},",
            r"    legend cell align=left,",
        ]
    lines.append(r"  ]")
    for i, lab in enumerate(_SERIES_ORDER):
        color = _RATIO_COLORS[i % len(_RATIO_COLORS)]
        mark = _SERIES_MARKS[lab]
        pts = series[lab]
        lines += addplot_scatter(
            lab,
            pts,
            with_legend=bool(legend_name),
            alpha=0.55,
            color=color,
            mark=mark,
        )
        lines += addplot_trendline(pts, color=color)
    return lines


def bound_ratio_figure(*, matched):
    """Discovery time / bound vs. n_R and |E_R| (2-panel groupplot)."""
    n = len(matched)
    series_n, stats_n = _ratio_series(matched, "n_R")
    series_m, stats_m = _ratio_series(matched, "m_R")
    legend_name = "compareBoundRatioLegend"

    caption = (
        rf"Ratio of ACO-PN discovery time to the theoretical and practical "
        rf"complexity bounds on the {n} $(k, \theta) = (2, 5)$ graphs, with "
        rf"log--log trends. Left: time / bound vs.\ $n_R$; Right: time / "
        rf"bound vs.\ $|E_R|$. An accurate bound should stay near-flat, "
        rf"meaning log--log slope $\approx 0$. The loose $n_R^2$ ratio "
        rf"declines with graph size "
        rf"({_fmt_log_log_regression_caption(stats_n[_NAIVE_LABEL])} vs.\ "
        rf"$n_R$; "
        rf"{_fmt_log_log_regression_caption(stats_m[_NAIVE_LABEL])} vs.\ "
        rf"$|E_R|$), while $n_E\cdot n_R+|E_R|$ stays nearer to flat "
        rf"({_fmt_log_log_regression_caption(stats_n[_PRACTICAL_LABEL])} "
        rf"vs.\ $n_R$; "
        rf"{_fmt_log_log_regression_caption(stats_m[_PRACTICAL_LABEL])} "
        rf"vs.\ $|E_R|$). The practical bound's slope is much closer to $0$ "
        rf"than the naive bound's, and a low $R^2$ suggests noise in the "
        rf"data rather than a definite nonzero slope."
    )

    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={",
        r"      group size=2 by 1,",
        r"      horizontal sep=52pt},",
        r"    width=0.45\textwidth,",
        r"    height=0.42\textwidth,",
        r"    grid=major,",
        r"    ylabel style={font=\small},",
        r"    xlabel style={font=\small},",
        r"  ]",
    ]
    lines += _emit_ratio_panel(
        title=r"vs.\ reduced nodes",
        xlabel=r"$n_R = |U_R|+|V_R|$",
        series=series_n,
        legend_name=legend_name,
    )
    lines += _emit_ratio_panel(
        title=r"vs.\ reduced edges",
        xlabel=r"$|E_R|$",
        series=series_m,
    )
    lines += [
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        r"",
        r"  \begin{center}",
        rf"  \pgfplotslegendfromname{{{legend_name}}}",
        r"  \end{center}",
        r"",
        rf"  \caption{{{caption}}}",
        r"  \label{fig:compare-bound-ratio}",
        r"\end{figure}",
    ]
    return lines


def density_size_figure(named_rows):
    variant_rows = {ACO_LABEL: named_rows}
    aco_order = [ACO_LABEL]
    full_size, _full_time, _full_heur_size, _full_heur_time = series_for_x(
        variant_rows, aco_order, edge_density
    )
    return size_figure(
        aco_order=aco_order,
        aco_size=full_size,
        heur_size=[],
        xlabel=r"Edge density $|E|/(|U|\,|V|)$",
        legend_name="compareSizeLegend",
        caption=r"ACO-PN solution size vs.\ graph edge density.",
        label="fig:compare-density-size",
    )


def theta_time_figure(named_rows):
    variant_rows = {ACO_LABEL: named_rows}
    aco_order = [ACO_LABEL]
    _unused_size, _unused_time, _unused_heur_size, theta_heur_time = (
        series_for_x(variant_rows, aco_order, theta_n_plus_m)
    )
    return size_figure(
        aco_order=[],
        aco_size={},
        heur_size=theta_heur_time,
        xlabel=r"$\theta(|U_R|+|V_R|)+|E_R|$",
        ylabel=r"Time (s)",
        caption=(
            r"$\theta$-heuristic wall time vs.\ "
            r"$\theta(|U_R|+|V_R|)+|E_R|$."
        ),
        label="fig:compare-theta-nm-time",
        width="0.72\\textwidth",
        height="0.48\\textwidth",
    )


def max_deg_time_figure(named_rows):
    variant_rows = {ACO_LABEL: named_rows}
    aco_order = [ACO_LABEL]
    _unused_size, max_deg_time, _unused_heur_size, _unused_heur_time = (
        series_for_x(variant_rows, aco_order, reduced_max_degree)
    )
    return size_figure(
        aco_order=aco_order,
        aco_size=max_deg_time,
        heur_size=[],
        xlabel=r"Maximum reduced degree $\Delta(G_R)$",
        ylabel=r"Discovery time (s)",
        legend_name="compareMaxDegTimeLegend",
        caption=(
            r"ACO-PN discovery time vs.\ maximum degree in the reduced graph."
        ),
        label="fig:compare-max-deg-time",
    )


def deg_size_time_figure(named_rows):
    matched = matched_discovery_bound_points(named_rows)
    return bound_ratio_figure(matched=matched)


BUILDERS = {
    "density-size": density_size_figure,
    "theta-time": theta_time_figure,
    "max-deg-time": max_deg_time_figure,
    "deg-size-time": deg_size_time_figure,
}
