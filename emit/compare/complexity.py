"""
Complexity / scaling compare figures.

Plot groups
-----------
  theta-time
    θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    ratio of discovery time to each bound vs |U_R|+|V_R| (matched graphs)

  density-size
    edge density vs |E(D*)|

  max-deg-time
    max reduced degree vs ACO-PN discovery time
"""

from __future__ import annotations

from .helpers import (
    ACO_LABEL,
    edge_density,
    log_log_regression_stats,
    reduced_max_degree,
    reduced_node_count,
    series_for_x,
    size_figure,
    solution_node_count,
    theta_n_plus_m,
)

PLOT_GROUPS = (
    "theta-time",
    "deg-size-time",
    "density-size",
    "max-deg-time",
)


def discovery_bound_row(row):
    """
    Per-graph discovery time and both complexity proxies, or None if incomplete.

    naive_bound = (|U_R|+|V_R|)^2
    practical_bound = (Δ(G_R)+|U_R|+|V_R|)·(|U_D|+|V_D|)
    """
    n_r = reduced_node_count(row)
    time_s = row.get("aco_time")
    if n_r is None or time_s is None or float(time_s) <= 0:
        return None
    max_deg = reduced_max_degree(row)
    n_d = solution_node_count(row)
    if max_deg is None or n_d is None:
        return None
    n_r = float(n_r)
    return {
        "n_R": n_r,
        "time": float(time_s),
        "naive_bound": n_r * n_r,
        "practical_bound": (float(max_deg) + n_r) * float(n_d),
    }


def matched_discovery_bound_points(named_rows):
    """Graphs with discovery time and both candidate complexity bounds."""
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
    return rf"slope ${slope:.2f}$, $R^2={r2:.2f}$"


def bound_ratio_figure(*, matched):
    """Discovery time divided by each bound vs. |U_R|+|V_R| on matched graphs."""
    naive_label = r"$T / (|U_R|+|V_R|)^2$"
    practical_label = (
        r"$T / ((\Delta_R+|U_R|+|V_R|)\cdot(|U_D|+|V_D|))$"
    )
    naive_pts = [(p["n_R"], p["time"] / p["naive_bound"]) for p in matched]
    practical_pts = [
        (p["n_R"], p["time"] / p["practical_bound"]) for p in matched
    ]
    naive_stats = log_log_regression_stats(naive_pts)
    practical_stats = log_log_regression_stats(practical_pts)
    n = len(matched)
    return size_figure(
        aco_order=[naive_label, practical_label],
        aco_size={
            naive_label: naive_pts,
            practical_label: practical_pts,
        },
        heur_size=[],
        xlabel=r"$|U_R|+|V_R|$",
        ylabel=r"Time / bound",
        legend_name="compareBoundRatioLegend",
        caption=(
            rf"Ratio of ACO-PN discovery time to each candidate complexity bound "
            rf"vs.\ $|U_R|+|V_R|$ on {n} matched graphs "
            rf"(dashed lines show log--log trends). "
            rf"A proportional bound should yield a flat ratio (log--log slope "
            rf"$\approx 0$). The naive $n_R^2$ bound declines "
            rf"({_fmt_log_log_regression_caption(naive_stats)}), while the "
            rf"practical bound is near-flat "
            rf"({_fmt_log_log_regression_caption(practical_stats)})."
        ),
        label="fig:compare-bound-ratio",
        mark_alpha=0.55,
        trend_lines=True,
        series_marks={naive_label: "*", practical_label: "square"},
    )


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
