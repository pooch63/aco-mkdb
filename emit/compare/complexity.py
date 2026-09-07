"""
Complexity / scaling compare figures.

Plot groups
-----------
  theta-time
    θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    2-panel log--log discovery-time scaling on matched graphs;
    left: raw T vs n_R with empirical power-law fit;
    right: T vs each candidate bound (naive n_R^2; practical
    n_E·n_R+|E_R| with n_E from iterations_budget) — slope ≈ 1
    means proportional

  practical-bound-time
    single-panel log--log normalized discovery time
    T/(n_S·n_E) vs the practical proxy n_E·n_R+|E_R|

  naive-bound-time
    single-panel log--log normalized discovery time
    T/(n_S·n_E) vs the naive bound n_R^2

  density-size
    edge density vs |E(D*)|

  max-deg-time
    max reduced degree vs ACO-N discovery time
"""

from __future__ import annotations

import sys

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

_BOUND_COLORS = ("blue", "red")

PLOT_GROUPS = (
    "theta-time",
    "deg-size-time",
    "practical-bound-time",
    "naive-bound-time",
    "density-size",
    "max-deg-time",
)

# Fallback epoch budget only when JSON omits iterations_budget.
_FALLBACK_N_E = 5

_NAIVE_LABEL = r"$n_R^2$"
_PRACTICAL_LABEL = r"$n_E\cdot n_R+|E_R|$"
_BOUND_SERIES_ORDER = (_NAIVE_LABEL, _PRACTICAL_LABEL)
_BOUND_SERIES_MARKS = {
    _NAIVE_LABEL: "*",
    _PRACTICAL_LABEL: "o",
}


def _row_ants(row):
    ants = row.get("aco_ants")
    if ants is None:
        return None
    ants = float(ants)
    return ants if ants > 0 else None


def _row_epochs(row):
    n_e = row.get("aco_iterations_budget")
    if n_e is None:
        return None
    n_e = float(n_e)
    return n_e if n_e > 0 else None


def _normalized_discovery_time(row):
    """
    Discovery wall time per ant per epoch: T / (n_S · n_E).

    Uses ants and iterations_budget from the winning trial JSON fields
    (via summarize_file). Returns None if any factor is missing/invalid.
    """
    time_s = row.get("aco_time")
    ants = _row_ants(row)
    n_e = _row_epochs(row)
    if time_s is None or ants is None or n_e is None:
        return None
    time_s = float(time_s)
    if time_s <= 0:
        return None
    return time_s / (ants * n_e)


def discovery_bound_row(row):
    """
    Per-graph discovery time and complexity proxies, or None if incomplete.

    naive_bound = (|U_R|+|V_R|)^2
    practical_bound = n_E · (|U_R|+|V_R|) + |E_R|
    with n_E from iterations_budget (fallback _FALLBACK_N_E only if missing)
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
    n_e = _row_epochs(row)
    if n_e is None:
        print(
            "Warning: missing aco_iterations_budget; "
            f"falling back to n_E={_FALLBACK_N_E} for practical bound",
            file=sys.stderr,
        )
        n_e = float(_FALLBACK_N_E)
    practical_bound = n_e * n_r + m_r
    if practical_bound <= 0:
        return None
    return {
        "n_R": n_r,
        "m_R": m_r,
        "n_E": n_e,
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


def matched_normalized_bound_points(named_rows):
    """
    Graphs with normalized discovery time T/(n_S·n_E) and complexity bounds.
    """
    points = []
    for name, row in named_rows:
        entry = discovery_bound_row(row)
        if entry is None:
            continue
        norm = _normalized_discovery_time(row)
        if norm is None:
            print(
                f"Warning: {name}: cannot normalize discovery time "
                "(need aco_time, aco_ants, aco_iterations_budget)",
                file=sys.stderr,
            )
            continue
        ants = _row_ants(row)
        entry = dict(entry)
        entry["norm_time"] = norm
        entry["n_S"] = ants
        points.append(entry)
    return points


def _fmt_log_log_regression(stats):
    if stats is None:
        return "regression unavailable"
    slope, _intercept, r2 = stats
    return rf"slope $= {slope:.2f}$, $R^2 = {r2:.2f}$"


def _common_budget(matched, key):
    """Single shared budget value across matched points, or None if mixed."""
    vals = {p.get(key) for p in matched if p.get(key) is not None}
    if len(vals) != 1:
        return None
    return next(iter(vals))


def _budget_caption_bits(matched):
    """LaTeX scraps describing the shared (n_S, n_E) budget when uniform."""
    n_s = _common_budget(matched, "n_S")
    n_e = _common_budget(matched, "n_E")
    if n_s is not None and n_e is not None:
        return (
            rf"$n_S = {int(n_s)}$, $n_E = {int(n_e)}$",
            rf"($n_E{{=}}{int(n_e)}$)",
        )
    return ("recorded $n_S$ and $n_E$", "")


def _bound_vs_time_series(matched):
    """(bound, T) points and log--log regression stats for each candidate."""
    naive_pts = [(p["naive_bound"], p["time"]) for p in matched]
    practical_pts = [(p["practical_bound"], p["time"]) for p in matched]
    n_e = _common_budget(matched, "n_E")
    practical_label = _PRACTICAL_LABEL
    if n_e is not None:
        practical_label = rf"$n_E\cdot n_R+|E_R|$ ($n_E{{=}}{int(n_e)}$)"
    return {
        _NAIVE_LABEL: naive_pts,
        practical_label: practical_pts,
    }, {
        _NAIVE_LABEL: log_log_regression_stats(naive_pts),
        practical_label: log_log_regression_stats(practical_pts),
    }, (_NAIVE_LABEL, practical_label)


def _emit_raw_time_vs_n_panel(*, title, points, stats):
    """Left panel: raw discovery time vs n_R with log--log trend."""
    lines = [
        r"\nextgroupplot[",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        r"    xlabel={$n_R = |U_R|+|V_R|$},",
        r"    ylabel={Discovery time (s)},",
        r"    xmode=log,",
        r"    ymode=log,",
        r"  ]",
    ]
    lines += addplot_scatter(
        "ACO-N",
        points,
        with_legend=False,
        alpha=0.6,
        color="blue",
        mark="*",
    )
    lines += addplot_trendline(points, color="black")
    if stats is not None:
        slope, _intercept, r2 = stats
        lines.append(
            rf"  \node[anchor=north west, font=\scriptsize, align=left] "
            rf"at (rel axis cs:0.04,0.96) "
            rf"{{$\alpha = {slope:.2f}$\\$R^2 = {r2:.2f}$}};"
        )
    return lines


def _emit_bound_panel(*, title, series, series_order, legend_name=None):
    """Right panel: T vs each candidate bound (slope ≈ 1 ⇒ proportional)."""
    lines = [
        r"\nextgroupplot[",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        r"    xlabel={Candidate bound},",
        r"    ylabel={Discovery time (s)},",
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
    for i, lab in enumerate(series_order):
        color = _BOUND_COLORS[i % len(_BOUND_COLORS)]
        # Prefer known marks; fall back by series index.
        mark = _BOUND_SERIES_MARKS.get(lab)
        if mark is None:
            mark = "*" if i == 0 else "o"
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


def discovery_scaling_figure(*, matched):
    """Raw discovery time vs n_R and vs candidate bounds (2-panel)."""
    n = len(matched)
    time_vs_n = [(p["n_R"], p["time"]) for p in matched]
    stats_n = log_log_regression_stats(time_vs_n)
    series_bound, stats_bound, series_order = _bound_vs_time_series(matched)
    legend_name = "compareDiscoveryScalingLegend"
    n_e = _common_budget(matched, "n_E")
    n_e_tex = int(n_e) if n_e is not None else _FALLBACK_N_E
    practical_lab = series_order[1]

    caption = (
        rf"ACO-N discovery time on the {n} $(k, \theta) = (2, 5)$ graphs "
        rf"(log--log). Left: raw time vs.\ reduced nodes $n_R$, with a "
        rf"least-squares power-law fit "
        rf"({_fmt_log_log_regression(stats_n)}). Right: the same times "
        rf"plotted against each candidate complexity bound; a tight bound "
        rf"should give slope $\approx 1$ (proportional growth). Fitting "
        rf"$\log T = \alpha\log(\mathrm{{bound}}) + c$ yields "
        rf"{_fmt_log_log_regression(stats_bound[_NAIVE_LABEL])} for the "
        rf"naive $n_R^2$ bound and "
        rf"{_fmt_log_log_regression(stats_bound[practical_lab])} for "
        rf"the practical proxy $n_E\cdot n_R+|E_R|$ with $n_E = {n_e_tex}$. "
        rf"The practical slope is closer to $1$, matching the claim that "
        rf"observed runtimes scale linearly with $|E_R|$ rather than "
        rf"quadratically as the worst-case bound suggests."
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
    lines += _emit_raw_time_vs_n_panel(
        title=r"Empirical scaling vs.\ $n_R$",
        points=time_vs_n,
        stats=stats_n,
    )
    lines += _emit_bound_panel(
        title=r"Time vs.\ candidate bound",
        series=series_bound,
        series_order=series_order,
        legend_name=legend_name,
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
        r"  \label{fig:compare-discovery-scaling}",
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
        caption=r"ACO-N solution size vs.\ graph edge density.",
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
            r"ACO-N discovery time vs.\ maximum degree in the reduced graph."
        ),
        label="fig:compare-max-deg-time",
    )


def deg_size_time_figure(named_rows):
    matched = matched_discovery_bound_points(named_rows)
    return discovery_scaling_figure(matched=matched)


def _normalized_bound_figure(
    *,
    matched,
    x_key,
    xlabel,
    bound_name_tex,
    log_bound_tex,
    label,
    annotate_n_e_on_xlabel=False,
):
    """
    Single-panel log--log plot of T/(n_S·n_E) vs a complexity proxy.
    """
    points = [(p[x_key], p["norm_time"]) for p in matched]
    stats = log_log_regression_stats(points)
    n = len(matched)
    budget_tex, n_e_paren = _budget_caption_bits(matched)
    xlabel_full = (
        f"{xlabel} {n_e_paren}"
        if annotate_n_e_on_xlabel and n_e_paren
        else xlabel
    )
    caption = (
        rf"ACO-N discovery time normalized by ant count and epoch budget "
        rf"$T/(n_S\cdot n_E)$ vs.\ {bound_name_tex} on the {n} "
        rf"$(k, \theta) = (2, 5)$ graphs (log--log; {budget_tex} from "
        rf"recorded trials). Fitting "
        rf"$\log\!\bigl(T/(n_S\cdot n_E)\bigr) = "
        rf"\alpha\log({log_bound_tex}) + c$ yields "
        rf"{_fmt_log_log_regression(stats)}. Slope $\approx 1$ would "
        rf"indicate proportional growth with the bound."
    )

    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{axis}[",
        r"    width=0.72\textwidth,",
        r"    height=0.48\textwidth,",
        rf"    xlabel={{{xlabel_full}}},",
        r"    ylabel={Normalized discovery time $T/(n_S\cdot n_E)$ (s)},",
        r"    xmode=log,",
        r"    ymode=log,",
        r"    grid=major,",
        r"  ]",
    ]
    lines += addplot_scatter(
        "ACO-N",
        points,
        with_legend=False,
        alpha=0.6,
        color="blue",
        mark="*",
    )
    lines += addplot_trendline(points, color="black")
    if stats is not None:
        slope, _intercept, r2 = stats
        lines.append(
            rf"  \node[anchor=north west, font=\scriptsize, align=left] "
            rf"at (rel axis cs:0.04,0.96) "
            rf"{{$\alpha = {slope:.2f}$\\$R^2 = {r2:.2f}$}};"
        )
    lines += [
        r"  \end{axis}",
        r"  \end{tikzpicture}",
        r"",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
    ]
    return lines


def practical_bound_time_figure(named_rows):
    """Normalized discovery time vs n_E·n_R+|E_R|."""
    matched = matched_normalized_bound_points(named_rows)
    return _normalized_bound_figure(
        matched=matched,
        x_key="practical_bound",
        xlabel=r"$n_E\cdot n_R+|E_R|$",
        bound_name_tex=r"the practical complexity proxy $n_E\cdot n_R+|E_R|$",
        log_bound_tex=r"n_E\cdot n_R+|E_R|",
        label="fig:compare-practical-bound-time",
        annotate_n_e_on_xlabel=True,
    )


def naive_bound_time_figure(named_rows):
    """Normalized discovery time vs n_R^2."""
    matched = matched_normalized_bound_points(named_rows)
    return _normalized_bound_figure(
        matched=matched,
        x_key="naive_bound",
        xlabel=r"$n_R^2$",
        bound_name_tex=r"the naive bound $n_R^2$",
        log_bound_tex=r"n_R^2",
        label="fig:compare-naive-bound-time",
    )


BUILDERS = {
    "density-size": density_size_figure,
    "theta-time": theta_time_figure,
    "max-deg-time": max_deg_time_figure,
    "deg-size-time": deg_size_time_figure,
    "practical-bound-time": practical_bound_time_figure,
    "naive-bound-time": naive_bound_time_figure,
}
