"""
Complexity / scaling compare figures.

Plot groups
-----------
  theta-time
    θ-heuristic wall time vs θ(|U|+|V|)+|E|

  deg-size-time
    2-panel log--log discovery-time scaling on matched graphs;
    left: raw t vs n with empirical power-law fit;
    right: t vs each candidate bound (naive n^2; practical
    T·n+m with T from iterations_budget) — slope ≈ 1
    means proportional

  bound-time
    2-panel log--log normalized discovery time t/(n_S·T):
    left vs naive bound n^2; right vs practical proxy
    T·n+m — shared y-axis so slopes can be compared

  density-size
    edge density vs |E(D_{best})|

  max-deg-time
    max reduced degree vs $F_N$ discovery time
"""

from __future__ import annotations

import math
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
    "bound-time",
    "density-size",
    "max-deg-time",
)

# Fallback epoch budget only when JSON omits iterations_budget.
_FALLBACK_T = 5

_NAIVE_LABEL = r"$n^2$"
_PRACTICAL_LABEL = r"$T\cdot n+m$"
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
    t = row.get("aco_iterations_budget")
    if t is None:
        return None
    t = float(t)
    return t if t > 0 else None


def _normalized_discovery_time(row):
    """
    Discovery wall time per ant per epoch: t / (n_S · T).

    Uses ants and iterations_budget from the winning trial JSON fields
    (via summarize_file). Returns None if any factor is missing/invalid.
    """
    time_s = row.get("aco_time")
    ants = _row_ants(row)
    t = _row_epochs(row)
    if time_s is None or ants is None or t is None:
        return None
    time_s = float(time_s)
    if time_s <= 0:
        return None
    return time_s / (ants * t)


def discovery_bound_row(row):
    """
    Per-graph discovery time and complexity proxies, or None if incomplete.

    naive_bound = (|U|+|V|)^2
    practical_bound = T · (|U|+|V|) + |E|
    with T from iterations_budget (fallback _FALLBACK_T only if missing)
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
    t = _row_epochs(row)
    if t is None:
        print(
            "Warning: missing aco_iterations_budget; "
            f"falling back to T={_FALLBACK_T} for practical bound",
            file=sys.stderr,
        )
        t = float(_FALLBACK_T)
    practical_bound = t * n_r + m_r
    if practical_bound <= 0:
        return None
    return {
        "n_R": n_r,
        "m_R": m_r,
        "T": t,
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
    Graphs with normalized discovery time t/(n_S·T) and complexity bounds.
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
    """LaTeX scraps describing the shared (n_S, T) budget when uniform."""
    n_s = _common_budget(matched, "n_S")
    t = _common_budget(matched, "T")
    if n_s is not None and t is not None:
        return (
            rf"$n_S = {int(n_s)}$, $T = {int(t)}$",
            rf"($T{{=}}{int(t)}$)",
        )
    return ("recorded $n_S$ and $T$", "")


def _bound_vs_time_series(matched):
    """(bound, t) points and log--log regression stats for each candidate."""
    naive_pts = [(p["naive_bound"], p["time"]) for p in matched]
    practical_pts = [(p["practical_bound"], p["time"]) for p in matched]
    t = _common_budget(matched, "T")
    practical_label = _PRACTICAL_LABEL
    if t is not None:
        practical_label = rf"$T\cdot n+m$ ($T{{=}}{int(t)}$)"
    return {
        _NAIVE_LABEL: naive_pts,
        practical_label: practical_pts,
    }, {
        _NAIVE_LABEL: log_log_regression_stats(naive_pts),
        practical_label: log_log_regression_stats(practical_pts),
    }, (_NAIVE_LABEL, practical_label)


def _emit_raw_time_vs_n_panel(*, title, points, stats):
    """Left panel: raw discovery time vs n with log--log trend."""
    lines = [
        r"\nextgroupplot[",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        r"    xlabel={$n = |U|+|V|$},",
        r"    ylabel={Discovery time (s)},",
        r"    xmode=log,",
        r"    ymode=log,",
        r"  ]",
    ]
    lines += addplot_scatter(
        ACO_LABEL,
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
    """Right panel: t vs each candidate bound (slope ≈ 1 ⇒ proportional)."""
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
    """Raw discovery time vs n and vs candidate bounds (2-panel)."""
    n = len(matched)
    time_vs_n = [(p["n_R"], p["time"]) for p in matched]
    stats_n = log_log_regression_stats(time_vs_n)
    series_bound, stats_bound, series_order = _bound_vs_time_series(matched)
    legend_name = "compareDiscoveryScalingLegend"
    n_e = _common_budget(matched, "T")
    n_e_tex = int(n_e) if n_e is not None else _FALLBACK_T
    practical_lab = series_order[1]

    caption = (
        rf"$F_N$ discovery time on the {n} $(k, \theta) = (2, 5)$ graphs "
        rf"(log--log). Left: raw time vs.\ nodes $n$, with a "
        rf"least-squares power-law fit "
        rf"({_fmt_log_log_regression(stats_n)}). Right: the same times "
        rf"plotted against each candidate complexity bound; a tight bound "
        rf"should give slope $\approx 1$ (proportional growth). Fitting "
        rf"$\log t = \alpha\log(\mathrm{{bound}}) + c$ yields "
        rf"{_fmt_log_log_regression(stats_bound[_NAIVE_LABEL])} for the "
        rf"naive $n^2$ bound and "
        rf"{_fmt_log_log_regression(stats_bound[practical_lab])} for "
        rf"the practical proxy $T\cdot n+m$ with $T = {n_e_tex}$. "
        rf"The practical slope is closer to $1$, matching the claim that "
        rf"observed runtimes scale linearly with $m$ rather than "
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
        title=r"Empirical scaling vs.\ $n$",
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
        caption=r"$F_N$ solution size vs.\ graph edge density.",
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
        xlabel=r"$\theta(|U|+|V|)+|E|$",
        ylabel=r"Time (s)",
        caption=(
            r"$\theta$-heuristic wall time vs.\ "
            r"$\theta(|U|+|V|)+|E|$."
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
            r"$F_N$ discovery time vs.\ maximum degree in the reduced graph."
        ),
        label="fig:compare-max-deg-time",
    )


def deg_size_time_figure(named_rows):
    matched = matched_discovery_bound_points(named_rows)
    return discovery_scaling_figure(matched=matched)


def _emit_normalized_bound_panel(
    *,
    title,
    xlabel,
    points,
    stats,
    ylabel=None,
):
    """One groupplot panel: t/(n_S·T) vs a complexity proxy."""
    lines = [
        r"\nextgroupplot[",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        rf"    xlabel={{{xlabel}}},",
        r"    xmode=log,",
        r"    ymode=log,",
    ]
    if ylabel is not None:
        lines.append(rf"    ylabel={{{ylabel}}},")
        lines.append(r"    ylabel style={font=\small, align=center},")
    lines.append(r"  ]")
    lines += addplot_scatter(
        ACO_LABEL,
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


def _log_pad_limits(values, *, pad=0.15):
    """Padded log-axis (ymin, ymax) from positive values, or None."""
    pos = [float(v) for v in values if v is not None and float(v) > 0]
    if not pos:
        return None
    lo, hi = min(pos), max(pos)
    if lo == hi:
        return 0.5 * lo, 2.0 * hi
    log_lo, log_hi = math.log10(lo), math.log10(hi)
    span = log_hi - log_lo
    return 10 ** (log_lo - pad * span), 10 ** (log_hi + pad * span)


def bound_time_figure(named_rows):
    """
    Normalized discovery time vs naive and practical bounds (2-panel).

    Left: t/(n_S·T) vs n^2 (theoretical worst case).
    Right: same times vs T·n+m (practical proxy).
    Shared y-axis lets the fitted slopes be compared directly.
    """
    matched = matched_normalized_bound_points(named_rows)
    n = len(matched)
    naive_pts = [(p["naive_bound"], p["norm_time"]) for p in matched]
    practical_pts = [(p["practical_bound"], p["norm_time"]) for p in matched]
    stats_naive = log_log_regression_stats(naive_pts)
    stats_practical = log_log_regression_stats(practical_pts)
    budget_tex, n_e_paren = _budget_caption_bits(matched)
    practical_xlabel = (
        rf"$T\cdot n+m$ {n_e_paren}"
        if n_e_paren
        else r"$T\cdot n+m$"
    )
    y_limits = _log_pad_limits(p["norm_time"] for p in matched)

    caption = (
        rf"$F_N$ discovery time normalized by ant count and epoch budget "
        rf"$t/(n_S\cdot T)$ on the {n} $(k, \theta) = (2, 5)$ graphs "
        rf"(log--log; {budget_tex} from recorded trials). Left: vs.\ the "
        rf"naive theoretical bound $n^2$ "
        rf"({_fmt_log_log_regression(stats_naive)}). Right: vs.\ the "
        rf"practical proxy $T\cdot n+m$ "
        rf"({_fmt_log_log_regression(stats_practical)}). Fitting "
        rf"$\log\!\bigl(t/(n_S\cdot T)\bigr) = "
        rf"\alpha\log(\mathrm{{bound}}) + c$, slope $\approx 1$ would "
        rf"indicate proportional growth. The practical slope is closer to "
        rf"$1$ than the naive $n^2$ fit, so while still loose the "
        rf"practical bound is a significantly better complexity scale."
    )

    ylabel = r"Normalized discovery time $t/(n_S\cdot T)$ (s)"
    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={",
        r"      group size=2 by 1,",
        r"      horizontal sep=52pt,",
        r"      ylabels at=edge left,",
        r"      yticklabels at=edge left},",
        r"    width=0.45\textwidth,",
        r"    height=0.42\textwidth,",
        r"    grid=major,",
        r"    xlabel style={font=\small},",
    ]
    if y_limits is not None:
        ymin, ymax = y_limits
        lines.append(rf"    ymin={ymin:.6g},")
        lines.append(rf"    ymax={ymax:.6g},")
    lines.append(r"  ]")
    lines += _emit_normalized_bound_panel(
        title=r"Naive bound $n^2$",
        xlabel=r"$n^2$",
        points=naive_pts,
        stats=stats_naive,
        ylabel=ylabel,
    )
    lines += _emit_normalized_bound_panel(
        title=r"Practical proxy $T\cdot n+m$",
        xlabel=practical_xlabel,
        points=practical_pts,
        stats=stats_practical,
    )
    lines += [
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        r"",
        rf"  \caption{{{caption}}}",
        r"  \label{fig:compare-bound-time}",
        r"\end{figure}",
    ]
    return lines


BUILDERS = {
    "density-size": density_size_figure,
    "theta-time": theta_time_figure,
    "max-deg-time": max_deg_time_figure,
    "deg-size-time": deg_size_time_figure,
    "bound-time": bound_time_figure,
}
