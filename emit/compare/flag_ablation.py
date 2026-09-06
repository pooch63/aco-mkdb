"""
P/N flag-ablation compare figures.

Plot groups
-----------
  flag-ablation
    2-panel groupplot (quality | discovery time) across ACO, ACO-P,
    ACO-N, and ACO-PN at a fixed ant count. Requires --flag-dir for each
    variant (see build.json flag_dirs).

    Quality is the per-graph mean of scored edge counts over counted
    replicates: θ-infeasible trials contribute 0 edges but remain in the
    mean (they are not dropped).

  flag-feasibility
    1-panel boxplot of per-graph θ-feasibility rate (%) for the same
    four variants, ordered ACO, ACO-N, ACO-P, ACO-PN (grouped by P).
"""

from __future__ import annotations

import math
import os
import statistics

from ..common import counted_trials, list_json_paths, load_json, aco_timed_out
from ..table import summarize_file

FLAG_VARIANTS = ("ACO", "ACO-P", "ACO-N", "ACO-PN")
FLAG_QUALITY_ORDER = ("ACO", "ACO-P", "ACO-N", "ACO-PN")
FLAG_TIME_ORDER = ("ACO", "ACO-N", "ACO-P", "ACO-PN")
FLAG_FEAS_ORDER = ("ACO", "ACO-N", "ACO-P", "ACO-PN")
FLAG_N_OFF = frozenset({"ACO", "ACO-P"})
FLAG_P_OFF = frozenset({"ACO", "ACO-N"})

PLOT_GROUPS = ("flag-ablation", "flag-feasibility")


def _five_number_summary(values, *, include_zeros=False):
    """Return min, q1, median, q3, max for boxplot prepared."""
    vals = []
    for v in values:
        if v is None:
            continue
        fv = float(v)
        if include_zeros:
            if fv < 0:
                continue
        elif fv <= 0:
            continue
        vals.append(fv)
    if not vals:
        return None
    vals.sort()
    n = len(vals)

    def quantile(p):
        if n == 1:
            return vals[0]
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return vals[lo]
        weight = idx - lo
        return vals[lo] * (1.0 - weight) + vals[hi] * weight

    return {
        "min": vals[0],
        "q1": quantile(0.25),
        "median": quantile(0.5),
        "q3": quantile(0.75),
        "max": vals[-1],
    }


def _dataset_key(path, data):
    return data.get("dataset") or os.path.splitext(os.path.basename(path))[0]


def _scored_edge_count(trial):
    """Edges for averaging: θ-infeasible trials score as 0."""
    if not trial.get("theta_feasible"):
        return 0
    final = trial.get("final_edges")
    if final is None:
        return 0
    return int(final)


def _variant_trial_stats(data, *, ants):
    """
    Mean scored edges and θ-feasibility rate over counted replicates.

    θ-infeasible trials contribute 0 to the edge mean but remain in the
    denominator. Returns None when no counted trials exist at ``ants``.
    """
    trials = [
        t
        for t in counted_trials(data.get("trials") or [], data)
        if t.get("ants") == ants and t.get("final_edges") is not None
    ]
    if not trials:
        return None
    scored = [_scored_edge_count(t) for t in trials]
    n_feas = sum(1 for t in trials if t.get("theta_feasible"))
    return {
        "mean_scored_edges": statistics.mean(scored),
        "theta_feas_rate": 100.0 * n_feas / len(trials),
        "n_trials": len(trials),
    }


def load_flag_ablation_matched(flag_dirs, *, ants=100):
    """
    Load per-graph rows for all four flag variants on the matched benchmark set.

    Returns (matched, skipped) where matched is a list of
    {"dataset": key, "ACO": row, ...} dicts. Each row includes table
    summarize_file fields plus ``mean_scored_edges`` and ``theta_feas_rate``.
    """
    missing = [label for label in FLAG_VARIANTS if label not in flag_dirs]
    if missing:
        raise SystemExit(
            "flag-ablation / flag-feasibility plots require --flag-dir for "
            f"each variant; missing: {', '.join(missing)}"
        )

    by_variant = {}
    skipped = []
    for label in FLAG_VARIANTS:
        directory = flag_dirs[label]
        by_variant[label] = {}
        for path in list_json_paths(directory):
            data = load_json(path)
            if data is None:
                skipped.append((path, "unreadable"))
                continue
            if aco_timed_out(data):
                limit = data.get("aco_timeout_s")
                reason = "aco timed out"
                if limit is not None:
                    reason = f"aco timed out ({limit}s)"
                skipped.append((path, reason))
                continue
            row = summarize_file(data, ants=ants)
            if row is None:
                skipped.append((path, "not a vary.jl ant-count result"))
                continue
            if row.get("aco_edges") is None:
                skipped.append((path, "no ACO trial at requested ant count"))
                continue
            stats = _variant_trial_stats(data, ants=ants)
            if stats is None:
                skipped.append((path, "no counted ACO trials at requested ant count"))
                continue
            row = dict(row)
            row.update(stats)
            by_variant[label][_dataset_key(path, data)] = row

    common = set.intersection(*(set(rows.keys()) for rows in by_variant.values()))
    matched = []
    for key in sorted(common):
        entry = {"dataset": key}
        for label in FLAG_VARIANTS:
            entry[label] = by_variant[label][key]
        matched.append(entry)
    return matched, skipped


def _variant_color(label, *, panel):
    """N panel: color by N; time / feasibility panels: color by P."""
    if panel == "quality":
        return "blue!70!black" if label in FLAG_N_OFF else "orange!85!black"
    return "blue!70!black" if label in FLAG_P_OFF else "red!75!black"


def _addplot_boxplot(x, stats, *, color):
    prepared = (
        "boxplot prepared={"
        f"lower whisker={stats['min']:.8g}, "
        f"lower quartile={stats['q1']:.8g}, "
        f"median={stats['median']:.8g}, "
        f"upper quartile={stats['q3']:.8g}, "
        f"upper whisker={stats['max']:.8g}"
        "}"
    )
    opts = [
        rf"boxplot/draw position={x}",
        prepared,
        "boxplot/draw direction=y",
        rf"draw={color}",
        rf"fill={color}!18",
    ]
    return [rf"\addplot+[{', '.join(opts)}] coordinates {{}};"]


def _pgf_xticklabels(labels):
    parts = []
    for label in labels:
        if "-" in label or " " in label:
            parts.append("{" + label + "}")
        else:
            parts.append(label)
    return "{" + ",".join(parts) + "}"


def _flag_ablation_caption(matched):
    n = len(matched)
    return (
        rf"Flag ablation at 100 ants on {n} matched graphs. "
        rf"Left: per-graph mean scored $|E|$ over counted replicates "
        rf"($\theta$-infeasible trials count as 0 edges but stay in the mean), "
        rf"grouped by neighbor-scope N. Right: discovery time, grouped by "
        rf"prefer-smaller-side P; either flag alone cuts time and ACO-PN is "
        rf"fastest."
    )


def flag_ablation_figure(matched):
    """2-panel groupplot for P/N flag ablation (quality | discovery time)."""
    if not matched:
        return [
            r"% flag-ablation: no graphs matched across all four variant directories",
        ]

    quality_xticks = list(FLAG_QUALITY_ORDER)
    time_xticks = list(FLAG_TIME_ORDER)
    # Slightly narrower panels leave room for larger horizontal sep.
    width = 0.44
    height = 0.44

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=52pt},",
        rf"    width={width:.2f}\textwidth,",
        rf"    height={height:.2f}\textwidth,",
        r"    grid=major,",
        r"  ]",
        r"\nextgroupplot[",
        r"    ylabel={Mean scored $|E|$},",
        r"    ylabel style={align=center},",
        r"    title={Solution quality},",
        r"    title style={font=\small},",
        rf"    xtick={{{','.join(str(i) for i in range(1, 5))}}},",
        rf"    xticklabels={_pgf_xticklabels(quality_xticks)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        r"    xmax=4.5,",
        r"]",
    ]

    for i, label in enumerate(FLAG_QUALITY_ORDER, start=1):
        stats = _five_number_summary(
            [g[label]["mean_scored_edges"] for g in matched],
            include_zeros=True,
        )
        if stats is None:
            continue
        color = _variant_color(label, panel="quality")
        lines += _addplot_boxplot(i, stats, color=color)

    lines += [
        r"  \node[font=\scriptsize] at (rel axis cs:0.25,-0.14) {N off};",
        r"  \node[font=\scriptsize] at (rel axis cs:0.75,-0.14) {N on};",
        r"\nextgroupplot[",
        r"    ylabel={Discovery time (s)},",
        r"    ylabel style={align=center},",
        r"    title={Discovery time},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        rf"    xtick={{{','.join(str(i) for i in range(1, 5))}}},",
        rf"    xticklabels={_pgf_xticklabels(time_xticks)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        r"    xmax=4.5,",
        r"]",
    ]

    for i, label in enumerate(FLAG_TIME_ORDER, start=1):
        stats = _five_number_summary(
            [g[label]["aco_time"] for g in matched if g[label].get("aco_time")]
        )
        if stats is None:
            continue
        color = _variant_color(label, panel="time")
        lines += _addplot_boxplot(i, stats, color=color)

    lines += [
        r"  \node[font=\scriptsize] at (rel axis cs:0.25,-0.14) {P off};",
        r"  \node[font=\scriptsize] at (rel axis cs:0.75,-0.14) {P on};",
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{_flag_ablation_caption(matched)}}}",
        r"  \label{fig:flag-ablation}",
        r"\end{figure}",
    ]
    return lines


def _flag_feasibility_caption(matched):
    n = len(matched)
    return (
        rf"Per-graph $\theta$-feasibility rate at 100 ants on {n} matched "
        rf"graphs (percentage of counted replicates with $\geq\theta$ "
        rf"vertices on both sides). Variants are ordered ACO, ACO-N, "
        rf"ACO-P, ACO-PN and colored by prefer-smaller-side P."
    )


def flag_feasibility_figure(matched):
    """1-panel boxplot of θ-feasibility rate, grouped by flag P."""
    if not matched:
        return [
            r"% flag-feasibility: no graphs matched across all four variant directories",
        ]

    xticks = list(FLAG_FEAS_ORDER)
    width = 0.55
    height = 0.44

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{axis}[",
        rf"    width={width:.2f}\textwidth,",
        rf"    height={height:.2f}\textwidth,",
        r"    grid=major,",
        r"    ylabel={$\theta$-feasibility rate (\%)},",
        r"    ylabel style={align=center},",
        r"    title={$\theta$-feasibility by flag variant},",
        r"    title style={font=\small},",
        rf"    xtick={{{','.join(str(i) for i in range(1, 5))}}},",
        rf"    xticklabels={_pgf_xticklabels(xticks)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        r"    xmax=4.5,",
        r"    ymin=0,",
        r"    ymax=105,",
        r"  ]",
    ]

    for i, label in enumerate(FLAG_FEAS_ORDER, start=1):
        stats = _five_number_summary(
            [g[label]["theta_feas_rate"] for g in matched],
            include_zeros=True,
        )
        if stats is None:
            continue
        color = _variant_color(label, panel="feasibility")
        lines += _addplot_boxplot(i, stats, color=color)

    lines += [
        r"  \node[font=\scriptsize] at (rel axis cs:0.25,-0.12) {P off};",
        r"  \node[font=\scriptsize] at (rel axis cs:0.75,-0.12) {P on};",
        r"  \end{axis}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{_flag_feasibility_caption(matched)}}}",
        r"  \label{fig:flag-feasibility}",
        r"\end{figure}",
    ]
    return lines


BUILDERS = {
    "flag-ablation": flag_ablation_figure,
    "flag-feasibility": flag_feasibility_figure,
}
