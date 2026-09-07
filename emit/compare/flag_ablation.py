"""
P/N flag-ablation compare figures.

Plot groups
-----------
  flag-ablation
    2-panel groupplot (quality | discovery time) across ACO, ACO-P,
    ACO-N, and ACO-PN at a fixed ant count. Requires --flag-dir for each
    variant (see build.json flag_dirs). All panels are ordered and
    colored by neighbor-scope N (ACO, ACO-P | ACO-N, ACO-PN).

    Quality is the per-graph mean of scored edge counts over counted
    replicates: θ-infeasible trials contribute 0 edges but remain in the
    mean (they are not dropped). Discovery time is the per-graph
    discovery cost (sum of counted same-ant wall_time_s).

  flag-ablation-replicates
    Same layout as flag-ablation, but each boxplot observation is a
    counted replicate: scored |E| (θ-infeasible → 0) and wall_time_s.

  flag-feasibility
    1-panel bar chart: percentage of matched graphs that admit at least
    one θ-feasible counted replicate, for the same four variants ordered
    ACO, ACO-P, ACO-N, ACO-PN (grouped by N).

  flag-feasibility-replicates
    Same layout, but the rate is the percentage of counted replicates
    that are θ-feasible (pooled over matched graphs).
"""

from __future__ import annotations

import math
import os
import statistics

from ..common import counted_trials, list_json_paths, load_json, aco_timed_out
from ..table import summarize_file

FLAG_VARIANTS = ("ACO", "ACO-P", "ACO-N", "ACO-PN")
# Group by neighbor-scope N: N off (ACO, ACO-P) then N on (ACO-N, ACO-PN).
FLAG_ORDER = ("ACO", "ACO-P", "ACO-N", "ACO-PN")
FLAG_N_OFF = frozenset({"ACO", "ACO-P"})

PLOT_GROUPS = (
    "flag-ablation",
    "flag-ablation-replicates",
    "flag-feasibility",
    "flag-feasibility-replicates",
)


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
    Mean scored edges, θ-feasibility rate, and per-replicate lists.

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
    wall_times = [
        float(t["wall_time_s"])
        for t in trials
        if t.get("wall_time_s") is not None
    ]
    n_feas = sum(1 for t in trials if t.get("theta_feasible"))
    return {
        "mean_scored_edges": statistics.mean(scored),
        "theta_feas_rate": 100.0 * n_feas / len(trials),
        "n_trials": len(trials),
        "n_feasible": n_feas,
        "scored_edges": scored,
        "wall_times": wall_times,
    }


def load_flag_ablation_matched(flag_dirs, *, ants=100):
    """
    Load per-graph rows for all four flag variants on the matched benchmark set.

    Returns (matched, skipped) where matched is a list of
    {"dataset": key, "ACO": row, ...} dicts. Each row includes table
    summarize_file fields plus ``mean_scored_edges``, ``theta_feas_rate``,
    and per-replicate ``scored_edges`` / ``wall_times`` lists.
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


def _variant_color(label):
    """Color by neighbor-scope N (off = blue, on = orange)."""
    return "blue!70!black" if label in FLAG_N_OFF else "orange!85!black"


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


def _n_group_labels():
    return [
        r"  \node[font=\scriptsize] at (rel axis cs:0.25,-0.10) {N off};",
        r"  \node[font=\scriptsize] at (rel axis cs:0.75,-0.10) {N on};",
    ]


def _quality_values(matched, label, *, unit):
    """Collect quality observations for one variant at graph or replicate unit."""
    if unit == "graph":
        return [g[label]["mean_scored_edges"] for g in matched]
    values = []
    for g in matched:
        values.extend(g[label].get("scored_edges") or [])
    return values


def _time_values(matched, label, *, unit):
    """Collect time observations for one variant at graph or replicate unit."""
    if unit == "graph":
        return [
            g[label]["aco_time"]
            for g in matched
            if g[label].get("aco_time")
        ]
    values = []
    for g in matched:
        values.extend(g[label].get("wall_times") or [])
    return values


def _flag_ablation_caption(matched, *, unit):
    n = len(matched)
    if unit == "graph":
        return (
            rf"Flag ablation at 100 ants on {n} matched graphs. "
            rf"Left: per-graph mean scored $|E|$ over counted replicates "
            rf"($\theta$-infeasible trials count as 0 edges but stay in the mean). "
            rf"Right: per-graph discovery time (sum of counted replicate "
            rf"wall times). Both panels are ordered and colored by "
            rf"neighbor-scope N."
        )
    n_repl = sum(g["ACO"]["n_trials"] for g in matched)
    return (
        rf"Flag ablation at 100 ants: counted replicates pooled over "
        rf"{n} matched graphs ({n_repl} replicates per variant). "
        rf"Left: scored $|E|$ per replicate ($\theta$-infeasible trials "
        rf"count as 0). Right: per-replicate wall time. Both panels are "
        rf"ordered and colored by neighbor-scope N."
    )


def flag_ablation_figure(matched, *, unit="graph"):
    """2-panel groupplot for P/N flag ablation (quality | time)."""
    label_name = (
        "flag-ablation" if unit == "graph" else "flag-ablation-replicates"
    )
    if not matched:
        return [
            rf"% {label_name}: no graphs matched across all four variant directories",
        ]

    xticks = list(FLAG_ORDER)
    width = 0.42
    height = 0.36
    quality_ylabel = (
        r"Mean scored $|E|$" if unit == "graph" else r"Scored $|E|$"
    )
    time_ylabel = (
        r"Discovery time (s)" if unit == "graph" else r"Wall time (s)"
    )
    fig_label = (
        r"fig:flag-ablation" if unit == "graph" else r"fig:flag-ablation-replicates"
    )

    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=52pt},",
        rf"    width={width:.2f}\textwidth,",
        rf"    height={height:.2f}\textwidth,",
        r"    grid=major,",
        r"    ylabel style={font=\small, align=center},",
        r"    y tick label style={font=\scriptsize},",
        r"  ]",
        r"\nextgroupplot[",
        rf"    ylabel={{{quality_ylabel}}},",
        r"    title={Solution quality},",
        r"    title style={font=\small},",
        rf"    xtick={{{','.join(str(i) for i in range(1, 5))}}},",
        rf"    xticklabels={_pgf_xticklabels(xticks)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        r"    xmax=4.5,",
        r"]",
    ]

    for i, label in enumerate(FLAG_ORDER, start=1):
        stats = _five_number_summary(
            _quality_values(matched, label, unit=unit),
            include_zeros=True,
        )
        if stats is None:
            continue
        color = _variant_color(label)
        lines += _addplot_boxplot(i, stats, color=color)

    lines += _n_group_labels()
    lines += [
        r"\nextgroupplot[",
        rf"    ylabel={{{time_ylabel}}},",
        r"    title={Discovery time}," if unit == "graph" else r"    title={Wall time},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        rf"    xtick={{{','.join(str(i) for i in range(1, 5))}}},",
        rf"    xticklabels={_pgf_xticklabels(xticks)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        r"    xmax=4.5,",
        r"]",
    ]

    for i, label in enumerate(FLAG_ORDER, start=1):
        stats = _five_number_summary(_time_values(matched, label, unit=unit))
        if stats is None:
            continue
        color = _variant_color(label)
        lines += _addplot_boxplot(i, stats, color=color)

    lines += _n_group_labels()
    lines += [
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{_flag_ablation_caption(matched, unit=unit)}}}",
        rf"  \label{{{fig_label}}}",
        r"\end{figure}",
    ]
    return lines


def _graph_theta_feasibility_rate(matched, label):
    """
    Percentage of matched graphs with at least one θ-feasible counted
    replicate for ``label``.
    """
    if not matched:
        return None
    n_ok = sum(1 for g in matched if float(g[label]["theta_feas_rate"]) > 0)
    return 100.0 * n_ok / len(matched)


def _replicate_theta_feasibility_rate(matched, label):
    """Percentage of counted replicates that are θ-feasible (pooled)."""
    n_trials = sum(int(g[label]["n_trials"]) for g in matched)
    if n_trials <= 0:
        return None
    n_feas = sum(int(g[label]["n_feasible"]) for g in matched)
    return 100.0 * n_feas / n_trials


def _flag_feasibility_caption(matched, *, unit):
    n = len(matched)
    if unit == "graph":
        return (
            rf"Percentage of {n} matched graphs with at least one "
            rf"$\theta$-feasible counted replicate at 100 ants. Variants are "
            rf"ordered ACO, ACO-P, ACO-N, ACO-PN and colored by "
            rf"neighbor-scope N."
        )
    n_repl = sum(g["ACO"]["n_trials"] for g in matched)
    return (
        rf"Percentage of counted replicates that are $\theta$-feasible at "
        rf"100 ants, pooled over {n} matched graphs ({n_repl} replicates "
        rf"per variant). Variants are ordered ACO, ACO-P, ACO-N, ACO-PN "
        rf"and colored by neighbor-scope N."
    )


def flag_feasibility_figure(matched, *, unit="graph"):
    """1-panel bar chart of θ-feasibility rate, grouped by N."""
    label_name = (
        "flag-feasibility"
        if unit == "graph"
        else "flag-feasibility-replicates"
    )
    if not matched:
        return [
            rf"% {label_name}: no graphs matched across all four variant directories",
        ]

    xticks = list(FLAG_ORDER)
    width = 0.55
    height = 0.36
    rate_fn = (
        _graph_theta_feasibility_rate
        if unit == "graph"
        else _replicate_theta_feasibility_rate
    )
    ylabel = (
        r"$\theta$-feasibility rate (\% graphs)"
        if unit == "graph"
        else r"$\theta$-feasibility rate (\% replicates)"
    )
    title = (
        r"$\theta$-feasibility by flag variant (graphs)"
        if unit == "graph"
        else r"$\theta$-feasibility by flag variant (replicates)"
    )
    fig_label = (
        r"fig:flag-feasibility"
        if unit == "graph"
        else r"fig:flag-feasibility-replicates"
    )

    lines = [
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \vspace{1.5em}",
        r"  \begin{tikzpicture}",
        r"  \begin{axis}[",
        rf"    width={width:.2f}\textwidth,",
        rf"    height={height:.2f}\textwidth,",
        r"    grid=major,",
        rf"    ylabel={{{ylabel}}},",
        r"    ylabel style={align=center},",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        rf"    xtick={{{','.join(str(i) for i in range(1, 5))}}},",
        rf"    xticklabels={_pgf_xticklabels(xticks)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        r"    xmax=4.5,",
        r"    ymin=0,",
        r"    ymax=100,",
        r"    ybar,",
        r"    bar width=18pt,",
        r"    bar shift=0,",
        r"  ]",
    ]

    for i, label in enumerate(FLAG_ORDER, start=1):
        rate = rate_fn(matched, label)
        if rate is None:
            continue
        color = _variant_color(label)
        lines.append(
            rf"  \addplot[ybar, bar shift=0, fill={color}!40, draw={color}] "
            rf"coordinates {{({i},{rate:.4g})}};"
        )

    lines += [
        r"  \node[font=\scriptsize] at (rel axis cs:0.25,-0.10) {N off};",
        r"  \node[font=\scriptsize] at (rel axis cs:0.75,-0.10) {N on};",
        r"  \end{axis}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{_flag_feasibility_caption(matched, unit=unit)}}}",
        rf"  \label{{{fig_label}}}",
        r"\end{figure}",
    ]
    return lines


def flag_ablation_replicates_figure(matched):
    return flag_ablation_figure(matched, unit="replicate")


def flag_feasibility_replicates_figure(matched):
    return flag_feasibility_figure(matched, unit="replicate")


BUILDERS = {
    "flag-ablation": flag_ablation_figure,
    "flag-ablation-replicates": flag_ablation_replicates_figure,
    "flag-feasibility": flag_feasibility_figure,
    "flag-feasibility-replicates": flag_feasibility_replicates_figure,
}
