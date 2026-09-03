"""
P/N flag-ablation compare figure.

Plot group
----------
  flag-ablation
    2-panel groupplot (quality | discovery time) across ACO, ACO-P,
    ACO-N, and ACO-PN at a fixed ant count. The quality panel shows that
    neither flag sacrifices $|E(D^*)|$; the time panel shows that either
    flag alone reduces discovery time and ACO-PN is fastest. Requires
    --flag-dir for each variant (see build.json flag_dirs).
"""

from __future__ import annotations

import math
import os

from ..common import list_json_paths, load_json
from ..table import summarize_file

FLAG_VARIANTS = ("ACO", "ACO-P", "ACO-N", "ACO-PN")
FLAG_QUALITY_ORDER = ("ACO", "ACO-P", "ACO-N", "ACO-PN")
FLAG_TIME_ORDER = ("ACO", "ACO-N", "ACO-P", "ACO-PN")
FLAG_N_OFF = frozenset({"ACO", "ACO-P"})
FLAG_P_OFF = frozenset({"ACO", "ACO-N"})

PLOT_GROUPS = ("flag-ablation",)


def _five_number_summary(values):
    """Return min, q1, median, q3, max for boxplot prepared."""
    vals = [float(v) for v in values if v is not None and float(v) > 0]
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


def _median(values):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _dataset_key(path, data):
    return data.get("dataset") or os.path.splitext(os.path.basename(path))[0]


def load_flag_ablation_matched(flag_dirs, *, ants=100):
    """
    Load per-graph rows for all four flag variants on the matched benchmark set.

    Returns (matched, skipped) where matched is a list of
    {"dataset": key, "ACO": row, ...} dicts.
    """
    missing = [label for label in FLAG_VARIANTS if label not in flag_dirs]
    if missing:
        raise SystemExit(
            "flag-ablation plot requires --flag-dir for each variant; "
            f"missing: {', '.join(missing)}"
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
            row = summarize_file(data, ants=ants)
            if row is None:
                skipped.append((path, "not a vary.jl ant-count result"))
                continue
            if row.get("aco_edges") is None:
                skipped.append((path, "no ACO trial at requested ant count"))
                continue
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
    """N panel: color by N; time panel: color by P."""
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


def _paired_line_coords(order, graph_entry, value_fn):
    parts = []
    for i, label in enumerate(order, start=1):
        value = value_fn(graph_entry[label])
        if value is None:
            return None
        parts.append(f"({i},{float(value):.8g})")
    return " ".join(parts)


def _flag_ablation_caption(matched):
    n = len(matched)
    med_edges = {
        label: _median([g[label]["aco_edges"] for g in matched])
        for label in FLAG_VARIANTS
    }
    med_times = {
        label: _median([g[label]["aco_time"] for g in matched])
        for label in FLAG_VARIANTS
    }

    quality_note = ""
    if all(med_edges[label] is not None for label in FLAG_VARIANTS):
        quality_note = (
            rf"median $|E(D^*)|$ is "
            rf"{med_edges['ACO']:.0f} (ACO), "
            rf"{med_edges['ACO-P']:.0f} (ACO-P), "
            rf"{med_edges['ACO-N']:.0f} (ACO-N), and "
            rf"{med_edges['ACO-PN']:.0f} (ACO-PN)"
        )

    time_note = ""
    med_time_aco = med_times["ACO"]
    med_time_pn = med_times["ACO-PN"]
    if (
        all(med_times[label] is not None for label in FLAG_VARIANTS)
        and med_time_pn > 0
    ):
        ratio = med_time_aco / med_time_pn
        time_note = (
            rf"median discovery time falls from {med_time_aco:.3g}\,s "
            rf"(ACO) to {med_times['ACO-P']:.3g}\,s (ACO-P), "
            rf"{med_times['ACO-N']:.3g}\,s (ACO-N), and "
            rf"{med_time_pn:.3g}\,s (ACO-PN), an "
            rf"${ratio:.1f}\times$ reduction from ACO to ACO-PN"
        )

    quality_clause = (
        rf"neither P nor N sacrifices solution quality"
        + (rf" ({quality_note})" if quality_note else "")
    )
    time_clause = (
        rf"either flag alone reduces discovery time, and ACO-PN is "
        rf"fastest of the four"
        + (rf" ({time_note})" if time_note else "")
    )

    body = (
        rf"Flag ablation at 100 ants on {n} benchmark graphs matched across "
        rf"all four variants. Left: $|E(D^*)|$ grouped by neighbor-scope "
        rf"limit (N); {quality_clause}. Right: discovery time (log scale) "
        rf"grouped by prefer-smaller-side (P); {time_clause}. Faint lines "
        rf"connect the same graph across variants."
    )
    return body


def flag_ablation_figure(matched):
    """2-panel groupplot for P/N flag ablation (quality | discovery time)."""
    if not matched:
        return [
            r"% flag-ablation: no graphs matched across all four variant directories",
        ]

    quality_xticks = list(FLAG_QUALITY_ORDER)
    time_xticks = list(FLAG_TIME_ORDER)
    width = 0.46
    height = 0.44

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=28pt},",
        rf"    width={width:.2f}\textwidth,",
        rf"    height={height:.2f}\textwidth,",
        r"    grid=major,",
        r"  ]",
        r"\nextgroupplot[",
        r"    ylabel={$|E(D^*)|$},",
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
            [g[label]["aco_edges"] for g in matched]
        )
        if stats is None:
            continue
        color = _variant_color(label, panel="quality")
        lines += _addplot_boxplot(i, stats, color=color)

    for graph in matched:
        coords = _paired_line_coords(
            FLAG_QUALITY_ORDER,
            graph,
            lambda row: row.get("aco_edges"),
        )
        if coords:
            lines.append(
                rf"\addplot[gray, opacity=0.14, mark=none, forget plot] "
                rf"coordinates {{{coords}}};"
            )

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

    for graph in matched:
        coords = _paired_line_coords(
            FLAG_TIME_ORDER,
            graph,
            lambda row: row.get("aco_time"),
        )
        if coords:
            lines.append(
                rf"\addplot[gray, opacity=0.14, mark=none, forget plot] "
                rf"coordinates {{{coords}}};"
            )

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


BUILDERS = {
    "flag-ablation": flag_ablation_figure,
}
