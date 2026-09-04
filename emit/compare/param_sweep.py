"""
k / θ parameter-sweep compare figures.

Plot groups
-----------
  k-sweep
    2-panel groupplot at fixed θ: ACO / θ-heuristic edge ratio (log scale)
    and ACO win rate across available k values. Uses every ``--param-dir``
    whose JSON reports the same θ (default: the modal θ among param dirs,
    typically 5). Graphs are matched across the selected directories.

  theta-sweep
    Same layout at fixed k (default: the modal k among param dirs,
    typically 3), comparing available θ values.

  param-runtime
    Combined 2-panel discovery-time figure: left varies k at fixed θ,
    right varies θ at fixed k (same matching rules as the quality sweeps).

Edge ratio $|E_{ACO}|/|E_θ|$ is the multiplicative form of percent gain
($1 + \\mathrm{pct}/100$), so a log $y$-axis is well-defined even when ACO
underperforms (ratio $< 1$). Incomplete suites are fine: emit uses the
intersection of graphs present in the selected directories and annotates *n*
in the caption. Adding a new ``vary_k*t*i_PN`` directory to ``param_dirs``
in build.json is enough for it to appear once JSON is available.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

from ..common import list_json_paths, load_json
from ..table import (
    SECTION_ACO,
    compare_section,
    summarize_file,
)
from .flag_ablation import (
    _addplot_boxplot,
    _pgf_xticklabels,
)

PLOT_GROUPS = ("k-sweep", "theta-sweep", "param-runtime")


def _five_number_summary(values):
    """min / q1 / median / q3 / max for positive values (log-scale boxplots)."""
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


def _dataset_key(path, data):
    label = data.get("dataset") or os.path.splitext(os.path.basename(path))[0]
    leaf = str(label).replace("\\", "/").split("/")[-1]
    return leaf.removesuffix("_ants")


def _xtick_label(k, theta, *, axis):
    """Short tick label for the swept parameter."""
    if axis == "k":
        return str(int(k))
    return str(int(theta))


def _config_label(k, theta):
    return rf"$k={int(k)},\,\theta={int(theta)}$"


def load_param_dirs(param_dirs, *, ants=100):
    """
    Load per-graph table rows for each labeled param directory.

    Returns (by_label, meta, skipped) where:
      by_label[label] = {dataset_key: row}
      meta[label] = {"k": int, "theta": int, "directory": str}
    """
    if not param_dirs:
        raise SystemExit(
            "k-sweep / theta-sweep require --param-dir=LABEL=DIR "
            "(see build.json param_dirs)"
        )

    by_label = {}
    meta = {}
    skipped = []
    for label, directory in param_dirs.items():
        by_label[label] = {}
        ks, thetas = [], []
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
            if row.get("k") is None or row.get("theta") is None:
                skipped.append((path, "missing k or theta"))
                continue
            key = _dataset_key(path, data)
            by_label[label][key] = row
            ks.append(int(row["k"]))
            thetas.append(int(row["theta"]))
        if not by_label[label]:
            print(
                f"Warning: param-dir {label!r} ({directory}): no usable graphs.",
                file=sys.stderr,
            )
            continue
        k_mode = Counter(ks).most_common(1)[0][0]
        theta_mode = Counter(thetas).most_common(1)[0][0]
        if len(set(ks)) > 1 or len(set(thetas)) > 1:
            print(
                f"Warning: param-dir {label!r} has mixed k/θ "
                f"(using k={k_mode}, θ={theta_mode}).",
                file=sys.stderr,
            )
        meta[label] = {
            "k": k_mode,
            "theta": theta_mode,
            "directory": directory,
        }
    return by_label, meta, skipped


def _select_sweep(meta, *, axis, fixed_value=None):
    """
    Pick labels for a sweep along ``axis`` ('k' or 'theta').

    Holds the other parameter fixed at ``fixed_value`` when given; otherwise
    uses the modal value of that parameter among all param dirs.
    """
    if not meta:
        return [], None

    other = "theta" if axis == "k" else "k"
    if fixed_value is None:
        counts = Counter(m[other] for m in meta.values())
        fixed_value = counts.most_common(1)[0][0]

    selected = [
        (label, meta[label])
        for label in meta
        if meta[label][other] == fixed_value
    ]
    selected.sort(key=lambda item: item[1][axis])
    seen = set()
    unique = []
    for label, info in selected:
        val = info[axis]
        if val in seen:
            print(
                f"Warning: duplicate {axis}={val} at {_config_label(info['k'], info['theta'])}; "
                f"keeping first param-dir only.",
                file=sys.stderr,
            )
            continue
        seen.add(val)
        unique.append((label, info))
    return unique, fixed_value


def match_sweep(by_label, selected_labels):
    """Intersection of datasets across selected labels, sorted."""
    if not selected_labels:
        return []
    common = set.intersection(*(set(by_label[lab].keys()) for lab in selected_labels))
    return sorted(common)


def _win_rate(rows_for_label):
    """Fraction of graphs where ACO beats the θ-heuristic (θ-feasible rules)."""
    if not rows_for_label:
        return None
    wins = sum(1 for row in rows_for_label if compare_section(row) == SECTION_ACO)
    return 100.0 * wins / len(rows_for_label)


def _edge_ratios(rows_for_label):
    """
    ACO / θ-heuristic edge ratios (multiplicative form of percent gain).

    Skips graphs with missing or non-positive heuristic edges so the values
    are safe for a log $y$-axis.
    """
    vals = []
    for row in rows_for_label:
        aco = row.get("aco_edges")
        heur = row.get("heur_edges")
        if aco is None or heur is None:
            continue
        heur = float(heur)
        if heur <= 0:
            continue
        ratio = float(aco) / heur
        if ratio > 0:
            vals.append(ratio)
    return vals


def _sweep_figure(
    by_label,
    selected,
    matched,
    *,
    axis,
    fixed_other,
    label,
    caption_lead,
):
    """Shared 2-panel figure: log edge-ratio boxplots | win-rate bars."""
    if not selected:
        return [
            rf"% {label}: no param dirs available for this sweep",
        ]
    if not matched:
        labels = ", ".join(lab for lab, _ in selected)
        return [
            rf"% {label}: no graphs matched across {labels}",
        ]

    n = len(matched)
    n_ticks = len(selected)
    tick_labels = [
        _xtick_label(info["k"], info["theta"], axis=axis)
        for _, info in selected
    ]
    xtick = ",".join(str(i) for i in range(1, n_ticks + 1))
    width = 0.46
    height = 0.44

    med_notes = []
    win_notes = []
    for lab, info in selected:
        rows = [by_label[lab][g] for g in matched]
        ratios = _edge_ratios(rows)
        wins = _win_rate(rows)
        cfg = _config_label(info["k"], info["theta"])
        if ratios:
            ratios_sorted = sorted(ratios)
            mid = ratios_sorted[len(ratios_sorted) // 2]
            # Use a plain f-string so "\\times" becomes a single TeX backslash.
            med_notes.append(f"{mid:.2g}$\\times$ ({cfg})")
        if wins is not None:
            win_notes.append(f"{wins:.0f}\\% ({cfg})")

    med_clause = (
        "median edge ratio is " + ", ".join(med_notes)
        if med_notes
        else "edge ratios are shown where the $\\theta$-heuristic returns a positive $|E|$"
    )
    win_clause = (
        "ACO win rates are " + ", ".join(win_notes)
        if win_notes
        else "win rates use the same $\\theta$-feasibility rules as Table~"
        r"\ref{tab:aco-heur-pn}"
    )

    if axis == "k":
        xlab = r"$k$"
        fixed_note = rf"$\theta = {int(fixed_other)}$"
    else:
        xlab = r"$\theta$"
        fixed_note = rf"$k = {int(fixed_other)}$"

    caption = (
        f"{caption_lead} at fixed {fixed_note} on {n} graphs matched across "
        f"all shown settings (100 ants, ACO-PN). Left: edge ratio "
        f"$|E_{{\\mathrm{{ACO}}}}|/|E_{{\\theta}}|$ on a log scale "
        f"(the multiplicative form of percent gain over the $\\theta$-heuristic; "
        f"the dashed line is parity); {med_clause}. Right: fraction of "
        f"matched graphs where ACO beats the heuristic under the "
        f"Table~\\ref{{tab:aco-heur-pn}} rules; {win_clause}."
    )

    colors = (
        "blue!70!black",
        "orange!85!black",
        "teal!70!black",
        "purple!70!black",
        "red!70!black",
    )

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
        r"    ylabel={$|E_{\mathrm{ACO}}|/|E_{\theta}|$},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Relative solution quality},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        rf"    xlabel={{{xlab}}},",
        rf"    xtick={{{xtick}}},",
        rf"    xticklabels={_pgf_xticklabels(tick_labels)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        rf"    xmax={n_ticks + 0.5:.1f},",
        r"]",
    ]

    for i, (lab, _info) in enumerate(selected, start=1):
        rows = [by_label[lab][g] for g in matched]
        stats = _five_number_summary(_edge_ratios(rows))
        if stats is None:
            continue
        color = colors[(i - 1) % len(colors)]
        lines += _addplot_boxplot(i, stats, color=color)

    # Parity line |E_ACO| = |E_θ| (ratio = 1).
    lines.append(
        rf"  \draw[dashed, gray] (axis cs:0.5,1) -- (axis cs:{n_ticks + 0.5:.1f},1);"
    )

    lines += [
        r"\nextgroupplot[",
        r"    ylabel={ACO win rate (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Win rate vs.\ $\theta$-heuristic},",
        r"    title style={font=\small},",
        rf"    xlabel={{{xlab}}},",
        rf"    xtick={{{xtick}}},",
        rf"    xticklabels={_pgf_xticklabels(tick_labels)},",
        r"    x tick label style={font=\scriptsize},",
        r"    ymin=0,",
        r"    ymax=100,",
        r"    xmin=0.5,",
        rf"    xmax={n_ticks + 0.5:.1f},",
        r"    ybar,",
        r"    bar width=12pt,",
        r"]",
    ]

    for i, (lab, _info) in enumerate(selected, start=1):
        rows = [by_label[lab][g] for g in matched]
        rate = _win_rate(rows)
        if rate is None:
            continue
        color = colors[(i - 1) % len(colors)]
        lines.append(
            rf"  \addplot[fill={color}!40, draw={color}] "
            rf"coordinates {{({i},{rate:.4g})}};"
        )

    lines += [
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"\end{figure}",
    ]
    return lines


def _discovery_times(rows_for_label):
    """Positive ACO discovery times (seconds) for log-scale boxplots."""
    vals = []
    for row in rows_for_label:
        t = row.get("aco_time")
        if t is None:
            continue
        t = float(t)
        if t > 0:
            vals.append(t)
    return vals


def _runtime_panel_lines(
    by_label,
    selected,
    matched,
    *,
    axis,
    fixed_other,
    colors,
    title,
):
    """Emit one groupplot panel of discovery-time boxplots."""
    n_ticks = len(selected)
    tick_labels = [
        _xtick_label(info["k"], info["theta"], axis=axis)
        for _, info in selected
    ]
    xtick = ",".join(str(i) for i in range(1, n_ticks + 1))
    if axis == "k":
        xlab = r"$k$"
        fixed_note = rf"$\theta = {int(fixed_other)}$"
    else:
        xlab = r"$\theta$"
        fixed_note = rf"$k = {int(fixed_other)}$"

    lines = [
        r"\nextgroupplot[",
        r"    ylabel={Discovery time (s)},",
        r"    ylabel style={align=center, font=\small},",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        rf"    xlabel={{{xlab}}},",
        rf"    xtick={{{xtick}}},",
        rf"    xticklabels={_pgf_xticklabels(tick_labels)},",
        r"    x tick label style={font=\scriptsize},",
        r"    xmin=0.5,",
        rf"    xmax={n_ticks + 0.5:.1f},",
        r"]",
    ]
    for i, (lab, _info) in enumerate(selected, start=1):
        rows = [by_label[lab][g] for g in matched]
        stats = _five_number_summary(_discovery_times(rows))
        if stats is None:
            continue
        color = colors[(i - 1) % len(colors)]
        lines += _addplot_boxplot(i, stats, color=color)

    med_notes = []
    for lab, info in selected:
        rows = [by_label[lab][g] for g in matched]
        times = _discovery_times(rows)
        if not times:
            continue
        times_sorted = sorted(times)
        mid = times_sorted[len(times_sorted) // 2]
        cfg = _config_label(info["k"], info["theta"])
        med_notes.append(f"{mid:.3g}\\,s ({cfg})")

    return {
        "lines": lines,
        "n": len(matched),
        "fixed_note": fixed_note,
        "med_notes": med_notes,
        "empty": not selected or not matched,
    }


def param_runtime_figure(by_label, meta, *, fixed_theta=None, fixed_k=None):
    """
    2-panel discovery-time figure: vary k at fixed θ | vary θ at fixed k.
    """
    colors = (
        "blue!70!black",
        "orange!85!black",
        "teal!70!black",
        "purple!70!black",
        "red!70!black",
    )

    k_selected, k_fixed = _select_sweep(meta, axis="k", fixed_value=fixed_theta)
    k_matched = match_sweep(by_label, [lab for lab, _ in k_selected])
    k_panel = _runtime_panel_lines(
        by_label,
        k_selected,
        k_matched,
        axis="k",
        fixed_other=k_fixed if k_fixed is not None else 5,
        colors=colors,
        title=r"Varying $k$",
    )

    t_selected, t_fixed = _select_sweep(meta, axis="theta", fixed_value=fixed_k)
    t_matched = match_sweep(by_label, [lab for lab, _ in t_selected])
    t_panel = _runtime_panel_lines(
        by_label,
        t_selected,
        t_matched,
        axis="theta",
        fixed_other=t_fixed if t_fixed is not None else 3,
        colors=colors,
        title=r"Varying $\theta$",
    )

    if k_panel["empty"] and t_panel["empty"]:
        return [r"% param-runtime: no usable k or θ sweep directories"]

    k_med = (
        "median discovery time is " + ", ".join(k_panel["med_notes"])
        if k_panel["med_notes"]
        else "left panel incomplete"
    )
    t_med = (
        "median discovery time is " + ", ".join(t_panel["med_notes"])
        if t_panel["med_notes"]
        else "right panel incomplete"
    )

    caption = (
        f"ACO-PN discovery time (100 ants, log scale) across preliminary "
        f"$(k, \\theta)$ suites. Left: vary $k$ at fixed {k_panel['fixed_note']} "
        f"on {k_panel['n']} matched graphs; {k_med}. Right: vary $\\theta$ at "
        f"fixed {t_panel['fixed_note']} on {t_panel['n']} matched graphs; {t_med}."
    )

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=28pt},",
        r"    width=0.46\textwidth,",
        r"    height=0.44\textwidth,",
        r"    grid=major,",
        r"  ]",
    ]
    if not k_panel["empty"]:
        lines += k_panel["lines"]
    else:
        lines += [
            r"\nextgroupplot[title={Varying $k$ (no data)}, title style={font=\small}]",
        ]
    if not t_panel["empty"]:
        lines += t_panel["lines"]
    else:
        lines += [
            r"\nextgroupplot[title={Varying $\theta$ (no data)}, title style={font=\small}]",
        ]
    lines += [
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{caption}}}",
        r"  \label{fig:param-runtime}",
        r"\end{figure}",
    ]
    return lines


def k_sweep_figure(by_label, meta, *, fixed_theta=None):
    selected, fixed = _select_sweep(meta, axis="k", fixed_value=fixed_theta)
    matched = match_sweep(by_label, [lab for lab, _ in selected])
    return _sweep_figure(
        by_label,
        selected,
        matched,
        axis="k",
        fixed_other=fixed if fixed is not None else 5,
        label="fig:k-sweep",
        caption_lead=r"Effect of defect budget $k$",
    )


def theta_sweep_figure(by_label, meta, *, fixed_k=None):
    selected, fixed = _select_sweep(meta, axis="theta", fixed_value=fixed_k)
    matched = match_sweep(by_label, [lab for lab, _ in selected])
    return _sweep_figure(
        by_label,
        selected,
        matched,
        axis="theta",
        fixed_other=fixed if fixed is not None else 3,
        label="fig:theta-sweep",
        caption_lead=r"Effect of minimum side size $\theta$",
    )


BUILDERS = {
    "k-sweep": k_sweep_figure,
    "theta-sweep": theta_sweep_figure,
    "param-runtime": param_runtime_figure,
}
