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

  density-wins
    Rolling ACO win rate vs reduced edge density
    $\\text{den}(G_R) = |E_R|/(|U_R|\\cdot|V_R|)$ (derived from JSON sizes/edges),
    pooled over all graph$\\times$suite observations across $(k, \\theta)$.

  param-density
    Two-panel log-scale boxplots of $\\text{den}(G_R)$: vary $k$ at fixed $\\theta$ |
    vary $\\theta$ at fixed $k$ (same matching as the quality sweeps).
    Shows that $(k, \\theta)$ reshape reduced density, so density-linked
    win trends are not reducible to ``ACO is worse at large $\\theta$'' alone.

  param-runtime
    2×2 groupplot of ACO discovery cost versus the θ-heuristic across
    $(k, \\theta)$: discovery-time ratio $T_{\\mathrm{ACO}}/T_θ$ (top) and
    absolute ACO discovery time (bottom), each as boxplots for vary-$k$
    (fixed θ) and vary-$θ$ (fixed k). Timed-out graphs are excluded from
    the matched boxes.

Edge ratio $|E_{ACO}|/|E_θ|$ is the multiplicative form of percent gain
($1 + \\mathrm{pct}/100$), so a log $y$-axis is well-defined even when ACO
underperforms (ratio $< 1$). Incomplete suites are fine: emit uses the
intersection of graphs present in the selected directories. Adding a new
``vary_k*t*i_PN`` directory to ``param_dirs`` in build.json is enough for
it to appear once JSON is available.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

from ..common import list_json_paths, load_json, aco_timed_out
from ..table import (
    SECTION_ACO,
    compare_section,
    summarize_file,
)
from .flag_ablation import (
    _addplot_boxplot,
    _pgf_xticklabels,
)
from .helpers import reduced_edge_density, scatter_coords

PLOT_GROUPS = (
    "k-sweep",
    "theta-sweep",
    "density-wins",
    "param-density",
    "param-runtime",
)

# Sliding window (graph×suite counts) for pooled rolling win rate vs density.
DENSITY_WIN_POOLED_WINDOW = 21


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
      meta[label] = {
          "k": int,
          "theta": int,
          "directory": str,
          "n_complete": int,
          "n_timeout": int,
          "n_seen": int,
      }
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
        n_seen = 0
        n_timeout = 0
        for path in list_json_paths(directory):
            data = load_json(path)
            if data is None:
                skipped.append((path, "unreadable"))
                continue
            n_seen += 1
            if aco_timed_out(data):
                n_timeout += 1
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
            # Still record timeout counts so runtime captions can mention them.
            if n_seen > 0:
                meta[label] = {
                    "k": None,
                    "theta": None,
                    "directory": directory,
                    "n_complete": 0,
                    "n_timeout": n_timeout,
                    "n_seen": n_seen,
                }
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
            "n_complete": len(by_label[label]),
            "n_timeout": n_timeout,
            "n_seen": n_seen,
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
        counts = Counter(
            m[other]
            for m in meta.values()
            if m.get(other) is not None and m.get(axis) is not None
        )
        if not counts:
            return [], None
        fixed_value = counts.most_common(1)[0][0]

    selected = [
        (label, meta[label])
        for label in meta
        if meta[label].get(axis) is not None
        and meta[label].get(other) is not None
        and meta[label][other] == fixed_value
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
    takeaway,
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
    # Narrower panels + larger sep so y-labels do not collide between panels.
    width = 0.42
    height = 0.44

    if axis == "k":
        xlab = r"$k$"
        fixed_note = rf"$\theta = {int(fixed_other)}$"
    else:
        xlab = r"$\theta$"
        fixed_note = rf"$k = {int(fixed_other)}$"

    caption = (
        f"{caption_lead} at fixed {fixed_note} on {n} graphs matched across "
        f"all shown settings (100 ants, ACO-PN). Left: edge ratio "
        f"$|E_{{\\mathrm{{ACO}}}}|/|E_{{\\theta}}|$ (log scale; dashed "
        f"parity). Right: ACO win rate vs.\\ the $\\theta$-heuristic. "
        f"{takeaway}"
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
        r"    group style={group size=2 by 1, horizontal sep=52pt},",
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
        # One \addplot per bar would otherwise get a multi-series bar shift.
        r"    bar shift=0,",
        r"]",
    ]

    for i, (lab, _info) in enumerate(selected, start=1):
        rows = [by_label[lab][g] for g in matched]
        rate = _win_rate(rows)
        if rate is None:
            continue
        color = colors[(i - 1) % len(colors)]
        lines.append(
            rf"  \addplot[ybar, bar shift=0, fill={color}!40, draw={color}] "
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
        takeaway=(
            r"Relative edge counts hold as $k$ grows, and ACO wins more "
            r"often at larger $k$."
        ),
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
        takeaway=(
            r"Raising $\theta$ modestly lowers relative edge count and win "
            r"rate, but ACO still beats the heuristic on a clear majority."
        ),
    )


def _odd_window(requested, n_points, *, min_window=5):
    """Largest odd window ≤ min(requested, n_points), or None if too small."""
    w = min(int(requested), int(n_points))
    if w % 2 == 0:
        w = max(w - 1, 1)
    if w < min_window:
        return None
    return w


def _rolling_win_rate_coords(points, window):
    """
    Count-based rolling win rate along graphs sorted by an x-metric.

    ``points`` is a list of (x, won_bool). Returns (x_center, win_pct)
    for each full window of ``window`` consecutive graphs.
    """
    if window < 2 or len(points) < window:
        return []
    ordered = sorted(points, key=lambda p: p[0])
    half = window // 2
    coords = []
    for i in range(0, len(ordered) - window + 1):
        chunk = ordered[i : i + window]
        x_center = chunk[half][0]
        wins = sum(1 for _x, won in chunk if won)
        coords.append((float(x_center), 100.0 * wins / window))
    return coords


def _suite_density_win_points(by_label, meta):
    """
    Per-suite lists of (density, won) for every usable graph.

    Returns (ordered_meta_items, by_lab_points) where ordered_meta_items is
    sorted (label, info) and by_lab_points[label] = [(delta_R, won), ...].
    """
    ordered = sorted(
        meta.items(),
        key=lambda item: (item[1]["k"], item[1]["theta"], item[0]),
    )
    by_lab = {}
    for lab, _info in ordered:
        pts = []
        for row in by_label.get(lab, {}).values():
            dens = reduced_edge_density(row)
            if dens is None or dens <= 0:
                continue
            pts.append((float(dens), compare_section(row) == SECTION_ACO))
        by_lab[lab] = pts
    return ordered, by_lab


def density_wins_figure(
    by_label,
    meta,
    *,
    pooled_window=DENSITY_WIN_POOLED_WINDOW,
):
    """
    Rolling ACO win rate vs reduced density, pooled over all $(k, \\theta)$
    graph×suite observations. No per-graph scatter.
    """
    if not meta or not by_label:
        return [r"% density-wins: no usable param directories"]

    ordered, by_lab = _suite_density_win_points(by_label, meta)
    pooled_pts = []

    for lab, info in ordered:
        win_pts = by_lab.get(lab, [])
        if not win_pts:
            print(
                f"Warning: density-wins: suite {lab!r} "
                f"({_config_label(info['k'], info['theta'])}) has no "
                f"usable reduced densities.",
                file=sys.stderr,
            )
            continue
        pooled_pts.extend(win_pts)

    if not pooled_pts:
        return [r"% density-wins: no plottable suite points"]

    pooled_w = _odd_window(pooled_window, len(pooled_pts))
    pooled_roll = (
        _rolling_win_rate_coords(pooled_pts, pooled_w) if pooled_w else []
    )
    if not pooled_roll:
        print(
            "Warning: density-wins: curve omitted "
            f"(need ≥5 graph×suite points; have {len(pooled_pts)}).",
            file=sys.stderr,
        )
        return [r"% density-wins: insufficient points for rolling curve"]

    caption = (
        r"ACO-PN win rate against the $\theta$-heuristic (100 ants) "
        r"versus reduced edge density "
        r"$\text{den}(G_R) = |E_R|/(|U_R|\,|V_R|)$ "
        r"(sliding-window rate over all graph$\times$suite observations; "
        r"$x$ is the window-median density). "
        r"Win rate falls as reduced density rises."
    )

    coords = scatter_coords(pooled_roll)
    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{axis}[",
        r"    width=0.72\textwidth,",
        r"    height=0.48\textwidth,",
        r"    grid=major,",
        r"    xmode=log,",
        r"    ymin=0,",
        r"    ymax=105,",
        r"    xlabel={$\text{den}(G_R)$ (window median)},",
        r"    ylabel={ACO win rate (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Across all $(k,\theta)$},",
        r"    title style={font=\small},",
        r"  ]",
        rf"  \addplot[very thick, mark=*, mark size=1.2pt, "
        rf"color=black!75] coordinates {{{coords}}};",
        r"  \end{axis}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{caption}}}",
        r"  \label{fig:density-wins}",
        r"\end{figure}",
    ]
    return lines


def _densities(rows):
    """Positive reduced edge densities for log-scale boxplots."""
    vals = []
    for row in rows:
        dens = reduced_edge_density(row)
        if dens is None:
            continue
        dens = float(dens)
        if dens > 0:
            vals.append(dens)
    return vals


def _density_panel_lines(
    by_label,
    selected,
    matched,
    *,
    axis,
    fixed_other,
    colors,
    title,
):
    """Emit one groupplot panel of reduced-density boxplots."""
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
        r"    ylabel={Reduced density $\text{den}(G_R)$},",
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
        stats = _five_number_summary(_densities(rows))
        if stats is None:
            continue
        color = colors[(i - 1) % len(colors)]
        lines += _addplot_boxplot(i, stats, color=color)

    return {
        "lines": lines,
        "n": len(matched),
        "fixed_note": fixed_note,
        "empty": not selected or not matched,
    }


def param_density_figure(by_label, meta, *, fixed_theta=None, fixed_k=None):
    """
    2-panel reduced-density figure: vary k at fixed θ | vary θ at fixed k.
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
    k_panel = _density_panel_lines(
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
    t_panel = _density_panel_lines(
        by_label,
        t_selected,
        t_matched,
        axis="theta",
        fixed_other=t_fixed if t_fixed is not None else 3,
        colors=colors,
        title=r"Varying $\theta$",
    )

    if k_panel["empty"] and t_panel["empty"]:
        return [r"% param-density: no usable k or θ sweep directories"]

    caption = (
        r"Reduced edge density $\text{den}(G_R) = |E_R|/(|U_R|\,|V_R|)$ "
        r"(log scale) after common-neighbor reduction "
        r"(threshold $\theta - k$). Left: vary $k$ at fixed "
        f"{k_panel['fixed_note']}. Right: vary $\\theta$ at fixed "
        f"{t_panel['fixed_note']}. Raising $k$ tends to leave sparser "
        r"reduced graphs, while raising $\theta$ tends to leave denser "
        r"ones, so density---not $\theta$ alone---tracks the win-rate "
        r"shift in Figure~\ref{fig:density-wins}."
    )

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=52pt},",
        r"    width=0.42\textwidth,",
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
        r"  \label{fig:param-density}",
        r"\end{figure}",
    ]
    return lines


def _time_ratios(rows):
    """ACO discovery / θ-heuristic wall-time ratios (positive only)."""
    vals = []
    for row in rows:
        aco = row.get("aco_time")
        heur = row.get("heur_time")
        if aco is None or heur is None:
            continue
        aco = float(aco)
        heur = float(heur)
        if aco <= 0 or heur <= 0:
            continue
        vals.append(aco / heur)
    return vals


def _discovery_times(rows):
    """Positive ACO discovery times (seconds)."""
    vals = []
    for row in rows:
        aco = row.get("aco_time")
        if aco is None:
            continue
        aco = float(aco)
        if aco > 0:
            vals.append(aco)
    return vals


def _runtime_match(by_label, selected_labels, *, metric):
    """
    Graphs present in every selected suite with usable timing fields.

    For ``metric=\"ratio\"`` both ACO discovery and θ-heuristic times must be
    positive; for ``metric=\"discovery\"`` only ACO discovery must be positive.
    """
    common = match_sweep(by_label, selected_labels)
    usable = []
    for key in common:
        ok = True
        for lab in selected_labels:
            row = by_label[lab][key]
            aco = row.get("aco_time")
            if aco is None or float(aco) <= 0:
                ok = False
                break
            if metric == "ratio":
                heur = row.get("heur_time")
                if heur is None or float(heur) <= 0:
                    ok = False
                    break
        if ok:
            usable.append(key)
    return usable


def _runtime_panel_lines(
    by_label,
    selected,
    matched,
    *,
    axis,
    fixed_other,
    colors,
    title,
    metric,
):
    """
    One groupplot panel of runtime boxplots.

    ``metric`` is ``\"ratio\"`` ($T_{ACO}/T_θ$) or ``\"discovery\"`` (seconds).
    """
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

    if metric == "ratio":
        ylabel = r"$T_{\mathrm{ACO}}/T_{\theta}$"
        value_fn = _time_ratios
    else:
        ylabel = r"ACO discovery time (s)"
        value_fn = _discovery_times

    lines = [
        r"\nextgroupplot[",
        rf"    ylabel={{{ylabel}}},",
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
        stats = _five_number_summary(value_fn(rows))
        if stats is None:
            continue
        color = colors[(i - 1) % len(colors)]
        lines += _addplot_boxplot(i, stats, color=color)

    return {
        "lines": lines,
        "n": len(matched),
        "fixed_note": fixed_note,
        "empty": not selected or not matched,
        "selected": selected,
    }


def param_runtime_figure(by_label, meta, *, fixed_theta=None, fixed_k=None):
    """
    2×2 runtime figure: time ratio and absolute discovery vs $k$ / $\\theta$.
    """
    colors = (
        "blue!70!black",
        "orange!85!black",
        "teal!70!black",
        "purple!70!black",
        "red!70!black",
    )

    k_selected, k_fixed = _select_sweep(meta, axis="k", fixed_value=fixed_theta)
    k_labels = [lab for lab, _ in k_selected]
    # Same matched set for both left-column panels (need both times).
    k_matched = _runtime_match(by_label, k_labels, metric="ratio")
    t_selected, t_fixed = _select_sweep(meta, axis="theta", fixed_value=fixed_k)
    t_labels = [lab for lab, _ in t_selected]
    t_matched = _runtime_match(by_label, t_labels, metric="ratio")

    k_ratio = _runtime_panel_lines(
        by_label,
        k_selected,
        k_matched,
        axis="k",
        fixed_other=k_fixed if k_fixed is not None else 5,
        colors=colors,
        title=r"Time ratio vs.\ $k$",
        metric="ratio",
    )
    t_ratio = _runtime_panel_lines(
        by_label,
        t_selected,
        t_matched,
        axis="theta",
        fixed_other=t_fixed if t_fixed is not None else 3,
        colors=colors,
        title=r"Time ratio vs.\ $\theta$",
        metric="ratio",
    )
    k_abs = _runtime_panel_lines(
        by_label,
        k_selected,
        k_matched,
        axis="k",
        fixed_other=k_fixed if k_fixed is not None else 5,
        colors=colors,
        title=r"Discovery time vs.\ $k$",
        metric="discovery",
    )
    t_abs = _runtime_panel_lines(
        by_label,
        t_selected,
        t_matched,
        axis="theta",
        fixed_other=t_fixed if t_fixed is not None else 3,
        colors=colors,
        title=r"Discovery time vs.\ $\theta$",
        metric="discovery",
    )

    if k_ratio["empty"] and t_ratio["empty"]:
        return [r"% param-runtime: no usable k or θ sweep directories"]

    caption = (
        r"ACO-PN discovery cost versus the $\theta$-heuristic (100 ants) "
        r"across $(k, \theta)$ suites. Top: ratio "
        r"$T_{\mathrm{ACO}}/T_{\theta}$ (log scale). Bottom: absolute ACO "
        r"discovery time (log scale). Left: vary $k$ at fixed "
        f"{k_ratio['fixed_note']}. Right: vary $\\theta$ at fixed "
        f"{t_ratio['fixed_note']}. ACO is typically hundreds of times "
        r"slower than the heuristic but still generally has a runtime of "
        r"a few seconds. Increasing $k$ lengthens search while increasing $\theta$ "
        r"shortens it, likely because a low $k$ and large $\theta$ enable common-neighbor reduction to prune more nodes and reduce the search space. "
        r"Timed-out graphs are excluded."
    )

    empty = r"\nextgroupplot[title={(no data)}, title style={font=\small}]"
    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 2, horizontal sep=52pt, "
        r"vertical sep=40pt},",
        r"    width=0.42\textwidth,",
        r"    height=0.38\textwidth,",
        r"    grid=major,",
        r"  ]",
    ]
    for panel in (k_ratio, t_ratio, k_abs, t_abs):
        lines += panel["lines"] if not panel["empty"] else [empty]
    lines += [
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{caption}}}",
        r"  \label{fig:param-runtime}",
        r"\end{figure}",
    ]
    return lines


BUILDERS = {
    "k-sweep": k_sweep_figure,
    "theta-sweep": theta_sweep_figure,
    "density-wins": density_wins_figure,
    "param-density": param_density_figure,
    "param-runtime": param_runtime_figure,
}
