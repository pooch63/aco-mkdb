"""
k / θ parameter-sweep compare figures.

Plot groups
-----------
  k-sweep
    Both win-rate lines in one compact, captionless pair that stays in
    the paragraph (no float, so it cannot force a float page). Left:
    ACO win rate versus defect budget k at fixed θ. Right: the same
    versus minimum side size θ at fixed k. Uses every ``--param-dir``
    whose JSON reports the shared fixed value (default: the modal value
    among param dirs, typically θ = 5 and k = 3). Graphs are matched
    across the selected directories of each panel.

  theta-sweep
    Drawn as the right panel of ``k-sweep``. This plot group emits only
    a comment so a paper that still names it does not repeat the pair.

  density-wins
    Two bar panels of ACO win rate, on the same graph×suite rows.
    Top: five equal-count bins of reduced density
    $\\text{den}(G_R) = |E_R|/(|U_R|\\cdot|V_R|)$.
    Bottom: each value of $\\theta - k$.
    Each panel is labeled with the correlation of that predictor with
    the win/loss outcome, so the two associations can be compared.

  param-runtime
    Single-column 2×2 log-scale runtime figure. Each panel is one
    instance: ACO and the θ-heuristic as marker lines versus $k$ at
    fixed θ, with the graph name under that panel. Axis boxes are
    small; the gaps are sized so tick labels and names stay readable.
    The four instances are evenly spaced by $|U|+|V|$ among graphs
    with a complete $k$ sweep. Timed-out graphs are excluded.

Incomplete suites are fine: emit uses the intersection of graphs present
in the selected directories. Adding a new ``vary_k*t*i_PN`` directory to
``param_dirs`` in build.json is enough for it to appear once JSON is
available.
"""

from __future__ import annotations

import math
import os
import sys
from collections import Counter

from ..common import (
    COL2_HEIGHT,
    COL2_WIDTH,
    aco_timed_out,
    begin_float,
    begin_tikz,
    end_float,
    end_tikz,
    list_json_paths,
    load_json,
)
from ..table import (
    SECTION_ACO,
    compare_section,
    summarize_file,
)
from .flag_ablation import _pgf_xticklabels
from .helpers import reduced_edge_density

PLOT_GROUPS = (
    "k-sweep",
    "theta-sweep",
    "density-wins",
    "param-runtime",
)


def _dataset_key(path, data):
    label = data.get("dataset") or os.path.splitext(os.path.basename(path))[0]
    leaf = str(label).replace("\\", "/").split("/")[-1]
    return leaf.removesuffix("_ants")


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


def _pad_axis(values, *, include=None, floor=None, ceil=None, frac=0.12):
    """Pad a numeric axis the way the quality line plots do."""
    vals = [float(v) for v in values]
    if include is not None:
        vals.append(float(include))
    lo, hi = min(vals), max(vals)
    span = hi - lo
    if span == 0:
        span = 1.0 if hi == 0 else abs(hi) * 0.2
        if span == 0:
            span = 1.0
    pad = frac * span
    lo -= pad
    hi += pad
    if floor is not None:
        lo = max(float(floor), lo)
    if ceil is not None:
        hi = min(float(ceil), hi)
    if lo >= hi:
        lo, hi = min(vals) - 1.0, max(vals) + 1.0
    return lo, hi


def _tick_label(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:g}"


def _rate_ticks(ymin, ymax):
    """A few y ticks so a short axis does not crowd its labels."""
    span = float(ymax) - float(ymin)
    step = 50.0
    for candidate in (5.0, 10.0, 20.0, 25.0, 50.0):
        if span / candidate <= 4.5:
            step = candidate
            break
    start = math.ceil((float(ymin) - 1e-9) / step) * step
    ticks = []
    value = start
    while value <= float(ymax) + 1e-6:
        ticks.append(value)
        value += step
    return ticks or [float(ymin), float(ymax)]


def _sweep_series(by_label, meta, *, axis, fixed_value=None):
    """
    One win-rate series, or comments when that sweep cannot be drawn.

    Returns (series or None, comments). ``series`` has points, a panel
    title for the fixed parameter, and the swept-axis label.
    """
    name = "k-sweep" if axis == "k" else "theta-sweep"
    selected, fixed = _select_sweep(meta, axis=axis, fixed_value=fixed_value)
    if not selected:
        return None, [rf"% {name}: no param dirs available for this sweep"]
    matched = match_sweep(by_label, [lab for lab, _ in selected])
    if not matched:
        labels = ", ".join(lab for lab, _ in selected)
        return None, [rf"% {name}: no graphs matched across {labels}"]

    points = []
    for lab, info in selected:
        rows = [by_label[lab][g] for g in matched]
        rate = _win_rate(rows)
        if rate is None:
            continue
        x = info["k"] if axis == "k" else info["theta"]
        points.append((float(x), float(rate)))
    if not points:
        return None, [rf"% {name}: no win rates"]

    if axis == "k":
        held = fixed if fixed is not None else 5
        title = rf"$\theta = {int(held)}$"
        xlabel = r"$k$"
    else:
        held = fixed if fixed is not None else 3
        title = rf"$k = {int(held)}$"
        xlabel = r"$\theta$"
    return {"points": points, "title": title, "xlabel": xlabel}, []


def _sweep_pair_figure(by_label, meta, *, fixed_theta=None, fixed_k=None):
    """
    Side-by-side win-rate lines, in the text rather than a float.

    Axis boxes stay a fraction of \\columnwidth and the picture is not
    resized, so the pair stays short enough to sit with its paragraph.
    The right panel shares the left y scale and hides its tick labels.
    There is no caption: the surrounding prose names both panels.
    """
    k_series, comments = _sweep_series(
        by_label, meta, axis="k", fixed_value=fixed_theta
    )
    t_series, t_comments = _sweep_series(
        by_label, meta, axis="theta", fixed_value=fixed_k
    )
    comments += t_comments
    series = [s for s in (k_series, t_series) if s is not None]
    if not series:
        return comments

    rates = [y for s in series for _, y in s["points"]]
    ymin, ymax = _pad_axis(rates, include=50.0, floor=0.0, ceil=100.0, frac=0.08)
    yticks = ",".join(f"{t:g}" for t in _rate_ticks(ymin, ymax))

    lines = comments + [
        r"\begin{center}",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        rf"  group style={{group size={len(series)} by 1,",
        r"    horizontal sep=0.55cm},",
        r"  scale only axis,",
        r"  width=0.32\columnwidth,",
        r"  height=0.26\columnwidth,",
        r"  tick label style={font=\footnotesize, inner sep=1pt},",
        r"  label style={font=\footnotesize, inner sep=1pt},",
        r"  title style={font=\footnotesize, inner sep=1.5pt},",
        r"  xlabel style={align=center},",
        r"  grid=major,",
        r"  grid style={gray!25},",
        r"  tick align=outside,",
        rf"  ymin={ymin:.4f},",
        rf"  ymax={ymax:.4f},",
        rf"  ytick={{{yticks}}},",
        r"]",
    ]
    for i, s in enumerate(series):
        xs = [x for x, _ in s["points"]]
        xmin, xmax = _pad_axis(xs, frac=0.18)
        xtick = ",".join(_tick_label(x) for x in xs)
        coords = " ".join(f"({x:g},{y:.4f})" for x, y in s["points"])
        lines += [
            r"\nextgroupplot[",
            rf"  xlabel={{{s['xlabel']}}},",
            rf"  title={{{s['title']}}},",
            rf"  xmin={xmin:g},",
            rf"  xmax={xmax:g},",
            rf"  xtick={{{xtick}}},",
        ]
        if i > 0:
            lines.append(r"  yticklabels={},")
        lines += [
            r"]",
            (
                r"\addplot[solid, thick, mark=*, blue!80!black] "
                rf"coordinates {{{coords}}};"
            ),
            (
                rf"\draw[dashed, gray] (axis cs:{xmin:g},50) -- "
                rf"(axis cs:{xmax:g},50);"
            ),
        ]
    lines += [
        r"\end{groupplot}",
        r"\path (group c1r1.west)",
        r"  node[xshift=-30pt, rotate=90, anchor=center,",
        r"       font=\footnotesize, inner sep=0pt]",
        r"  {ACO win rate (\%)};",
        r"\end{tikzpicture}",
        r"\end{center}",
    ]
    return lines


def k_sweep_figure(by_label, meta, *, fixed_theta=None):
    return _sweep_pair_figure(by_label, meta, fixed_theta=fixed_theta)


def theta_sweep_figure(by_label, meta, *, fixed_k=None):
    del by_label, meta, fixed_k
    return [r"% theta-sweep is the right panel of the k-sweep pair"]


def _pearson(xs, ys):
    """Correlation of two equal-length sequences, or None if undefined."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0.0 or den_y == 0.0:
        return None
    return num / (den_x * den_y)


def _density_win_observations(by_label, meta):
    """One record per graph×suite row with a positive reduced density."""
    ordered = sorted(
        meta.items(),
        key=lambda item: (
            item[1].get("k") if item[1].get("k") is not None else 10**9,
            item[1].get("theta") if item[1].get("theta") is not None else 10**9,
            item[0],
        ),
    )
    obs = []
    for lab, info in ordered:
        if info.get("k") is None or info.get("theta") is None:
            continue
        rows = by_label.get(lab, {})
        added = 0
        for key, row in rows.items():
            dens = reduced_edge_density(row)
            if dens is None or dens <= 0:
                continue
            obs.append(
                {
                    "graph": key,
                    "k": float(info["k"]),
                    "theta": float(info["theta"]),
                    "density": float(dens),
                    "gap": float(info["theta"]) - float(info["k"]),
                    "won": compare_section(row) == SECTION_ACO,
                }
            )
            added += 1
        if rows and added == 0:
            print(
                f"Warning: density-wins: suite {lab!r} "
                f"({_config_label(info['k'], info['theta'])}) has no "
                f"usable reduced densities.",
                file=sys.stderr,
            )
    return obs


def _equal_count_rate_bins(obs, n_bins):
    """Equal-count bins of density. Each bin stores its median and win rate."""
    ordered = sorted(obs, key=lambda ob: ob["density"])
    n = len(ordered)
    if n_bins < 1 or n < n_bins:
        return []
    base, extra = divmod(n, n_bins)
    bins = []
    start = 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        chunk = ordered[start : start + size]
        start += size
        if not chunk:
            continue
        dens = sorted(ob["density"] for ob in chunk)
        mid = len(dens) // 2
        if len(dens) % 2:
            median = dens[mid]
        else:
            median = 0.5 * (dens[mid - 1] + dens[mid])
        wins = sum(1 for ob in chunk if ob["won"])
        bins.append(
            {
                "n": len(chunk),
                "median": median,
                "rate": wins / len(chunk),
            }
        )
    return bins


def _rates_by_gap(obs):
    """Win rate at each distinct value of θ − k."""
    groups = {}
    for ob in obs:
        groups.setdefault(ob["gap"], []).append(ob)
    rows = []
    for gap in sorted(groups):
        chunk = groups[gap]
        wins = sum(1 for ob in chunk if ob["won"])
        rows.append(
            {
                "gap": gap,
                "n": len(chunk),
                "rate": wins / len(chunk),
            }
        )
    return rows


def _density_tick(median):
    """Short tick label for a bin-median density."""
    if median >= 0.1:
        return f"{median:.2f}"
    if median >= 0.001:
        return f"{median:.3f}"
    return f"{median:.1e}"


def _gap_tick(gap):
    if float(gap).is_integer():
        return str(int(gap))
    return f"{gap:g}"


def _fmt_r(value):
    return f"{value:.2f}"


def _win_rate_bars(title, xlabel, tick_labels, rates, *, color):
    """One groupplot panel: win-rate bars on a shared 0–100% axis."""
    n_ticks = len(tick_labels)
    xtick = ",".join(str(i) for i in range(1, n_ticks + 1))
    lines = [
        r"\nextgroupplot[",
        r"    ylabel={ACO win rate (\%)},",
        r"    ylabel style={align=center, font=\small},",
        rf"    title={{{title}}},",
        r"    title style={font=\small},",
        rf"    xlabel={{{xlabel}}},",
        rf"    xtick={{{xtick}}},",
        rf"    xticklabels={_pgf_xticklabels(tick_labels)},",
        r"    x tick label style={font=\scriptsize},",
        r"    ymin=0,",
        r"    ymax=100,",
        r"    ytick={0,25,50,75,100},",
        r"    xmin=0.5,",
        rf"    xmax={n_ticks + 0.5:.1f},",
        r"    ybar,",
        r"    bar width=9pt,",
        r"    bar shift=0,",
        r"]",
    ]
    for i, rate in enumerate(rates, start=1):
        lines.append(
            rf"  \addplot[ybar, bar shift=0, fill={color}!40, draw={color}] "
            rf"coordinates {{({i},{100.0 * rate:.4g})}};"
        )
    return lines


def density_wins_figure(by_label, meta):
    """
    ACO win rate against reduced density and against θ − k.

    Both panels use the same rows and the same 0–100% axis. The panel
    title is the correlation of that predictor with the binary outcome.
    """
    if not meta or not by_label:
        return [r"% density-wins: no usable param directories"]

    obs = _density_win_observations(by_label, meta)
    if len(obs) < 8:
        return [r"% density-wins: no plottable suite points"]

    n_bins = 5 if len(obs) >= 25 else 4
    bins = _equal_count_rate_bins(obs, n_bins)
    gaps = _rates_by_gap(obs)
    if len(bins) < 2 or len(gaps) < 2:
        return [r"% density-wins: not enough bins to compare"]

    outcomes = [1.0 if ob["won"] else 0.0 for ob in obs]
    r_density = _pearson([ob["density"] for ob in obs], outcomes)
    r_gap = _pearson([ob["gap"] for ob in obs], outcomes)
    if r_density is None or r_gap is None:
        return [r"% density-wins: correlation was undefined"]

    n_obs = len(obs)
    n_wins = int(sum(outcomes))
    print(
        f"# density-wins: n={n_obs} wins={n_wins} "
        f"r_density={r_density:.3f} r_theta_minus_k={r_gap:.3f}",
        file=sys.stderr,
    )
    for b in bins:
        print(
            f"# density-wins density bin: median={b['median']:.6g} "
            f"n={b['n']} rate={b['rate']:.3f}",
            file=sys.stderr,
        )
    for row in gaps:
        print(
            f"# density-wins theta-k: value={row['gap']:.6g} "
            f"n={row['n']} rate={row['rate']:.3f}",
            file=sys.stderr,
        )

    dens_title = rf"$\text{{den}}(G_R)$, $r = {_fmt_r(r_density)}$"
    gap_title = rf"$\theta - k$, $r = {_fmt_r(r_gap)}$"
    caption = (
        r"ACO win rate (100 ants) against reduced density and against "
        r"$\theta - k$, on the same graph$\times$suite rows "
        rf"($n = {n_obs}$, {n_wins} wins). "
        rf"Top: {n_bins} equal-count bins of "
        r"$\text{den}(G_R) = |E_R|/(|U_R|\,|V_R|)$, labeled by the "
        r"median density in the bin. "
        r"Bottom: the win rate at each value of $\theta - k$. "
        r"$r$ is the correlation of that predictor with the win/loss "
        r"outcome. The predictor with the larger $|r|$ lines up more "
        r"closely with whether ACO beats the $\theta$-heuristic."
    )

    env, lines = begin_float("figure", star=False)
    lines += begin_tikz()
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=56pt},",
        rf"    width={COL2_WIDTH},",
        rf"    height={COL2_HEIGHT},",
        r"    grid=major,",
        r"  ]",
    ]
    lines += _win_rate_bars(
        dens_title,
        r"$\text{den}(G_R)$",
        [_density_tick(b["median"]) for b in bins],
        [b["rate"] for b in bins],
        color="blue!70!black",
    )
    lines += _win_rate_bars(
        gap_title,
        r"$\theta - k$",
        [_gap_tick(row["gap"]) for row in gaps],
        [row["rate"] for row in gaps],
        color="orange!85!black",
    )
    lines += [
        r"  \end{groupplot}",
    ]
    lines += end_tikz()
    lines += [
        rf"  \caption{{{caption}}}",
        r"  \label{fig:density-wins}",
    ]
    lines += end_float(env)
    return lines


# One color and marker shape per method, shared across the four instances.
_RUNTIME_SERIES = (
    {
        "legend": "ACO",
        "key": "aco_time",
        "color": "acoMark",
        "mark": "*",
    },
    {
        "legend": r"$\theta$-heuristic",
        "key": "heur_time",
        "color": "heurMark",
        "mark": "triangle*",
    },
)
_RUNTIME_PANELS = 4


def _positive_time(row, key):
    value = row.get(key)
    if value is None:
        return None
    value = float(value)
    if value <= 0:
        return None
    return value


def _runtime_match(by_label, selected_labels):
    """Graphs present in every selected suite with both times positive."""
    common = match_sweep(by_label, selected_labels)
    usable = []
    for key in common:
        ok = True
        for lab in selected_labels:
            row = by_label[lab][key]
            if _positive_time(row, "aco_time") is None:
                ok = False
                break
            if _positive_time(row, "heur_time") is None:
                ok = False
                break
        if ok:
            usable.append(key)
    return usable


def _graph_nodes(row):
    """Original bipartition size $|U|+|V|$, stable across $(k, \\theta)$."""
    n_u = row.get("nU")
    n_v = row.get("nV")
    if n_u is None or n_v is None:
        return None
    return float(n_u) + float(n_v)


def _instance_title(key):
    """Title-case dataset key, matching table display names."""
    return str(key).replace("_", " ").replace("-", " ").title()


def _coord_body(points):
    return " ".join(f"({x:.6g},{y:.6g})" for x, y in points)


def _pick_runtime_instances(by_label, selected):
    """
    Up to four graphs with a complete time sweep, evenly spaced by size.

    Order is increasing $|U|+|V|$ so the panels run from a small instance
    to a large one, left to right and top to bottom.
    """
    if not selected:
        return []
    labels = [lab for lab, _ in selected]
    ranked = []
    for key in _runtime_match(by_label, labels):
        size = _graph_nodes(by_label[labels[0]][key])
        if size is None:
            continue
        ranked.append((size, key))
    ranked.sort()
    n = len(ranked)
    if n == 0:
        return []
    if n <= _RUNTIME_PANELS:
        return [key for _, key in ranked]
    idxs = [
        min(n - 1, (i * (n - 1)) // (_RUNTIME_PANELS - 1))
        for i in range(_RUNTIME_PANELS)
    ]
    # Integer division can repeat an index when n is small; keep first wins.
    picked = []
    seen = set()
    for idx in idxs:
        key = ranked[idx][1]
        if key in seen:
            continue
        seen.add(key)
        picked.append(key)
    return picked


def _instance_panel_lines(
    by_label, selected, graph, *, with_legend, with_ylabel
):
    """One instance: ACO and θ-heuristic time versus $k$."""
    ks = [int(info["k"]) for _, info in selected]
    series_pts = []
    for spec in _RUNTIME_SERIES:
        pts = []
        for (lab, info), k in zip(selected, ks):
            value = _positive_time(by_label[lab][graph], spec["key"])
            if value is None:
                continue
            pts.append((k, value))
        series_pts.append((spec, pts))
    if not any(pts for _, pts in series_pts):
        return None

    k_min, k_max = min(ks), max(ks)
    xtick = ",".join(str(k) for k in ks)
    # Y-axis title only on the left column. The right column still
    # draws its own log tick labels; repeating "Time (s)" there is what
    # pushed the pair wider than one column.
    lines = [r"\nextgroupplot["]
    if with_ylabel:
        lines += [
            r"    ylabel={Time (s)},",
            r"    ylabel style={font=\scriptsize, inner sep=1pt},",
        ]
    lines += [
        r"    xlabel={$k$},",
        r"    xlabel style={font=\scriptsize, inner sep=1pt},",
        rf"    xtick={{{xtick}}},",
        rf"    xmin={k_min - 0.35:.2f},",
        rf"    xmax={k_max + 0.35:.2f},",
        r"    enlarge y limits=0.08,",
    ]
    if with_legend:
        lines += [
            r"    legend to name=paramRuntimeLegend,",
            r"    legend columns=-1,",
            r"    legend cell align=left,",
            r"    legend image post style={scale=0.9},",
            r"    legend style={draw=black!60, fill=white, font=\scriptsize, "
            r"inner xsep=3pt, inner ysep=1pt, "
            r"/tikz/every even column/.append style={column sep=0.7em}},",
        ]
    lines.append(r"]")
    for spec, pts in series_pts:
        if not pts:
            continue
        color = spec["color"]
        lines.append(
            rf"  \addplot[semithick, mark={spec['mark']}, mark size=1.7pt, "
            rf"color={color}, "
            rf"mark options={{solid, fill={color}, draw={color}}}] "
            rf"coordinates {{{_coord_body(pts)}}};"
        )
        if with_legend:
            lines.append(rf"  \addlegendentry{{{spec['legend']}}}")
    return lines


def param_runtime_figure(by_label, meta, *, fixed_theta=None, fixed_k=None):
    """
    2×2 runtime figure: one instance per panel, ACO vs the θ-heuristic.
    """
    k_selected, k_fixed = _select_sweep(meta, axis="k", fixed_value=fixed_theta)
    instances = _pick_runtime_instances(by_label, k_selected)
    if not instances:
        return [r"% param-runtime: no instance with a complete k sweep"]

    theta = int(k_fixed if k_fixed is not None else 5)
    panels = []
    panel_graphs = []
    for i, graph in enumerate(instances):
        block = _instance_panel_lines(
            by_label,
            k_selected,
            graph,
            with_legend=(i == 0),
            with_ylabel=(i % 2 == 0),
        )
        if block:
            panels.append(block)
            panel_graphs.append(graph)
    if not panels:
        return [r"% param-runtime: no instance with a complete k sweep"]

    names = ", ".join(_instance_title(g) for g in panel_graphs)
    caption = (
        rf"Discovery time on four instances (100 ants, $\theta = {theta}$, "
        r"log scale). Each panel is one graph. Circles are ACO and "
        r"triangles are the $\theta$-heuristic. The instances are evenly "
        rf"spaced by $|U|+|V|$ among graphs finished at every "
        rf"$k$ in the sweep: {names}. Timed-out graphs are excluded."
    )

    env, lines = begin_float("figure", star=False)
    # Natural size inside one column. Axis boxes are a fraction of
    # \columnwidth; the seps are absolute so $k$, the log ticks, and
    # the graph name keep a readable gap when the boxes shrink.
    # \resizebox would scale those labels down with the picture.
    lines += begin_tikz(resize=False)
    lines += [
        r"  \definecolor{acoMark}{RGB}{31,119,180}",
        r"  \definecolor{heurMark}{RGB}{214,39,150}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 2,",
        r"      horizontal sep=0.95cm, vertical sep=1.12cm},",
        r"    scale only axis,",
        r"    width=0.28\columnwidth,",
        r"    height=0.22\columnwidth,",
        r"    ymode=log,",
        r"    grid=major,",
        r"    major grid style={black!22, thin},",
        r"    minor tick num=0,",
        r"    tick align=outside,",
        r"    tick style={black, thin},",
        r"    tick label style={font=\scriptsize, inner sep=1pt},",
        r"    label style={font=\scriptsize},",
        r"    ylabel near ticks,",
        r"  ]",
    ]
    for block in panels:
        lines += block
    lines += [
        r"  \end{groupplot}",
        r"  \path (group c1r1.north) -- (group c2r1.north)",
        r"    node[midway, above=2pt, inner sep=0pt]",
        r"    {\pgfplotslegendfromname{paramRuntimeLegend}};",
    ]
    # Graph name sits under $k$. A second line inside xlabel does not
    # grow the axis bounding box, so the next row was drawn on top of it.
    # vertical sep above leaves room for both lines.
    for i, graph in enumerate(panel_graphs):
        col = (i % 2) + 1
        row = (i // 2) + 1
        letter = chr(ord("a") + i)
        name = _instance_title(graph)
        lines.append(
            rf"  \node[anchor=north, yshift=-1pt, font=\scriptsize, "
            rf"inner sep=1pt] "
            rf"at (group c{col}r{row}.outer south) {{({letter}) {name}}};"
        )
    lines += end_tikz(resize=False)
    lines += [
        rf"  \caption{{{caption}}}",
        r"  \label{fig:param-runtime}",
    ]
    lines += end_float(env)
    return lines


BUILDERS = {
    "k-sweep": k_sweep_figure,
    "theta-sweep": theta_sweep_figure,
    "density-wins": density_wins_figure,
    "param-runtime": param_runtime_figure,
}
