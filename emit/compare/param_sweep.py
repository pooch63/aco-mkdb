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
    One column, two panels. Upper: ACO win rate versus k at fixed θ,
    three bars per k (equal-count low / medium / high thirds of reduced
    density). Lower: the same versus θ at fixed k. Same fixed values as
    k-sweep. No quintiles.

  param-runtime
    Discovery time on four instances. Which sweeps are drawn is
    ``runtime_sweeps`` in build.json (passed as ``--runtime-sweep``):
    each ``k`` entry is time versus k at one fixed θ, and each
    ``theta`` entry is time versus θ at one fixed k. The swept values
    are every completed suite at that fixed parameter in ``param_dirs``,
    including any value beyond the listed ``values``.     One color per
    sweep; circles are ACO and triangles are the θ-heuristic. A legend
    under each 2×2 names the color and the marker. Each panel's axis
    is the range that graph has. The four instances are evenly spaced
    by $|U|+|V|$ among graphs finished at every plotted suite. A
    listed (k, θ) with no completed JSON is skipped with a warning.

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


# Equal-count thirds within one swept value. Darker is denser.
_N_TERTILES = 3
_TERTILE_COLORS = ("blue!60", "blue!75!black", "black")
_TERTILE_NAMES = ("low", "mid", "high")


def _sweep_observations(by_label, selected, *, axis):
    """Rows in one k-sweep or θ-sweep, tagged with the swept value."""
    obs = []
    for lab, info in selected:
        for key, row in by_label.get(lab, {}).items():
            dens = reduced_edge_density(row)
            if dens is None or dens <= 0:
                continue
            obs.append(
                {
                    "graph": key,
                    "k": float(info["k"]),
                    "theta": float(info["theta"]),
                    "density": float(dens),
                    "value": float(info[axis]),
                    "won": compare_section(row) == SECTION_ACO,
                }
            )
    return obs


def _obs_sort_key(ob):
    """Stable order so equal densities always fall in the same bin."""
    return (ob["density"], ob["graph"], ob["k"], ob["theta"])


def _split_equal_count(obs, n_bins):
    """Equal-count chunks of ``obs``, densest last. Empty if too few rows."""
    ordered = sorted(obs, key=_obs_sort_key)
    n = len(ordered)
    if n_bins < 1 or n < n_bins:
        return []
    base, extra = divmod(n, n_bins)
    chunks = []
    start = 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        chunks.append(ordered[start : start + size])
        start += size
    return chunks


def _median(values):
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _chunk_stats(chunk):
    dens = [ob["density"] for ob in chunk]
    wins = sum(1 for ob in chunk if ob["won"])
    return {
        "n": len(chunk),
        "rate": wins / len(chunk),
        "median": _median(dens),
        "lo": min(dens),
        "hi": max(dens),
    }


def _gap_tick(gap):
    if float(gap).is_integer():
        return str(int(gap))
    return f"{gap:g}"


def _density_tick(median):
    """Short label for a bin-median reduced density."""
    if median >= 0.1:
        return f"{median:.2f}"
    if median >= 0.001:
        return f"{median:.3f}"
    return f"{median:.1e}"


def _tertile_rows(obs, group_key):
    """
    Win rate in density thirds at each fixed value of ``group_key``.

    Thirds are equal-count within that value, so ``high`` is the densest
    third at that reduction setting, not a global density cutoff.
    """
    groups = {}
    for ob in obs:
        groups.setdefault(ob[group_key], []).append(ob)
    rows = []
    for value in sorted(groups):
        chunks = _split_equal_count(groups[value], _N_TERTILES)
        if len(chunks) < _N_TERTILES or any(len(c) < 3 for c in chunks):
            print(
                f"Warning: density-wins: skip {group_key}="
                f"{_gap_tick(value)} (n={len(groups[value])}); "
                "fewer than three graphs in a density third.",
                file=sys.stderr,
            )
            continue
        rows.append(
            {"value": value, "tertiles": [_chunk_stats(c) for c in chunks]}
        )
    return rows


def _log_tertiles(tag, tertile_rows):
    """Print the bar percentages so the build log matches the figure."""
    print(
        f"# density-wins {tag}: density thirds (low / mid / high)",
        file=sys.stderr,
    )
    for row in tertile_rows:
        cells = "  ".join(
            f"{_TERTILE_NAMES[i]}={100.0 * t['rate']:.0f}% "
            f"n={t['n']} den=[{_density_tick(t['lo'])},"
            f"{_density_tick(t['hi'])}]"
            for i, t in enumerate(row["tertiles"])
        )
        print(
            f"#   {_gap_tick(row['value'])}: {cells}",
            file=sys.stderr,
        )


def _percent_coords(xs, rate_by_x):
    parts = []
    for x in xs:
        rate = rate_by_x.get(x)
        label = _gap_tick(x)
        if rate is None:
            parts.append(f"({label},nan)")
        else:
            parts.append(f"({label},{100.0 * rate:.4g})")
    return " ".join(parts)


def _rate_axis(xs, *, title, xlabel, extra):
    xmin = min(xs) - 0.7
    xmax = max(xs) + 0.7
    xtick = ",".join(_gap_tick(x) for x in xs)
    lines = [
        r"  \nextgroupplot[",
        rf"    title={{{title}}},",
        r"    title style={font=\scriptsize, inner sep=1pt},",
        r"    ylabel={ACO win rate (\%)},",
        r"    ylabel style={font=\scriptsize},",
        r"    ymin=0,",
        r"    ymax=100,",
        r"    ytick={0,50,100},",
        r"    ymajorgrids=true,",
        r"    xmajorgrids=false,",
        r"    grid style={gray!25},",
        r"    tick label style={font=\scriptsize},",
        rf"    xmin={xmin:.4g},",
        rf"    xmax={xmax:.4g},",
        rf"    xtick={{{xtick}}},",
    ]
    if xlabel:
        lines.append(rf"    xlabel={{{xlabel}}},")
        lines.append(r"    xlabel style={font=\scriptsize},")
    lines.extend(extra)
    lines.append(r"  ]")
    return lines, xmin, xmax


def _fifty_line(xmin, xmax):
    return (
        rf"  \draw[dashed, gray] (axis cs:{xmin:.4g},50) -- "
        rf"(axis cs:{xmax:.4g},50);"
    )


def _tertile_panel(rows, *, title, xlabel):
    """Grouped bars: three density thirds at each parameter value."""
    xs = [row["value"] for row in rows]
    # Narrower bars once several parameter values share the column.
    bar_pt = 5.2 if len(xs) >= 6 else 7.5
    lines, xmin, xmax = _rate_axis(
        xs,
        title=title,
        xlabel=xlabel,
        extra=[
            r"    ybar,",
            rf"    bar width={bar_pt:.1f}pt,",
        ],
    )
    for index, color in enumerate(_TERTILE_COLORS):
        rates = {row["value"]: row["tertiles"][index]["rate"] for row in rows}
        coords = _percent_coords(xs, rates)
        lines.append(
            rf"  \addplot[fill={color}, draw=black!55] "
            rf"coordinates {{{coords}}};"
        )
    lines.append(_fifty_line(xmin, xmax))
    return lines


def _sweep_panel(by_label, meta, *, axis):
    """Density-third bars along one sweep, or a comment if it cannot be drawn."""
    selected, fixed = _select_sweep(meta, axis=axis)
    if not selected or fixed is None:
        name = "k" if axis == "k" else "theta"
        return None, [rf"% density-wins: no {name} sweep"]
    obs = _sweep_observations(by_label, selected, axis=axis)
    rows = _tertile_rows(obs, "value")
    if axis == "k":
        tag = f"k at theta={int(fixed)}"
        title = rf"$\theta = {int(fixed)}$"
        xlabel = r"$k$"
    else:
        tag = f"theta at k={int(fixed)}"
        title = rf"$k = {int(fixed)}$"
        xlabel = r"$\theta$"
    _log_tertiles(tag, rows)
    if len(rows) < 2:
        return None, [rf"% density-wins: not enough points for {tag}"]
    return _tertile_panel(rows, title=title, xlabel=xlabel), []


def density_wins_figure(by_label, meta):
    """
    Win rate versus k and versus θ, split into density thirds.

    One column. The upper panel varies k at fixed θ; the lower varies
    θ at fixed k. The three bars are equal-count thirds of reduced
    density within that swept value.
    """
    if not meta or not by_label:
        return [r"% density-wins: no usable param directories"]

    k_panel, comments = _sweep_panel(by_label, meta, axis="k")
    t_panel, t_comments = _sweep_panel(by_label, meta, axis="theta")
    comments += t_comments
    panels = [p for p in (k_panel, t_panel) if p is not None]
    if not panels:
        return comments

    env, lines = begin_float("figure", star=False)
    lines += begin_tikz()
    lines += [
        r"  \begin{groupplot}[",
        rf"    group style={{group size=1 by {len(panels)}, vertical sep=54pt}},",
        rf"    width={COL2_WIDTH},",
        r"    height=0.38\columnwidth,",
        r"  ]",
    ]
    for panel in panels:
        lines += panel
    lines += [r"  \end{groupplot}"]
    lines += end_tikz()
    lines += [
        r"  \caption{ACO win rate by reduced-density third.}",
        r"  \label{fig:density-wins}",
    ]
    lines += end_float(env)
    return comments + lines


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
    picked = []
    seen = set()
    for idx in idxs:
        key = ranked[idx][1]
        if key in seen:
            continue
        seen.add(key)
        picked.append(key)
    return picked


# One color per configured sweep. ACO is a filled circle, the
# θ-heuristic a triangle, both in that sweep's color.
_RUNTIME_PALETTE = (
    (31, 119, 180),
    (255, 127, 14),
    (44, 160, 44),
    (214, 39, 40),
    (148, 103, 189),
    (140, 86, 75),
    (227, 119, 194),
    (127, 127, 127),
)


def _completed_sweep_points(meta, *, axis, fixed):
    """Every completed suite at this fixed parameter, sorted by the swept value."""
    fixed = int(fixed)
    found = []
    for label, info in meta.items():
        k = info.get("k")
        theta = info.get("theta")
        if k is None or theta is None:
            continue
        if axis == "k" and int(theta) == fixed:
            found.append((int(k), label, info))
        elif axis == "theta" and int(k) == fixed:
            found.append((int(theta), label, info))
    found.sort(key=lambda item: (item[0], item[1]))
    unique = []
    seen = set()
    for x, label, info in found:
        if x in seen:
            print(
                f"Warning: param-runtime: duplicate {axis}={x} "
                f"at fixed {fixed}; keeping the first suite.",
                file=sys.stderr,
            )
            continue
        seen.add(x)
        unique.append((x, label, info))
    return unique


def resolve_runtime_sweeps(meta, sweeps):
    """
    Turn build.json runtime_sweeps into plotted points.

    ``sweeps`` is a list of ``{"axis", "fixed", "values"}``. ``axis``
    ``"k"`` varies k at ``fixed`` θ; ``"theta"`` varies θ at ``fixed`` k.
    The plotted values are every completed suite at that fixed parameter,
    so a ``values`` list cannot hide a finished ``(k, θ)``. Listed values
    with no JSON are skipped with a warning. ``None`` keeps the previous
    single modal k-sweep.
    """
    if not sweeps:
        selected, fixed = _select_sweep(meta, axis="k")
        if not selected or fixed is None:
            return []
        return [
            {
                "axis": "k",
                "fixed": int(fixed),
                "points": [
                    (int(info["k"]), lab, info) for lab, info in selected
                ],
            }
        ]
    resolved = []
    for spec in sweeps:
        axis = spec["axis"]
        fixed = int(spec["fixed"])
        available = _completed_sweep_points(meta, axis=axis, fixed=fixed)
        by_x = {x: (x, lab, info) for x, lab, info in available}
        requested = [int(raw) for raw in spec.get("values") or []]
        wanted = sorted(set(requested) | set(by_x))
        points = []
        for x in wanted:
            if x not in by_x:
                if axis == "k":
                    where = f"k={x}, theta={fixed}"
                else:
                    where = f"k={fixed}, theta={x}"
                print(
                    f"Warning: param-runtime: no completed suite for {where}.",
                    file=sys.stderr,
                )
                continue
            points.append(by_x[x])
        if not points:
            print(
                f"Warning: param-runtime: no points on {axis} sweep "
                f"at fixed {fixed}.",
                file=sys.stderr,
            )
            continue
        resolved.append({"axis": axis, "fixed": fixed, "points": points})
    return resolved


def _half_series(resolved, axis):
    series = []
    held = [sw for sw in resolved if sw["axis"] == axis]
    for i, sw in enumerate(held):
        if axis == "k":
            legend = rf"$\theta = {sw['fixed']}$"
        else:
            legend = rf"$k = {sw['fixed']}$"
        series.append(
            {
                "legend": legend,
                "color": f"rt{i}",
                "rgb": _RUNTIME_PALETTE[i % len(_RUNTIME_PALETTE)],
                "pairs": [(x, lab) for x, lab, _ in sw["points"]],
            }
        )
    return series


def _panel_xs(by_label, series, graph):
    """Swept values this graph can actually plot, across every series."""
    xs = set()
    for s in series:
        for x, lab in s["pairs"]:
            row = by_label.get(lab, {}).get(graph)
            if row is None:
                continue
            if any(
                _positive_time(row, spec["key"]) is not None
                for spec in _RUNTIME_SERIES
            ):
                xs.add(x)
    return sorted(xs)


def _legend_entry(spec, series):
    """Method and the fixed parameter, for the key under the group."""
    return rf"{spec['legend']} ({series['legend']})"


def _multi_series_panel(
    by_label, series, graph, *, xlabel, with_ylabel, legend_name=None,
):
    """One instance: every sweep on this axis, ACO and the θ-heuristic."""
    xs = _panel_xs(by_label, series, graph)
    if not xs:
        return None
    lines = [r"\nextgroupplot["]
    if with_ylabel:
        lines += [
            r"    ylabel={Time (s)},",
            r"    ylabel style={font=\scriptsize, inner sep=1pt},",
        ]
    lines += [
        rf"    xlabel={{{xlabel}}},",
        r"    xlabel style={font=\scriptsize, inner sep=1pt},",
        rf"    xtick={{{','.join(str(x) for x in xs)}}},",
        rf"    xmin={min(xs) - 0.35:.2f},",
        rf"    xmax={max(xs) + 0.35:.2f},",
        r"    enlarge y limits=0.08,",
    ]
    if legend_name:
        lines += [
            rf"    legend to name={legend_name},",
            r"    legend columns=2,",
            r"    legend cell align=left,",
            r"    legend style={",
            r"      draw=black!45, fill=white, font=\scriptsize,",
            r"      inner sep=3pt, row sep=1pt,",
            r"      /tikz/every even column/.append style={column sep=0.85em}},",
        ]
    lines.append(r"]")
    drew = False
    for s in series:
        color = s["color"]
        for spec in _RUNTIME_SERIES:
            pts = []
            for x, lab in s["pairs"]:
                row = by_label.get(lab, {}).get(graph)
                if row is None:
                    continue
                value = _positive_time(row, spec["key"])
                if value is None:
                    continue
                pts.append((x, value))
            if not pts:
                continue
            drew = True
            dashed = "" if spec["key"] == "aco_time" else "dashed, "
            lines.append(
                rf"  \addplot[{dashed}semithick, mark={spec['mark']}, "
                rf"mark size=1.6pt, color={color}, "
                rf"mark options={{solid, fill={color}, draw={color}}}] "
                rf"coordinates {{{_coord_body(pts)}}};"
            )
            if legend_name:
                lines.append(
                    rf"  \addlegendentry{{{_legend_entry(spec, s)}}}"
                )
    if not drew:
        return None
    return lines


def _runtime_half(by_label, series, instances, *, xlabel, legend_name):
    """One 2×2 group: four instances, sweeps on ``series``."""
    lines = begin_tikz(resize=False)
    for s in series:
        r, g, b = s["rgb"]
        lines.append(
            rf"  \definecolor{{{s['color']}}}{{RGB}}{{{r},{g},{b}}}"
        )
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 2,",
        # One gap between the two columns; the column is wide enough
        # for a slightly larger split. Vertical room is only the
        # xlabel plus the panel name.
        r"      horizontal sep=1.5cm, vertical sep=1.48cm},",
        r"    scale only axis,",
        r"    width=0.28\columnwidth,",
        r"    height=0.22\columnwidth,",
        r"    ymode=log,",
        r"    grid=major,",
        r"    major grid style={black!22, thin},",
        # minor tick num is ignored on a log axis; an empty list
        # is what drops the marks between the grid lines.
        r"    minor tick num=0,",
        r"    minor xtick={},",
        r"    minor ytick={},",
        r"    tick align=outside,",
        r"    tick style={black, thin},",
        r"    tick label style={font=\scriptsize, inner sep=1pt},",
        r"    label style={font=\scriptsize},",
        r"    ylabel near ticks,",
        r"  ]",
    ]
    panel_graphs = []
    legend_open = False
    for i, graph in enumerate(instances):
        block = _multi_series_panel(
            by_label,
            series,
            graph,
            xlabel=xlabel,
            with_ylabel=(i % 2 == 0),
            legend_name=None if legend_open else legend_name,
        )
        if block:
            lines += block
            panel_graphs.append(graph)
            legend_open = True
    if not panel_graphs:
        return [], []
    lines.append(r"  \end{groupplot}")
    for i, graph in enumerate(panel_graphs):
        col = (i % 2) + 1
        row = (i // 2) + 1
        letter = chr(ord("a") + i)
        name = _instance_title(graph)
        # Axis south is the frame. Tick labels and the xlabel hang
        # below it; -26pt clears that xlabel without a large gap.
        lines.append(
            rf"  \node[anchor=north, font=\scriptsize, inner sep=0pt, "
            rf"align=center] "
            rf"at ([yshift=-26pt]group c{col}r{row}.south) "
            rf"{{({letter}) {name}}};"
        )
    if legend_open:
        lines.append(
            rf"  \node[anchor=north, inner sep=0pt] "
            rf"at ([yshift=-3pt]current bounding box.south) "
            rf"{{\pgfplotslegendfromname{{{legend_name}}}}};"
        )
    lines += end_tikz(resize=False)
    return lines, panel_graphs


def param_runtime_figure(by_label, meta, *, fixed_theta=None, fixed_k=None,
                         sweeps=None):
    """
    Runtime on four instances for every sweep in ``runtime_sweeps``.

    ``fixed_theta`` / ``fixed_k`` are unused; the lists in build.json
    decide the axes. ``sweeps is None`` falls back to the modal k sweep.
    """
    del fixed_theta, fixed_k
    resolved = resolve_runtime_sweeps(meta, sweeps)
    if not resolved:
        return [r"% param-runtime: no sweeps to plot"]

    chosen = []
    seen = set()
    for sw in resolved:
        for _x, lab, info in sw["points"]:
            if lab in seen:
                continue
            seen.add(lab)
            chosen.append((lab, info))
    instances = _pick_runtime_instances(by_label, chosen)
    if not instances:
        return [r"% param-runtime: no instance finished every plotted suite"]

    halves = []
    if _half_series(resolved, "k"):
        halves.append(("k", r"$k$"))
    if _half_series(resolved, "theta"):
        halves.append(("theta", r"$\theta$"))
    series_groups = [_half_series(resolved, axis) for axis, _xlabel in halves]

    caption = r"Discovery time versus $k$ and $\theta$."

    env, lines = begin_float("figure", star=False)
    first_names = None
    for (axis, xlabel), series in zip(halves, series_groups):
        block, panel_graphs = _runtime_half(
            by_label,
            series,
            instances,
            xlabel=xlabel,
            legend_name="paramRuntimeK" if axis == "k" else "paramRuntimeTheta",
        )
        if not block:
            continue
        if first_names is None:
            first_names = panel_graphs
        elif lines and lines[-1] != r"  \centering":
            lines.append(r"  \par\vspace{0.35em}")
        lines += block
    if first_names is None:
        return [r"% param-runtime: no instance finished every plotted suite"]
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
