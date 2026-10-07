"""
Anytime, best-of-R, equal-budget, and pheromone-ablation figures.

Data model
----------
``scripts/budget-trace.bash`` (via ``vary.jl``) runs one ant count for
``T`` epochs and ``R`` counted replicates, and stores ``epoch_trace`` on
each trial: the incumbent after every completed epoch (edge count,
θ-feasibility, cumulative search time). Algorithm 1 has no schedule that
depends on ``T``, so epoch ``t`` of that trace is the run that would have
been returned by stopping there.

These figures read only that recorded trace. They do not credit a trial
from ``iterations_to_best``, and they do not treat the first ``R`` seeds
as an estimate of the best-of-``R`` expectation.

Plot groups
-----------
  iteration-budget
      Median and IQR of ``|E(B_t)| / |E(D_θ)|`` vs epoch (log x), faint
      per-graph curves, cumulative time on a second axis, and a separate
      panel for ``P(B_t is θ-feasible)``.
  replicate-budget
      For each ``R``, the mean across graphs of the expected maximum over
      ``R`` replicates drawn without replacement, with a bootstrap CI,
      plus the per-graph coefficient of variation of ``|E(B)|``.
  budget-allocation
      Median relative quality over the ``(T, R)`` grid, with contours of
      constant ``T·R``, and those equal-cost slices drawn as curves.
  pheromone-ablation
      The anytime median with pheromone on, against ``Δτ = 0`` (τ stays
      at 1) and any recorded ρ sweep, at the same ant count.
"""

from __future__ import annotations

import math
import random
import statistics
import sys

from ..common import (
    aco_timed_out,
    begin_float,
    begin_tikz,
    counted_trials,
    display_name,
    end_float,
    end_tikz,
    list_json_paths,
    load_json,
)

PLOT_GROUPS = (
    "iteration-budget",
    "replicate-budget",
    "budget-allocation",
    "pheromone-ablation",
)
DEFAULT_ANTS = 100
N_BOOT = 1000
BOOT_SEED = 0

# Log-spaced epochs used for the allocation heatmap (full traces still
# feed the anytime curves). Integer epochs nearest these targets.
_HEATMAP_EPOCHS = (
    1, 2, 3, 4, 5, 7, 10, 14, 20, 28, 40, 56, 80, 100,
)


def expected_max(values, r):
    """
    Exact E[max of an r-subset drawn without replacement].

    With the sample sorted nondecreasing, the maximum equals ``x_(j)``
    with probability C(j-1, r-1) / C(n, r) for j = r…n.
    """
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if r < 1 or r > n:
        return None
    comb = 1.0
    for i in range(1, r + 1):
        comb *= (n - r + i) / i
    prob = 1.0 / comb
    total = xs[r - 1] * prob
    for j in range(r + 1, n + 1):
        prob *= (j - 1) / (j - r)
        total += xs[j - 1] * prob
    return total


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    weight = pos - lo
    return float(sorted_vals[lo]) * (1.0 - weight) + float(sorted_vals[hi]) * weight


def _bootstrap_mean_ci(values, rng, n_boot=N_BOOT):
    """Percentile interval for the mean. None when fewer than 2 graphs."""
    n = len(values)
    if n < 2:
        return None, None
    means = []
    for _ in range(n_boot):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.mean(draw))
    means.sort()
    return _quantile(means, 0.025), _quantile(means, 0.975)


def _index_trace(points):
    by_t = {}
    for pt in points or []:
        t = pt.get("t")
        edges = pt.get("edges")
        if t is None or edges is None:
            continue
        by_t[int(t)] = {
            "edges": int(edges),
            "theta_feasible": pt.get("theta_feasible") is True,
            "elapsed_s": (
                float(pt["elapsed_s"]) if pt.get("elapsed_s") is not None else None
            ),
        }
    return by_t


def load_suite(json_paths, *, ants=DEFAULT_ANTS):
    """
    Counted replicates that recorded an epoch trace, one entry per graph.

    JIT warmup replicates are omitted. The process-wide ``max_replicates``
    cap is ignored so R = 1…R_max uses every counted replicate.
    """
    skipped = []
    graphs = []
    for path in json_paths:
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
        heur = (data.get("heuristic") or {}).get("final_edges")
        if heur is None:
            skipped.append((path, "missing heuristic.final_edges"))
            continue
        reps = []
        for trial in counted_trials(
            data.get("trials") or [], data, max_replicates=None
        ):
            if ants is not None and trial.get("ants") != ants:
                continue
            by_t = _index_trace(trial.get("epoch_trace"))
            if by_t:
                reps.append(by_t)
        if not reps:
            skipped.append((path, f"no epoch_trace at ants={ants}"))
            continue
        graphs.append({
            "name": display_name(path, data),
            "heur": int(heur),
            "ants": ants,
            "pheromone": data.get("pheromone"),
            "evaporation": data.get("evaporation"),
            "reps": reps,
        })
    return graphs, skipped


def _epochs(graphs):
    seen = set()
    for graph in graphs:
        for rep in graph["reps"]:
            seen.update(rep)
    return sorted(seen)


def _at(rep, t, field):
    pt = rep.get(t)
    if pt is None:
        return None
    return pt[field]


def _per_graph_median_ratio(graph, epochs):
    heur = graph["heur"]
    if heur <= 0:
        return [None] * len(epochs)
    out = []
    for t in epochs:
        ratios = []
        for rep in graph["reps"]:
            edges = _at(rep, t, "edges")
            if edges is None:
                continue
            ratios.append(edges / heur)
        out.append(statistics.median(ratios) if ratios else None)
    return out


def _per_graph_feas_rate(graph, epochs):
    out = []
    for t in epochs:
        flags = [
            _at(rep, t, "theta_feasible")
            for rep in graph["reps"]
            if _at(rep, t, "theta_feasible") is not None
        ]
        out.append(statistics.mean(flags) if flags else None)
    return out


def _per_graph_median_time(graph, epochs):
    out = []
    for t in epochs:
        times = [
            _at(rep, t, "elapsed_s")
            for rep in graph["reps"]
            if _at(rep, t, "elapsed_s") is not None
        ]
        out.append(statistics.median(times) if times else None)
    return out


def _across(columns):
    """Median, q25, q75 down a list of per-graph series."""
    n = len(columns[0]) if columns else 0
    med, lo, hi = [], [], []
    for i in range(n):
        vals = sorted(v for col in columns if (v := col[i]) is not None)
        if not vals:
            med.append(None)
            lo.append(None)
            hi.append(None)
            continue
        med.append(statistics.median(vals))
        lo.append(_quantile(vals, 0.25))
        hi.append(_quantile(vals, 0.75))
    return med, lo, hi


def _full_budget_edges(graph):
    """Edge counts at the longest epoch shared by the graph's replicates."""
    if not graph["reps"]:
        return [], None
    cap = min(max(rep) for rep in graph["reps"] if rep)
    edges = []
    for rep in graph["reps"]:
        if cap in rep:
            edges.append(rep[cap]["edges"])
    return edges, cap


def _factor_splits(cost, t_max, r_max):
    splits = []
    for t in range(1, t_max + 1):
        if cost % t:
            continue
        r = cost // t
        if 1 <= r <= r_max:
            splits.append((t, r))
    return splits


def _coords(xs, ys, *, digits=4):
    parts = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        parts.append(f"({x:g},{y:.{digits}f})")
    return " ".join(parts)


def _log_ticks(xmax):
    marks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    kept = [m for m in marks if m <= xmax]
    if not kept or kept[-1] != xmax:
        if xmax not in kept and xmax > 1:
            kept.append(xmax)
    return "{" + ",".join(str(m) for m in kept) + "}"


def _missing_figure(label, message):
    env, lines = begin_float("figure", star=False)
    lines += [
        rf"  \caption{{{message}}}",
        rf"  \label{{{label}}}",
    ]
    lines += end_float(env)
    return lines


def _style_keys():
    return [
        r"    scale only axis,",
        r"    tick label style={font=\scriptsize},",
        r"    label style={font=\small},",
        r"    title style={font=\small},",
        r"    every axis plot/.append style={line join=round},",
        r"    grid=major,",
        r"    grid style={gray!25},",
        r"    tick style={black!60},",
        r"    scaled ticks=false,",
    ]


def anytime_figure(graphs):
    """Median/IQR relative quality, time, and θ-feasibility vs epoch."""
    if not graphs:
        return _missing_figure(
            "fig:iteration-budget",
            "Epoch traces are not recorded yet. "
            r"Run \texttt{scripts/budget-trace.bash} to produce this figure.",
        )

    epochs = _epochs(graphs)
    ratios = [_per_graph_median_ratio(g, epochs) for g in graphs]
    feas = [_per_graph_feas_rate(g, epochs) for g in graphs]
    times = [_per_graph_median_time(g, epochs) for g in graphs]
    # Graphs whose θ-heuristic returned no edges have an undefined ratio.
    ratio_graphs = [col for col, g in zip(ratios, graphs) if g["heur"] > 0]
    med, lo, hi = _across(ratio_graphs)
    feas_med, feas_lo, feas_hi = _across(feas)
    time_med, _, _ = _across(times)

    positive_hi = [v for v in hi if v is not None]
    ymax = 2.0
    if positive_hi:
        ymax = max(1.35, _quantile(sorted(positive_hi), 0.9) * 1.2)
    med_max = max((v for v in med if v is not None), default=1.0)
    ymax = max(ymax, med_max * 1.05)

    xtick = _log_ticks(epochs[-1])
    width = r"0.42\textwidth"
    height = r"0.34\textwidth"

    env, lines = begin_float("figure", star=False)
    lines += begin_tikz()
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=42pt},",
        *_style_keys(),
        rf"    width={width},",
        rf"    height={height},",
        r"    xmode=log,",
        r"    log ticks with fixed point,",
        r"    xlabel={Epoch $t$},",
        r"    xmin=1,",
        rf"    xmax={epochs[-1]},",
        rf"    xtick={xtick},",
        r"    log basis x=10,",
        r"  ]",
        r"  \nextgroupplot[",
        r"    ylabel={$|E(B_t)|\,/\,|E(D_\theta)|$},",
        r"    title={Relative edge count},",
        r"    ymin=0,",
        rf"    ymax={ymax:.4f},",
        r"    legend style={",
        r"      at={(0.03,0.97)}, anchor=north west,",
        r"      font=\scriptsize, draw=none,",
        r"      fill=white, fill opacity=0.85, text opacity=1,",
        r"      legend cell align=left},",
        r"    legend columns=1,",
        r"  ]",
    ]
    for col in ratio_graphs:
        coords = _coords(epochs, col)
        if not coords:
            continue
        lines.append(
            rf"  \addplot[black, opacity=0.18, thin, forget plot] "
            rf"coordinates {{{coords}}};"
        )
    upper = _coords(epochs, hi)
    lower = _coords(epochs, lo)
    if upper and lower:
        lines += [
            rf"  \addplot[name path=ratiohi, draw=none, forget plot] "
            rf"coordinates {{{upper}}};",
            rf"  \addplot[name path=ratiolo, draw=none, forget plot] "
            rf"coordinates {{{lower}}};",
            r"  \addplot[blue!70!black, fill opacity=0.20, draw=none, forget plot] "
            r"fill between[of=ratiohi and ratiolo];",
        ]
    med_coords = _coords(epochs, med)
    if med_coords:
        lines.append(
            rf"  \addplot[very thick, blue!70!black] coordinates {{{med_coords}}};"
        )
        lines.append(r"  \addlegendentry{median}")
    if upper and lower:
        lines.append(
            r"  \addlegendimage{area legend, fill=blue!70!black, fill opacity=0.45}"
        )
        lines.append(r"  \addlegendentry{IQR}")
    lines.append(
        rf"  \addplot[dotted, black!55] coordinates {{(1,1) ({epochs[-1]},1)}};"
    )
    lines.append(r"  \addlegendentry{$\theta$-heuristic}")
    lines += [
        r"  \nextgroupplot[",
        r"    ylabel={$P(B_t$ is $\theta$-feasible$)$},",
        r"    title={$\theta$-feasibility},",
        r"    ymin=0,",
        r"    ymax=1.05,",
        r"  ]",
    ]
    for col in feas:
        coords = _coords(epochs, col)
        if not coords:
            continue
        lines.append(
            rf"  \addplot[black, opacity=0.18, thin, forget plot] "
            rf"coordinates {{{coords}}};"
        )
    f_upper = _coords(epochs, feas_hi)
    f_lower = _coords(epochs, feas_lo)
    if f_upper and f_lower:
        lines += [
            rf"  \addplot[name path=feashi, draw=none, forget plot] "
            rf"coordinates {{{f_upper}}};",
            rf"  \addplot[name path=feaslo, draw=none, forget plot] "
            rf"coordinates {{{f_lower}}};",
            r"  \addplot[teal!70!black, fill opacity=0.20, draw=none, forget plot] "
            r"fill between[of=feashi and feaslo];",
        ]
    feas_coords = _coords(epochs, feas_med)
    if feas_coords:
        lines.append(
            rf"  \addplot[very thick, teal!70!black] coordinates {{{feas_coords}}};"
        )
    lines += [r"  \end{groupplot}"]

    time_coords = _coords(epochs, time_med)
    time_vals = [v for v in time_med if v is not None and v > 0]
    if time_coords and time_vals:
        tmin = min(time_vals)
        tmax = max(time_vals)
        ymode = r"    ymode=log," if tmax / tmin > 8 else ""
        lines += [
            r"  \begin{axis}[",
            rf"    at={{(group c1r1.south west)}},",
            r"    anchor=south west,",
            r"    scale only axis,",
            rf"    width={width},",
            rf"    height={height},",
            r"    axis y line*=right,",
            r"    axis x line=none,",
            r"    xtick=\empty,",
            r"    xmode=log,",
            r"    xmin=1,",
            rf"    xmax={epochs[-1]},",
            ymode,
            r"    ylabel={Cumulative time (s)},",
            r"    ylabel style={font=\small},",
            r"    tick label style={font=\scriptsize},",
            r"    scaled ticks=false,",
            r"    log ticks with fixed point,",
            r"    legend style={",
            r"      at={(0.97,0.03)}, anchor=south east,",
            r"      font=\scriptsize, draw=none,",
            r"      fill=white, fill opacity=0.85, text opacity=1},",
            r"  ]",
            rf"  \addplot[thick, dashed, orange!85!black] coordinates {{{time_coords}}};",
            r"  \addlegendentry{median time}",
            r"  \end{axis}",
        ]
    lines += end_tikz()
    n_rep = sum(len(g["reps"]) for g in graphs)
    t_max = epochs[-1]
    lines += [
        (
            rf"  \caption{{Anytime quality at {graphs[0]['ants']} ants "
            rf"($T{{=}}{t_max}$ epochs, {len(graphs)} graphs, "
            rf"{n_rep} counted replicates). "
            rf"Each faint curve is one graph (median over its replicates). "
            rf"The band is the interquartile range across graphs. "
            rf"The right axis is median cumulative search time. "
            rf"$\theta$-feasibility is the fraction of replicates whose "
            rf"incumbent has at least $\theta$ vertices on both sides.}}"
        ),
        r"  \label{fig:iteration-budget}",
    ]
    lines += end_float(env)
    return lines


def replicate_figure(graphs):
    """Mean best-of-R relative quality with bootstrap CI, and per-graph CV."""
    if not graphs:
        return _missing_figure(
            "fig:replicate-budget",
            "Counted epoch traces are not recorded yet. "
            r"Run \texttt{scripts/budget-trace.bash} to produce this figure.",
        )

    scored = []
    cvs = []
    r_max = 0
    t_used = None
    for graph in graphs:
        edges, cap = _full_budget_edges(graph)
        if cap is None or not edges:
            continue
        t_used = cap if t_used is None else min(t_used, cap)
        r_max = max(r_max, len(edges))
        if graph["heur"] > 0 and len(edges) >= 2:
            mean = statistics.mean(edges)
            if mean > 0:
                cvs.append(statistics.stdev(edges) / mean)
        if graph["heur"] > 0:
            scored.append((graph["heur"], edges))

    if not scored or r_max < 1:
        return _missing_figure(
            "fig:replicate-budget",
            "Epoch traces did not contain enough replicates to estimate best-of-$R$.",
        )

    budgets = list(range(1, r_max + 1))
    means = []
    lo_band = []
    hi_band = []
    rng = random.Random(BOOT_SEED)
    for r in budgets:
        eligible = [expected_max(edges, r) / heur for heur, edges in scored if len(edges) >= r]
        eligible = [v for v in eligible if v is not None]
        if not eligible:
            means.append(None)
            lo_band.append(None)
            hi_band.append(None)
            continue
        means.append(statistics.mean(eligible))
        lo, hi = _bootstrap_mean_ci(eligible, rng)
        lo_band.append(lo)
        hi_band.append(hi)

    mean_max = max((v for v in means if v is not None), default=1.0)
    hi_max = max((v for v in hi_band if v is not None), default=mean_max)
    ymax = max(1.35, hi_max * 1.08, mean_max * 1.08)

    xtick = "{" + ",".join(
        str(b) for b in budgets if b == 1 or b == r_max or b % 5 == 0
    ) + "}"

    env, lines = begin_float("figure", star=False)
    lines += begin_tikz()
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=42pt},",
        *_style_keys(),
        r"    width=0.42\textwidth,",
        r"    height=0.34\textwidth,",
        r"  ]",
        r"  \nextgroupplot[",
        r"    xlabel={Replicates $R$},",
        r"    ylabel={$E[\max |E|]\,/\,|E(D_\theta)|$},",
        r"    title={Best of $R$ restarts},",
        r"    xmin=0.5,",
        rf"    xmax={r_max + 0.5},",
        rf"    xtick={xtick},",
        r"    ymin=0,",
        rf"    ymax={ymax:.4f},",
        r"    legend style={",
        r"      at={(0.03,0.97)}, anchor=north west,",
        r"      font=\scriptsize, draw=none,",
        r"      fill=white, fill opacity=0.85, text opacity=1,",
        r"      legend cell align=left},",
        r"  ]",
    ]
    upper = _coords(budgets, hi_band)
    lower = _coords(budgets, lo_band)
    if upper and lower:
        lines += [
            rf"  \addplot[name path=boothi, draw=none, forget plot] coordinates {{{upper}}};",
            rf"  \addplot[name path=bootlo, draw=none, forget plot] coordinates {{{lower}}};",
            r"  \addplot[blue!70!black, fill opacity=0.20, draw=none, forget plot] "
            r"fill between[of=boothi and bootlo];",
        ]
    mean_coords = _coords(budgets, means)
    if mean_coords:
        lines.append(
            rf"  \addplot[very thick, mark=*, mark size=1.1pt, blue!70!black] "
            rf"coordinates {{{mean_coords}}};"
        )
        lines.append(r"  \addlegendentry{mean}")
    if upper and lower:
        lines.append(
            r"  \addlegendimage{area legend, fill=blue!70!black, fill opacity=0.45}"
        )
        lines.append(r"  \addlegendentry{95\% bootstrap CI}")
    lines.append(
        rf"  \addplot[dotted, black!55] coordinates {{(1,1) ({r_max},1)}};"
    )
    lines.append(r"  \addlegendentry{$\theta$-heuristic}")
    lines += [
        r"  \nextgroupplot[",
        r"    xlabel={Graphs, sorted by CV},",
        r"    ylabel={Coefficient of variation},",
        r"    title={Spread across restarts},",
        r"    xmin=0.5,",
        rf"    xmax={max(len(cvs), 1) + 0.5},",
        r"    ymin=0,",
        r"    xtick=\empty,",
        r"    scaled y ticks=false,",
        r"    yticklabel={\pgfmathprintnumber[fixed, precision=2]{\tick}},",
    ]
    if cvs:
        cv_max = max(cvs)
        lines.append(rf"    ymax={max(0.05, cv_max * 1.15):.4f},")
    lines += [
        r"    legend style={",
        r"      at={(0.97,0.97)}, anchor=north east,",
        r"      font=\scriptsize, draw=none,",
        r"      fill=white, fill opacity=0.85, text opacity=1},",
        r"  ]",
    ]
    if cvs:
        ordered = sorted(cvs)
        cv_coords = " ".join(f"({i},{v:.4f})" for i, v in enumerate(ordered, start=1))
        med_cv = statistics.median(ordered)
        lines += [
            rf"  \addplot[only marks, mark=*, mark size=1.3pt, teal!70!black] "
            rf"coordinates {{{cv_coords}}};",
            rf"  \addplot[dashed, black!60] coordinates {{(1,{med_cv:.4f}) ({len(ordered)},{med_cv:.4f})}};",
            r"  \addlegendentry{median CV}",
        ]
    else:
        lines.append(r"  % no graph had a positive mean edge count")
    lines += [r"  \end{groupplot}"]
    lines += end_tikz()
    n_rep = sum(len(edges) for _heur, edges in scored)
    cv_note = ""
    if cvs:
        cv_note = (
            f" Median coefficient of variation is {statistics.median(cvs):.3f}."
        )
    lines += [
        (
            rf"  \caption{{Best of $R$ independent replicates at epoch {t_used} "
            rf"({len(scored)} graphs, up to {r_max} counted replicates). "
            rf"For each graph, $E[\max]$ is the expected best edge count "
            rf"over subsets of size $R$ drawn without replacement from that "
            rf"graph's replicates; the curve is the mean of those values "
            rf"relative to $|E(D_\theta)|$, and the band is a bootstrap "
            rf"confidence interval over graphs. "
            rf"The bottom panel is the coefficient of variation of $|E(B)|$ "
            rf"across replicates of each graph.{cv_note}}}"
        ),
        r"  \label{fig:replicate-budget}",
    ]
    lines += end_float(env)
    return lines


def _quality_grid(graphs):
    """Median relative E[max of r draws at epoch t] on a log-spaced epoch grid."""
    t_cap = max(_epochs(graphs))
    r_cap = max(len(g["reps"]) for g in graphs)
    epochs = [t for t in _HEATMAP_EPOCHS if t <= t_cap]
    if t_cap not in epochs:
        epochs.append(t_cap)
    rs = list(range(1, r_cap + 1))
    # Precompute, per graph, the edge list at each heatmap epoch.
    prepared = []
    for graph in graphs:
        if graph["heur"] <= 0:
            continue
        columns = {}
        for t in epochs:
            edges = [
                rep[t]["edges"] for rep in graph["reps"] if t in rep
            ]
            if edges:
                columns[t] = edges
        if columns:
            prepared.append((graph["heur"], columns))
    grid = {}
    for t in epochs:
        for r in rs:
            vals = []
            for heur, columns in prepared:
                edges = columns.get(t)
                if edges is None or len(edges) < r:
                    continue
                est = expected_max(edges, r)
                if est is not None:
                    vals.append(est / heur)
            if vals:
                grid[(t, r)] = statistics.median(vals)
    epochs, rs = _dense_rectangle(epochs, rs, grid)
    grid = {(t, r): grid[(t, r)] for t in epochs for r in rs if (t, r) in grid}
    return epochs, rs, grid


def _dense_rectangle(epochs, rs, grid):
    """Largest top-left rectangle of (epoch, replicate) cells that were estimated."""
    for r_hi in range(len(rs), 0, -1):
        sub_r = rs[:r_hi]
        use_t = [t for t in epochs if all((t, r) in grid for r in sub_r)]
        if len(use_t) >= 2 and len(sub_r) >= 1:
            return use_t, sub_r
    return [], []


def allocation_figure(graphs):
    """Heatmap of median relative quality with iso-cost contours and slices."""
    if not graphs:
        return _missing_figure(
            "fig:budget-allocation",
            "Epoch traces are not recorded yet. "
            r"Run \texttt{scripts/budget-trace.bash} to produce this figure.",
        )
    epochs, rs, grid = _quality_grid(graphs)
    if not grid:
        return _missing_figure(
            "fig:budget-allocation",
            "Epoch traces did not contain a $(T, R)$ grid to plot.",
        )

    vals = list(grid.values())
    zmin = min(vals)
    zmax = max(vals)
    if zmax <= zmin:
        zmax = zmin + 1.0

    t_max = epochs[-1]
    r_max = rs[-1]
    # Costs that fit inside the recorded rectangle, including the corners
    # (1, C) and (C, 1) when both sides are long enough.
    corner = min(t_max, r_max)
    costs = []
    for cost in (corner, 100, 300, 1000):
        splits = _factor_splits(cost, t_max, r_max)
        if len(splits) >= 3 and cost not in costs:
            costs.append(cost)
    if not costs and corner >= 2:
        costs = [corner]
    slice_ts = [
        t
        for cost in costs
        for t, _r in _factor_splits(cost, t_max, r_max)
    ]
    slice_xmax = max(slice_ts) if slice_ts else t_max

    env, lines = begin_float("figure", star=True)
    lines += begin_tikz()
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=18mm},",
        *_style_keys(),
        r"    width=0.40\textwidth,",
        r"    height=0.36\textwidth,",
        r"  ]",
        r"  \nextgroupplot[",
        r"    view={0}{90},",
        r"    xlabel={Epochs $T$},",
        r"    ylabel={Replicates $R$},",
        r"    title={Quality over $(T,R)$},",
        r"    colormap/viridis,",
        r"    colorbar,",
        r"    colorbar style={font=\scriptsize, ylabel={median $|E|/|E(D_\theta)|$}},",
        rf"    point meta min={zmin:.4f},",
        rf"    point meta max={zmax:.4f},",
        rf"    zmin={zmin:.4f},",
        rf"    zmax={zmax:.4f},",
        rf"    xtick={{{_log10_tick_spec(epochs)}}},",
        rf"    xticklabels={{{_log10_tick_labels(epochs)}}},",
        rf"    ytick={{{_log10_tick_spec(rs)}}},",
        rf"    yticklabels={{{_log10_tick_labels(rs)}}},",
        r"  ]",
    ]
    # One scanline per replicate count; T runs across the row.
    mesh_rows = []
    for r in rs:
        row = " ".join(
            f"({math.log10(t):.5f},{math.log10(r):.5f},{grid[(t, r)]:.5f})"
            for t in epochs
        )
        mesh_rows.append(row)
    lines += [
        rf"  \addplot3[surf, shader=interp, mesh/rows={len(rs)}, "
        rf"mesh/cols={len(epochs)}] coordinates {{",
        "\n".join(mesh_rows),
        r"  };",
    ]
    # Iso-cost lines in the same log10 plane. Drawn after the surface so
    # they sit on the top view; a second 2-D axis overlays them cleanly.
    lines.append(r"  \nextgroupplot[")
    lines += [
        r"    xlabel={Epochs $T$},",
        r"    ylabel={$|E|/|E(D_\theta)|$},",
        r"    title={Equal ant-epoch cost},",
        r"    xmode=log,",
        r"    log ticks with fixed point,",
        r"    xmin=1,",
        rf"    xmax={slice_xmax},",
        rf"    xtick={_log_ticks(slice_xmax)},",
        r"    ymin=0,",
        rf"    ymax={max(1.35, zmax * 1.08):.4f},",
        r"    legend style={",
        r"      at={(0.03,0.97)}, anchor=north west,",
        r"      font=\scriptsize, draw=none,",
        r"      fill=white, fill opacity=0.85, text opacity=1,",
        r"      legend cell align=left},",
        r"  ]",
    ]
    slice_colors = (
        "blue!70!black",
        "orange!85!black",
        "teal!70!black",
        "violet!70!black",
    )
    for idx, cost in enumerate(costs):
        splits = _factor_splits(cost, t_max, r_max)
        xs = [t for t, _r in splits]
        ys = [grid.get((t, r)) for t, r in splits]
        coords = _coords(xs, ys)
        if not coords:
            continue
        color = slice_colors[idx % len(slice_colors)]
        lines.append(
            rf"  \addplot[very thick, mark=*, mark size=1.2pt, {color}] "
            rf"coordinates {{{coords}}};"
        )
        lines.append(rf"  \addlegendentry{{$T R={cost}$}}")
    lines.append(
        rf"  \addplot[dotted, black!55, forget plot] coordinates {{(1,1) ({t_max},1)}};"
    )
    lines += [r"  \end{groupplot}"]

    # Contours over the heatmap. Same log10 limits as the surf axis.
    x_hi = math.log10(t_max)
    y_hi = math.log10(r_max)
    lines += [
        r"  \begin{axis}[",
        r"    at={(group c1r1.south west)},",
        r"    anchor=south west,",
        r"    scale only axis,",
        r"    width=0.40\textwidth,",
        r"    height=0.36\textwidth,",
        r"    axis lines=none,",
        r"    ticks=none,",
        r"    xmin=0,",
        rf"    xmax={x_hi:.5f},",
        r"    ymin=0,",
        rf"    ymax={y_hi:.5f},",
        r"    clip=true,",
        r"  ]",
    ]
    for idx, cost in enumerate(costs):
        # log10(R) = log10(C) - log10(T), restricted to the rectangle.
        t_lo = max(1, math.ceil(cost / r_max))
        t_hi = min(t_max, cost)  # R >= 1
        if t_hi <= t_lo:
            continue
        color = "white"
        lines.append(
            rf"  \addplot[thick, {color}] coordinates {{"
            rf"({math.log10(t_lo):.5f},{math.log10(cost / t_lo):.5f}) "
            rf"({math.log10(t_hi):.5f},{math.log10(cost / t_hi):.5f})}};"
        )
        mid_t = math.sqrt(t_lo * t_hi)
        mid_r = cost / mid_t
        if 1 <= mid_r <= r_max:
            lines.append(
                rf"  \node[font=\scriptsize, text=white, inner sep=1pt] "
                rf"at ({{axis cs:{math.log10(mid_t):.5f},{math.log10(mid_r):.5f}}}) "
                rf"{{${cost}$}};"
            )
    lines += [r"  \end{axis}"]
    lines += end_tikz()
    lines += [
        (
            rf"  \caption{{Allocation of a fixed ant-epoch budget $A\cdot T\cdot R$ "
            rf"at {graphs[0]['ants']} ants ({len(graphs)} graphs). "
            rf"Color is the median across graphs of $E[\max |E|]/|E(D_\theta)|$, "
            rf"where the expectation is over $R$ replicates drawn without "
            rf"replacement and each replicate is read at epoch $T$. "
            rf"White lines are constant $T\cdot R$. The right panel is that "
            rf"same median along those contours.}}"
        ),
        r"  \label{fig:budget-allocation}",
    ]
    lines += end_float(env)
    return lines


def _log10_tick_values(values):
    """Nice log ticks inside the grid, always including the endpoints."""
    lo, hi = min(values), max(values)
    present = [v for v in (1, 2, 5, 10, 20, 50, 100, 200, 500) if lo <= v <= hi]
    if lo not in present:
        present = [lo, *present]
    if hi not in present:
        present.append(hi)
    return present


def _log10_tick_spec(values):
    return ",".join(f"{math.log10(v):.4f}" for v in _log10_tick_values(values))


def _log10_tick_labels(values):
    return ",".join(str(v) for v in _log10_tick_values(values))


def _series_legend(label):
    if label in ("dt0", "delta0", "no-pheromone"):
        return r"$\Delta\tau=0$"
    if label.startswith("rho"):
        raw = label[3:].lstrip("= ")
        return rf"$\rho={raw}$"
    return label.replace("_", r"\_")


_ABLATION_STYLES = (
    ("red!70!black", "dashed"),
    ("orange!85!black", "solid"),
    ("violet!75!black", "solid"),
    ("green!45!black", "solid"),
    ("brown!70!black", "solid"),
)


def pheromone_figure(graphs, ablation):
    """Anytime median with pheromone on, against Δτ=0 and optional ρ curves."""
    if not graphs and not ablation:
        return _missing_figure(
            "fig:pheromone-ablation",
            "Pheromone and ablation traces are not recorded yet. "
            r"Run \texttt{ABLATION=1 RHO\_SWEEP=1 scripts/budget-trace.bash}.",
        )
    if not graphs:
        return _missing_figure(
            "fig:pheromone-ablation",
            r"The pheromone-on epoch trace is not recorded yet.",
        )

    series = [("baseline", graphs)]
    for label, suite in ablation.items():
        if suite:
            series.append((label, suite))

    # Shared epoch axis: intersection of each series' epoch set, then union
    # so a shorter ablation still draws as far as it goes.
    all_epochs = sorted(set().union(*(_epochs(suite) for _lab, suite in series)))
    if not all_epochs:
        return _missing_figure(
            "fig:pheromone-ablation",
            "Recorded traces had no epochs to plot.",
        )

    env, lines = begin_float("figure", star=False)
    lines += begin_tikz()
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=42pt},",
        *_style_keys(),
        r"    width=0.42\textwidth,",
        r"    height=0.34\textwidth,",
        r"    xmode=log,",
        r"    log ticks with fixed point,",
        r"    xlabel={Epoch $t$},",
        r"    xmin=1,",
        rf"    xmax={all_epochs[-1]},",
        rf"    xtick={_log_ticks(all_epochs[-1])},",
        r"  ]",
        r"  \nextgroupplot[",
        r"    ylabel={median $|E(B_t)|\,/\,|E(D_\theta)|$},",
        r"    title={Relative edge count},",
        r"    ymin=0,",
        r"    legend style={",
        r"      at={(0.03,0.97)}, anchor=north west,",
        r"      font=\scriptsize, draw=none,",
        r"      fill=white, fill opacity=0.85, text opacity=1,",
        r"      legend cell align=left},",
        r"  ]",
    ]

    ymax = 1.35
    drawn = 0
    for idx, (label, suite) in enumerate(series):
        epochs = _epochs(suite)
        cols = [
            _per_graph_median_ratio(g, epochs)
            for g in suite if g["heur"] > 0
        ]
        med, _, _ = _across(cols)
        coords = _coords(epochs, med)
        if not coords:
            continue
        peak = max((v for v in med if v is not None), default=1.0)
        ymax = max(ymax, peak * 1.08)
        if label == "baseline":
            style = r"very thick, blue!70!black"
            legend = r"$\Delta\tau=1$, $\rho=0.05$"
        else:
            color, dash = _ABLATION_STYLES[(idx - 1) % len(_ABLATION_STYLES)]
            style = rf"very thick, {dash}, {color}"
            legend = _series_legend(label)
        lines.append(rf"  \addplot[{style}] coordinates {{{coords}}};")
        lines.append(rf"  \addlegendentry{{{legend}}}")
        drawn += 1
    lines.append(
        rf"  \addplot[dotted, black!55] coordinates {{(1,1) ({all_epochs[-1]},1)}};"
    )
    lines.append(r"  \addlegendentry{$\theta$-heuristic}")
    # ymax was learned while emitting plots; pgfplots needs it on the axis.
    # Set it with \pgfplotsset before the group by rewriting: inject via a
    # placeholder we fill below. Use a generous default if we patch in place.
    lines += [
        r"  \nextgroupplot[",
        r"    ylabel={$P(B_t$ is $\theta$-feasible$)$},",
        r"    title={$\theta$-feasibility},",
        r"    ymin=0,",
        r"    ymax=1.05,",
        r"  ]",
    ]
    for idx, (label, suite) in enumerate(series):
        epochs = _epochs(suite)
        cols = [_per_graph_feas_rate(g, epochs) for g in suite]
        med, _, _ = _across(cols)
        coords = _coords(epochs, med)
        if not coords:
            continue
        if label == "baseline":
            style = r"very thick, blue!70!black"
        else:
            color, dash = _ABLATION_STYLES[(idx - 1) % len(_ABLATION_STYLES)]
            style = rf"very thick, {dash}, {color}"
        lines.append(rf"  \addplot[{style}] coordinates {{{coords}}};")
    lines += [r"  \end{groupplot}"]
    lines += end_tikz()

    # Patch the quality-panel ymax now that series peaks are known.
    ymax_line = rf"    ymax={ymax:.4f},"
    patched = []
    inserted = False
    for line in lines:
        patched.append(line)
        if (not inserted) and "Relative edge count" in line:
            patched.append(ymax_line)
            inserted = True
    lines = patched

    n_graphs = len(graphs)
    control = "The $\\Delta\\tau=0$ control is not in this results tree."
    if any(label in ("dt0", "delta0", "no-pheromone") for label, _s in series):
        control = (
            r"$\Delta\tau=0$ keeps every trail at $1$, so $T$ epochs are "
            r"$T$ independent colonies with the same number of ants."
        )
    lines += [
        (
            rf"  \caption{{Pheromone control at the same ant count "
            rf"({n_graphs} graphs in the pheromone-on run). "
            rf"Curves are medians across graphs of $|E(B_t)|/|E(D_\theta)|$. "
            rf"{control}}}"
        ),
        r"  \label{fig:pheromone-ablation}",
    ]
    lines += end_float(env)
    if drawn == 0:
        return _missing_figure(
            "fig:pheromone-ablation",
            "Epoch traces had no positive $\\theta$-heuristic edge counts to ratio against.",
        )
    return lines


def iteration_budget_figure(graphs):
    return anytime_figure(graphs)


def replicate_budget_figure(graphs):
    return replicate_figure(graphs)


BUILDERS = {
    "iteration-budget": lambda graphs, _ablation: anytime_figure(graphs),
    "replicate-budget": lambda graphs, _ablation: replicate_figure(graphs),
    "budget-allocation": lambda graphs, _ablation: allocation_figure(graphs),
    "pheromone-ablation": lambda graphs, ablation: pheromone_figure(graphs, ablation),
}


def summarize_for_log(graphs):
    if not graphs:
        return "0 graphs"
    n_rep = sum(len(g["reps"]) for g in graphs)
    t_max = max(_epochs(graphs) or [0])
    return f"{len(graphs)} graph(s), {n_rep} counted replicate(s), T<={t_max}"
