"""
Epoch/replicate budget figure and the pheromone-ablation figure.

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
  budget
      Same 2×2 as the quality figure. Top row: median percent deviation
      from the θ-heuristic (θ-infeasible outcomes score −100%). Bottom
      row: median θ-feasibility. Columns are epoch budget ``T`` and
      replicate count ``R``. For ``R``, deviation is the expected best
      of an ``R``-subset at the end of the trace, and feasibility is the
      probability that subset contains a θ-feasible replicate.
  pheromone-ablation
      The anytime median with pheromone on, against ``Δτ = 0`` (τ stays
      at 1) and any recorded ρ sweep, at the same ant count.
"""

from __future__ import annotations

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
from ..quality import (
    _axis_limits,
    _shared_ylim,
    _tick_values,
    _xscale_opts,
    pct_deviation,
)

PLOT_GROUPS = (
    "budget",
    "pheromone-ablation",
)
DEFAULT_ANTS = 100

# Knots drawn on the budget figure, in the same style as the quality
# panels (a handful of marked points, not every epoch).
_T_MARKS = (1, 2, 5, 10, 20, 50, 100)
_R_MARKS = (1, 2, 5, 10, 20, 30)
_PAPER_DEFAULT = 5.0


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


def _coords(xs, ys, *, digits=4):
    parts = []
    for x, y in zip(xs, ys):
        if y is None:
            continue
        parts.append(f"({x:g},{y:.{digits}f})")
    return " ".join(parts)


def _missing_figure(label, message):
    env, lines = begin_float("figure", star=False)
    lines += [
        rf"  \caption{{{message}}}",
        rf"  \label{{{label}}}",
    ]
    lines += end_float(env)
    return lines


def _sparse_log_ticks(xmax):
    """Three labels. A compact panel cannot hold the full 1,2,5,… set."""
    marks = [m for m in (1, 2, 5, 10, 20, 50, 100, 200, 500) if m <= xmax]
    if not marks or marks[-1] != xmax:
        marks.append(xmax)
    kept = _tick_values(marks, "log10")
    return "{" + ",".join(f"{m:g}" for m in kept) + "}"


def _ncr(n, k):
    if k < 0 or k > n:
        return 0.0
    c = 1.0
    for i in range(k):
        c *= (n - i) / (i + 1.0)
    return c


def _p_any_feasible(n, n_feas, r):
    """Probability an r-subset contains at least one θ-feasible replicate."""
    if r < 1 or r > n:
        return None
    if n - n_feas < r:
        return 1.0
    total = _ncr(n, r)
    if total == 0.0:
        return None
    return 1.0 - _ncr(n - n_feas, r) / total


def _kept_marks(available, preferred):
    """Preferred knots that the traces actually recorded."""
    have = set(available)
    marks = [m for m in preferred if m in have]
    if len(marks) >= 2:
        return marks
    vals = sorted(have)
    if len(vals) <= 7:
        return vals
    return [vals[0], vals[len(vals) // 2], vals[-1]]


def _epoch_graph_values(graph, t):
    """Mean % deviation (infeasible = −100) and % of replicates feasible."""
    devs = []
    flags = []
    heur = graph["heur"]
    for rep in graph["reps"]:
        pt = rep.get(t)
        if pt is None:
            continue
        feas = pt["theta_feasible"] is True
        flags.append(feas)
        if heur is None or heur <= 0:
            continue
        if not feas:
            devs.append(-100.0)
            continue
        dev = pct_deviation(pt["edges"], heur)
        if dev is not None:
            devs.append(dev)
    dev = statistics.mean(devs) if devs else None
    feas_pct = 100.0 * sum(flags) / len(flags) if flags else None
    return dev, feas_pct


def _final_outcomes(graph):
    """(deviation or −100, feasible) at the longest shared epoch."""
    reps = [rep for rep in graph["reps"] if rep]
    if not reps:
        return []
    cap = min(max(rep) for rep in reps)
    heur = graph["heur"]
    out = []
    for rep in reps:
        pt = rep.get(cap)
        if pt is None:
            continue
        feas = pt["theta_feasible"] is True
        if heur is None or heur <= 0:
            dev = None
        elif not feas:
            dev = -100.0
        else:
            dev = pct_deviation(pt["edges"], heur)
        out.append((dev, feas))
    return out


def _median_at(graphs, marks, value_at):
    series = {}
    for mark in marks:
        vals = []
        for graph in graphs:
            val = value_at(graph, mark)
            if val is not None:
                vals.append(val)
        if vals:
            series[mark] = statistics.median(vals)
    return series


def budget_series(graphs):
    """
    Four median series: deviation and feasibility against T and against R.

    T uses the mean over counted replicates at that epoch, then the median
    across graphs (same reduction as the quality figure). R is evaluated
    at the end of each graph's trace: deviation is E[max score], and
    feasibility is P(the subset contains a θ-feasible replicate).
    """
    epochs = _epochs(graphs)
    t_marks = _kept_marks(epochs, _T_MARKS)
    r_cap = max((len(g["reps"]) for g in graphs), default=0)
    r_marks = _kept_marks(range(1, r_cap + 1), _R_MARKS)

    def dev_at_t(graph, t):
        return _epoch_graph_values(graph, t)[0]

    def feas_at_t(graph, t):
        return _epoch_graph_values(graph, t)[1]

    def dev_at_r(graph, r):
        scores = [dev for dev, _feas in _final_outcomes(graph) if dev is not None]
        if len(scores) < r:
            return None
        return expected_max(scores, r)

    def feas_at_r(graph, r):
        flags = [feas for _dev, feas in _final_outcomes(graph)]
        if len(flags) < r:
            return None
        prob = _p_any_feasible(len(flags), sum(flags), r)
        if prob is None:
            return None
        return 100.0 * prob

    return {
        "dev_t": _median_at(graphs, t_marks, dev_at_t),
        "dev_r": _median_at(graphs, r_marks, dev_at_r),
        "feas_t": _median_at(graphs, t_marks, feas_at_t),
        "feas_r": _median_at(graphs, r_marks, feas_at_r),
    }


def _panel_body(xy, xmin, xmax, ymin, ymax, default, *, zero_line):
    lines = []
    if xy:
        coords = " ".join(
            f"({x:g},{y:.4f})"
            for x, y in sorted(xy.items(), key=lambda item: float(item[0]))
        )
        lines.append(
            r"\addplot[solid, thick, mark=*, mark size=1.1pt, blue!80!black] "
            rf"coordinates {{{coords}}};"
        )
    else:
        lines.append(
            r"\node[font=\scriptsize, align=center, text=black!55] "
            r"at (rel axis cs:0.5,0.55) {not recorded};"
        )
    if zero_line:
        lines.append(
            rf"\draw[dashed, gray] (axis cs:{xmin:g},0) -- (axis cs:{xmax:g},0);"
        )
    if default is not None and xmin < float(default) < xmax:
        lines.append(
            rf"\draw[dotted, black!55] (axis cs:{default:g},{ymin:.4f}) -- "
            rf"(axis cs:{default:g},{ymax:.4f});"
        )
    return lines


def _budget_panel(xlabel, xy, scale, ymin, ymax, *, hide_y, zero_line, ytick):
    if xy:
        xs = sorted(float(x) for x in xy)
        xmin, xmax, scale = _axis_limits(xs, scale)
        xtick = "{" + ",".join(f"{x:g}" for x in _tick_values(xs, scale)) + "}"
    else:
        xmin, xmax, scale, xtick = 0.5, 8.0, "linear", None
    opts = [
        r"\nextgroupplot[",
        rf"  xlabel={{{xlabel}}},",
    ]
    if hide_y:
        opts.append(r"  yticklabels={},")
    opts += [
        rf"  ymin={ymin:.4f},",
        rf"  ymax={ymax:.4f},",
    ]
    opts += _xscale_opts(scale)
    opts += [
        rf"  xmin={xmin:g},",
        rf"  xmax={xmax:g},",
    ]
    if xtick is not None:
        opts.append(rf"  xtick={xtick},")
    if ytick is not None:
        opts.append(rf"  ytick={{{ytick}}},")
    opts.append(r"]")
    opts += _panel_body(
        xy, xmin, xmax, ymin, ymax, _PAPER_DEFAULT, zero_line=zero_line,
    )
    return opts


def budget_figure(graphs):
    """Quality-style 2×2: deviation and θ-feasibility versus T and R."""
    if not graphs:
        return _missing_figure(
            "fig:budget",
            "Epoch traces are not recorded yet. "
            r"Run \texttt{scripts/budget-trace.bash} to produce this figure.",
        )
    series = budget_series(graphs)
    if not any(series.values()):
        return _missing_figure(
            "fig:budget",
            "Epoch traces had no deviation or $\\theta$-feasibility to plot.",
        )
    dev_lo, dev_hi = _shared_ylim([series["dev_t"], series["dev_r"]])
    feas_lo, feas_hi = -8.0, 108.0
    panels = (
        (r"Epochs $T$", series["dev_t"], "log2", dev_lo, dev_hi, False, True, None),
        (r"Replicates $R$", series["dev_r"], "log2", dev_lo, dev_hi, True, True, None),
        (r"Epochs $T$", series["feas_t"], "log2", feas_lo, feas_hi, False, False, "0,50,100"),
        (r"Replicates $R$", series["feas_r"], "log2", feas_lo, feas_hi, True, False, "0,50,100"),
    )
    body = [
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 2,",
        r"    horizontal sep=0.48cm, vertical sep=0.82cm},",
        r"  scale only axis,",
        r"  width=0.34\columnwidth,",
        r"  height=0.26\columnwidth,",
        r"  tick label style={font=\scriptsize, inner sep=1pt},",
        r"  label style={font=\scriptsize, inner sep=1pt},",
        r"  xlabel style={align=center},",
        r"  grid=major,",
        r"  grid style={gray!25},",
        r"  tick align=outside,",
        r"]",
    ]
    for xlabel, xy, scale, ymin, ymax, hide_y, zero_line, ytick in panels:
        body += _budget_panel(
            xlabel, xy, scale, ymin, ymax,
            hide_y=hide_y, zero_line=zero_line, ytick=ytick,
        )
    body += [
        r"\end{groupplot}",
        r"\path (group c1r1.west)",
        r"  node[xshift=-32pt, rotate=90, anchor=center,",
        r"       font=\scriptsize, inner sep=0pt]",
        r"  {Dev.\ from $\theta$-heuristic (\%)};",
        r"\path (group c1r2.west)",
        r"  node[xshift=-32pt, rotate=90, anchor=center,",
        r"       font=\scriptsize, inner sep=0pt]",
        r"  {$\theta$-feasible (\%)};",
    ]
    env, lines = begin_float("figure", star=False)
    lines += begin_tikz(resize=False)
    lines += body
    lines += end_tikz(resize=False)
    lines += [
        r"  \caption{Median deviation from the $\theta$-heuristic, "
        r"and $\theta$-feasibility, versus epoch budget and replicate count.}",
        r"  \label{fig:budget}",
    ]
    lines += end_float(env)
    print(
        "# budget: "
        + ", ".join(f"{key}={len(val)}" for key, val in series.items()),
        file=sys.stderr,
    )
    return lines


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
    lines += begin_tikz(resize=False)
    lines += [
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=1.15cm},",
        r"    scale only axis,",
        r"    width=0.30\columnwidth,",
        r"    height=0.26\columnwidth,",
        r"    tick label style={font=\scriptsize, inner sep=1pt},",
        r"    label style={font=\scriptsize, inner sep=1pt},",
        r"    title style={font=\scriptsize, inner sep=1pt},",
        r"    xlabel style={align=center},",
        r"    ylabel style={inner sep=1pt},",
        r"    every axis plot/.append style={line join=round},",
        r"    grid=major,",
        r"    grid style={gray!25},",
        r"    tick align=outside,",
        r"    scaled ticks=false,",
        r"    xmode=log,",
        r"    log ticks with fixed point,",
        r"    xlabel={Epoch $t$},",
        r"    xmin=1,",
        rf"    xmax={all_epochs[-1]},",
        rf"    xtick={_sparse_log_ticks(all_epochs[-1])},",
        r"  ]",
        r"  \nextgroupplot[",
        r"    ylabel={$|E(B_t)|/|E(D_\theta)|$},",
        r"    title={Relative edge count},",
        r"    ymin=0,",
        r"    legend to name=pheromonelegend,",
        r"    legend columns=-1,",
        r"    legend style={font=\scriptsize, draw=none, fill=none,",
        r"      inner sep=0pt, legend cell align=left},",
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
        r"    ylabel={$P(\theta$-feasible$)$},",
        r"    title={$\theta$-feasibility},",
        r"    ymin=0,",
        r"    ymax=1.05,",
        r"    ytick={0,0.5,1},",
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
    lines += [
        r"  \end{groupplot}",
        r"  \path (group c1r1.outer south) -- (group c2r1.outer south)",
        r"    node[midway, below=2pt, anchor=north, inner sep=0pt]",
        r"    {\pgfplotslegendfromname{pheromonelegend}};",
    ]
    lines += end_tikz(resize=False)

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

    lines += [
        r"  \caption{Anytime quality with and without pheromone.}",
        r"  \label{fig:pheromone-ablation}",
    ]
    lines += end_float(env)
    if drawn == 0:
        return _missing_figure(
            "fig:pheromone-ablation",
            "Epoch traces had no positive $\\theta$-heuristic edge counts to ratio against.",
        )
    return lines


BUILDERS = {
    "budget": lambda graphs, _ablation: budget_figure(graphs),
    "pheromone-ablation": lambda graphs, ablation: pheromone_figure(graphs, ablation),
}


def summarize_for_log(graphs):
    if not graphs:
        return "0 graphs"
    n_rep = sum(len(g["reps"]) for g in graphs)
    t_max = max(_epochs(graphs) or [0])
    return f"{len(graphs)} graph(s), {n_rep} counted replicate(s), T<={t_max}"
