"""
Replicate-budget analysis from ordered counted (non-JIT) trials.

Data model (read this before changing anything)
----------------------------------------------
Each vary.jl ``*_ants.json`` trial is one independent **replicate**
(``run`` / seed). When ``aco_runs > 1``, run 1 (or ``jit_warmup=true``)
is the real-graph Julia JIT warmup and is **never** used here — only
``counted_trials`` remain. Process-wide ``max_replicates`` is ignored
here (``max_replicates=None``) so the figure can always show
``R = 1…R_max`` over the full recorded counted set.

Counted replicates on a graph are ordered by ascending ``run``. Credited
replicate budget ``R`` means: keep only the first ``R`` counted
replicates (the earliest seeds after discarding JIT), take the best
among them, and score that graph. This is a retrospective prefix of the
recorded replicate sequence — we do not re-run ACO with fewer seeds.

``replicates-to-best`` (RTB) for a graph is the 1-based index in that
ordered counted list of the first replicate whose ``final_edges`` equals
the best ``final_edges`` among all counted replicates on that graph.

Plot group
----------
  replicate-budget
    3-panel groupplot at fixed ant count (default 100) for $F_N$:
      Left: percentage of **graphs** where ACO beats the θ-heuristic,
            and percentage that admit a θ-feasible ACO solution, as a
            function of replicate budget R ∈ {1…R_max}.
            Per graph: best among the first R counted replicates.
      Middle: mean and median percent edge increase of ACO over the
            θ-heuristic among θ-feasible **graphs**, log-scaled
            y-axis, as R grows (same per-graph best).
      Right: CDF of replicates-to-best over **graphs**: for each R,
            the percentage of graphs whose eventual best among all
            counted replicates was already reached by counted
            replicate R (RTB ≤ R).
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict

from ..common import counted_trials, list_json_paths, load_json, aco_timed_out
from ..quality import pct_deviation

PLOT_GROUPS = ("replicate-budget",)
DEFAULT_ANTS = 100


def _ordered_counted_at_ants(data, ants):
    """
    Counted (non-JIT) replicates at this ant count, ordered by run.

    JIT warmup is excluded via counted_trials. Ordering by ``run`` makes
    R=1 the first *counted* seed (typically raw run 2), never the warmup.

    Always uses the full counted set (``max_replicates=None``): this plot's
    job is to show R = 1…R_max, so it ignores the process-wide emit cap.
    """
    trials = [
        t
        for t in counted_trials(
            data.get("trials") or [], data, max_replicates=None
        )
        if t.get("ants") == ants and t.get("final_edges") is not None
    ]
    return sorted(trials, key=lambda t: int(t.get("run", 10**9)))


def _replicates_to_best(ordered):
    """1-based index of first counted replicate matching the eventual best."""
    if not ordered:
        return None
    best_edges = max(int(t["final_edges"]) for t in ordered)
    for i, t in enumerate(ordered, start=1):
        if int(t["final_edges"]) == best_edges:
            return i
    return None


def summarize_replicate_budget(json_paths, *, ants=DEFAULT_ANTS):
    """
    Aggregate per-replicate-budget win / feasibility / RTB-CDF stats.

    All panels are per **graph**. JIT warmup replicates are never included.

    Returns None when no usable files exist; otherwise a dict with:
      n_graphs, n_replicates, budgets, wins, feasible_graphs,
      reach_graphs, mean_pct, median_pct, full_wins
    """
    skipped = []
    loaded = []

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
        ordered = _ordered_counted_at_ants(data, ants)
        if not ordered:
            skipped.append((path, f"no counted trials at ants={ants}"))
            continue
        heur = (data.get("heuristic") or {}).get("final_edges")
        if heur is None:
            skipped.append((path, "missing heuristic.final_edges"))
            continue
        loaded.append((ordered, int(heur)))

    if not loaded:
        return None, skipped

    max_budget = max(len(ordered) for ordered, _ in loaded)
    if max_budget < 1:
        return None, skipped

    by_budget_wins = defaultdict(int)
    by_budget_feas = defaultdict(int)
    by_budget_pct = defaultdict(list)
    reach_graphs = defaultdict(int)
    n_replicates = 0
    full_wins = 0

    for ordered, heur in loaded:
        n_replicates += len(ordered)

        feas_full = [t for t in ordered if t.get("theta_feasible")]
        if feas_full:
            best_full = max(int(t["final_edges"]) for t in feas_full)
            if best_full > heur:
                full_wins += 1

        rtb = _replicates_to_best(ordered)
        if rtb is not None:
            for budget in range(rtb, max_budget + 1):
                reach_graphs[budget] += 1

        for budget in range(1, max_budget + 1):
            prefix = ordered[:budget]
            if not prefix:
                continue
            eligible = [t for t in prefix if t.get("theta_feasible")]
            if not eligible:
                continue
            by_budget_feas[budget] += 1
            best = max(eligible, key=lambda t: int(t["final_edges"]))
            edges = int(best["final_edges"])
            if edges > heur:
                by_budget_wins[budget] += 1
            pct = pct_deviation(edges, heur)
            if pct is not None:
                by_budget_pct[budget].append(pct)

    budgets = list(range(1, max_budget + 1))
    summary = {
        "n_graphs": len(loaded),
        "n_replicates": n_replicates,
        "ants": ants,
        "budgets": budgets,
        "wins": [by_budget_wins[b] for b in budgets],
        "feasible_graphs": [by_budget_feas[b] for b in budgets],
        "reach_graphs": [reach_graphs[b] for b in budgets],
        "mean_pct": [
            statistics.mean(by_budget_pct[b]) if by_budget_pct[b] else None
            for b in budgets
        ],
        "median_pct": [
            statistics.median(by_budget_pct[b]) if by_budget_pct[b] else None
            for b in budgets
        ],
        "full_wins": full_wins,
    }
    return summary, skipped


def load_replicate_budget(directory, *, ants=DEFAULT_ANTS):
    """Load summary from a vary.jl directory."""
    return summarize_replicate_budget(list_json_paths(directory), ants=ants)


def _as_pct(count, total):
    if not total:
        return 0.0
    return 100.0 * count / total


def _pct_coords(budgets, values):
    """Build pgfplots coordinates; skip None / non-positive (log-unsafe)."""
    parts = []
    for b, v in zip(budgets, values):
        if v is None or v <= 0:
            continue
        parts.append(f"({b},{v:.4f})")
    return " ".join(parts)


def _caption(summary):
    ants = summary["ants"]
    budgets = summary["budgets"]

    return (
        f"Replicate budget at {ants} ants "
        f"($k{{=}}2$, $\\theta{{=}}5$; best among the first $R$ counted "
        f"replicates, up to $R_{{\\max}}{{=}}{budgets[-1]}$). "
        f"A single counted replicate already recovers essentially all "
        f"head-to-head wins; additional seeds mainly improve late "
        f"replicates-to-best coverage and slight $\\theta$-feasibility "
        f"gains rather than new wins against the $\\theta$-heuristic."
    )


def replicate_budget_figure(summary):
    """3-panel groupplot: win/feas % | % edge increase (log) | RTB CDF."""
    if not summary:
        return [
            r"% replicate-budget: no usable vary.jl trials "
            r"(need counted ants trials with final_edges, "
            r"heuristic.final_edges)",
        ]

    budgets = summary["budgets"]
    n_graphs = summary["n_graphs"]
    win_pct = [_as_pct(w, n_graphs) for w in summary["wins"]]
    feas_pct = [_as_pct(f, n_graphs) for f in summary["feasible_graphs"]]
    reach_pct = [
        _as_pct(r, n_graphs) for r in summary["reach_graphs"]
    ]

    win_coords = " ".join(
        f"({b},{p:.4f})" for b, p in zip(budgets, win_pct)
    )
    feas_coords = " ".join(
        f"({b},{p:.4f})" for b, p in zip(budgets, feas_pct)
    )
    mean_coords = _pct_coords(budgets, summary["mean_pct"])
    median_coords = _pct_coords(budgets, summary["median_pct"])
    cdf_coords = " ".join(
        f"({b},{p:.4f})" for b, p in zip(budgets, reach_pct)
    )
    xtick = "{" + ",".join(str(b) for b in budgets) + "}"

    pct_vals = [
        v
        for v in (
            list(summary["mean_pct"])
            + list(summary["median_pct"])
        )
        if v is not None and v > 0
    ]
    ymin_log = max(1.0, min(pct_vals) / 2.0) if pct_vals else 1.0
    ymax_log = max(pct_vals) * 1.5 if pct_vals else 100.0

    lines = [
        # Stacked under fig:iteration-budget in §4.6; slight pull-up.
        r"\begin{figure}[H]",
        r"  \vspace{-1em}",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={",
        r"      group size=3 by 1,",
        r"      horizontal sep=56pt},",
        r"    width=0.28\textwidth,",
        r"    height=0.40\textwidth,",
        r"    grid=major,",
        r"    xlabel={Replicate budget $R$},",
        r"    ylabel style={font=\small},",
        rf"    xtick={xtick},",
        r"    xmin=0.5,",
        rf"    xmax={budgets[-1] + 0.5:.1f},",
        r"  ]",
        r"\nextgroupplot[",
        r"    ylabel={Percentage of graphs (\%)},",
        r"    title={Quality vs.\ $\theta$-heuristic},",
        r"    title style={font=\small},",
        r"    ymin=0,",
        r"    ymax=105,",
        r"    legend style={",
        r"      at={(0.5,-0.32)}, anchor=north,",
        r"      font=\scriptsize, draw=none, fill=none,",
        r"      row sep=1pt},",
        r"    legend columns=-1,",
        r"    legend cell align=left,",
        r"  ]",
        rf"  \addplot[thick, mark=*, blue!70!black] coordinates {{{win_coords}}};",
        r"  \addlegendentry{beats $\theta$-heuristic}",
        rf"  \addplot[thick, dashed, mark=square*, red!70!black] coordinates {{{feas_coords}}};",
        r"  \addlegendentry{$\theta$-feasible}",
        r"\nextgroupplot[",
        r"    ylabel={Edge increase (\%)},",
        r"    title={Percent increase vs.\ $\theta$},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        rf"    ymin={ymin_log:.4f},",
        rf"    ymax={ymax_log:.4f},",
        r"    legend style={",
        r"      at={(0.5,-0.32)}, anchor=north,",
        r"      font=\scriptsize, draw=none, fill=none,",
        r"      row sep=1pt},",
        r"    legend columns=-1,",
        r"    legend cell align=left,",
        r"  ]",
    ]
    if mean_coords:
        lines.append(
            rf"  \addplot[thick, mark=*, orange!85!black] "
            rf"coordinates {{{mean_coords}}};"
        )
        lines.append(r"  \addlegendentry{mean}")
    if median_coords:
        lines.append(
            rf"  \addplot[thick, dashed, mark=triangle*, "
            rf"teal!70!black] coordinates {{{median_coords}}};"
        )
        lines.append(r"  \addlegendentry{median}")
    if not mean_coords and not median_coords:
        lines.append(
            r"  % no positive percent-increase aggregates to plot"
        )
    lines += [
        r"\nextgroupplot[",
        r"    ylabel={Graphs at eventual best (\%)},",
        r"    title={CDF of replicates-to-best},",
        r"    title style={font=\small},",
        r"    ymin=0,",
        r"    ymax=105,",
        r"  ]",
        rf"  \addplot[thick, mark=*, green!50!black] coordinates {{{cdf_coords}}};",
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{_caption(summary)}}}",
        r"  \label{fig:replicate-budget}",
        r"\end{figure}",
    ]
    return lines


def build_from_paths(json_paths, *, ants=DEFAULT_ANTS):
    """Return (latex_lines, skipped) for compare-mode wiring."""
    summary, skipped = summarize_replicate_budget(json_paths, ants=ants)
    if summary is None:
        print(
            "Warning: replicate-budget: no usable graphs "
            f"(ants={ants}).",
            file=sys.stderr,
        )
        return replicate_budget_figure(None), skipped
    print(
        f"# replicate-budget: {summary['n_graphs']} graph(s), "
        f"{summary['n_replicates']} counted replicate(s), ants={ants}; "
        f"wins@R={dict(zip(summary['budgets'], summary['wins']))}",
        file=sys.stderr,
    )
    return replicate_budget_figure(summary), skipped


BUILDERS = {
    "replicate-budget": replicate_budget_figure,
}
