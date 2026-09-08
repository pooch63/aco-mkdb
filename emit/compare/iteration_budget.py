"""
Epoch-budget analysis from recorded iterations_to_best (ETB).

Data model (read this before changing anything)
----------------------------------------------
Each vary.jl ``*_ants.json`` trial is one independent **replicate**
(``run`` / seed), always executed for the full colony budget
``iterations_budget`` (= ``T`` epochs, typically 5). The JSON does
**not** contain separate experiments that stopped early at T=1,2,….

Within each replicate, ``iterations_to_best`` (ETB) is the epoch index
in ``1…T`` at which that replicate first reached the ``final_edges``
it reports. So:

  - ``run``              → which replicate (stochastic restart)
  - ``iterations_budget`` → epochs that replicate was allowed to run
  - ``iterations_to_best`` → epoch *inside* that replicate when its
                             eventual best |E| first appeared

This figure is a **retrospective truncation** of those full-budget
replicates: for credited budget T we keep only replicates with
ETB ≤ T and use their recorded ``final_edges``. Intermediate edge
counts between epoch 1 and ETB are not stored, so short-budget quality
is a conservative lower bound.

Plot group
----------
  iteration-budget
    3-panel groupplot at fixed ant count (default 100) for $F_N$:
      Left: percentage of **graphs** where ACO beats the θ-heuristic,
            and percentage that admit a θ-feasible ACO solution, as a
            function of credited epoch budget T ∈ {1…T_max}.
            Per graph: best among counted replicates with ETB ≤ T.
      Middle: mean and median percent edge increase of ACO over the
            θ-heuristic among θ-feasible **graphs**, log-scaled
            y-axis, as T grows (same per-graph best).
      Right: CDF of epochs-to-best over **graphs**: for each graph,
            take the counted replicate(s) with that graph's best
            final_edges, then the earliest ETB among those ties; for
            each T, the percentage of graphs whose epochs-to-best is
            ≤ T (i.e. the eventual best-of-replicates edge count was
            already achieved by epoch T).

Methodology
-----------
Each vary.jl trial runs a fixed epoch budget (typically T = 5) and
records iterations_to_best (epochs-to-best). For budget T we credit a
trial's final_edges only when ETB ≤ T — a conservative lower bound on
quality under a shorter budget, because intermediate edge counts before
ETB are not stored. JIT warmup replicates are omitted via counted_trials.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict

from ..common import counted_trials, list_json_paths, load_json, aco_timed_out
from ..quality import pct_deviation

PLOT_GROUPS = ("iteration-budget",)
DEFAULT_ANTS = 100
DEFAULT_MAX_BUDGET = 5


def _trials_at_ants(data, ants):
    """Counted (non-JIT) replicates at this ant count with usable ETB."""
    return [
        t
        for t in counted_trials(data.get("trials") or [], data)
        if t.get("ants") == ants
        and t.get("final_edges") is not None
        and t.get("iterations_to_best") is not None
    ]


def _max_budget(trials, default=DEFAULT_MAX_BUDGET):
    budgets = [
        int(t["iterations_budget"])
        for t in trials
        if t.get("iterations_budget") is not None
    ]
    if budgets:
        return max(budgets)
    itbs = [int(t["iterations_to_best"]) for t in trials]
    return max(default, max(itbs) if itbs else default)


def _epochs_to_best(trials):
    """
    Per-graph epochs-to-best: earliest ETB among replicates that match
    the graph's best final_edges (conservative / fastest claim on ties).
    """
    if not trials:
        return None
    best_edges = max(int(t["final_edges"]) for t in trials)
    return min(
        int(t["iterations_to_best"])
        for t in trials
        if int(t["final_edges"]) == best_edges
    )


def summarize_iteration_budget(json_paths, *, ants=DEFAULT_ANTS):
    """
    Aggregate per-budget win / feasibility / ETB-CDF stats.

    All panels are per **graph**.

    Returns None when no usable files exist; otherwise a dict with:
      n_graphs, n_trials, budgets, wins, feasible_graphs,
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
        trials = _trials_at_ants(data, ants)
        if not trials:
            skipped.append((path, f"no counted trials at ants={ants}"))
            continue
        heur = (data.get("heuristic") or {}).get("final_edges")
        if heur is None:
            skipped.append((path, "missing heuristic.final_edges"))
            continue
        loaded.append((trials, int(heur)))

    if not loaded:
        return None, skipped

    max_budget = DEFAULT_MAX_BUDGET
    for trials, _heur in loaded:
        max_budget = max(max_budget, _max_budget(trials))

    by_budget_wins = defaultdict(int)
    by_budget_feas = defaultdict(int)
    by_budget_pct = defaultdict(list)
    reach_graphs = defaultdict(int)
    n_trials = 0
    full_wins = 0

    for trials, heur in loaded:
        # trials = counted replicates on this graph at fixed ants.
        n_trials += len(trials)

        feas_full = [t for t in trials if t.get("theta_feasible")]
        if feas_full:
            best_full = max(int(t["final_edges"]) for t in feas_full)
            if best_full > heur:
                full_wins += 1

        # CDF: one ETB per graph (earliest among best-edge replicates).
        etb = _epochs_to_best(trials)
        if etb is not None:
            for budget in range(etb, max_budget + 1):
                reach_graphs[budget] += 1

        # Win / feas / % increase: one best replicate per graph per T.
        for budget in range(1, max_budget + 1):
            eligible = [
                t
                for t in trials
                if int(t["iterations_to_best"]) <= budget
                and t.get("theta_feasible")
            ]
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
        "n_trials": n_trials,
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


def load_iteration_budget(directory, *, ants=DEFAULT_ANTS):
    """Load summary from a vary.jl directory."""
    return summarize_iteration_budget(list_json_paths(directory), ants=ants)


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
    n_graphs = summary["n_graphs"]
    reach_by_budget = dict(zip(budgets, summary["reach_graphs"]))
    # Right-panel CDF: share of graphs whose best-of-replicates ETB ≤ 3.
    epoch_ref = 3
    if epoch_ref in reach_by_budget and n_graphs:
        reach_pct = f"{_as_pct(reach_by_budget[epoch_ref], n_graphs):.1f}\\%"
    else:
        reach_pct = "--"

    return (
        f"Retrospective epoch-budget analysis at {ants} ants "
        f"($k{{=}}2, \\theta{{=}}5, T\\in\\{{1,\\ldots,5\\}}$). "
        f"Most of $F_N$'s advantage over the $\\theta$-heuristic requires "
        f"few epochs. Win rate and $\\theta$-feasibility largely plateau by "
        f"3 epochs. Median and mean edge increase slightly after 3 epochs, but "
        f"the majority of increase happens in the first 3 epochs. Indeed, by "
        f"epoch 3, {reach_pct} of graphs have already reached their best edge count "
        f"over all counted replicates."
    )


def iteration_budget_figure(summary):
    """3-panel groupplot: win/feas % | % edge increase (log) | ETB CDF."""
    if not summary:
        return [
            r"% iteration-budget: no usable vary.jl trials "
            r"(need ants, iterations_to_best, heuristic.final_edges)",
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

    # Log-safe floor slightly below the smallest plotted aggregate.
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
        # [H] keeps the figure in §4.6; htbp can defer it past §5.
        r"\begin{figure}[H]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={",
        r"      group size=3 by 1,",
        # Extra sep so long y-labels do not collide with the next panel.
        r"      horizontal sep=56pt},",
        r"    width=0.28\textwidth,",
        r"    height=0.40\textwidth,",
        r"    grid=major,",
        # T is a retrospective credit threshold on full-budget replicates,
        # not a separately measured shorter-budget experiment.
        r"    xlabel={Epoch budget $T$},",
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
        r"    title={CDF of epochs-to-best},",
        r"    title style={font=\small},",
        r"    ymin=0,",
        r"    ymax=105,",
        r"  ]",
        rf"  \addplot[thick, mark=*, green!50!black] coordinates {{{cdf_coords}}};",
        r"  \end{groupplot}",
        r"  \end{tikzpicture}",
        rf"  \caption{{{_caption(summary)}}}",
        r"  \label{fig:iteration-budget}",
        r"\end{figure}",
    ]
    return lines


def build_from_paths(json_paths, *, ants=DEFAULT_ANTS):
    """Return (latex_lines, skipped) for compare-mode wiring."""
    summary, skipped = summarize_iteration_budget(json_paths, ants=ants)
    if summary is None:
        print(
            "Warning: iteration-budget: no usable graphs "
            f"(ants={ants}).",
            file=sys.stderr,
        )
        return iteration_budget_figure(None), skipped
    print(
        f"# iteration-budget: {summary['n_graphs']} graph(s), "
        f"{summary['n_trials']} replicate(s), ants={ants}; "
        f"wins@T={dict(zip(summary['budgets'], summary['wins']))}",
        file=sys.stderr,
    )
    return iteration_budget_figure(summary), skipped


BUILDERS = {
    "iteration-budget": iteration_budget_figure,
}
