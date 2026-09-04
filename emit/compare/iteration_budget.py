"""
Iteration-budget analysis from recorded iterations_to_best (ITB).

Plot group
----------
  iteration-budget
    2-panel groupplot at fixed ant count (default 100) for ACO-PN:
      Left: percentage of graphs where ACO beats the θ-heuristic, and
            percentage that admit a θ-feasible ACO solution, as a
            function of epoch budget I ∈ {1…n_E}.
      Right: cumulative distribution function (CDF) of iterations-to-best:
            for each I, the percentage of counted (non-JIT) trials whose
            eventual best was already found by epoch I.

Methodology
-----------
Each vary.jl trial runs a fixed iteration budget (typically n_E = 5) and
records iterations_to_best. For budget I we credit a trial's final_edges
only when ITB ≤ I — a conservative lower bound on quality under a shorter
budget, because intermediate edge counts before ITB are not stored.
JIT warmup replicates are omitted via counted_trials.
"""

from __future__ import annotations

import statistics
import sys
from collections import defaultdict

from ..common import counted_trials, list_json_paths, load_json
from ..quality import pct_deviation

PLOT_GROUPS = ("iteration-budget",)
DEFAULT_ANTS = 100
DEFAULT_MAX_BUDGET = 5


def _trials_at_ants(data, ants):
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


def summarize_iteration_budget(json_paths, *, ants=DEFAULT_ANTS):
    """
    Aggregate per-budget win / feasibility / ITB-CDF stats.

    Returns None when no usable files exist; otherwise a dict with:
      n_graphs, n_trials, budgets, wins, feasible_graphs,
      reach_trials, mean_pct, median_pct, full_wins
    """
    skipped = []
    loaded = []

    for path in json_paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
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
    reach_trials = defaultdict(int)
    n_trials = 0
    full_wins = 0

    for trials, heur in loaded:
        n_trials += len(trials)

        feas_full = [t for t in trials if t.get("theta_feasible")]
        if feas_full:
            best_full = max(int(t["final_edges"]) for t in feas_full)
            if best_full > heur:
                full_wins += 1

        for t in trials:
            itb = int(t["iterations_to_best"])
            for budget in range(1, max_budget + 1):
                if itb <= budget:
                    reach_trials[budget] += 1

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
        "reach_trials": [reach_trials[b] for b in budgets],
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


def _caption(summary):
    n = summary["n_graphs"]
    ants = summary["ants"]
    budgets = summary["budgets"]
    wins = summary["wins"]
    feas = summary["feasible_graphs"]
    reach = summary["reach_trials"]
    n_trials = summary["n_trials"]

    win_pct = [_as_pct(w, n) for w in wins]
    feas_pct = [_as_pct(f, n) for f in feas]

    w1, w_full = win_pct[0], win_pct[-1]
    plateau_i = budgets[-1]
    for b, w in zip(budgets, wins):
        if w == wins[-1]:
            plateau_i = b
            break
    # Midpoint for CDF callouts: third epoch when available.
    i_mid = budgets[min(2, len(budgets) - 1)]
    w_mid = win_pct[min(2, len(win_pct) - 1)]
    reach1 = _as_pct(reach[0], n_trials)
    reach_mid = _as_pct(reach[min(2, len(reach) - 1)], n_trials)

    med = summary["median_pct"]
    med_note = ""
    if med[0] is not None and med[-1] is not None:
        med_note = (
            f" Median percent edge deviation among $\\theta$-feasible "
            f"graphs stays near ${med[0]:.0f}\\%$ at $I{{=}}1$ and "
            f"${med[-1]:.0f}\\%$ at $I{{=}}{budgets[-1]}$."
        )

    return (
        f"Effect of ACO epoch budget $I$ at {ants} ants on {n} graphs "
        f"($k{{=}}2$, $\\theta{{=}}5$), using counted non-JIT replicates "
        f"per graph. Left: percentage of graphs where ACO-PN beats the "
        f"$\\theta$-heuristic (solid) and where a $\\theta$-feasible "
        f"solution exists (dashed), crediting a trial only when "
        f"iterations-to-best $\\leq I$ (conservative for $I < n_E$). "
        f"Win rate rises from {w1:.0f}\\% at $I{{=}}1$ to {w_mid:.0f}\\% "
        f"at $I{{=}}{i_mid}$ and reaches the full-budget rate of "
        f"{w_full:.0f}\\% by $I{{=}}{plateau_i}$. $\\theta$-feasibility "
        f"rises from {feas_pct[0]:.0f}\\% to {feas_pct[-1]:.0f}\\%. "
        f"Right: cumulative distribution function (CDF) of "
        f"iterations-to-best over {n_trials} trials---the percentage of "
        f"trials whose eventual best was already found by epoch $I$ "
        f"({reach1:.0f}\\% by $I{{=}}1$, {reach_mid:.0f}\\% by "
        f"$I{{=}}{i_mid}$).{med_note}"
    )


def iteration_budget_figure(summary):
    """2-panel groupplot: win/feasibility percentages | ITB CDF."""
    if not summary:
        return [
            r"% iteration-budget: no usable vary.jl trials "
            r"(need ants, iterations_to_best, heuristic.final_edges)",
        ]

    budgets = summary["budgets"]
    n_graphs = summary["n_graphs"]
    n_trials = summary["n_trials"]
    win_pct = [_as_pct(w, n_graphs) for w in summary["wins"]]
    feas_pct = [_as_pct(f, n_graphs) for f in summary["feasible_graphs"]]
    reach_pct = [
        _as_pct(r, n_trials) for r in summary["reach_trials"]
    ]

    win_coords = " ".join(
        f"({b},{p:.4f})" for b, p in zip(budgets, win_pct)
    )
    feas_coords = " ".join(
        f"({b},{p:.4f})" for b, p in zip(budgets, feas_pct)
    )
    cdf_coords = " ".join(
        f"({b},{p:.4f})" for b, p in zip(budgets, reach_pct)
    )
    xtick = "{" + ",".join(str(b) for b in budgets) + "}"

    lines = [
        r"\begin{figure}[htbp]",
        r"  \centering",
        r"  \begin{tikzpicture}",
        r"  \begin{groupplot}[",
        r"    group style={group size=2 by 1, horizontal sep=32pt},",
        r"    width=0.46\textwidth,",
        r"    height=0.44\textwidth,",
        r"    grid=major,",
        r"    xlabel={Epoch budget $I$},",
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
        r"      at={(0.03,0.97)}, anchor=north west,",
        r"      font=\scriptsize, draw=none, fill=none,",
        r"      row sep=1pt},",
        r"    legend cell align=left,",
        r"  ]",
        rf"  \addplot[thick, mark=*, blue!70!black] coordinates {{{win_coords}}};",
        r"  \addlegendentry{beats $\theta$-heuristic}",
        rf"  \addplot[thick, dashed, mark=square*, red!70!black] coordinates {{{feas_coords}}};",
        r"  \addlegendentry{$\theta$-feasible}",
        r"\nextgroupplot[",
        r"    ylabel={Trials at eventual best (\%)},",
        r"    title={CDF of iterations-to-best},",
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
        f"{summary['n_trials']} trial(s), ants={ants}; "
        f"wins@I={dict(zip(summary['budgets'], summary['wins']))}",
        file=sys.stderr,
    )
    return iteration_budget_figure(summary), skipped


BUILDERS = {
    "iteration-budget": iteration_budget_figure,
}
