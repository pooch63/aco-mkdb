"""
metrics mode — per-instance solver records → aggregate comparison.

Reads a compare-methods JSON whose methods include the θ-heuristic,
double-greedy, clique-order-edge, and the ACO variants. Each method run
already stores the headline solution (best θ-feasible replicate, else the
largest edge count) and, when ``replicates`` is set, one record per seed
with ``final_edges``, ``nU``, ``nV``, ``theta_feasible``, ``wall_time_s``,
and ``seed``. This module only reads those fields.

Aggregates, on the instances present for every listed method:

  Win rate
      A shared win is a θ-feasible, k-valid solution whose edge count
      equals the best such count on that instance. Algorithms with fewer
      edges do not win. An instance on which every algorithm is
      θ-infeasible is excluded.

  Ratio to the best known
      r_A(I) = |E_A(I)| / max_B |E_B(I)| when A is θ-feasible, and 0
      otherwise. The reported value is the geometric mean. A zero ratio
      makes the mean zero.

  Feasibility rate
      Fraction of instances with a θ-feasible solution.

  Optimality gap
      Mean (E* - |E_A|) / E* on instances where branch-and-pivot finished
      in ``--compare-dir``. An infeasible solution scores a gap of 1.
      Instances whose recorded optimum is missing, or exceeded by any
      algorithm, are skipped.

  Performance profile
      Dolan–Moré profile of solution quality. The ratio is E* / |E_A|
      (1 is best). A θ-infeasible solution never enters a finite factor.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

from .common import (
    COLUMN_HEIGHT,
    COLUMN_WIDTH,
    begin_float,
    begin_tikz,
    end_float,
    end_tikz,
    list_json_paths,
    load_json,
    write_tex,
)

# Stable paper order. Methods absent from the JSON are omitted.
_ORDER = (
    ("heuristic", r"$\theta$-heuristic"),
    ("double-greedy", "Double-greedy"),
    ("clique-order-edge", "Clique-edge"),
    ("aco-default", "ACO-default"),
    ("aco-1thread", "ACO-1thread"),
    ("aco-matched", "ACO-matched"),
)

_COLORS = (
    "blue",
    "red",
    "green!60!black",
    "orange",
    "purple",
    "brown",
)

_TAUS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 4.0, 8.0)


def _label(method_id):
    for mid, label in _ORDER:
        if mid == method_id:
            return label
    return str(method_id).replace("_", r"\_")


def _ordered_ids(present):
    known = [mid for mid, _label in _ORDER if mid in present]
    extra = sorted(mid for mid in present if mid not in dict(_ORDER))
    return known + extra


def _scored_edges(run):
    """θ-feasible, k-valid edge count, or None when the run does not count."""
    if not isinstance(run, dict) or run.get("status") == "error":
        return None
    if run.get("k_valid", True) is False:
        return None
    if run.get("theta_feasible") is not True:
        return None
    edges = run.get("final_edges")
    if edges is None:
        return None
    return int(edges)


def _edges_or_zero(run):
    scored = _scored_edges(run)
    return 0 if scored is None else scored


def _time_s(run):
    if not isinstance(run, dict):
        return None
    seconds = run.get("wall_time_s")
    if seconds is None:
        return None
    return float(seconds)


def _dataset_key(row):
    return str(row.get("dataset") or "")


def _load_suite(path):
    """Return (rows, k, theta) from a compare-methods JSON file."""
    data = load_json(path)
    if not isinstance(data, dict):
        print(f"# metrics: could not read {path}", file=sys.stderr)
        return [], None, None
    rows = []
    for row in data.get("results") or []:
        if not isinstance(row, dict) or row.get("status") == "error":
            continue
        runs = row.get("runs")
        if not isinstance(runs, dict) or not runs:
            continue
        rows.append(row)
    return rows, data.get("k"), data.get("theta")


def _instance_runs(rows):
    """
    Graphs that have every method present on every other graph.

    Returns (datasets, method_ids, by_dataset) where by_dataset[dataset][id]
    is the headline run dict.
    """
    by_dataset = {}
    for row in rows:
        key = _dataset_key(row)
        if not key:
            continue
        runs = {}
        for mid, run in (row.get("runs") or {}).items():
            if isinstance(run, dict) and run.get("status") != "error":
                runs[str(mid)] = run
        if runs:
            by_dataset[key] = runs
    if not by_dataset:
        return [], [], {}
    common = None
    for runs in by_dataset.values():
        ids = set(runs)
        common = ids if common is None else common & ids
    if not common:
        print("# metrics: no method is present on every instance", file=sys.stderr)
        return [], [], {}
    method_ids = _ordered_ids(common)
    datasets = sorted(by_dataset)
    return datasets, method_ids, by_dataset


def _best_edges(runs, method_ids):
    scored = [_scored_edges(runs.get(mid)) for mid in method_ids]
    scored = [edges for edges in scored if edges is not None]
    if not scored:
        return None
    return max(scored)


def _win_rate(datasets, method_ids, by_dataset, method_id):
    counted = 0
    wins = 0
    for key in datasets:
        runs = by_dataset[key]
        best = _best_edges(runs, method_ids)
        if best is None:
            continue
        counted += 1
        edges = _scored_edges(runs.get(method_id))
        if edges is not None and edges == best:
            wins += 1
    if counted == 0:
        return None, 0
    return 100.0 * wins / counted, counted


def _ratio(runs, method_ids, method_id):
    best = _best_edges(runs, method_ids)
    if best is None or best <= 0:
        return None
    return _edges_or_zero(runs.get(method_id)) / float(best)


def _geometric_mean(values):
    if not values:
        return None
    if any(value <= 0.0 for value in values):
        return 0.0
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _feasibility(datasets, by_dataset, method_id):
    if not datasets:
        return None
    ok = sum(
        1
        for key in datasets
        if _scored_edges(by_dataset[key].get(method_id)) is not None
    )
    return 100.0 * ok / len(datasets)


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def load_optima(compare_dir):
    """
    Map dataset → optimal edge count from finished branch-and-pivot runs.

    A file counts only when ``pivot_theta`` finished and recorded
    ``final_edges``. Timeouts are skipped.
    """
    optima = {}
    if not compare_dir:
        return optima
    directory = Path(compare_dir)
    if not directory.is_dir():
        print(f"# metrics: pivot directory not found: {directory}", file=sys.stderr)
        return optima
    for path in list_json_paths(directory):
        data = load_json(path)
        if not isinstance(data, dict):
            continue
        block = data.get("pivot_theta")
        if not isinstance(block, dict):
            continue
        if block.get("timed_out") is True or block.get("status") == "timeout":
            continue
        edges = block.get("final_edges")
        dataset = data.get("dataset")
        if edges is None or not dataset:
            continue
        try:
            optima[str(dataset)] = int(edges)
        except (TypeError, ValueError):
            print(f"# metrics: bad pivot edge count in {path}", file=sys.stderr)
    return optima


def _optimum_for(key, optima):
    if key in optima:
        return optima[key]
    leaf = key.split("/")[-1]
    return optima.get(leaf)


def _gap_instances(datasets, by_dataset, method_ids, optima):
    """
    Instances with a finished pivot whose optimum no algorithm exceeds.

    An exceeded optimum is a different instance than this suite, so the
    gap is not computed there.
    """
    kept = []
    for key in datasets:
        optimum = _optimum_for(key, optima)
        if optimum is None or optimum <= 0:
            continue
        runs = by_dataset[key]
        if any((_scored_edges(runs.get(mid)) or 0) > optimum for mid in method_ids):
            print(
                f"# metrics: {key} exceeds recorded optimum {optimum}; "
                "excluded from the gap",
                file=sys.stderr,
            )
            continue
        kept.append((key, optimum))
    return kept


def _gap_for(run, optimum):
    edges = _scored_edges(run)
    if edges is None:
        return 1.0
    return (float(optimum) - float(edges)) / float(optimum)


def _quality_ratio(runs, method_ids, method_id):
    """Dolan–Moré ratio E*/|E_A|. Infeasible solutions stay at infinity."""
    best = _best_edges(runs, method_ids)
    edges = _scored_edges(runs.get(method_id))
    if best is None or best <= 0 or edges is None or edges <= 0:
        return math.inf
    return float(best) / float(edges)


def _profile(ratios, taus):
    if not ratios:
        return []
    n = len(ratios)
    points = []
    for tau in taus:
        count = sum(1 for ratio in ratios if ratio <= tau + 1e-9)
        points.append((tau, count / n))
    return points


def _fmt_pct(value):
    if value is None:
        return "--"
    return f"{value:.0f}"


def _fmt_ratio(value):
    if value is None:
        return "--"
    return f"{value:.2f}"


def _fmt_gap(value):
    if value is None:
        return "--"
    return f"{value:.0f}"


def _fmt_time(value):
    if value is None:
        return "--"
    if value < 0.01:
        return f"{value:.4f}"
    if value < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _table(method_ids, stats, *, k, theta, n_wins, n_gap):
    k_bit = r"$k$" if k is None else str(int(k))
    t_bit = r"$\theta$" if theta is None else str(int(theta))
    gap_n = f"{n_gap} instance" + ("" if n_gap == 1 else "s")
    lines = [
        r"\begin{table}[!t]",
        r"  \centering",
        r"  \caption{Solution quality and runtime of different algorithms at "
        + rf"$k = {k_bit}$, $\theta = {t_bit}$"
        r"  \label{tab:metrics}",
        r"  \footnotesize",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{@{}lrrrrr@{}}",
        r"    \toprule",
        r"    Algorithm & Win & Feas. & Gap & $t$ \\",
        r"    \midrule",
    ]
    for mid in method_ids:
        row = stats[mid]
        lines.append(
            "    "
            + " & ".join(
                [
                    _label(mid),
                    _fmt_pct(row["win"]),
                    _fmt_pct(row["feas"]),
                    _fmt_gap(row["gap"]),
                    _fmt_time(row["time"]),
                ]
            )
            + r" \\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    if n_wins == 0:
        lines.insert(0, r"% metrics: no instance had a $\theta$-feasible solution")
    return "\n".join(lines)


def _figure(method_ids, profiles):
    xmax = 8.0
    env, lines = begin_float("figure")
    lines += begin_tikz()
    lines += [
        r"  \begin{axis}[",
        r"    scale only axis,",
        rf"    width={COLUMN_WIDTH},",
        rf"    height={COLUMN_HEIGHT},",
        r"    xmode=log,",
        r"    log ticks with fixed point,",
        r"    xlabel={best $|E|$ / this $|E|$},",
        r"    ylabel={fraction of instances},",
        r"    xmin=1,",
        rf"    xmax={xmax:g},",
        r"    ymin=0,",
        r"    ymax=1,",
        r"    ytick={0,0.5,1},",
        r"    xtick={1,2,4,8},",
        r"    tick label style={font=\footnotesize},",
        r"    label style={font=\footnotesize},",
        r"    legend style={font=\scriptsize, at={(0.5,-0.42)}, anchor=north,",
        r"      legend columns=2, legend cell align=left,",
        r"      fill=white, draw=none, inner sep=1pt},",
        r"    clip=false,",
        r"    legend image post style={sharp plot},",
        r"    grid=major,",
        r"    grid style={gray!25},",
        r"    tick align=outside,",
        r"  ]",
    ]
    for index, mid in enumerate(method_ids):
        points = profiles.get(mid) or []
        if not points:
            continue
        color = _COLORS[index % len(_COLORS)]
        coords = " ".join(f"({tau:g},{frac:.4f})" for tau, frac in points)
        lines += [
            rf"  \addplot[{color}, thick] coordinates {{{coords}}};",
            rf"  \addlegendentry{{{_label(mid)}}}",
        ]
    lines += [r"  \end{axis}"]
    lines += end_tikz()
    lines += [
        r"  \caption{Performance profile of solution quality. The horizontal "
        r"axis is the best $\theta$-feasible edge count divided by the "
        r"algorithm's edge count, so $1$ matches the best and $2$ is half "
        r"as many edges. A $\theta$-infeasible solution is never within a "
        r"finite factor.}",
        r"  \label{fig:performance-profile}",
    ]
    lines += end_float(env)
    return "\n".join(lines)


def _unavailable(reason):
    print(f"# metrics: {reason}", file=sys.stderr)
    return (
        r"\begin{center}" "\n"
        r"\footnotesize Comparison metrics are not available." "\n"
        r"\end{center}"
    )


def run(path, output, *, compare_dir=None, plots="main"):
    source = Path(path)
    if source.is_dir():
        files = list(source.glob("k*.json"))
        source = files[0] if files else source / "k2t5.json"
    if not source.is_file():
        write_tex(_unavailable(f"missing {source}"), output)
        return

    rows, k, theta = _load_suite(source)
    datasets, method_ids, by_dataset = _instance_runs(rows)
    if not method_ids:
        write_tex(_unavailable(f"no completed instances in {source}"), output)
        return

    optima = load_optima(compare_dir)
    if not optima:
        print(
            "# metrics: no finished branch-and-pivot optimum; gap column is --",
            file=sys.stderr,
        )

    gap_keys = _gap_instances(datasets, by_dataset, method_ids, optima)
    stats = {}
    profiles = {}
    n_wins = None
    n_gap = len(gap_keys)
    for mid in method_ids:
        win, n_wins = _win_rate(datasets, method_ids, by_dataset, mid)
        ratios = []
        for key in datasets:
            ratio = _ratio(by_dataset[key], method_ids, mid)
            if ratio is not None:
                ratios.append(ratio)
        if gap_keys:
            gap = 100.0 * sum(
                _gap_for(by_dataset[key].get(mid), optimum)
                for key, optimum in gap_keys
            ) / len(gap_keys)
        else:
            gap = None
        times = [
            seconds
            for key in datasets
            if (seconds := _time_s(by_dataset[key].get(mid))) is not None
        ]
        stats[mid] = {
            "win": win,
            "ratio": _geometric_mean(ratios),
            "feas": _feasibility(datasets, by_dataset, mid),
            "gap": gap,
            "time": _median(times),
        }
        qratios = [
            _quality_ratio(by_dataset[key], method_ids, mid) for key in datasets
        ]
        profiles[mid] = _profile(qratios, _TAUS)
        print(
            f"# metrics {mid}: win={stats[mid]['win']} "
            f"r={stats[mid]['ratio']} feas={stats[mid]['feas']} "
            f"gap={stats[mid]['gap']} t={stats[mid]['time']}",
            file=sys.stderr,
        )

    parts = []
    if plots in ("main", "table"):
        parts.append(
            _table(
                method_ids,
                stats,
                k=k,
                theta=theta,
                n_wins=n_wins or 0,
                n_gap=n_gap or 0,
            )
        )
    if plots in ("main", "profile"):
        if any(profiles.values()):
            parts.append(_figure(method_ids, profiles))
        else:
            print("# metrics: performance profile has no ratios", file=sys.stderr)
    if not parts:
        parts.append(_unavailable(f"nothing to emit for --plots={plots}"))
    write_tex("\n\n".join(parts), output)
