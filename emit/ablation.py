"""
ablation mode — multi-method compare JSON → table rows and a summary sentence.

Reads the checkpoint written by ``bin/compare-methods.jl`` when the config
has a ``methods`` array (one injected graph, several solvers). θ-infeasible
or k-invalid outcomes score as 0 edges. Graphs missing any listed method are
omitted from every row so the medians stay paired.

``--field=rows`` (default) emits tabular rows.
``--field=summary`` emits one paragraph for the ablation section.
"""

from __future__ import annotations

import statistics
import sys

from .common import load_json, write_tex

# id, table variant, "what changes" cell, prose name
ROWS = (
    ("aco", "Every ant", r"deposit while walking; no end fill", "every-ant deposit"),
    (
        "no_pheromone",
        "No pheromone",
        r"$\Delta\tau=0$ ($\tau\equiv 1$)",
        "no pheromone",
    ),
    (
        "best_only",
        "Best-only",
        r"only the epoch-best deposits",
        "best-only deposit",
    ),
    (
        "greedy_fill_end",
        "Greedy fill",
        r"pack each side when an ant stops",
        "greedy fill on the best-only deposit",
    ),
    (
        "ga",
        r"GA [3]",
        r"crossover (GCLIQUE), 500 generations",
        "the genetic algorithm",
    ),
)


def _fmt_edges(value):
    if value is None:
        return "--"
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-9:
        return str(int(rounded))
    return f"{float(value):.1f}"


def _fmt_time(value):
    if value is None:
        return "--"
    value = float(value)
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _fmt_feas(value):
    if value is None:
        return "--"
    return str(int(round(float(value))))


def _scored_edges(run):
    """Edge count for the median. Infeasible and k-invalid outcomes score 0."""
    if not isinstance(run, dict):
        return None
    if run.get("final_edges") is None:
        return None
    valid = run.get("beats_valid")
    if valid is None:
        valid = bool(run.get("theta_feasible")) and run.get("k_valid") is not False
    if not valid:
        return 0
    return int(run["final_edges"])


def _feasible(run):
    if not isinstance(run, dict):
        return False
    if run.get("beats_valid") is not None:
        return bool(run.get("beats_valid"))
    return bool(run.get("theta_feasible")) and run.get("k_valid") is not False


def _paired_rows(data, ids):
    rows = []
    skipped = 0
    for row in data.get("results") or []:
        if row.get("status") != "ok":
            skipped += 1
            continue
        runs = row.get("runs") or {}
        if not all(_scored_edges(runs.get(i)) is not None for i in ids):
            skipped += 1
            continue
        rows.append(row)
    return rows, skipped


def _variant_stats(rows, method_id):
    edges = []
    times = []
    feas = 0
    for row in rows:
        run = row["runs"][method_id]
        edges.append(_scored_edges(run))
        if _feasible(run):
            feas += 1
        if run.get("wall_time_s") is not None:
            times.append(float(run["wall_time_s"]))
    n = len(rows)
    return {
        "median": statistics.median(edges) if edges else None,
        "feas": (100.0 * feas / n) if n else None,
        "time": statistics.median(times) if times else None,
    }


def _versus_counts(rows, method_id):
    wins = losses = ties = 0
    for row in rows:
        verdict = str((row.get("versus") or {}).get(method_id, ""))
        if verdict == "challenger":
            wins += 1
        elif verdict == "baseline":
            losses += 1
        elif verdict == "tie":
            ties += 1
    return wins, losses, ties


def _valid(run):
    if not isinstance(run, dict):
        return False
    if run.get("beats_valid") is not None:
        return bool(run.get("beats_valid"))
    return bool(run.get("theta_feasible")) and run.get("k_valid") is not False


def _pair_counts(rows, challenger_id, baseline_id):
    """Paper compare_results: a valid solution beats an invalid one, else more edges."""
    wins = losses = ties = 0
    for row in rows:
        runs = row["runs"]
        challenger = runs[challenger_id]
        baseline = runs[baseline_id]
        c_valid = _valid(challenger)
        b_valid = _valid(baseline)
        c_edges = int(challenger.get("final_edges") or 0)
        b_edges = int(baseline.get("final_edges") or 0)
        if c_valid and not b_valid:
            wins += 1
        elif b_valid and not c_valid:
            losses += 1
        elif c_edges > b_edges:
            wins += 1
        elif b_edges > c_edges:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties


def build_rows_tex(data):
    ids = [row[0] for row in ROWS]
    paired, skipped = _paired_rows(data, ids)
    if not paired:
        print(
            "Warning: ablation JSON has no complete graphs; emitting placeholders",
            file=sys.stderr,
        )
        lines = []
        for _method_id, variant, _change, _prose in ROWS:
            lines.append(f"{variant} & -- & -- & -- \\\\")
        return "\n".join(lines)
    if skipped:
        print(
            f"Warning: ablation omitted {skipped} incomplete graph(s); "
            f"using {len(paired)} paired graphs",
            file=sys.stderr,
        )
    lines = []
    for method_id, variant, _change, _prose in ROWS:
        stats = _variant_stats(paired, method_id)
        lines.append(
            f"{variant} & {_fmt_edges(stats['median'])} & "
            f"{_fmt_feas(stats['feas'])} & {_fmt_time(stats['time'])} \\\\"
        )
    return "\n".join(lines)


def build_summary_tex(data):
    ids = [row[0] for row in ROWS]
    paired, _skipped = _paired_rows(data, ids)
    if not paired:
        return "Ablation results are unavailable."
    by_id = {}
    for method_id, variant, _change, prose in ROWS:
        stats = _variant_stats(paired, method_id)
        if method_id == "greedy_fill_end":
            wins, losses, ties = _pair_counts(paired, method_id, "best_only")
            opponent = "best-only deposit"
        else:
            wins, losses, ties = _versus_counts(paired, method_id)
            opponent = "every-ant deposit"
        by_id[method_id] = {
            "variant": variant,
            "prose": prose,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "opponent": opponent,
            **stats,
        }
    ref = by_id["aco"]
    n = len(paired)
    sentences = [
        (
            f"On {n} graphs, with $F_N$, $A=100$, and $T=5$ held fixed, "
            f"every-ant deposit with no end fill has median "
            f"$|E|={_fmt_edges(ref['median'])}$, "
            f"$\\theta$-feasibility {_fmt_feas(ref['feas'])}\\%, "
            f"and median time {_fmt_time(ref['time'])}\\,s."
        )
    ]
    for method_id, _variant, _change, prose in ROWS:
        if method_id == "aco":
            continue
        item = by_id[method_id]
        record = (
            f"{prose[0].upper()}{prose[1:]} has median "
            f"$|E|={_fmt_edges(item['median'])}$ "
            f"({_fmt_feas(item['feas'])}\\% $\\theta$-feasible, "
            f"median {_fmt_time(item['time'])}\\,s)"
        )
        opponent = item["opponent"]
        if item["losses"] == 0 and item["wins"] == 0:
            record += f", tying {opponent} on all {item['ties']} graphs."
        elif item["losses"] == 0:
            record += (
                f", ahead of {opponent} on {item['wins']} graphs "
                f"and tied on {item['ties']}."
            )
        else:
            record += (
                f", ahead of {opponent} on {item['wins']} graphs, "
                f"behind on {item['losses']}, and tied on {item['ties']}."
            )
        sentences.append(record)
    ranked = sorted(
        by_id.values(),
        key=lambda item: (
            -(item["median"] if item["median"] is not None else -1),
            -(item["feas"] if item["feas"] is not None else -1),
        ),
    )
    leader = ranked[0]
    if leader["prose"] == "ACO":
        sentences.append("Every-ant deposit has the highest median edge count.")
    else:
        name = leader["prose"]
        sentences.append(
            f"{name[0].upper()}{name[1:]} has the highest median edge count."
        )
    return " ".join(sentences)


def run(path, output, field=None):
    field = field or "rows"
    if field not in ("rows", "summary"):
        raise SystemExit(f"ablation --field must be rows or summary, got {field!r}")
    data = load_json(path)
    if not isinstance(data, dict) or "results" not in data:
        print(f"Warning: could not read ablation results from {path}", file=sys.stderr)
        data = {"results": []}
    tex = build_rows_tex(data) if field == "rows" else build_summary_tex(data)
    write_tex(tex, output)
