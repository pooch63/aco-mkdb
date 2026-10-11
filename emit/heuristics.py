"""
heuristics mode — multi-method compare JSON → one compact full-width table.

Reads the suite written by ``bin/compare-methods.jl`` when the config lists
several methods (``results[]`` with a ``runs`` object per method). Each row
is one graph. Each method contributes the solution's ``nU``, ``nV``, and
``final_edges``, plus ``wall_time_s`` converted to milliseconds. Bold marks
the largest θ-feasible, k-valid edge count on that graph; a tie bolds every
method that matches. Wall time is not bold. Missing fields become ``--``
and a stderr warning. This module only reads the JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import load_json, series_name, write_tex

_LABELS = {
    "heuristic": r"$\theta$-heuristic",
    "clique-order-edge": "Clique-order-edge",
    "double-greedy": "Double-greedy",
}


def _fmt_count(value):
    if value is None:
        return "--"
    return str(int(value))


def _fmt_ms(seconds):
    """Seconds from the JSON, displayed in milliseconds."""
    if seconds is None:
        return "--"
    ms = float(seconds) * 1000.0
    if ms < 10:
        return f"{ms:.1f}"
    return f"{ms:.0f}"


def _label(method_id):
    if method_id in _LABELS:
        return _LABELS[method_id]
    return str(method_id).replace("_", r"\_")


def _methods(data):
    raw = data.get("methods") if isinstance(data, dict) else None
    found = []
    if isinstance(raw, list):
        for spec in raw:
            if isinstance(spec, dict) and spec.get("id"):
                found.append(str(spec["id"]))
            elif isinstance(spec, str) and spec:
                found.append(spec)
    return found


def _run_cells(run, warned):
    if not isinstance(run, dict) or run.get("status") == "error":
        if not isinstance(run, dict):
            warned.add("runs")
        return ["--", "--", "--", "--"], None
    edges = run.get("final_edges")
    if edges is None:
        warned.add("final_edges")
    nU = run.get("nU")
    nV = run.get("nV")
    if nU is None:
        warned.add("nU")
    if nV is None:
        warned.add("nV")
    seconds = run.get("wall_time_s")
    if seconds is None:
        warned.add("wall_time_s")
    feasible = bool(run.get("theta_feasible")) and bool(run.get("k_valid", True))
    score = int(edges) if feasible and edges is not None else None
    return [
        _fmt_count(nU),
        _fmt_count(nV),
        _fmt_count(edges),
        _fmt_ms(seconds),
    ], score


def render(data):
    """LaTeX for every recorded graph in a multi-method compare file."""
    if not isinstance(data, dict):
        print("# heuristics table: expected a compare-methods object", file=sys.stderr)
        data = {}
    methods = _methods(data)
    rows = data.get("results") or []
    if not isinstance(rows, list):
        print("# heuristics table: results is not a list", file=sys.stderr)
        rows = []
    if not methods or not rows:
        print("# heuristics table: no methods or graphs in the results file", file=sys.stderr)
        return (
            r"\begin{center}" "\n"
            r"\scriptsize Construction sizes are not available." "\n"
            r"\end{center}"
        )

    warned = set()
    body = []
    k = data.get("k")
    theta = data.get("theta")
    k_tex = str(int(k)) if isinstance(k, int) or (isinstance(k, float) and k == int(k)) else "varied"
    theta_tex = (
        str(int(theta))
        if isinstance(theta, int) or (isinstance(theta, float) and theta == int(theta))
        else "varied"
    )
    for row in rows:
        if not isinstance(row, dict) or "dataset" not in row:
            continue
        name = series_name("", row).replace("_", r"\_")
        runs = row.get("runs") if isinstance(row.get("runs"), dict) else {}
        cells = [name]
        scores = []
        edge_text = []
        for method_id in methods:
            shown, score = _run_cells(runs.get(method_id), warned)
            cells.extend(shown)
            edge_text.append(shown[2])
            scores.append(score)
        ranked = [(i, score) for i, score in enumerate(scores) if score is not None]
        if ranked:
            best = max(score for _, score in ranked)
            winner_text = {
                edge_text[i] for i, score in ranked if score == best and edge_text[i] != "--"
            }
            for i, text in enumerate(edge_text):
                if text in winner_text:
                    # |U| |V| |E| t, so |E| is the third cell of the block.
                    cells[1 + 4 * i + 2] = rf"\textbf{{{text}}}"
        body.append(" & ".join(cells) + r" \\")

    if warned:
        fields = ", ".join(sorted(warned))
        print(
            f"# heuristics table: missing {fields} on at least one graph",
            file=sys.stderr,
        )
    if not body:
        print("# heuristics table: no usable rows", file=sys.stderr)
        return (
            r"\begin{center}" "\n"
            r"\scriptsize Construction sizes are not available." "\n"
            r"\end{center}"
        )

    groups = []
    rules = []
    sub = ["Graph"]
    col = 2
    for method_id in methods:
        groups.append(rf"& \multicolumn{{4}}{{c}}{{{_label(method_id)}}}")
        rules.append(rf"\cmidrule(lr){{{col}-{col + 3}}}")
        sub.append(r"& $|U|$ & $|V|$ & $|E|$ & $t$")
        col += 4
    spec = "@{}l" + " rrrr" * len(methods) + "@{}"
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\abovecaptionskip}{1pt}",
        r"\setlength{\belowcaptionskip}{0pt}",
        r"\caption{Solution side sizes $|U|$ and $|V|$, edge count $|E|$, "
        r"and wall time $t$ in milliseconds. "
        r"Bold is the largest $\theta$-feasible $|E|$ on that graph. "
        rf"Experiments use $k={k_tex}$ and $\theta={theta_tex}$." + "}",
        r"\label{tab:heuristics}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\setlength{\aboverulesep}{0.12ex}",
        r"\setlength{\belowrulesep}{0.12ex}",
        r"\setlength{\cmidrulesep}{0pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        "".join(groups) + r" \\",
        "".join(rules),
        "".join(sub) + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def run(path, output):
    path = Path(path)
    data = load_json(path)
    if data is None:
        print(f"# heuristics table: could not read {path}", file=sys.stderr)
        write_tex(render({}), output)
        return
    write_tex(render(data), output)
