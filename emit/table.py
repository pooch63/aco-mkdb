"""
table mode — vary.jl ant-count JSON → LaTeX table.

Columns match the paper table:

    Dataset | |U_G| | |V_G| | |E_G| | |U_R| | |V_R| | |E_R|
            | ACO: |E(D*)| Time ITB TTB | θ-Heuristic: |E(D*)| Time

|U_G| / |V_G| / |E_G| come from the original graph (`graph.nU` / `graph.nV`
and top-level `edge_count`). |U_R| / |V_R| / |E_R| are the reduced graph
sizes. ACO reports the best trial (max `final_edges`, then min
`time_to_best_s`, then min `wall_time_s`).
"""

from __future__ import annotations

import sys

from .common import display_name, load_json, report_skipped, write_tex


def fmt_int(value):
    if value is None:
        return "--"
    return str(int(value))


def fmt_time(value):
    """Format a wall/elapsed time stored in seconds as milliseconds."""
    if value is None:
        return "--"
    return f"{float(value) * 1000:.1f}"


def fmt_itb(value):
    if value is None:
        return "--"
    return str(int(value))


def trial_sort_key(trial):
    """Best trial: most edges, then fastest time-to-best, then fastest wall time."""
    edges = trial.get("final_edges")
    ttb = trial.get("time_to_best_s")
    wall = trial.get("wall_time_s")
    return (
        -(int(edges) if edges is not None else -10**18),
        float(ttb) if ttb is not None else float("inf"),
        float(wall) if wall is not None else float("inf"),
    )


def select_best_trial(trials, ants=None):
    candidates = []
    for t in trials:
        if t.get("final_edges") is None:
            continue
        if ants is not None and t.get("ants") != ants:
            continue
        candidates.append(t)
    if not candidates:
        return None
    return min(candidates, key=trial_sort_key)


def summarize_file(data, ants=None):
    """Pull one table row from a vary.jl ant-count JSON, or None if unusable."""
    if data.get("vary") not in (None, "ant-count") and "trials" not in data:
        return None

    graph = data.get("graph") or {}
    heuristic = data.get("heuristic") or {}
    trials = data.get("trials") or []
    best = select_best_trial(trials, ants=ants)

    edge_count = data.get("edge_count")
    if edge_count is None:
        edge_count = graph.get("edges")

    return {
        "k": data.get("k"),
        "theta": data.get("theta"),
        "nU": graph.get("nU"),
        "nV": graph.get("nV"),
        "edge_count": edge_count,
        "reduced_nU": graph.get("reduced_nU"),
        "reduced_nV": graph.get("reduced_nV"),
        "reduced_edges": graph.get("reduced_edges"),
        "aco_edges": None if best is None else best.get("final_edges"),
        "aco_time": None if best is None else best.get("wall_time_s"),
        "aco_itb": None if best is None else best.get("iterations_to_best"),
        "aco_ttb": None if best is None else best.get("time_to_best_s"),
        "heur_edges": heuristic.get("final_edges"),
        "heur_time": heuristic.get("wall_time_s"),
    }


def row_tex(name, row):
    cells = [
        name,
        fmt_int(row["nU"]),
        fmt_int(row["nV"]),
        fmt_int(row["edge_count"]),
        fmt_int(row["reduced_nU"]),
        fmt_int(row["reduced_nV"]),
        fmt_int(row["reduced_edges"]),
        fmt_int(row["aco_edges"]),
        fmt_time(row["aco_time"]),
        fmt_itb(row["aco_itb"]),
        fmt_time(row["aco_ttb"]),
        fmt_int(row["heur_edges"]),
        fmt_time(row["heur_time"]),
    ]
    return "    " + " & ".join(cells) + r" \\"


def caption_k_theta(rows):
    ks = {r["k"] for r in rows if r["k"] is not None}
    thetas = {r["theta"] for r in rows if r["theta"] is not None}
    k = ks.pop() if len(ks) == 1 else "?"
    theta = thetas.pop() if len(thetas) == 1 else "?"
    return rf"$k={k}, \theta={theta}$"


def build_table(named_rows):
    caption = caption_k_theta([row for _, row in named_rows])
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        r"",
        r"  \begin{adjustwidth}{-5cm}{-5cm}",
        r"  \centering",
        r"  \setlength{\tabcolsep}{3.5pt} % Reduced spacing between columns",
        r"  \begin{tabular}{l *{12}{r}} % 1 left-aligned column + 12 right-aligned columns",
        r"    \toprule",
        r"    Dataset & $|U_G|$ & $|V_G|$ & $|E_G|$ & $|U_R|$ & $|V_R|$ & $|E_R|$"
        r" & \multicolumn{4}{c}{ACO} & \multicolumn{2}{c}{$\theta$-Heuristic} \\",
        r"    \cmidrule(lr){8-11} \cmidrule(lr){12-13}",
        r"    & & & & & & & $|E(D^*)|$ & Time & ITB & TTB & $|E(D^*)|$ & Time \\",
        r"    \midrule",
    ]
    for name, row in named_rows:
        lines.append(row_tex(name, row))
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \end{adjustwidth}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def run(json_paths, output, ants=None):
    named_rows = []
    skipped = []

    for path in json_paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue

        row = summarize_file(data, ants=ants)
        if row is None:
            skipped.append((path, "not a vary.jl ant-count result"))
            continue

        named_rows.append((display_name(path, data), row))

    if not named_rows:
        raise SystemExit("No usable vary JSON files -- nothing to dump.")

    named_rows.sort(
        key=lambda nr: (nr[1]["edge_count"] is None, nr[1]["edge_count"] or 0, nr[0])
    )

    tex = build_table(named_rows)
    write_tex(tex, output)

    print(
        f"# table: {len(named_rows)} dataset(s)"
        + (f"; ants={ants}" if ants is not None else ""),
        file=sys.stderr,
    )
    report_skipped(skipped)
