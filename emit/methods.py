"""
methods mode — load.jl ``--benchmark=…`` JSON → per-graph multi-method table.

Reads one benchmark JSON per graph (as written by ``scripts/evaluate.bash``
with ``METHODS=…``) and renders, for each method, the returned biclique
sides / edges and the timed wall clock:

    Dataset | |U_R| | |V_R| | |E_R|
            | <method>: |U_D| |V_D| |E(D)| Time   (one group per method)

Original graph sizes are omitted; the sizes table lists them once.
Reduced-size columns that no input file recorded are dropped. The first
method in ``--methods`` is the *focal* method; rows are grouped into three
blocks by its outcome against every other method under the paper win rule
(θ-feasible and more edges, or θ-feasible while the other is not):

  1. focal beats all other methods
  2. tie for best (no method beats the focal one, but it does not beat all)
  3. some other method beats the focal one

Within each row the largest θ-feasible |E(D)| is bold; θ-infeasible
solutions are marked with a dagger.
"""

from __future__ import annotations

import os
import sys

from .common import display_name, fmt_thousands, load_json, report_skipped, write_tex

DEFAULT_METHODS = ("tow", "aco", "heuristic")
METHOD_LABELS = {
    "tow": "TOW",
    "tug": "TOW",
    "aco": "ACO",
    "heuristic": r"$\theta$-Heuristic",
    "edges": "Edge ACO",
    "diffusion": "Diffusion ACO",
    "ga": "GA",
    "tabu": "Tabu",
    "vns": "VNS",
    "retry": "Retry",
    "pivot": "Pivot",
}

# Reduced graph sizes only. Original |U|, |V|, |E| are in the sizes table.
# A column is shown when some file recorded it.
GRAPH_COLUMNS = (
    ("reduced_nU", r"$U_k$"),
    ("reduced_nV", r"$V_k$"),
    ("reduced_edges", r"$|E_R|$"),
)

SECTION_WIN = "win"
SECTION_TIE = "tie"
SECTION_LOSS = "loss"


def method_label(name):
    return METHOD_LABELS.get(name, name.upper())


def fmt_int(value):
    return "--" if value is None else str(int(value))


def fmt_ms(value):
    return "--" if value is None else f"{float(value) * 1000:.1f}"


def method_cell(data, name, path):
    """Pull |U|, |V|, |E|, wall time for one method block, or None if absent."""
    block = data.get(name)
    if not isinstance(block, dict):
        print(
            f"# Warning: {os.path.basename(path)} has no '{name}' block",
            file=sys.stderr,
        )
        return None
    sol = block.get("solution") or {}
    edges = block.get("final_edges", sol.get("edges"))
    return {
        "nU": block.get("nU", sol.get("nU")),
        "nV": block.get("nV", sol.get("nV")),
        "edges": edges,
        "time": block.get("wall_time_s"),
    }


def is_feasible(cell, theta):
    if cell is None or theta is None:
        return False
    if cell["nU"] is None or cell["nV"] is None or cell["edges"] is None:
        return False
    return int(cell["nU"]) >= int(theta) and int(cell["nV"]) >= int(theta)


def beats(a, b, theta):
    """Paper win rule: a is θ-feasible and (b is not, or a has more edges)."""
    if not is_feasible(a, theta):
        return False
    if not is_feasible(b, theta):
        return True
    return int(a["edges"]) > int(b["edges"])


def summarize_file(path, data, methods):
    if not any(isinstance(data.get(m), dict) for m in methods):
        return None
    graph = data.get("graph") or {}
    return {
        "k": data.get("k"),
        "theta": data.get("theta"),
        "nU": graph.get("nU"),
        "nV": graph.get("nV"),
        "edge_count": data.get("edge_count"),
        "reduced_nU": graph.get("reduced_nU"),
        "reduced_nV": graph.get("reduced_nV"),
        "reduced_edges": graph.get("reduced_edges"),
        "cells": {m: method_cell(data, m, path) for m in methods},
        "settings": {
            m: (data.get(m) or {}).get("settings") for m in methods
        },
        "aco_budget": (
            (data.get("aco") or {}).get("ants"),
            (data.get("aco") or {}).get("iterations_budget"),
        ),
    }


def section_of(row, methods):
    focal, others = methods[0], methods[1:]
    theta = row["theta"]
    fc = row["cells"][focal]
    if any(beats(row["cells"][o], fc, theta) for o in others):
        return SECTION_LOSS
    if others and all(beats(fc, row["cells"][o], theta) for o in others):
        return SECTION_WIN
    return SECTION_TIE


def focal_sort_key(named, focal):
    name, row = named
    cell = row["cells"][focal]
    edges = None if cell is None else cell["edges"]
    return (edges is None, edges or 0, name)


def row_cells(name, row, methods, graph_cols):
    theta = row["theta"]
    feasible_edges = [
        int(c["edges"])
        for c in row["cells"].values()
        if is_feasible(c, theta)
    ]
    best = max(feasible_edges) if feasible_edges else None

    cells = [name] + [
        fmt_thousands(row[key]) if key in ("nU", "nV", "reduced_nU", "reduced_nV")
        else fmt_int(row[key])
        for key, _ in graph_cols
    ]
    for m in methods:
        c = row["cells"][m]
        if c is None:
            cells += ["--"] * 4
            continue
        edges = fmt_int(c["edges"])
        if c["edges"] is not None:
            if not is_feasible(c, theta):
                edges += r"$^\dagger$"
            elif best is not None and int(c["edges"]) == best:
                edges = rf"\textbf{{{edges}}}"
        cells += [fmt_int(c["nU"]), fmt_int(c["nV"]), edges, fmt_ms(c["time"])]
    return cells


def row_tex(name, row, methods, graph_cols):
    return "    " + " & ".join(row_cells(name, row, methods, graph_cols)) + r" \\"


def record_counts(rows, focal, other):
    w = t = l = 0
    for row in rows:
        a, b, theta = row["cells"][focal], row["cells"][other], row["theta"]
        if beats(a, b, theta):
            w += 1
        elif beats(b, a, theta):
            l += 1
        else:
            t += 1
    return w, t, l


def single_value(values):
    vals = {v for v in values if v is not None}
    return vals.pop() if len(vals) == 1 else None


def settings_text(rows, methods):
    """Describe recorded hyperparameters when every file agrees."""
    parts = []
    if "aco" in methods:
        budget = single_value(r["aco_budget"] for r in rows)
        if budget and budget[0] is not None:
            parts.append(f"ACO: {budget[0]} ants, {budget[1]} epochs")
    for m in methods:
        recorded = [r["settings"].get(m) for r in rows if r["settings"].get(m)]
        if not recorded:
            continue
        frozen = single_value(tuple(sorted(s.items())) for s in recorded)
        if frozen is None:
            print(
                f"# Warning: '{m}' settings differ across files; omitted from caption",
                file=sys.stderr,
            )
            continue
        s = dict(frozen)
        s.pop("seed", None)
        desc = ", ".join(
            f"{k.replace('_', ' ')}={str(v).lower() if isinstance(v, bool) else v}"
            for k, v in sorted(s.items())
        )
        parts.append(f"{method_label(m)}: {desc}")
    return "; ".join(parts)


def build_tex(named_rows, methods, label):
    rows = [r for _, r in named_rows]
    focal = methods[0]
    k = single_value(r["k"] for r in rows)
    theta = single_value(r["theta"] for r in rows)
    kt = rf"$k={k if k is not None else '?'}, \theta={theta if theta is not None else '?'}$"

    graph_cols = [
        (key, header)
        for key, header in GRAPH_COLUMNS
        if any(r[key] is not None for r in rows)
    ]

    sections = {SECTION_WIN: [], SECTION_TIE: [], SECTION_LOSS: []}
    for named in named_rows:
        sections[section_of(named[1], methods)].append(named)
    for key in sections:
        sections[key].sort(key=lambda n: focal_sort_key(n, focal))
    blocks = [
        sections[s]
        for s in (SECTION_WIN, SECTION_TIE, SECTION_LOSS)
        if sections[s]
    ]

    records = []
    for other in methods[1:]:
        w, t, l = record_counts(rows, focal, other)
        records.append(
            f"vs.\\ {method_label(other)}: {w} wins, {t} ties, {l} losses"
        )
    settings = settings_text(rows, methods)

    caption = (
        rf"{method_label(focal)} vs.\ "
        + ", ".join(method_label(m) for m in methods[1:])
        + rf" ({kt}, {len(rows)} graphs). {method_label(focal)} "
        + "; ".join(records)
        + ". Rows grouped by whether "
        + rf"{method_label(focal)} beats every other method, ties for best, "
        + r"or is beaten. Bold: largest $\theta$-feasible $|E(D)|$ per row; "
        + r"$^\dagger$: not $\theta$-feasible. Times in ms."
        + (f" Settings --- {settings}." if settings else "")
    )

    n_graph = len(graph_cols)
    n_cols = 1 + n_graph + 4 * len(methods)
    group_headers = []
    cmidrules = []
    start = 2 + n_graph
    for m in methods:
        group_headers.append(rf"\multicolumn{{4}}{{c}}{{{method_label(m)}}}")
        cmidrules.append(rf"\cmidrule(lr){{{start}-{start + 3}}}")
        start += 4
    sub = r"$|U_D|$ & $|V_D|$ & $|E(D)|$ & Time"

    lines = [
        r"\begin{landscapetable}",
        r"\begin{table}[p]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"",
        r"  \small",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \begin{adjustbox}{max width=\linewidth, max totalheight=0.82\textheight}",
        rf"  \begin{{tabular}}{{l *{{{n_cols - 1}}}{{r}}}}",
        r"    \toprule",
        "    Dataset"
        + "".join(f" & {h}" for _, h in graph_cols)
        + "".join(f" & {h}" for h in group_headers)
        + r" \\",
        "    " + " ".join(cmidrules),
        "    " + " & ".join([""] * (1 + n_graph) + [sub] * len(methods)) + r" \\",
        r"    \midrule",
    ]
    for i, block in enumerate(blocks):
        for name, row in block:
            lines.append(row_tex(name, row, methods, graph_cols))
        lines.append(r"    \midrule" if i < len(blocks) - 1 else r"    \bottomrule")
    lines += [
        r"  \end{tabular}",
        r"  \end{adjustbox}",
        r"\end{table}",
        r"\end{landscapetable}",
    ]
    return "\n".join(lines), {s: len(v) for s, v in sections.items()}


PREAMBLE = r"\usepackage{adjustbox}"

SUBSET_FULL = "full"
SUBSET_COMPACT = "compact"

# Group-row titles. Kept shorter than the four numeric columns they span.
_DENSE_METHOD = {
    "tow": "TOW",
    "tug": "TOW",
    "aco": "ACO",
    "heuristic": r"$\theta$",
    "edges": "Edge",
    "diffusion": "Diff.",
    "ga": "GA",
    "tabu": "Tabu",
    "vns": "VNS",
    "retry": "Retry",
    "pivot": "Pivot",
}
_GRAPH_HEAD = {
    "nU": "$U$",
    "nV": "$V$",
    "edge_count": "$E$",
    "reduced_nU": r"$U_k$",
    "reduced_nV": r"$V_k$",
    "reduced_edges": "$E$",
}


def _dense_header(methods, graph_cols):
    original = [key for key, _ in graph_cols if not str(key).startswith("reduced")]
    reduced = [key for key, _ in graph_cols if str(key).startswith("reduced")]
    groups = [(1, None)]
    headers = ["Dataset"]
    if original:
        groups.append((len(original), "Graph"))
        headers.extend(_GRAPH_HEAD[key] for key in original)
    if reduced:
        groups.append((len(reduced), "Reduced"))
        headers.extend(_GRAPH_HEAD[key] for key in reduced)
    for m in methods:
        groups.append((4, _DENSE_METHOD.get(m, method_label(m))))
        headers.extend(["$U$", "$V$", "$E$", "Time"])
    return groups, headers


def build_compact_tex(named_rows, methods, label):
    """Every dataset, packed into side-by-side blocks."""
    from .dense_table import render_dense_table

    rows = [r for _, r in named_rows]
    focal = methods[0]
    k = single_value(r["k"] for r in rows)
    theta = single_value(r["theta"] for r in rows)
    kt = rf"$k={k if k is not None else '?'}, \theta={theta if theta is not None else '?'}$"
    graph_cols = [
        (key, header)
        for key, header in GRAPH_COLUMNS
        if any(r[key] is not None for r in rows)
    ]
    sections = {SECTION_WIN: [], SECTION_TIE: [], SECTION_LOSS: []}
    for named in named_rows:
        sections[section_of(named[1], methods)].append(named)
    for key in sections:
        sections[key].sort(key=lambda n: focal_sort_key(n, focal))
    blocks = [
        [
            row_cells(name, row, methods, graph_cols)
            for name, row in sections[s]
        ]
        for s in (SECTION_WIN, SECTION_TIE, SECTION_LOSS)
        if sections[s]
    ]
    groups, headers = _dense_header(methods, graph_cols)
    others = " and ".join(method_label(m) for m in methods[1:])
    caption = rf"{method_label(focal)} vs.\ {others} ({kt})"
    tex = render_dense_table(
        caption=caption,
        label=label,
        column_headers=headers,
        groups=groups,
        sections=blocks,
    )
    return tex, {s: len(v) for s, v in sections.items()}


def run(json_paths, output, methods=None, label=None, subset=SUBSET_FULL):
    methods = tuple(methods or DEFAULT_METHODS)
    if not methods:
        raise SystemExit("methods mode needs at least one method")

    named_rows = []
    skipped = []
    for path in json_paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue
        row = summarize_file(path, data, methods)
        if row is None:
            skipped.append((path, f"no {'/'.join(methods)} benchmark blocks"))
            continue
        named_rows.append((display_name(path, data), row))

    if not named_rows:
        raise SystemExit("No usable benchmark JSON files -- nothing to emit.")

    if label is None:
        directory = os.path.basename(os.path.dirname(os.path.abspath(json_paths[0])))
        label = f"tab:methods-{directory}"
    if subset == SUBSET_COMPACT:
        from .dense_table import PREAMBLE as DENSE_PREAMBLE
        tex, counts = build_compact_tex(named_rows, methods, label)
        preamble = DENSE_PREAMBLE
    else:
        if subset != SUBSET_FULL:
            print(
                f"# Warning: unknown methods subset {subset!r}; emitting full table",
                file=sys.stderr,
            )
        tex, counts = build_tex(named_rows, methods, label)
        preamble = PREAMBLE
    write_tex(tex, output, preamble=preamble)

    print(
        f"# methods: {len(named_rows)} dataset(s); focal={methods[0]} "
        f"(win={counts[SECTION_WIN]}, tie={counts[SECTION_TIE]}, "
        f"loss={counts[SECTION_LOSS]})",
        file=sys.stderr,
    )
    report_skipped(skipped)
