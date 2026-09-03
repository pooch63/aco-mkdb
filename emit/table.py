"""
table mode — vary.jl ant-count JSON → LaTeX table.

Columns match the paper table:

    Dataset | |U_G| | |V_G| | |E_G| | |U_R| | |V_R| | |E_R|
            | ACO: |U(D*)| |V(D*)| |E(D*)| Discovery ITB TTB
            | θ-Heuristic: |U(D*)| |V(D*)| |E(D*)| Time

|U_G| / |V_G| / |E_G| come from the original graph (`graph.nU` / `graph.nV`
and top-level `edge_count`). |U_R| / |V_R| / |E_R| are the reduced graph
sizes. ACO reports the best counted trial (max `final_edges`, then min
`time_to_best_s`, then min `wall_time_s`). The first replicate per ant
count is a Julia JIT warmup when `aco_runs > 1` and is omitted. Discovery
is the sum of counted same-ant `wall_time_s` values. ITB / TTB are from
that winning replicate only.

Rows are grouped into three sections (each sorted ascending by ACO's
|E(D*)|), separated by midrules in one landscape table:

  1. ACO beat the θ-heuristic (θ-feasible ACO with strictly more edges,
     or ACO θ-feasible while the heuristic is not)
  2. θ-heuristic beat ACO (symmetric)
  3. Tie, or both failed θ-feasibility (need ≥θ vertices on both sides;
     more edges without θ on both sides still counts as a failure)

--subset=full (default) emits every row (appendix).
--subset=highlights emits the largest relative ACO wins plus the closest
θ-heuristic wins (main text).
"""

from __future__ import annotations

import sys

from .common import (
    aco_discovery_cost,
    counted_trials,
    display_name,
    load_json,
    report_skipped,
    select_best_trial_any,
    write_tex,
)

SUBSET_FULL = "full"
SUBSET_HIGHLIGHTS = "highlights"

# How many rows to keep in each highlights block.
HIGHLIGHT_BEST = 5
HIGHLIGHT_LEAST_BAD = 3


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


def select_best_trial(trials, ants=None):
    if ants is None:
        return select_best_trial_any(trials)
    candidates = [t for t in trials if t.get("final_edges") is not None and t.get("ants") == ants]
    return select_best_trial_any(candidates)


def summarize_file(data, ants=None):
    """Pull one table row from a vary.jl ant-count JSON, or None if unusable."""
    if data.get("vary") not in (None, "ant-count") and "trials" not in data:
        return None

    graph = data.get("graph") or {}
    heuristic = data.get("heuristic") or {}
    trials = counted_trials(data.get("trials") or [], data)
    best = select_best_trial(trials, ants=ants)

    edge_count = data.get("edge_count")
    if edge_count is None:
        edge_count = graph.get("edges")

    # Prefer recomputing discovery so legacy JSON that summed the JIT run
    # still emits post-warmup totals.
    discovery = None
    if best is not None:
        discovery = aco_discovery_cost(trials, best, until_found=False, data=data)
    if discovery is None:
        discovery = data.get("aco_discovery_s")

    return {
        "k": data.get("k"),
        "theta": data.get("theta"),
        "nU": graph.get("nU"),
        "nV": graph.get("nV"),
        "edge_count": edge_count,
        "reduced_nU": graph.get("reduced_nU"),
        "reduced_nV": graph.get("reduced_nV"),
        "reduced_edges": graph.get("reduced_edges"),
        "reduced_max_degree": graph.get("reduced_max_degree"),
        "reduced_avg_degree": graph.get("reduced_avg_degree"),
        "aco_nU": None if best is None else best.get("nU"),
        "aco_nV": None if best is None else best.get("nV"),
        "aco_edges": None if best is None else best.get("final_edges"),
        "aco_time": discovery,
        "aco_itb": None if best is None else best.get("iterations_to_best"),
        "aco_ttb": None if best is None else best.get("time_to_best_s"),
        "heur_nU": heuristic.get("nU"),
        "heur_nV": heuristic.get("nV"),
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
        fmt_int(row["aco_nU"]),
        fmt_int(row["aco_nV"]),
        fmt_int(row["aco_edges"]),
        fmt_time(row["aco_time"]),
        fmt_itb(row["aco_itb"]),
        fmt_time(row["aco_ttb"]),
        fmt_int(row["heur_nU"]),
        fmt_int(row["heur_nV"]),
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


def table_labels(rows):
    """
    LaTeX \\label{} ids for the full and highlights tables.

    Primary paper setting (k=2, θ=5) keeps the historical ids; other
    (k, θ) pairs get a suffix so multiple tables can coexist.
    """
    ks = {r["k"] for r in rows if r["k"] is not None}
    thetas = {r["theta"] for r in rows if r["theta"] is not None}
    k = ks.pop() if len(ks) == 1 else None
    theta = thetas.pop() if len(thetas) == 1 else None
    if k == 2 and theta == 5:
        return "tab:aco-heur-pn", "tab:aco-heur-pn-highlights"
    if k is not None and theta is not None:
        suffix = f"k{int(k)}t{int(theta)}"
        return (
            f"tab:aco-heur-pn-{suffix}",
            f"tab:aco-heur-pn-{suffix}-highlights",
        )
    return "tab:aco-heur-pn", "tab:aco-heur-pn-highlights"


def is_theta_feasible(nU, nV, theta):
    """True iff both sides have at least θ vertices."""
    if theta is None or nU is None or nV is None:
        return False
    return int(nU) >= int(theta) and int(nV) >= int(theta)


# Section ids for outcome grouping (order is table order).
SECTION_ACO = "aco"
SECTION_HEUR = "heur"
SECTION_TIE = "tie"


def compare_section(row):
    """
    Which table section a row belongs to.

    A side that lacks ≥θ vertices on both parts has failed, regardless of
    edge count. Among θ-feasible solutions, more edges wins; equal edges
    (or mutual failure) is a tie.
    """
    aco_ok = is_theta_feasible(row["aco_nU"], row["aco_nV"], row["theta"])
    heur_ok = is_theta_feasible(row["heur_nU"], row["heur_nV"], row["theta"])

    if aco_ok and not heur_ok:
        return SECTION_ACO
    if heur_ok and not aco_ok:
        return SECTION_HEUR
    if not aco_ok and not heur_ok:
        return SECTION_TIE

    aco_edges = int(row["aco_edges"] or 0)
    heur_edges = int(row["heur_edges"] or 0)
    if aco_edges > heur_edges:
        return SECTION_ACO
    if heur_edges > aco_edges:
        return SECTION_HEUR
    return SECTION_TIE


def aco_edge_sort_key(named_row):
    _, row = named_row
    return (row["aco_edges"] is None, row["aco_edges"] or 0, named_row[0])


def sectioned_rows(named_rows):
    """Split into (aco-win, heur-win, tie/fail) lists, each sorted by ACO |E(D*)|."""
    sections = {SECTION_ACO: [], SECTION_HEUR: [], SECTION_TIE: []}
    for named in named_rows:
        sections[compare_section(named[1])].append(named)
    for key in sections:
        sections[key].sort(key=aco_edge_sort_key)
    return (
        sections[SECTION_ACO],
        sections[SECTION_HEUR],
        sections[SECTION_TIE],
    )


def tabular_header_lines():
    return [
        r"  \small",
        r"  \setlength{\tabcolsep}{2.5pt}",
        r"  \resizebox{\linewidth}{!}{%",
        r"  \begin{tabular}{l *{16}{r}} % 1 left-aligned column + 16 right-aligned columns",
        r"    \toprule",
        r"    Dataset & $|U_G|$ & $|V_G|$ & $|E_G|$ & $|U_R|$ & $|V_R|$ & $|E_R|$"
        r" & \multicolumn{6}{c}{ACO-PN} & \multicolumn{4}{c}{$\theta$-Heuristic} \\",
        r"    \cmidrule(lr){8-13} \cmidrule(lr){14-17}",
        r"    & & & & & & & $|U_{D^*}|$ & $|V_{D^*}|$ & $|E(D^*)|$ & Discovery & ITB & TTB"
        r" & $|U_{D^*}|$ & $|V_{D^*}|$ & $|E(D^*)|$ & Time \\",
        r"    \midrule",
    ]


def tabular_body_lines(sections):
    """Render one or more row sections; midrule between sections, bottomrule at end."""
    lines = []
    for i, section in enumerate(sections):
        for name, row in section:
            lines.append(row_tex(name, row))
        if i < len(sections) - 1:
            lines.append(r"    \midrule")
        else:
            lines.append(r"    \bottomrule")
    return lines


def build_table_tex(caption, label, sections, *, landscape=True, placement="p"):
    """Render the ACO vs θ comparison tabular, optionally in a landscape page."""
    lines = []
    if landscape:
        lines.append(r"\begin{landscapetable}")
    lines += [
        rf"\begin{{table}}[{placement}]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        r"",
    ]
    lines += tabular_header_lines()
    lines += tabular_body_lines(sections)
    lines += [
        r"  \end{tabular}",
        r"  }",
        r"\end{table}",
    ]
    if landscape:
        lines.append(r"\end{landscapetable}")
    return "\n".join(lines)


def improvement_ratio(row):
    """ACO |E(D*)| / θ-heuristic |E(D*)|; missing or zero heuristic → large."""
    aco = row.get("aco_edges")
    if aco is None:
        return float("-inf")
    heur = row.get("heur_edges")
    if heur is None or int(heur) <= 0:
        return float(aco) + 1.0e9
    return float(aco) / float(heur)


def edge_deficit(row):
    """θ-heuristic |E| − ACO |E|; smaller (incl. negative) is less bad for ACO."""
    aco = int(row.get("aco_edges") or 0)
    heur = int(row.get("heur_edges") or 0)
    return heur - aco


def pick_highlight_sections(aco_rows, heur_rows):
    """
    Representative subset for the main-text table.

    Top: largest relative ACO wins (by |E(D*)| ratio). Bottom: closest
    θ-heuristic wins (smallest edge deficits). Within each block, keep the
    same ascending ACO-|E| order as the full appendix table.
    """
    best = sorted(
        aco_rows,
        key=lambda named: (-improvement_ratio(named[1]), aco_edge_sort_key(named)),
    )[:HIGHLIGHT_BEST]
    best.sort(key=aco_edge_sort_key)

    least_bad = sorted(
        heur_rows,
        key=lambda named: (edge_deficit(named[1]), aco_edge_sort_key(named)),
    )[:HIGHLIGHT_LEAST_BAD]
    return best, least_bad


def build_table(named_rows, subset=SUBSET_FULL):
    """Build ACO vs θ LaTeX: full appendix table or main-text highlights."""
    rows_only = [row for _, row in named_rows]
    kt = caption_k_theta(rows_only)
    full_label, highlights_label = table_labels(rows_only)
    aco_rows, heur_rows, tie_rows = sectioned_rows(named_rows)

    if subset == SUBSET_HIGHLIGHTS:
        best, least_bad = pick_highlight_sections(aco_rows, heur_rows)
        sections = [s for s in (best, least_bad) if s]
        if not sections:
            raise ValueError("No highlight rows to emit for ACO vs θ table")
        return build_table_tex(
            (
                rf"Representative ACO-PN vs.\ $\theta$-heuristic comparisons "
                rf"({kt}). Top: largest relative $|E(D^*)|$ gains for ACO-PN. "
                rf"Bottom: closest $\theta$-heuristic wins (smallest edge "
                rf"deficits). Full results are in Table~\ref{{{full_label}}}."
            ),
            highlights_label,
            sections,
            landscape=False,
            placement="H",
        )

    if subset != SUBSET_FULL:
        print(
            f"# Warning: unknown table subset {subset!r}; emitting full table",
            file=sys.stderr,
        )

    sections = [s for s in (aco_rows, heur_rows, tie_rows) if s]
    return build_table_tex(
        kt,
        full_label,
        sections,
        landscape=True,
        placement="p",
    )


def run(json_paths, output, ants=None, subset=SUBSET_FULL):
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

    aco_rows, heur_rows, tie_rows = sectioned_rows(named_rows)
    tex = build_table(named_rows, subset=subset)
    write_tex(tex, output)

    highlight_note = ""
    if subset == SUBSET_HIGHLIGHTS:
        best, least_bad = pick_highlight_sections(aco_rows, heur_rows)
        highlight_note = (
            f"; highlights best={len(best)}, least-bad={len(least_bad)}"
        )

    print(
        f"# table: {len(named_rows)} dataset(s)"
        f" (aco-win={len(aco_rows)}, heur-win={len(heur_rows)}, "
        f"tie/fail={len(tie_rows)})"
        + (f"; ants={ants}" if ants is not None else "")
        + f"; subset={subset}"
        + highlight_note,
        file=sys.stderr,
    )
    report_skipped(skipped)
