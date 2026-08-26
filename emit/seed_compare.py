"""
seed-compare mode — compare-seeds.jl (+ vary.jl) → paper table + figures:

  1. Table of every dataset: graph/reduced nodes & edges, pivot time saved
     (or "--" when ACO did not beat θ), and percent reduction
  2. Bar chart of every dataset: pivot time saved (s) and percent reduction

Reads full pivot comparisons and beat_heuristic=false skip markers from
compare-seeds.jl. Pass --vary-dir (or rely on compare_* → vary_* name
mapping / per-file source_vary) for full graph sizes and datasets that
still lack a compare JSON.
"""

from __future__ import annotations

import glob
import os
import statistics
import sys

from .common import (
    aco_discovery_cost,
    display_name,
    infer_vary_dir,
    load_json,
    report_skipped,
    resolve_path,
    select_best_trial_any,
    series_name,
    write_tex,
)


def select_best_beating_trial(trials):
    """Mirror compare-seeds.jl: max edges among beats_heuristic, then min TTB."""
    beating = [t for t in trials if t.get("beats_heuristic") in (True, 1)]
    if not beating:
        return None
    best_edges = max(int(t.get("final_edges") or 0) for t in beating)
    tied = [t for t in beating if int(t.get("final_edges") or 0) == best_edges]

    def trial_time(t):
        tb = t.get("time_to_best_s")
        if tb is not None:
            return float(tb)
        wt = t.get("wall_time_s")
        return float(wt) if wt is not None else float("inf")

    return min(tied, key=trial_time)


def aco_mean_wct_s(trials):
    """Mean ACO wall_time_s over all trials (all ant counts / replicates)."""
    times = [
        float(t["wall_time_s"])
        for t in trials
        if t.get("wall_time_s") is not None
    ]
    return statistics.mean(times) if times else None


def _first_int(*values):
    for v in values:
        if v is not None:
            return int(v)
    return None


def graph_nU(*graphs):
    return _first_int(*(g.get("nU") for g in graphs if g))


def graph_nV(*graphs):
    return _first_int(*(g.get("nV") for g in graphs if g))


def full_vertex_count(*graphs):
    """Full graph |U| + |V| from a vary.jl / compare-seeds graph block."""
    u, v = graph_nU(*graphs), graph_nV(*graphs)
    if u is not None and v is not None:
        return u + v
    return None


def graph_edge_count(*sources):
    """Full graph |E| from graph.edges or top-level edge_count."""
    for src in sources:
        if not src:
            continue
        graph = src.get("graph") if isinstance(src, dict) else None
        if graph:
            edges = graph.get("edges")
            if edges is not None:
                return int(edges)
        if isinstance(src, dict):
            edge_count = src.get("edge_count")
            if edge_count is not None:
                return int(edge_count)
            if graph is None:
                edges = src.get("edges")
                if edges is not None:
                    return int(edges)
    return None


def reduced_nU(*graphs):
    return _first_int(*(g.get("reduced_nU") for g in graphs if g))


def reduced_nV(*graphs):
    return _first_int(*(g.get("reduced_nV") for g in graphs if g))


def reduced_vertex_count(*graphs):
    """
    Reduced graph |U_R| + |V_R| from a vary.jl / compare-seeds graph block.

    Prefers reduced_nU/reduced_nV; falls back to original nU/nV when reduced
    sizes are absent. Returns None if no usable pair is found.
    """
    ru, rv = reduced_nU(*graphs), reduced_nV(*graphs)
    if ru is not None and rv is not None:
        return ru + rv
    return full_vertex_count(*graphs)


def reduced_edge_count(*graphs):
    """Reduced graph |E_R| from reduced_edges on a graph block."""
    return _first_int(*(g.get("reduced_edges") for g in graphs if g))


def attach_size_metrics(summary, data=None, vary_data=None):
    """Add full/reduced graph size fields onto a summary dict."""
    vary_graph = (vary_data or {}).get("graph") if vary_data else None
    data_graph = (data or {}).get("graph") if data else None
    graphs = (vary_graph, data_graph)
    summary["nU"] = graph_nU(*graphs)
    summary["nV"] = graph_nV(*graphs)
    summary["full_vertex_count"] = full_vertex_count(*graphs)
    summary["graph_edges"] = graph_edge_count(vary_data, data)
    summary["reduced_nU"] = reduced_nU(*graphs)
    summary["reduced_nV"] = reduced_nV(*graphs)
    summary["vertex_count"] = reduced_vertex_count(*graphs)
    summary["reduced_edges"] = reduced_edge_count(*graphs)
    return summary


def load_vary_for_compare(data, compare_path, vary_by_leaf=None):
    """
    Resolve vary.jl JSON for a compare-seeds file.

    Prefers an already-indexed vary_by_leaf entry, then source_vary /
    aco_trial.source_vary paths recorded in the compare JSON.
    """
    leaf = os.path.splitext(os.path.basename(compare_path))[0]
    if vary_by_leaf and leaf in vary_by_leaf:
        return vary_by_leaf[leaf][1]

    candidates = []
    if data.get("source_vary"):
        candidates.append(data["source_vary"])
    trial = data.get("aco_trial") or {}
    if trial.get("source_vary"):
        candidates.append(trial["source_vary"])

    for raw in candidates:
        path = resolve_path(raw, relative_to=compare_path)
        if path and os.path.isfile(path):
            return load_json(path)
    return None


def summarize_no_beat_marker(data, vary_data=None):
    """
    Summarize a compare-seeds.jl skip marker (beat_heuristic=false).

    Prefers fields written into the marker so plotting does not need the
    vary file; falls back to summarize_no_beat_vary when discovery stats
    are missing.
    """
    discovery = data.get("aco_discovery_s")
    mean_aco = data.get("aco_mean_wct_s")
    inclusive = data.get("inclusive_reduction_s")
    best = data.get("best_trial") or {}
    heur = data.get("heuristic") or {}
    heur_edges = data.get("heuristic_edges")
    if heur_edges is None:
        heur_edges = heur.get("final_edges")

    if discovery is None and vary_data is not None:
        return summarize_no_beat_vary(vary_data)

    if discovery is None:
        return None

    discovery = float(discovery)
    if inclusive is None:
        inclusive = -discovery

    return {
        "beat_heuristic": False,
        "skip_reason": data.get("skip_reason"),
        "theta_wall_time_s": None,
        "aco_seed_wall_time_s": None,
        "time_reduction_s": None,
        "time_reduction_pct": float(data.get("time_reduction_pct") or 0.0),
        "aco_discovery_s": discovery,
        "inclusive_reduction_s": float(inclusive),
        "inclusive_reduction_pct": None,
        "aco_time_to_best_s": best.get("time_to_best_s"),
        "aco_final_edges": best.get("final_edges"),
        "heuristic_edges": heur_edges,
        "ants": best.get("ants"),
        "run": best.get("run"),
        "aco_mean_wct_s": (
            float(mean_aco) if mean_aco is not None else None
        ),
    }


def summarize_seed_compare(data, compare_path=None, vary_data=None):
    """
    Extract pivot timing fields from a compare-seeds.jl JSON.

    Handles both full pivot comparisons (beat_heuristic=true / legacy files
    with pivot_theta) and skip markers (beat_heuristic=false).

    When vary_data (or aco_trial.source_vary) is available, also computes
    ACO discovery cost and inclusive net time change for beat cases.

    Returns dict with dataset label fields, or None if not a seed-compare file.
    """
    if data.get("compare") != "seeds":
        return None

    if data.get("skipped_existing"):
        return None

    # Explicit no-beat / missing-UV marker from compare-seeds.jl.
    if data.get("beat_heuristic") is False:
        return summarize_no_beat_marker(data, vary_data=vary_data)

    theta = data.get("pivot_theta") or {}
    aco = data.get("pivot_aco_seed") or {}
    trial = data.get("aco_trial") or {}

    theta_t = theta.get("wall_time_s")
    aco_t = aco.get("wall_time_s")
    if theta_t is None or aco_t is None:
        return None

    reduction_s = data.get("time_reduction_s")
    if reduction_s is None:
        reduction_s = float(theta_t) - float(aco_t)

    reduction_pct = data.get("time_reduction_pct")
    if reduction_pct is None:
        reduction_pct = (
            100.0 * float(reduction_s) / float(theta_t) if float(theta_t) > 0 else 0.0
        )

    discovery_s = data.get("aco_discovery_s")
    if discovery_s is None and vary_data is None and trial.get("source_vary"):
        src = resolve_path(trial["source_vary"], relative_to=compare_path)
        if src:
            vary_data = load_json(src)

    if discovery_s is None and vary_data is not None:
        trials = vary_data.get("trials") or []
        winning = select_best_beating_trial(trials)
        if winning is None and trial:
            # Fall back to the trial recorded in the compare JSON.
            winning = {
                "ants": trial.get("ants"),
                "run": trial.get("run"),
                "time_to_best_s": trial.get("time_to_best_s"),
                "wall_time_s": trial.get("wall_time_s"),
            }
        discovery_s = aco_discovery_cost(trials, winning, until_found=True)

    if discovery_s is None and trial.get("time_to_best_s") is not None:
        # No vary file: at least count time-to-best of the seeded trial.
        discovery_s = float(trial["time_to_best_s"])
    elif discovery_s is not None:
        discovery_s = float(discovery_s)

    mean_aco_wct = data.get("aco_mean_wct_s")
    if mean_aco_wct is None and vary_data is not None:
        mean_aco_wct = aco_mean_wct_s(vary_data.get("trials") or [])
    elif mean_aco_wct is not None:
        mean_aco_wct = float(mean_aco_wct)

    inclusive_s = data.get("inclusive_reduction_s")
    inclusive_pct = data.get("inclusive_reduction_pct")
    if inclusive_s is None and discovery_s is not None:
        inclusive_s = float(theta_t) - discovery_s - float(aco_t)
    if inclusive_pct is None and inclusive_s is not None and float(theta_t) > 0:
        inclusive_pct = 100.0 * float(inclusive_s) / float(theta_t)

    return {
        "beat_heuristic": True,
        "theta_wall_time_s": float(theta_t),
        "aco_seed_wall_time_s": float(aco_t),
        "time_reduction_s": float(reduction_s),
        "time_reduction_pct": float(reduction_pct),
        "aco_discovery_s": discovery_s,
        "inclusive_reduction_s": (
            float(inclusive_s) if inclusive_s is not None else None
        ),
        "inclusive_reduction_pct": (
            float(inclusive_pct) if inclusive_pct is not None else None
        ),
        "aco_time_to_best_s": trial.get("time_to_best_s"),
        "aco_final_edges": trial.get("final_edges"),
        "heuristic_edges": trial.get("heuristic_edges"),
        "ants": trial.get("ants"),
        "run": trial.get("run"),
        "aco_mean_wct_s": mean_aco_wct,
    }


def summarize_from_vary_only(data):
    """
    Build a seed-compare row from vary.jl alone (no compare-seeds JSON).

    Used for datasets skipped by compare-seeds.jl (no ACO beat) and for
    datasets where ACO beat θ but compare-seeds has not been run yet.
    """
    trials = data.get("trials") or []
    if not trials:
        return None

    mean_aco = aco_mean_wct_s(trials)
    best_beat = select_best_beating_trial(trials)

    if best_beat is not None:
        discovery_s = aco_discovery_cost(trials, best_beat, until_found=True)
        # Pivot not timed yet: discovery is sunk cost with unknown benefit.
        inclusive_s = -float(discovery_s) if discovery_s is not None else None
        return {
            "beat_heuristic": True,
            "theta_wall_time_s": None,
            "aco_seed_wall_time_s": None,
            "time_reduction_s": None,
            "time_reduction_pct": None,
            "aco_discovery_s": discovery_s,
            "inclusive_reduction_s": inclusive_s,
            "inclusive_reduction_pct": None,
            "aco_time_to_best_s": best_beat.get("time_to_best_s"),
            "aco_final_edges": best_beat.get("final_edges"),
            "heuristic_edges": (data.get("heuristic") or {}).get("final_edges"),
            "ants": best_beat.get("ants"),
            "run": best_beat.get("run"),
            "aco_mean_wct_s": mean_aco,
        }

    return summarize_no_beat_vary(data)


def summarize_no_beat_vary(data):
    """
    For a vary.jl file where ACO never beat θ: discovery cost is a pure
    time increase (no pivot benefit). Returns None if any trial beat θ.

    Cost is the sum of wall_time_s over every ACO trial (all ant counts) —
    the full search that failed to improve on θ.
    """
    trials = data.get("trials") or []
    if not trials:
        return None
    if select_best_beating_trial(trials) is not None:
        return None

    best = select_best_trial_any(trials)
    total = 0.0
    any_time = False
    for t in trials:
        wt = t.get("wall_time_s")
        if wt is None:
            continue
        total += float(wt)
        any_time = True
    if not any_time:
        return None

    return {
        "beat_heuristic": False,
        "theta_wall_time_s": None,
        "aco_seed_wall_time_s": None,
        "time_reduction_s": None,
        "time_reduction_pct": 0.0,
        "aco_discovery_s": total,
        # No pivot speedup: entire ACO search is overhead.
        "inclusive_reduction_s": -total,
        "inclusive_reduction_pct": None,
        "aco_time_to_best_s": None if best is None else best.get("time_to_best_s"),
        "aco_final_edges": None if best is None else best.get("final_edges"),
        "heuristic_edges": (data.get("heuristic") or {}).get("final_edges"),
        "ants": None if best is None else best.get("ants"),
        "run": None if best is None else best.get("run"),
        "aco_mean_wct_s": aco_mean_wct_s(trials),
    }


def fmt_int(value):
    if value is None:
        return "--"
    return str(int(value))


def fmt_nodes(nU, nV, total=None):
    """Format |U|+|V| when sides are known, else a precomputed total, else --."""
    if nU is not None and nV is not None:
        return str(int(nU) + int(nV))
    if total is not None:
        return str(int(total))
    return "--"


def fmt_saved_s(row):
    """Pivot time saved (s), or -- when ACO did not beat θ."""
    if not row.get("beat_heuristic"):
        return "--"
    value = row.get("time_reduction_s")
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def fmt_pct(row):
    """Pivot percent reduction, or -- when ACO did not beat θ."""
    if not row.get("beat_heuristic"):
        return "--"
    value = row.get("time_reduction_pct")
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def _tex_escape_label(label):
    return (
        str(label)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
    )


def _symbolic_coord(label):
    """pgfplots symbolic x coord: hyphenated, no braces/commas."""
    return str(label).replace(" ", "-").replace("_", "-").replace(",", "")


def build_seed_compare_table(rows):
    """
    LaTeX table: every dataset with graph/reduced sizes and pivot savings.

    Pivot time saved and percent reduction are "--" when ACO did not beat θ.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Pivot seed comparison: ACO vs.\ $\theta$-heuristic}",
        r"  \label{tab:seed-compare}",
        r"  \setlength{\tabcolsep}{4pt}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Dataset & Nodes & Edges & Reduced nodes & Reduced edges"
        r" & Time saved (s) & Reduction (\%) \\",
        r"    \midrule",
    ]
    for r in rows:
        cells = [
            _tex_escape_label(r.get("display_name") or r.get("name") or ""),
            fmt_nodes(r.get("nU"), r.get("nV"), r.get("full_vertex_count")),
            fmt_int(r.get("graph_edges")),
            fmt_nodes(r.get("reduced_nU"), r.get("reduced_nV"), r.get("vertex_count")),
            fmt_int(r.get("reduced_edges")),
            fmt_saved_s(r),
            fmt_pct(r),
        ]
        lines.append("    " + " & ".join(cells) + r" \\")
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def build_seed_compare_bars(rows):
    """
    Bar charts for every dataset: pivot time saved (s) and percent reduction.

    No-beat rows are omitted from the y coordinates (no bar) but remain on
    the symbolic x-axis so every example is visible.
    """
    if not rows:
        return ""

    symbols = [_symbolic_coord(r.get("display_name") or r.get("name") or f"d{i}")
               for i, r in enumerate(rows)]
    # Ensure uniqueness for pgfplots symbolic coords.
    seen = {}
    unique_symbols = []
    for s in symbols:
        n = seen.get(s, 0)
        seen[s] = n + 1
        unique_symbols.append(s if n == 0 else f"{s}-{n + 1}")

    time_coords = []
    pct_coords = []
    for r, sym in zip(rows, unique_symbols):
        if not r.get("beat_heuristic"):
            continue
        ts = r.get("time_reduction_s")
        pct = r.get("time_reduction_pct")
        if ts is not None:
            time_coords.append(f"({sym},{float(ts):.6f})")
        if pct is not None:
            pct_coords.append(f"({sym},{float(pct):.6f})")

    sym_list = ",".join(unique_symbols)
    n = len(rows)
    # Widen a bit when there are many datasets so labels remain readable.
    width = "0.95\\textwidth" if n <= 12 else "1.15\\textwidth"

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=70pt},",
        rf"    width={width},",
        r"    height=0.38\textwidth,",
        r"    ybar,",
        r"    bar width=6pt,",
        r"    grid=major,",
        rf"    symbolic x coords={{{sym_list}}},",
        # Explicit xtick so no-beat datasets still appear (empty slot).
        rf"    xtick={{{sym_list}}},",
        r"    x tick label style={rotate=60, anchor=east, font=\scriptsize},",
        r"    yticklabel style={font=\small},",
        r"    enlarge x limits=0.06,",
        r"]",

        r"\nextgroupplot[",
        r"    ylabel={Pivot time saved (s)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Pivot time saved per dataset "
        r"(no bar $=$ ACO did not beat $\theta$)},",
        r"    title style={font=\small},",
        r"]",
    ]
    if time_coords:
        lines.append(
            r"\addplot+[fill=black!55, draw=black] coordinates {"
            + " ".join(time_coords)
            + r"};"
        )

    lines += [
        r"\nextgroupplot[",
        r"    ylabel={Pivot time savings (\%)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Percent pivot reduction per dataset},",
        r"    title style={font=\small},",
        r"]",
    ]
    if pct_coords:
        lines.append(
            r"\addplot+[fill=black!35, draw=black] coordinates {"
            + " ".join(pct_coords)
            + r"};"
        )

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
    ]
    return "\n".join(lines)


def build_seed_compare_latex(rows):
    """
    Build seed-compare LaTeX: per-example table + per-example bar charts.
    """
    if not rows:
        raise ValueError("No seed-compare rows to plot")

    # Stable order: fewer edges first (matches table mode), then name.
    ordered = sorted(
        rows,
        key=lambda r: (
            r.get("graph_edges") is None,
            r.get("graph_edges") or 0,
            r.get("display_name") or r.get("name") or "",
        ),
    )

    n_total = len(ordered)
    n_fail = sum(1 for r in ordered if not r.get("beat_heuristic"))
    note = (
        rf"We examined ${n_total}$ graphs. On ${n_fail}$ of them, ACO did "
        r"not beat the $\theta$-heuristic and therefore did not yield a "
        r"seed that could reduce pivot time (shown as ``--'' / missing bars)."
    )

    parts = [
        build_seed_compare_table(ordered),
        "",
        note,
        "",
        build_seed_compare_bars(ordered),
    ]
    return "\n".join(parts)


def run(json_paths, output, vary_dir=None):
    rows = []
    skipped = []
    seen_names = set()

    vary_by_leaf = {}
    if vary_dir:
        for vpath in sorted(glob.glob(os.path.join(vary_dir, "*.json"))):
            vdata = load_json(vpath)
            if vdata is None:
                continue
            leaf = os.path.splitext(os.path.basename(vpath))[0]
            if leaf.endswith("_ants"):
                leaf = leaf[: -len("_ants")]
            vary_by_leaf[leaf] = (vpath, vdata)

    for path in json_paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue

        leaf = os.path.splitext(os.path.basename(path))[0]
        vary_data = load_vary_for_compare(data, path, vary_by_leaf=vary_by_leaf)

        summary = summarize_seed_compare(
            data, compare_path=path, vary_data=vary_data
        )
        if summary is None:
            skipped.append((path, "not a compare-seeds result (or missing timings)"))
            continue

        attach_size_metrics(summary, data=data, vary_data=vary_data)
        name = series_name(path, data)
        rows.append({
            "name": name,
            "display_name": display_name(path, data),
            **summary,
        })
        seen_names.add(leaf)
        seen_names.add(name)

    # Every vary dataset: no-beat overhead, or beat without compare JSON yet.
    if vary_dir:
        for leaf, (vpath, vdata) in sorted(vary_by_leaf.items()):
            name = series_name(vpath, vdata)
            if leaf in seen_names or name in seen_names:
                continue
            summary = summarize_from_vary_only(vdata)
            if summary is None:
                continue
            attach_size_metrics(summary, data=vdata, vary_data=vdata)
            rows.append({
                "name": name,
                "display_name": display_name(vpath, vdata),
                **summary,
            })
            seen_names.add(leaf)

    if not rows:
        raise SystemExit(
            "No compare-seeds JSON files with pivot timings -- nothing to plot."
        )

    tex = build_seed_compare_latex(rows)
    write_tex(tex, output)

    beat = [r for r in rows if r.get("beat_heuristic")]
    nobeat = [r for r in rows if not r.get("beat_heuristic")]
    parts = [f"# seed-compare: {len(rows)} dataset(s)"]
    if beat:
        with_nodes = sum(1 for r in beat if r.get("full_vertex_count") is not None)
        with_edges = sum(1 for r in beat if r.get("graph_edges") is not None)
        with_reduced = sum(1 for r in beat if r.get("vertex_count") is not None)
        with_pct = sum(1 for r in beat if r.get("time_reduction_pct") is not None)
        parts.append(
            f"{len(beat)} ACO beat θ ({with_nodes} with nodes, "
            f"{with_edges} with |E|, {with_reduced} with reduced nodes, "
            f"{with_pct} with pivot % savings)"
        )
        compared = [
            r["time_reduction_pct"]
            for r in beat
            if r.get("time_reduction_pct") is not None
        ]
        if compared:
            parts.append(
                f"mean pivot-only reduction = {statistics.mean(compared):.2f}%"
            )
    if nobeat:
        waste = statistics.mean(r["aco_discovery_s"] for r in nobeat)
        parts.append(
            f"{len(nobeat)} no-beat (mean ACO overhead = {waste:.4f}s)"
        )
    print("; ".join(parts), file=sys.stderr)
    report_skipped(skipped)


def resolve_vary_dir(directory, vary_dir=None):
    """Resolve --vary-dir, falling back to compare_* → vary_* inference."""
    resolved = vary_dir or infer_vary_dir(directory)
    if resolved and not os.path.isdir(resolved):
        print(
            f"# Warning: --vary-dir not found: {resolved}",
            file=sys.stderr,
        )
        return None
    if resolved:
        print(f"# Using vary dir: {resolved}", file=sys.stderr)
    return resolved


