"""
heuristics mode — one comparison table per $(k, \\theta)$ sweep.

A directory of ``k{k}t{theta}.json`` files (``results/method_sweep``) emits
one full-width table per file. Each row is one graph. Double-greedy,
clique-order-edge, and any other method in that file come from the file.
The $\\theta$-heuristic and ACO come from the matching ant-count suite
(``--param-dir=k2t5=vary_k2t5i_PN``, or a sibling ``vary_k{k}t{theta}i_PN``).
ACO is the best counted (non-JIT) trial at ``--ants``, by edge count.
That column's time is the sum of those counted replicates at the same ant
count, including every counted replicate even when ``--max-replicates``
caps the other figures.

A single compare-methods JSON still emits one table. ``--vary-dir`` fills
any ACO or $\\theta$-heuristic column that file does not already contain.

Each caption is only the $(k, \\theta)$ setting. Bold marks the largest
$\\theta$-feasible, $k$-valid edge count on that graph; a tie bolds every
method that matches. Wall time is not bold. Missing fields become ``--``
and a stderr warning. This module only reads the JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import (
    aco_timed_out,
    counted_trials,
    list_json_paths,
    load_json,
    series_name,
    write_tex,
)
from .table import select_best_trial

_LABELS = {
    "heuristic": r"$\theta$-heuristic",
    "aco": "ACO",
    "clique-order-edge": "Clique-order-edge",
    "double-greedy": "Double-greedy",
}

# Same columns on every sweep table, then any extra method in the file.
_COLUMN_ORDER = (
    "heuristic",
    "aco",
    "double-greedy",
    "clique-order-edge",
)


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


def _dataset_leaf(label):
    return str(label).replace("\\", "/").split("/")[-1].removesuffix("_ants")


def _as_int(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    return None


def _k_valid(missing, k):
    if missing is None or k is None:
        return True
    try:
        return int(missing) <= int(k)
    except (TypeError, ValueError):
        return True


def _setting_caption(k, theta):
    k_i = _as_int(k)
    theta_i = _as_int(theta)
    if k_i is None or theta_i is None:
        return r"Varied $k$ and $\theta$"
    return rf"$k = {k_i}$, $\theta = {theta_i}$"


def _setting_label(k, theta):
    k_i = _as_int(k)
    theta_i = _as_int(theta)
    if k_i is None or theta_i is None:
        return "tab:heuristics"
    return f"tab:heuristics-k{k_i}t{theta_i}"


def _kt_key(path, data):
    k = _as_int(data.get("k")) if isinstance(data, dict) else None
    theta = _as_int(data.get("theta")) if isinstance(data, dict) else None
    name = Path(path).name
    if k is None or theta is None:
        return (10**9, 10**9, name)
    return (k, theta, name)


def _heuristic_run(data):
    heur = data.get("heuristic") or {}
    if not isinstance(heur, dict) or heur.get("final_edges") is None:
        return None
    return {
        "final_edges": heur.get("final_edges"),
        "nU": heur.get("nU"),
        "nV": heur.get("nV"),
        "wall_time_s": heur.get("wall_time_s"),
        "theta_feasible": bool(heur.get("theta_feasible")),
        "k_valid": _k_valid(heur.get("missing"), data.get("k")),
        "status": "ok",
    }


def _aco_run(data, ants):
    """Best counted trial, or None when ACO timed out or has no trial."""
    if aco_timed_out(data):
        return None
    trials = counted_trials(
        data.get("trials") or [], data, max_replicates=None
    )
    best = select_best_trial(trials, ants=ants)
    if best is None:
        return None
    ants_used = best.get("ants")
    total = 0.0
    any_time = False
    for trial in trials:
        if trial.get("ants") != ants_used:
            continue
        seconds = trial.get("wall_time_s")
        if seconds is None:
            continue
        total += float(seconds)
        any_time = True
    return {
        "final_edges": best.get("final_edges"),
        "nU": best.get("nU"),
        "nV": best.get("nV"),
        "wall_time_s": total if any_time else None,
        "theta_feasible": bool(best.get("theta_feasible")),
        "k_valid": _k_valid(best.get("missing"), data.get("k")),
        "status": "ok",
    }


def load_vary_runs(vary_dir, ants):
    """
    Map a graph leaf to ACO and $\\theta$-heuristic runs.

    ``aco`` is None when that file timed out or has no counted trial.
    The summed ACO wall time uses every counted replicate, not the
    process-wide ``max_replicates`` cap.
    """
    by_leaf = {}
    if not vary_dir:
        return by_leaf
    directory = Path(vary_dir)
    if not directory.is_dir():
        print(
            f"# heuristics table: ACO directory not found: {directory}",
            file=sys.stderr,
        )
        return by_leaf
    for path in list_json_paths(directory):
        data = load_json(path)
        if data is None:
            print(f"# heuristics table: could not read {path}", file=sys.stderr)
            continue
        leaf = _dataset_leaf(data.get("dataset") or Path(path).stem)
        aco = _aco_run(data, ants)
        if aco_timed_out(data):
            print(f"# heuristics table: ACO timed out on {leaf}", file=sys.stderr)
        elif aco is None:
            print(
                f"# heuristics table: no ACO trial for {leaf}"
                + (f" at {ants} ants" if ants is not None else ""),
                file=sys.stderr,
            )
        by_leaf[leaf] = {
            "aco": aco,
            "heuristic": _heuristic_run(data),
            "source_edges": data.get("edge_count"),
        }
    return by_leaf


def load_aco_runs(vary_dir, ants):
    """Map a graph leaf to an ACO run dict. ``None`` means a timeout."""
    runs = {}
    for leaf, pair in load_vary_runs(vary_dir, ants).items():
        runs[leaf] = pair.get("aco")
    return runs


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


def _column_methods(found, vary_runs):
    """θ-heuristic and ACO, then the methods recorded in the sweep file."""
    present = list(found)
    if vary_runs is not None:
        for method_id in ("heuristic", "aco"):
            if method_id not in present:
                present.append(method_id)
    ordered = [method_id for method_id in _COLUMN_ORDER if method_id in present]
    ordered.extend(method_id for method_id in present if method_id not in ordered)
    return ordered


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


def _fill_from_vary(runs, row, vary_runs, methods, missing):
    """Copy ACO and the θ-heuristic onto a row that does not already have them."""
    if vary_runs is None:
        return 0
    if not vary_runs:
        for method_id in ("heuristic", "aco"):
            if method_id in methods and method_id not in runs:
                runs[method_id] = {"status": "error"}
        return 0
    leaf = _dataset_leaf(row["dataset"])
    pair = vary_runs.get(leaf)
    mismatches = 0
    if pair is None:
        for method_id in ("heuristic", "aco"):
            if method_id in methods and method_id not in runs:
                missing[method_id].append(leaf)
                runs[method_id] = {"status": "error"}
        return 0
    source_edges = pair.get("source_edges")
    row_edges = row.get("edge_count")
    checked = False
    for method_id in ("heuristic", "aco"):
        if method_id not in methods or method_id in runs:
            continue
        run = pair.get(method_id)
        if run is None:
            missing[method_id].append(leaf)
            runs[method_id] = {"status": "error"}
            continue
        runs[method_id] = run
        if not checked and source_edges is not None and row_edges is not None:
            checked = True
            if int(source_edges) != int(row_edges):
                mismatches = 1
    return mismatches


def render(data, vary_runs=None, ants=None, *, caption=None, label="tab:heuristics"):
    """LaTeX for every recorded graph in one multi-method compare file."""
    if not isinstance(data, dict):
        print("# heuristics table: expected a compare-methods object", file=sys.stderr)
        data = {}
    methods = _column_methods(_methods(data), vary_runs)
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
    missing = {"aco": [], "heuristic": []}
    density_mismatch = 0
    body = []
    if caption is None:
        caption = _setting_caption(data.get("k"), data.get("theta"))
    for row in rows:
        if not isinstance(row, dict) or "dataset" not in row:
            continue
        name = series_name("", row).replace("_", r"\_")
        runs = dict(row.get("runs") if isinstance(row.get("runs"), dict) else {})
        density_mismatch += _fill_from_vary(runs, row, vary_runs, methods, missing)
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

    tag = caption.replace("$", "").replace("\\", "")
    if warned:
        fields = ", ".join(sorted(warned))
        print(
            f"# heuristics table ({tag}): missing {fields} on at least one graph",
            file=sys.stderr,
        )
    if missing["aco"]:
        print(
            f"# heuristics table ({tag}): no ACO trial for "
            + ", ".join(sorted(set(missing["aco"]))),
            file=sys.stderr,
        )
    if missing["heuristic"]:
        print(
            f"# heuristics table ({tag}): no θ-heuristic for "
            + ", ".join(sorted(set(missing["heuristic"]))),
            file=sys.stderr,
        )
    if density_mismatch:
        print(
            f"# heuristics table ({tag}): graph size differs on "
            f"{density_mismatch} graph(s) between the vary suite and "
            "the comparison file.",
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
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
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


def _vary_dir_for(path, data, param_dirs):
    stem = Path(path).stem
    if param_dirs and stem in param_dirs:
        return param_dirs[stem]
    k = _as_int(data.get("k")) if isinstance(data, dict) else None
    theta = _as_int(data.get("theta")) if isinstance(data, dict) else None
    if k is None or theta is None:
        return None
    candidate = Path(path).resolve().parent.parent / f"vary_k{k}t{theta}i_PN"
    if candidate.is_dir():
        return candidate
    return None


def render_directory(directory, ants=None, param_dirs=None):
    """One table per ``k{k}t{theta}.json`` in ``directory``, ordered by $(k, \\theta)$."""
    paths = list_json_paths(directory)
    loaded = []
    for path in paths:
        data = load_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            print(
                f"# heuristics table: skipping {path} (not a compare-methods file)",
                file=sys.stderr,
            )
            continue
        loaded.append((path, data))
    loaded.sort(key=lambda item: _kt_key(item[0], item[1]))
    if not loaded:
        print(f"# heuristics table: no sweep files in {directory}", file=sys.stderr)
        return (
            r"\begin{center}" "\n"
            r"\scriptsize Construction sizes are not available." "\n"
            r"\end{center}"
        )
    parts = []
    for path, data in loaded:
        vary_dir = _vary_dir_for(path, data, param_dirs)
        if vary_dir is None:
            print(
                f"# heuristics table: no vary suite for {Path(path).name}",
                file=sys.stderr,
            )
            vary_runs = {}
        else:
            vary_runs = load_vary_runs(vary_dir, ants)
        k = data.get("k")
        theta = data.get("theta")
        parts.append(
            render(
                data,
                vary_runs,
                ants=ants,
                caption=_setting_caption(k, theta),
                label=_setting_label(k, theta),
            )
        )
        print(
            f"# heuristics table: {Path(path).name} "
            f"({len(data.get('results') or [])} graph(s))",
            file=sys.stderr,
        )
    return "\n\n".join(parts)


def run(path, output, vary_dir=None, ants=None, param_dirs=None):
    path = Path(path)
    if path.is_dir():
        write_tex(
            render_directory(path, ants=ants, param_dirs=param_dirs),
            output,
        )
        return
    data = load_json(path)
    vary_runs = load_vary_runs(vary_dir, ants) if vary_dir else None
    if data is None:
        print(f"# heuristics table: could not read {path}", file=sys.stderr)
        write_tex(render({}, vary_runs, ants=ants), output)
        return
    k = data.get("k")
    theta = data.get("theta")
    write_tex(
        render(
            data,
            vary_runs,
            ants=ants,
            caption=_setting_caption(k, theta),
            label=_setting_label(k, theta),
        ),
        output,
    )
