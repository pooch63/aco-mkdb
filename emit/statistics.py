"""
statistics mode — vary.jl ant-count JSON → inline LaTeX statistics.

Emits short text fragments for %%STATISTICS:field%% placeholders in
main.tex (win/loss counts, cross-run edge-count variability, Wilcoxon test,
missing-edge counts at a fixed subgraph size during construction).

Fields (pass via --field=… or placeholder args):
  aco-wins, heur-wins, ties, n-graphs — integer counts
  variance — sentence on min/max/std of |E(D*)| across replicates
  wilcoxon — sentence on paired Wilcoxon signed-rank test vs θ-heuristic
  missing-at-5 — full sentence comparing ACO-N vs plain ACO at |S|=5
  aco-missing-at-5, aco-n-missing-at-5 — numeric means only
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

from .common import list_json_paths, load_json, report_skipped, write_tex
from .table import SECTION_ACO, SECTION_HEUR, SECTION_TIE, compare_section, summarize_file

FIELDS = (
    "aco-wins",
    "heur-wins",
    "ties",
    "n-graphs",
    "variance",
    "wilcoxon",
    "missing-at-5",
    "aco-missing-at-5",
    "aco-n-missing-at-5",
)

MISSING_AT_SIZE_FIELDS = frozenset(
    {"missing-at-5", "aco-missing-at-5", "aco-n-missing-at-5"}
)


def _trials_at_ants(trials, ants):
    return [
        t
        for t in trials
        if t.get("final_edges") is not None and t.get("ants") == ants
    ]


def collect_outcomes(json_paths, ants=None):
    """Per-graph best-trial outcome vs the θ-heuristic."""
    rows = []
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
        section = compare_section(row)
        aco_edges = row.get("aco_edges")
        heur_edges = row.get("heur_edges")
        rows.append(
            {
                "section": section,
                "aco_edges": None if aco_edges is None else int(aco_edges),
                "heur_edges": None if heur_edges is None else int(heur_edges),
                "trials": _trials_at_ants(data.get("trials") or [], ants),
            }
        )

    return rows, skipped


def count_outcomes(rows):
    counts = {SECTION_ACO: 0, SECTION_HEUR: 0, SECTION_TIE: 0}
    for row in rows:
        counts[row["section"]] += 1
    return counts


def edge_count_spreads(rows):
    """Within-graph std/min/max of final_edges across replicates."""
    stds, mins, maxs = [], [], []
    for row in rows:
        edges = [int(t["final_edges"]) for t in row["trials"]]
        if not edges:
            continue
        mins.append(min(edges))
        maxs.append(max(edges))
        if len(edges) >= 2:
            stds.append(statistics.pstdev(edges))
        else:
            stds.append(0.0)
    return stds, mins, maxs


def wilcoxon_signed_rank(differences):
    """Two-sided Wilcoxon signed-rank test on paired differences."""
    diffs = [float(d) for d in differences if d != 0]
    n = len(diffs)
    if n == 0:
        return None, None, None

    ranked = sorted((abs(d), i, d) for i, d in enumerate(diffs))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[ranked[k][1]] = avg_rank
        i = j

    w_plus = sum(r for r, d in zip(ranks, diffs) if d > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return w_plus, 0.0, 1.0
    z = (w_plus - mean_w) / math.sqrt(var_w)
    p = math.erfc(abs(z) / math.sqrt(2.0))
    return w_plus, z, p


def fmt_int(value):
    return str(int(value))


def fmt_num(value, digits=1):
    return f"{float(value):.{digits}f}"


def fmt_missing(value):
    return f"{float(value):.2f}"


def fmt_p_value(p):
    if p is None:
        return "--"
    if p < 0.001:
        return r"$p < 0.001$"
    if p < 0.01:
        return r"$p < 0.01$"
    return rf"$p = {p:.3f}$"


def _parse_julia_mean(stdout):
    for line in stdout.splitlines():
        if line.startswith("MEAN,"):
            value = line.split(",", 1)[1].strip()
            return None if not value else float(value)
    return None


def _newest_mtime(paths):
    latest = 0.0
    for path in paths:
        try:
            latest = max(latest, os.path.getmtime(path))
        except OSError:
            continue
    return latest


def default_missing_at_cache_path():
    return Path(__file__).resolve().parents[1] / "paper" / "generated" / ".missing-at-5-cache.json"


def measure_missing_at_5(vary_base, *, ants=100, target_size=5, cache_path=None):
    """Pooled mean missing at |S|=target_size for base ACO vs ACO-N."""
    if cache_path is None:
        cache_path = default_missing_at_cache_path()
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "missing-at-size.jl"
    aco_dir = os.path.abspath(vary_base.rstrip(os.sep))
    aco_n_dir = aco_dir + "N"
    if not os.path.isdir(aco_dir):
        raise SystemExit(f"Missing ACO vary directory: {aco_dir}")
    if not os.path.isdir(aco_n_dir):
        raise SystemExit(f"Missing ACO-N vary directory: {aco_n_dir}")

    json_paths = list_json_paths(aco_dir) + list_json_paths(aco_n_dir)
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.is_file():
        try:
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            if _newest_mtime(json_paths) <= cache_file.stat().st_mtime:
                return float(cached["aco"]), float(cached["aco_n"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    result_aco = subprocess.run(
        ["julia", str(script), aco_dir, f"--ants={ants}", f"--size={target_size}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    result_n = subprocess.run(
        ["julia", str(script), aco_n_dir, f"--ants={ants}", f"--size={target_size}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    mean_aco = _parse_julia_mean(result_aco.stdout)
    mean_n = _parse_julia_mean(result_n.stdout)
    if mean_aco is None or mean_n is None:
        raise SystemExit("missing-at-5 measurement returned no samples")

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(
                {
                    "aco": mean_aco,
                    "aco_n": mean_n,
                    "ants": ants,
                    "target_size": target_size,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return mean_aco, mean_n


def build_missing_at_5_text(mean_aco, mean_aco_n):
    return (
        "Indeed, we find that with ACO-N, the average number of missing edges "
        "after a subgraph has reached 5 vertices is "
        f"{fmt_missing(mean_aco_n)}, much smaller than the average of "
        f"{fmt_missing(mean_aco)} edges using plain ACO."
    )


def build_variance_text(rows):
    stds, mins, maxs = edge_count_spreads(rows)
    if not stds:
        return "Cross-run edge-count variability was not available."
    return (
        "Across the five replicates per graph, $|E(D^*)|$ had a mean "
        f"within-graph standard deviation of {fmt_num(statistics.mean(stds))} "
        f"(range {fmt_int(min(mins))}--{fmt_int(max(maxs))})."
    )


def build_wilcoxon_text(rows):
    paired = [
        (row["aco_edges"], row["heur_edges"])
        for row in rows
        if row["aco_edges"] is not None and row["heur_edges"] is not None
    ]
    if len(paired) < 2:
        return "A paired Wilcoxon signed-rank test was not available."

    diffs = [aco - heur for aco, heur in paired]
    _w_plus, _z, p = wilcoxon_signed_rank(diffs)
    if p is None:
        return "A paired Wilcoxon signed-rank test was not available."

    direction = "favors ACO" if sum(diffs) > 0 else "does not favor ACO"
    return (
        f"A Wilcoxon signed-rank test on paired $|E(D^*)|$ counts {direction} "
        f"({fmt_p_value(p)})."
    )


def render_field(field, rows, *, vary_base=None, ants=None, cache_path=None):
    counts = count_outcomes(rows)
    if field == "aco-wins":
        return fmt_int(counts[SECTION_ACO])
    if field == "heur-wins":
        return fmt_int(counts[SECTION_HEUR])
    if field == "ties":
        return fmt_int(counts[SECTION_TIE])
    if field == "n-graphs":
        return fmt_int(len(rows))
    if field == "variance":
        return build_variance_text(rows)
    if field == "wilcoxon":
        return build_wilcoxon_text(rows)
    if field in MISSING_AT_SIZE_FIELDS:
        if vary_base is None:
            raise SystemExit(
                f"statistics field {field!r} requires vary_base in build.json"
            )
        mean_aco, mean_aco_n = measure_missing_at_5(
            vary_base, ants=ants or 100, cache_path=cache_path
        )
        if field == "aco-missing-at-5":
            return fmt_missing(mean_aco)
        if field == "aco-n-missing-at-5":
            return fmt_missing(mean_aco_n)
        return build_missing_at_5_text(mean_aco, mean_aco_n)
    raise ValueError(f"Unknown statistics field {field!r}")


def run(json_paths, output, ants=None, field=None, vary_base=None, cache_path=None):
    if field is None:
        raise SystemExit("statistics mode requires --field=…")

    if field not in FIELDS:
        raise SystemExit(
            f"Unknown statistics field {field!r}; choose from: {', '.join(FIELDS)}"
        )

    rows, skipped = collect_outcomes(json_paths, ants=ants)
    if field not in MISSING_AT_SIZE_FIELDS and not rows:
        raise SystemExit("No usable vary JSON files -- nothing to summarize.")

    tex = render_field(
        field,
        rows,
        vary_base=vary_base,
        ants=ants,
        cache_path=cache_path,
    )
    write_tex(tex, output)

    if rows:
        counts = count_outcomes(rows)
        print(
            f"# statistics [{field}]: {len(rows)} graph(s)"
            f" (aco={counts[SECTION_ACO]}, heur={counts[SECTION_HEUR]}, "
            f"tie={counts[SECTION_TIE]})"
            + (f"; ants={ants}" if ants is not None else ""),
            file=sys.stderr,
        )
    else:
        print(f"# statistics [{field}]: missing-at-5", file=sys.stderr)
    report_skipped(skipped)
