"""
statistics mode — vary.jl ant-count JSON → inline LaTeX statistics.

Emits short text fragments for %%STATISTICS:field%% placeholders in
main.tex (win/loss counts, cross-run edge-count variability, Wilcoxon test,
θ-feasibility rates, missing-edge counts at a fixed subgraph size during
construction, and pivot seed-compare timeout coverage among ACO wins).

Fields (pass via --field=… or placeholder args):
  aco-wins, heur-wins, ties, n-graphs, aco-nonwins — integer counts
  variance — mean within-graph std of |E(D*)| across replicates,
    with the min–max range of those per-graph stds
  wilcoxon — sentence on paired Wilcoxon signed-rank test vs θ-heuristic
    (non-θ-feasible solutions scored as 0 edges)
  aco-theta-feasibility-rate — % of counted trials that are θ-feasible
  theta-heuristic-feasibility-rate — % of graphs where the θ-heuristic is
    θ-feasible
  missing-at-5 — full sentence comparing ACO-N vs plain ACO at |S|=5
  aco-missing-at-5, aco-n-missing-at-5 — numeric means only
  pivot-tested-wins, pivot-excluded-wins — ACO-win graphs with / without a
    completed θ vs ACO-seed pivot comparison (compare-seeds JSON; dual
    timeouts and missing compare files count as excluded)
  pivot-tested-wins-mean-nR, pivot-excluded-wins-mean-nR — mean reduced
    |U_R|+|V_R| among those win subsets
  pivot-tested-wins-mean-eR, pivot-excluded-wins-mean-eR — mean |E_R|

missing-at-5 fields read pre-recorded missing_at_size from vary JSON only.
Pivot-coverage fields also read compare-seeds JSON via --compare-dir.
"""

from __future__ import annotations

import math
import os
import statistics
import sys

from .common import (
    aco_timed_out,
    counted_trials,
    list_json_paths,
    load_json,
    report_skipped,
    write_tex,
)
from .result_fields import pool_missing_at_size_mean, validate_missing_at_size_dirs
from .seed_compare import (
    both_pivots_timed_out,
    reduced_edge_count,
    reduced_vertex_count,
)
from .table import (
    SECTION_ACO,
    SECTION_HEUR,
    SECTION_TIE,
    compare_section,
    is_theta_feasible,
    summarize_file,
)


FIELDS = (
    "aco-wins",
    "heur-wins",
    "ties",
    "n-graphs",
    "aco-nonwins",
    "variance",
    "wilcoxon",
    "aco-theta-feasibility-rate",
    "theta-heuristic-feasibility-rate",
    "missing-at-5",
    "aco-missing-at-5",
    "aco-n-missing-at-5",
    "pivot-tested-wins",
    "pivot-excluded-wins",
    "pivot-tested-wins-mean-nR",
    "pivot-excluded-wins-mean-nR",
    "pivot-tested-wins-mean-eR",
    "pivot-excluded-wins-mean-eR",
)

FEASIBILITY_RATE_FIELDS = frozenset(
    {"aco-theta-feasibility-rate", "theta-heuristic-feasibility-rate"}
)

MISSING_AT_SIZE_FIELDS = frozenset(
    {"missing-at-5", "aco-missing-at-5", "aco-n-missing-at-5"}
)

PIVOT_COVERAGE_FIELDS = frozenset(
    {
        "pivot-tested-wins",
        "pivot-excluded-wins",
        "pivot-tested-wins-mean-nR",
        "pivot-excluded-wins-mean-nR",
        "pivot-tested-wins-mean-eR",
        "pivot-excluded-wins-mean-eR",
    }
)


def _trials_at_ants(trials, ants, data=None):
    return [
        t
        for t in counted_trials(trials, data)
        if t.get("final_edges") is not None
        and (ants is None or t.get("ants") == ants)
    ]


def _heuristic_theta_feasible(data, row):
    """Read heuristic.theta_feasible from JSON, else derive from side sizes."""
    heur = data.get("heuristic") or {}
    flag = heur.get("theta_feasible")
    if flag is not None:
        return bool(flag)
    return is_theta_feasible(row.get("heur_nU"), row.get("heur_nV"), row.get("theta"))


def _vary_leaf(path):
    """Basename of a vary JSON path without ``_ants`` / ``.json``."""
    leaf = os.path.splitext(os.path.basename(path))[0]
    if leaf.endswith("_ants"):
        leaf = leaf[: -len("_ants")]
    return leaf


def _reduced_sizes(data):
    """Reduced |U_R|+|V_R| and |E_R| from vary.jl top-level / graph blocks."""
    graph = data.get("graph") if isinstance(data, dict) else None
    n_r = reduced_vertex_count(data, graph)
    e_r = reduced_edge_count(data, graph)
    return n_r, e_r


def collect_outcomes(json_paths, ants=None):
    """Per-graph best-trial outcome vs the θ-heuristic."""
    rows = []
    skipped = []

    for path in json_paths:
        data = load_json(path)
        if data is None:
            skipped.append((path, "unreadable"))
            continue
        if aco_timed_out(data):
            limit = data.get("aco_timeout_s")
            reason = "aco timed out"
            if limit is not None:
                reason = f"aco timed out ({limit}s)"
            skipped.append((path, reason))
            continue
        row = summarize_file(data, ants=ants)
        if row is None:
            skipped.append((path, "not a vary.jl ant-count result"))
            continue
        section = compare_section(row)
        aco_edges = row.get("aco_edges")
        heur_edges = row.get("heur_edges")
        trials = _trials_at_ants(data.get("trials") or [], ants, data)
        aco_ok = is_theta_feasible(row.get("aco_nU"), row.get("aco_nV"), row.get("theta"))
        heur_ok = _heuristic_theta_feasible(data, row)
        n_r, e_r = _reduced_sizes(data)
        rows.append(
            {
                "leaf": _vary_leaf(path),
                "section": section,
                "aco_edges": None if aco_edges is None else int(aco_edges),
                "heur_edges": None if heur_edges is None else int(heur_edges),
                "trials": trials,
                "aco_theta_feasible": aco_ok,
                "heur_theta_feasible": heur_ok,
                "n_R": n_r,
                "e_R": e_r,
            }
        )

    return rows, skipped


def pivot_comparison_completed(compare_data):
    """
    True when compare-seeds recorded both pivot timings without dual timeout.

    Dual-timeout JSON is omitted from the seed-compare table; missing or
    no-beat markers never ran pivots and do not count as completed.
    """
    if not compare_data or compare_data.get("compare") != "seeds":
        return False
    if compare_data.get("beat_heuristic") is False:
        return False
    if both_pivots_timed_out(compare_data):
        return False
    theta = compare_data.get("pivot_theta") or {}
    aco = compare_data.get("pivot_aco_seed") or {}
    return (
        theta.get("wall_time_s") is not None
        and aco.get("wall_time_s") is not None
    )


def index_compare_by_leaf(compare_dir):
    """Map dataset leaf → compare-seeds JSON (or None if unreadable)."""
    if not compare_dir or not os.path.isdir(compare_dir):
        return {}
    indexed = {}
    for path in list_json_paths(compare_dir):
        leaf = os.path.splitext(os.path.basename(path))[0]
        indexed[leaf] = load_json(path)
    return indexed


def split_aco_wins_by_pivot(rows, compare_dir):
    """
    Partition ACO-win rows into pivot-tested vs pivot-excluded.

    Tested = compare-seeds JSON exists with both pivot wall times and not
    dual-timeout. Excluded = every other ACO win (dual timeout, missing
    compare file, or incomplete timings).
    """
    by_leaf = index_compare_by_leaf(compare_dir)
    tested, excluded = [], []
    for row in rows:
        if row.get("section") != SECTION_ACO:
            continue
        compare_data = by_leaf.get(row.get("leaf"))
        if pivot_comparison_completed(compare_data):
            tested.append(row)
        else:
            excluded.append(row)
    return tested, excluded


def mean_int_or_none(values):
    """Rounded mean of numeric values, or None when empty."""
    nums = [float(v) for v in values if v is not None]
    if not nums:
        return None
    return int(round(statistics.mean(nums)))


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
    if value is None:
        return "--"
    return f"{float(value):.2f}"


def fmt_p_value(p):
    if p is None:
        return "--"
    if p < 0.001:
        return r"$p < 0.001$"
    if p < 0.01:
        return r"$p < 0.01$"
    return rf"$p = {p:.3f}$"


def fmt_rate(value, digits=1):
    """Format a percentage for inline LaTeX (e.g. 90.8\\%)."""
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}\\%"


def theta_feasibility_rates(rows):
    """
    Pooled ACO trial θ-feasibility % and per-graph θ-heuristic feasibility %.

    ACO rate matches quality.py: fraction of counted trials (at the
    configured ant count) with theta_feasible == True. Heuristic rate is
    the fraction of graphs whose recorded heuristic solution is θ-feasible.
    """
    aco_ok = aco_tot = 0
    for row in rows:
        for trial in row["trials"]:
            aco_tot += 1
            if bool(trial.get("theta_feasible")):
                aco_ok += 1

    heur_ok = sum(1 for row in rows if row.get("heur_theta_feasible"))
    heur_tot = len(rows)

    aco_rate = None if aco_tot == 0 else 100.0 * aco_ok / aco_tot
    heur_rate = None if heur_tot == 0 else 100.0 * heur_ok / heur_tot
    return aco_rate, heur_rate, aco_ok, aco_tot, heur_ok, heur_tot


def measure_missing_at_5(vary_base, *, ants=100, target_size=5):
    """Pooled mean missing at |S|=target_size for base ACO vs ACO-N (JSON only)."""
    aco_dir = os.path.abspath(vary_base.rstrip(os.sep))
    aco_n_dir = aco_dir + "N"
    if not os.path.isdir(aco_dir):
        print(
            f"Warning: missing ACO vary directory: {aco_dir}",
            file=sys.stderr,
        )
        return None, None
    if not os.path.isdir(aco_n_dir):
        print(
            f"Warning: missing ACO-N vary directory: {aco_n_dir}",
            file=sys.stderr,
        )
        return None, None

    validate_missing_at_size_dirs(
        aco_dir, aco_n_dir, ants=ants, target_size=target_size
    )

    mean_aco, n_aco, total_aco, _ = pool_missing_at_size_mean(
        aco_dir, ants=ants, target_size=target_size
    )
    mean_n, n_n, total_n, _ = pool_missing_at_size_mean(
        aco_n_dir, ants=ants, target_size=target_size
    )
    print(
        f"# statistics: missing-at-{target_size} from JSON "
        f"(ACO {n_aco}/{total_aco}, ACO-N {n_n}/{total_n} graphs)",
        file=sys.stderr,
    )
    return mean_aco, mean_n


def build_missing_at_5_text(mean_aco, mean_aco_n):
    if mean_aco is None or mean_aco_n is None:
        return "Missing-at-size construction statistics were not available."
    return (
        "Indeed, we find that with ACO-N, the average number of missing edges "
        "after a subgraph has reached 5 vertices is "
        f"{fmt_missing(mean_aco_n)}, much smaller than the average of "
        f"{fmt_missing(mean_aco)} edges using plain ACO."
    )


def build_variance_text(rows):
    stds, _mins, _maxs = edge_count_spreads(rows)
    if not stds:
        return "Cross-run edge-count variability was not available."
    return (
        "Across the five replicates per graph, $|E(D^*)|$ had a mean "
        f"within-graph standard deviation of {fmt_num(statistics.mean(stds))} "
        f"(range {fmt_num(min(stds))}--{fmt_num(max(stds))})."
    )


def _wilcoxon_scored_edges(row):
    """
    Paired |E(D*)| for Wilcoxon: non-θ-feasible subgraphs score as 0 edges.

    Matches the table win/loss rule that a side without ≥θ vertices on both
    parts has failed, regardless of raw edge count.
    """
    aco = row.get("aco_edges")
    heur = row.get("heur_edges")
    if aco is None or heur is None:
        return None
    aco_scored = int(aco) if row.get("aco_theta_feasible") else 0
    heur_scored = int(heur) if row.get("heur_theta_feasible") else 0
    return aco_scored, heur_scored


def build_wilcoxon_text(rows):
    paired = []
    for row in rows:
        scored = _wilcoxon_scored_edges(row)
        if scored is not None:
            paired.append(scored)
    if len(paired) < 2:
        return "A paired Wilcoxon signed-rank test was not available."

    diffs = [aco - heur for aco, heur in paired]
    _w_plus, _z, p = wilcoxon_signed_rank(diffs)
    if p is None:
        return "A paired Wilcoxon signed-rank test was not available."

    direction = "favors ACO" if sum(diffs) > 0 else "does not favor ACO"
    return (
        f"A Wilcoxon signed-rank test on paired $|E(D^*)|$ counts "
        f"(scoring non-$\\theta$-feasible solutions as $0$) {direction} "
        f"({fmt_p_value(p)})."
    )


def render_field(field, rows, *, vary_base=None, ants=None, compare_dir=None):
    counts = count_outcomes(rows)
    if field == "aco-wins":
        return fmt_int(counts[SECTION_ACO])
    if field == "heur-wins":
        return fmt_int(counts[SECTION_HEUR])
    if field == "ties":
        return fmt_int(counts[SECTION_TIE])
    if field == "n-graphs":
        return fmt_int(len(rows))
    if field == "aco-nonwins":
        return fmt_int(counts[SECTION_HEUR] + counts[SECTION_TIE])
    if field == "variance":
        return build_variance_text(rows)
    if field == "wilcoxon":
        return build_wilcoxon_text(rows)
    if field in FEASIBILITY_RATE_FIELDS:
        aco_rate, heur_rate, *_ = theta_feasibility_rates(rows)
        if field == "aco-theta-feasibility-rate":
            if aco_rate is None:
                print(
                    "Warning: no counted trials for ACO θ-feasibility rate",
                    file=sys.stderr,
                )
            return fmt_rate(aco_rate)
        if heur_rate is None:
            print(
                "Warning: no graphs for θ-heuristic feasibility rate",
                file=sys.stderr,
            )
        return fmt_rate(heur_rate)
    if field in MISSING_AT_SIZE_FIELDS:
        if vary_base is None:
            print(
                f"Warning: statistics field {field!r} requires vary_base in build.json",
                file=sys.stderr,
            )
            if field == "missing-at-5":
                return "Missing-at-size construction statistics were not available."
            return "--"
        mean_aco, mean_aco_n = measure_missing_at_5(
            vary_base, ants=ants or 100
        )
        if field == "aco-missing-at-5":
            return fmt_missing(mean_aco)
        if field == "aco-n-missing-at-5":
            return fmt_missing(mean_aco_n)
        return build_missing_at_5_text(mean_aco, mean_aco_n)
    if field in PIVOT_COVERAGE_FIELDS:
        if not compare_dir or not os.path.isdir(compare_dir):
            print(
                f"Warning: statistics field {field!r} requires compare_dir "
                f"in build.json (got {compare_dir!r})",
                file=sys.stderr,
            )
            return "--"
        tested, excluded = split_aco_wins_by_pivot(rows, compare_dir)
        if field == "pivot-tested-wins":
            return fmt_int(len(tested))
        if field == "pivot-excluded-wins":
            return fmt_int(len(excluded))
        if field == "pivot-tested-wins-mean-nR":
            mean = mean_int_or_none(r.get("n_R") for r in tested)
            if mean is None:
                print(
                    "Warning: no reduced n_R for pivot-tested ACO wins",
                    file=sys.stderr,
                )
                return "--"
            return fmt_int(mean)
        if field == "pivot-excluded-wins-mean-nR":
            mean = mean_int_or_none(r.get("n_R") for r in excluded)
            if mean is None:
                print(
                    "Warning: no reduced n_R for pivot-excluded ACO wins",
                    file=sys.stderr,
                )
                return "--"
            return fmt_int(mean)
        if field == "pivot-tested-wins-mean-eR":
            mean = mean_int_or_none(r.get("e_R") for r in tested)
            if mean is None:
                print(
                    "Warning: no reduced |E_R| for pivot-tested ACO wins",
                    file=sys.stderr,
                )
                return "--"
            return fmt_int(mean)
        if field == "pivot-excluded-wins-mean-eR":
            mean = mean_int_or_none(r.get("e_R") for r in excluded)
            if mean is None:
                print(
                    "Warning: no reduced |E_R| for pivot-excluded ACO wins",
                    file=sys.stderr,
                )
                return "--"
            return fmt_int(mean)
    raise ValueError(f"Unknown statistics field {field!r}")


def run(
    json_paths,
    output,
    ants=None,
    field=None,
    vary_base=None,
    compare_dir=None,
    cache_path=None,
):
    if field is None:
        raise SystemExit("statistics mode requires --field=…")

    if field not in FIELDS:
        raise SystemExit(
            f"Unknown statistics field {field!r}; choose from: {', '.join(FIELDS)}"
        )

    rows, skipped = collect_outcomes(json_paths, ants=ants)
    if field not in MISSING_AT_SIZE_FIELDS and not rows:
        print(
            "Warning: no usable vary JSON files for statistics.",
            file=sys.stderr,
        )

    tex = render_field(
        field,
        rows,
        vary_base=vary_base,
        ants=ants,
        compare_dir=compare_dir,
    )
    write_tex(tex, output)

    if rows:
        counts = count_outcomes(rows)
        extra = ""
        if field in FEASIBILITY_RATE_FIELDS:
            aco_rate, heur_rate, aco_ok, aco_tot, heur_ok, heur_tot = (
                theta_feasibility_rates(rows)
            )
            extra = (
                f"; aco_feas={aco_ok}/{aco_tot}"
                f" ({fmt_rate(aco_rate) if aco_rate is not None else '--'})"
                f"; heur_feas={heur_ok}/{heur_tot}"
                f" ({fmt_rate(heur_rate) if heur_rate is not None else '--'})"
            )
        if field in PIVOT_COVERAGE_FIELDS and compare_dir and os.path.isdir(
            compare_dir
        ):
            tested, excluded = split_aco_wins_by_pivot(rows, compare_dir)
            extra += (
                f"; pivot_tested={len(tested)}, pivot_excluded={len(excluded)}"
            )
        print(
            f"# statistics [{field}]: {len(rows)} graph(s)"
            f" (aco={counts[SECTION_ACO]}, heur={counts[SECTION_HEUR]}, "
            f"tie={counts[SECTION_TIE]})"
            + (f"; ants={ants}" if ants is not None else "")
            + extra,
            file=sys.stderr,
        )
    else:
        print(f"# statistics [{field}]: no rows", file=sys.stderr)
    report_skipped(skipped)
