"""
seed-compare mode — compare-seeds.jl (+ vary.jl) → paper figures:

  1. Scatter (ACO beat θ only): reduced vertex count vs ACO edges saved
  2. Scatter (ACO beat θ only): ACO edges saved vs % pivot time savings
  3. Short note: N graphs examined; M where ACO did not beat θ
  4. Scatter (no-beat only): reduced vertex count vs ACO search cost

Reads full pivot comparisons and beat_heuristic=false skip markers from
compare-seeds.jl. Pass --vary-dir (or rely on compare_* → vary_* name
mapping) for reduced graph sizes and datasets that still lack a compare JSON.
"""

from __future__ import annotations

import glob
import os
import statistics
import sys

from .common import (
    infer_vary_dir,
    load_json,
    report_skipped,
    resolve_path,
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


def select_best_trial_any(trials):
    """Best trial by edges even if it did not beat the heuristic."""
    usable = [t for t in trials if t.get("final_edges") is not None]
    if not usable:
        return None
    best_edges = max(int(t["final_edges"]) for t in usable)
    tied = [t for t in usable if int(t["final_edges"]) == best_edges]

    def trial_time(t):
        tb = t.get("time_to_best_s")
        if tb is not None:
            return float(tb)
        wt = t.get("wall_time_s")
        return float(wt) if wt is not None else float("inf")

    return min(tied, key=trial_time)


def aco_discovery_cost(trials, winning_trial, until_found=True):
    """
    Wall time spent with the winning ant count until the chosen replicate.

    Sums full wall_time_s of same-ant runs with run < winning.run, then adds
    time_to_best_s of the winning run (fallback: that run's wall_time_s).

    If until_found is False (ACO never beat θ), sums wall_time_s of *all*
    same-ant replicates — the full search budget at that ant count.
    """
    if winning_trial is None:
        return None

    ants = winning_trial.get("ants")
    win_run = winning_trial.get("run")
    if ants is None:
        return None

    same = [t for t in trials if t.get("ants") == ants]
    if not same:
        return None

    if not until_found or win_run is None:
        total = 0.0
        any_time = False
        for t in same:
            wt = t.get("wall_time_s")
            if wt is None:
                continue
            total += float(wt)
            any_time = True
        return total if any_time else None

    win_run = int(win_run)
    total = 0.0
    saw_win = False
    for t in same:
        r = t.get("run")
        if r is None:
            continue
        r = int(r)
        if r < win_run:
            wt = t.get("wall_time_s")
            if wt is not None:
                total += float(wt)
        elif r == win_run:
            saw_win = True
            ttb = t.get("time_to_best_s")
            if ttb is not None:
                total += float(ttb)
            else:
                wt = t.get("wall_time_s")
                if wt is not None:
                    total += float(wt)

    return total if saw_win else None


def aco_mean_wct_s(trials):
    """Mean ACO wall_time_s over all trials (all ant counts / replicates)."""
    times = [
        float(t["wall_time_s"])
        for t in trials
        if t.get("wall_time_s") is not None
    ]
    return statistics.mean(times) if times else None


def reduced_vertex_count(*graphs):
    """
    Reduced graph |U_R| + |V_R| from a vary.jl / compare-seeds graph block.

    Prefers reduced_nU/reduced_nV; falls back to original nU/nV when reduced
    sizes are absent. Returns None if no usable pair is found.
    """
    for g in graphs:
        if not g:
            continue
        ru, rv = g.get("reduced_nU"), g.get("reduced_nV")
        if ru is not None and rv is not None:
            return int(ru) + int(rv)
    for g in graphs:
        if not g:
            continue
        u, v = g.get("nU"), g.get("nV")
        if u is not None and v is not None:
            return int(u) + int(v)
    return None


def edges_saved(aco_edges, heur_edges):
    """Extra biclique edges found by ACO over the θ-heuristic, or None."""
    if aco_edges is None or heur_edges is None:
        return None
    return int(aco_edges) - int(heur_edges)


def attach_size_metrics(summary, data=None, vary_data=None):
    """Add vertex_count and edges_saved onto a summary dict (in place)."""
    vary_graph = (vary_data or {}).get("graph") if vary_data else None
    data_graph = (data or {}).get("graph") if data else None
    summary["vertex_count"] = reduced_vertex_count(vary_graph, data_graph)
    summary["edges_saved"] = edges_saved(
        summary.get("aco_final_edges"),
        summary.get("heuristic_edges"),
    )
    return summary


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


def _scatter_coords(rows, x_key, y_key, *, require_positive_x=False):
    """Build a pgfplots coordinates {...} body from row dicts."""
    parts = []
    for r in rows:
        x, y = r.get(x_key), r.get(y_key)
        if x is None or y is None:
            continue
        x, y = float(x), float(y)
        if require_positive_x and x <= 0:
            continue
        parts.append(f"({x:.6f},{y:.6f})")
    return " ".join(parts)


def build_seed_compare_latex(rows):
    """
    Build seed-compare figures:

      1–2. ACO-beat scatters (vertex size vs edges saved; edges saved vs %
           pivot time savings)
      3.   Note: N examined, M where ACO did not beat θ
      4.   No-beat scatter: vertex size vs ACO search cost
    """
    if not rows:
        raise ValueError("No seed-compare rows to plot")

    beat = [r for r in rows if r.get("beat_heuristic")]
    nobeat = [r for r in rows if not r.get("beat_heuristic")]
    n_total = len(rows)
    n_fail = len(nobeat)

    coords_size_edges = _scatter_coords(
        beat, "vertex_count", "edges_saved", require_positive_x=True
    )
    coords_edges_time = _scatter_coords(beat, "edges_saved", "time_reduction_pct")
    coords_fail_cost = _scatter_coords(
        nobeat, "vertex_count", "aco_discovery_s", require_positive_x=True
    )

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=55pt},",
        r"    title style={yshift=-3pt},",
        r"    width=0.85\textwidth,",
        r"    height=0.38\textwidth,",
        r"    grid=major,",
        r"    only marks,",
        r"    mark=*,",
        r"    x tick label style={font=\small},",
        r"    yticklabel style={font=\small},",
        r"]",

        r"\nextgroupplot[",
        r"    xlabel={Reduced graph vertices $|U_R|+|V_R|$},",
        r"    ylabel={ACO edges saved},",
        r"    xlabel style={font=\small},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={ACO quality gain vs reduced graph size},",
        r"    title style={font=\small},",
        r"    xmode=log,",
        r"]",
    ]

    if coords_size_edges:
        lines.append(
            r"\addplot+[only marks, mark=*] coordinates {"
            + coords_size_edges
            + r"};"
        )

    lines += [
        r"\nextgroupplot[",
        r"    xlabel={ACO edges saved},",
        r"    ylabel={Pivot time savings (\%)},",
        r"    xlabel style={font=\small},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Pivot speedup vs ACO quality gain},",
        r"    title style={font=\small},",
        r"]",
    ]

    if coords_edges_time:
        lines.append(
            r"\addplot+[only marks, mark=square*] coordinates {"
            + coords_edges_time
            + r"};"
        )

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"",
        (
            rf"We examined ${n_total}$ graphs. On ${n_fail}$ of them, ACO did "
            r"not beat the $\theta$-heuristic and therefore did not yield a "
            r"seed that could reduce pivot time. Even in those unsuccessful "
            r"cases, however, the ACO search cost stays modest:"
        ),
        r"",
        r"\begin{tikzpicture}",
        r"\begin{axis}[",
        r"    width=0.85\textwidth,",
        r"    height=0.38\textwidth,",
        r"    grid=major,",
        r"    only marks,",
        r"    mark=triangle*,",
        r"    xmode=log,",
        r"    xlabel={Reduced graph vertices $|U_R|+|V_R|$},",
        r"    ylabel={ACO search cost (s)},",
        r"    xlabel style={font=\small},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={ACO overhead when it fails to beat $\theta$},",
        r"    title style={font=\small},",
        r"    x tick label style={font=\small},",
        r"    yticklabel style={font=\small},",
        r"]",
    ]

    if coords_fail_cost:
        lines.append(
            r"\addplot+[only marks, mark=triangle*] coordinates {"
            + coords_fail_cost
            + r"};"
        )

    lines += [
        r"\end{axis}",
        r"\end{tikzpicture}",
    ]

    return "\n".join(lines)


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
        vary_data = None
        if leaf in vary_by_leaf:
            vary_data = vary_by_leaf[leaf][1]

        summary = summarize_seed_compare(
            data, compare_path=path, vary_data=vary_data
        )
        if summary is None:
            skipped.append((path, "not a compare-seeds result (or missing timings)"))
            continue

        attach_size_metrics(summary, data=data, vary_data=vary_data)
        name = series_name(path, data)
        rows.append({"name": name, **summary})
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
            rows.append({"name": name, **summary})
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
        with_edges = sum(1 for r in beat if r.get("edges_saved") is not None)
        with_pct = sum(1 for r in beat if r.get("time_reduction_pct") is not None)
        parts.append(
            f"{len(beat)} ACO beat θ ({with_edges} with edges saved, "
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
