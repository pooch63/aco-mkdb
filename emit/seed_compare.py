"""
seed-compare mode — compare-seeds.jl output → 1x2 groupplot:

  - signed net time change vs θ-only pivot, including ACO discovery
    cost (negative = time increase; every vary dataset)
  - absolute wall times: θ / θ+ACO pivot when compared, plus ACO
    discovery and mean ACO WCT for every dataset

Reads full pivot comparisons and beat_heuristic=false skip markers
written by compare-seeds.jl. Pass --vary-dir (or rely on compare_* →
vary_* name mapping) for datasets that still lack a compare JSON.
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


def build_seed_compare_latex(rows):
    """
    Build a 1x2 groupplot:

      1. Signed time reduction vs θ-only pivot, including ACO discovery
         (every dataset; negative = time increase)
      2. Absolute wall times: mean ACO WCT (every dataset), θ / θ+ACO
         pivot when compared, and ACO discovery cost when there was no
         pivot benefit (time-increase cases)

    X positions are categorical symbolic coordinates (dataset names).
    """
    if not rows:
        raise ValueError("No seed-compare rows to plot")

    # Stable alphabetical order by display name.
    rows = sorted(rows, key=lambda r: r["name"])
    symbols = ",".join(r["name"] for r in rows)

    # Prefer % when every row has an inclusive % (all have θ pivot times).
    # Otherwise plot seconds so no-beat / uncompared rows still appear.
    use_pct = all(r.get("inclusive_reduction_pct") is not None for r in rows)
    red_key = "inclusive_reduction_pct" if use_pct else "inclusive_reduction_s"

    def _red_val(r):
        return r.get(red_key)

    red_pos = " ".join(
        f"({r['name']},{_red_val(r):.6f})"
        for r in rows
        if _red_val(r) is not None and _red_val(r) >= 0
    )
    red_neg = " ".join(
        f"({r['name']},{_red_val(r):.6f})"
        for r in rows
        if _red_val(r) is not None and _red_val(r) < 0
    )

    if use_pct:
        red_ylabel = r"Time reduction incl.\ ACO (\%)"
        red_title = (
            r"Net speedup vs $\theta$-only pivot "
        )
    else:
        red_ylabel = r"Time reduction incl.\ ACO (s)"
        red_title = (
            r"Net change vs $\theta$-only pivot "
        )

    theta_coords = " ".join(
        f"({r['name']},{r['theta_wall_time_s']:.6f})"
        for r in rows
        if r.get("theta_wall_time_s") is not None
    )
    aco_pivot_coords = " ".join(
        f"({r['name']},{r['aco_seed_wall_time_s']:.6f})"
        for r in rows
        if r.get("aco_seed_wall_time_s") is not None
    )
    aco_mean_coords = " ".join(
        f"({r['name']},{r['aco_mean_wct_s']:.6f})"
        for r in rows
        if r.get("aco_mean_wct_s") is not None
    )
    # Discovery / overhead: every dataset that has it (highlights time-increase
    # cases on the WCT panel even when pivot was never run).
    discovery_coords = " ".join(
        f"({r['name']},{r['aco_discovery_s']:.6f})"
        for r in rows
        if r.get("aco_discovery_s") is not None
    )

    # Explicit tick list so every dataset is labeled on both panels, even
    # when a series only has points for ACO-success / pivot-compared rows.
    xtick_list = ",".join(r["name"] for r in rows)

    lines = [
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"    group style={group size=1 by 2, vertical sep=55pt},",
        r"    title style={yshift=-3pt},",
        r"    width=0.85\textwidth,",
        r"    height=0.38\textwidth,",
        r"    grid=major,",
        r"    symbolic x coords={" + symbols + r"},",
        r"    xtick={" + xtick_list + r"},",
        r"    x tick label style={rotate=45, anchor=east, font=\small},",
        r"    yticklabel style={font=\small},",
        r"]",

        r"\nextgroupplot[",
        rf"    ylabel={{{red_ylabel}}},",
        r"    ylabel style={align=center, font=\small},",
        rf"    title={{{red_title}}},",
        r"    title style={font=\small},",
        r"    legend to name=seedcomparereductionlegend,",
        r"    legend columns=2,",
        r"    legend style={draw=none, fill=white, font=\small},",
        r"]",
    ]

    if red_pos:
        lines.append(
            r"\addplot+[ybar, bar width=8pt, fill=black!40] coordinates {"
            + red_pos
            + r"};"
        )
        lines.append(r"\addlegendentry{time saved}")
    if red_neg:
        lines.append(
            r"\addplot+[ybar, bar width=8pt, fill=black!15] coordinates {"
            + red_neg
            + r"};"
        )
        lines.append(r"\addlegendentry{time increase}")

    lines += [
        r"\nextgroupplot[",
        r"    ylabel={Wall time (s)},",
        r"    ylabel style={align=center, font=\small},",
        r"    title={Pivot WCT and ACO discovery cost},",
        r"    title style={font=\small},",
        r"    ymode=log,",
        r"    legend to name=seedcomparelegend,",
        r"    legend columns=2,",
        r"    legend style={draw=none, fill=white, font=\small},",
        r"]",
    ]

    if theta_coords:
        lines.append(
            r"\addplot+[only marks, mark=*] coordinates {" + theta_coords + r"};"
        )
        lines.append(r"\addlegendentry{$\theta$ pivot}")
    if aco_pivot_coords:
        lines.append(
            r"\addplot+[only marks, mark=square*] coordinates {"
            + aco_pivot_coords
            + r"};"
        )
        lines.append(r"\addlegendentry{$\theta$ + ACO pivot}")
    if discovery_coords:
        lines.append(
            r"\addplot+[only marks, mark=triangle*] coordinates {"
            + discovery_coords
            + r"};"
        )
        lines.append(r"\addlegendentry{ACO discovery}")
    if aco_mean_coords:
        lines.append(
            r"\addplot+[only marks, mark=x] coordinates {" + aco_mean_coords + r"};"
        )
        lines.append(r"\addlegendentry{mean ACO WCT}")

    lines += [
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"",
        r"\begin{center}",
        r"\ref{seedcomparereductionlegend}\qquad\ref{seedcomparelegend}",
        r"\end{center}",
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
        compared = [
            r["time_reduction_pct"]
            for r in beat
            if r.get("time_reduction_pct") is not None
        ]
        if compared:
            parts.append(
                f"{len(compared)} pivot-compared; mean pivot-only reduction = "
                f"{statistics.mean(compared):.2f}%"
            )
        pending = len(beat) - len(compared)
        if pending:
            parts.append(f"{pending} ACO beat without compare JSON")
        incl = [
            r["inclusive_reduction_pct"]
            for r in beat
            if r.get("inclusive_reduction_pct") is not None
        ]
        if incl:
            parts.append(
                f"mean inclusive reduction = {statistics.mean(incl):.2f}%"
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
