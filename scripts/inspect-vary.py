#!/usr/bin/env python3
"""
Inspect a vary.jl ant-count JSON: θ-heuristic vs ACO (ants=100 by default).

Reports:
  - θ-heuristic result
  - first ants=N trial that beats the heuristic (trial # + full same-ants
    wall-time budget across all replicates)
  - best ants=N trial (trial # + same full same-ants wall-time budget)

Usage:
    python3 scripts/inspect-vary.py results/vary_k2t5i/foo_ants.json
    python3 scripts/inspect-vary.py results/vary_k2t5i
    python3 scripts/inspect-vary.py results/vary_k2t5i --ants=50
    python3 scripts/inspect-vary.py results/vary_k2t5i/foo_ants.json --uv
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

DEFAULT_ANTS = 100


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


def filter_trials(trials, ants):
    """Same-ant trials sorted by run number."""
    same = [t for t in trials if t.get("ants") == ants]
    same.sort(key=lambda t: int(t["run"]) if t.get("run") is not None else 10**18)
    return same


def select_best_trial(trials):
    usable = [t for t in trials if t.get("final_edges") is not None]
    if not usable:
        return None
    return min(usable, key=trial_sort_key)


def select_first_beating(trials):
    """First trial (by run) with beats_heuristic==true."""
    for t in trials:
        if t.get("beats_heuristic") in (True, 1):
            return t
    return None


def discovery_cost(trials, winning_trial):
    """
    Full same-ants wall-time budget (every replicate), matching
    emit.common.aco_discovery_cost / compare-seeds.jl.
    """
    if winning_trial is None:
        return None

    ants = winning_trial.get("ants")
    if ants is None:
        return None

    total = 0.0
    any_time = False
    for t in trials:
        if t.get("ants") != ants:
            continue
        wt = t.get("wall_time_s")
        if wt is None:
            continue
        total += float(wt)
        any_time = True
    return total if any_time else None


def fmt_time(seconds):
    if seconds is None:
        return "--"
    ms = float(seconds) * 1000.0
    if ms < 1000.0:
        return f"{ms:.1f} ms"
    return f"{float(seconds):.4f} s"


def fmt_side(label, block, *, discovery_s=None, n_runs=None):
    if not block:
        return f"  {label}: (missing)"
    edges = block.get("final_edges")
    nU = block.get("nU")
    nV = block.get("nV")
    wall = block.get("wall_time_s")
    parts = [
        f"edges={edges}",
        f"|U|={nU}",
        f"|V|={nV}",
        f"time={fmt_time(wall)}",
    ]
    if block.get("run") is not None:
        run = block["run"]
        if n_runs is not None:
            parts.insert(0, f"trial {run}/{n_runs}")
        else:
            parts.insert(0, f"trial {run}")
    if "theta_feasible" in block and block["theta_feasible"] is not None:
        parts.append(f"θ-feasible={block['theta_feasible']}")
    if "ants" in block and block["ants"] is not None:
        parts.append(f"ants={block['ants']}")
    if "seed" in block and block["seed"] is not None:
        parts.append(f"seed={block['seed']}")
    if "beats_heuristic" in block and block["beats_heuristic"] is not None:
        parts.append(f"beats_heuristic={block['beats_heuristic']}")
    if "time_to_best_s" in block and block["time_to_best_s"] is not None:
        parts.append(f"TTB={fmt_time(block['time_to_best_s'])}")
    if "iterations_to_best" in block and block["iterations_to_best"] is not None:
        parts.append(f"ITB={block['iterations_to_best']}")
    if discovery_s is not None:
        parts.append(f"time run1→here={fmt_time(discovery_s)}")
    return f"  {label}: " + "  ".join(parts)


def print_uv(label, block):
    if not block:
        return
    U = block.get("U")
    V = block.get("V")
    if U is None and V is None:
        return
    print(f"  {label} U = {U}")
    print(f"  {label} V = {V}")


def resolve_paths(path):
    """Return sorted JSON paths from a file or a directory of *.json files."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.json")))
    return None


def inspect_file(path, ants=DEFAULT_ANTS, show_uv=False):
    """Print heuristic vs ACO for one file. Return 0 on success, 1 on error."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"error: cannot read {path}: {e}", file=sys.stderr)
        return 1

    if "trials" not in data and data.get("vary") not in (None, "ant-count"):
        print(
            f"error: {path} does not look like a vary.jl ant-count JSON",
            file=sys.stderr,
        )
        return 1

    dataset = data.get("dataset") or path
    heuristic = data.get("heuristic") or {}
    all_trials = data.get("trials") or []
    trials = filter_trials(all_trials, ants)
    first_beat = select_first_beating(trials)
    best = select_best_trial(trials)
    first_disc = discovery_cost(trials, first_beat)
    best_disc = discovery_cost(trials, best)
    n_runs = len(trials)

    print(f"file: {path}")
    print(f"dataset: {dataset}")
    print(f"k={data.get('k')}  θ={data.get('theta')}  "
          f"ants={ants}  trials={n_runs}/{len(all_trials)}")
    print()
    print(fmt_side("θ-heuristic", heuristic))

    if first_beat is None:
        print(f"  first beat (ants={ants}): (none — no trial beat θ-heuristic)")
    else:
        print(fmt_side(
            f"first beat (ants={ants})",
            first_beat,
            discovery_s=first_disc,
            n_runs=n_runs,
        ))

    print(fmt_side(
        f"best ACO (ants={ants})",
        best,
        discovery_s=best_disc,
        n_runs=n_runs,
    ))

    heur_edges = heuristic.get("final_edges")
    aco_edges = None if best is None else best.get("final_edges")
    if heur_edges is not None and aco_edges is not None:
        delta = int(aco_edges) - int(heur_edges)
        if heur_edges:
            pct = 100.0 * delta / float(heur_edges)
            print(f"\n  best vs heuristic: {delta:+d} edges ({pct:+.1f}%)")
        else:
            print(f"\n  best vs heuristic: {delta:+d} edges")

    if first_beat is not None and first_disc is not None:
        print(f"  first-beat discovery (full ants={ants} budget; win run "
              f"{first_beat.get('run')}): {fmt_time(first_disc)}")
    if best is not None and best_disc is not None:
        print(f"  best-trial discovery (full ants={ants} budget; win run "
              f"{best.get('run')}): {fmt_time(best_disc)}")

    if show_uv:
        print()
        print_uv("θ-heuristic", heuristic)
        print_uv("first beat", first_beat)
        print_uv("best ACO", best)

    if not trials:
        print(
            f"\nwarning: no ACO trials with ants={ants}",
            file=sys.stderr,
        )
        return 1

    if best is None:
        print("\nwarning: no usable ACO trial", file=sys.stderr)
        return 1

    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Show θ-heuristic vs ACO from a vary.jl JSON file "
                    "or every *.json in a directory. Defaults to ants=100.",
    )
    parser.add_argument(
        "path",
        help="Path to a vary.jl *_ants.json file, or a directory of them",
    )
    parser.add_argument(
        "--ants",
        type=int,
        default=DEFAULT_ANTS,
        help=f"Only consider ACO trials with this ant count (default: {DEFAULT_ANTS})",
    )
    parser.add_argument(
        "--uv",
        action="store_true",
        help="Also print the U/V vertex sets for heuristic / first beat / best",
    )
    args = parser.parse_args(argv)

    paths = resolve_paths(args.path)
    if paths is None:
        print(f"error: path not found: {args.path}", file=sys.stderr)
        return 1
    if not paths:
        print(f"error: no .json files in {args.path}", file=sys.stderr)
        return 1

    failures = 0
    for i, path in enumerate(paths):
        if i > 0:
            print("\n" + "=" * 72 + "\n")
        failures += inspect_file(path, ants=args.ants, show_uv=args.uv)

    if len(paths) > 1:
        print(
            f"\n# inspected {len(paths)} file(s)"
            + (f"; {failures} failed" if failures else ""),
            file=sys.stderr,
        )

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
