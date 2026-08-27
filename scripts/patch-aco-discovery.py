#!/usr/bin/env python3
"""
Rewrite aco_discovery_s / inclusive_reduction_* in compare-seeds JSON files
so discovery charges the full same-ants wall-time budget (every replicate),
not time truncated at the winning run / time-to-best.

Does not re-run pivots. For beat_heuristic=true files, reloads the vary JSON
referenced by aco_trial.source_vary (or re-selects the best beating trial),
sums wall_time_s over all same-ants trials, and refreshes inclusive savings
from the already-recorded pivot wall times.

No-beat skip markers already store the full trial-sum discovery cost; they
are left unchanged unless --force-nobeat is set.

Usage:
    python3 scripts/patch-aco-discovery.py compare_k2t5i_konect-small
    python3 scripts/patch-aco-discovery.py compare_k2t5i_konect-small --dry-run
    python3 scripts/patch-aco-discovery.py path/to/one.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from emit.common import aco_discovery_cost  # noqa: E402
from emit.seed_compare import select_best_beating_trial  # noqa: E402


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    with path.open("w") as f:
        json.dump(data, f, separators=(",", ":"))
        f.write("\n")


def resolve_vary(source: str | None, compare_path: Path) -> Path | None:
    if not source:
        return None
    # Collapse accidental double slashes from older writers.
    cleaned = source.replace("//", "/")
    candidates = [
        Path(cleaned),
        ROOT / cleaned,
        compare_path.parent / cleaned,
        ROOT / Path(cleaned).name,
    ]
    # Also try beside common vary_* dirs if only a basename was stored oddly.
    name = Path(cleaned).name
    for d in ROOT.glob("vary_*"):
        if d.is_dir():
            candidates.append(d / name)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def patch_one(path: Path, *, dry_run: bool, force_nobeat: bool) -> str:
    data = load_json(path)
    if data.get("compare") != "seeds":
        return f"skip {path.name}: not a seed-compare JSON"

    beat = data.get("beat_heuristic")
    old_disc = data.get("aco_discovery_s")

    if beat is False or beat == 0:
        if not force_nobeat:
            return f"skip {path.name}: no-beat (already full-budget)"
        # Recompute total wall sum from source vary if present.
        src = data.get("source_vary")
        vary_path = resolve_vary(src, path)
        if vary_path is None:
            return f"skip {path.name}: no-beat, missing vary ({src!r})"
        vary = load_json(vary_path)
        trials = vary.get("trials") or []
        total = 0.0
        n = 0
        for t in trials:
            wt = t.get("wall_time_s")
            if wt is None:
                continue
            total += float(wt)
            n += 1
        if n == 0:
            return f"skip {path.name}: no timed trials in {vary_path}"
        new_disc = total
        data["aco_discovery_s"] = new_disc
        data["inclusive_reduction_s"] = -new_disc
        if "inclusive_reduction_pct" in data:
            data["inclusive_reduction_pct"] = None
        if not dry_run:
            write_json(path, data)
        return (
            f"{'would update' if dry_run else 'updated'} {path.name}: "
            f"discovery {old_disc} → {new_disc:.6g} (no-beat)"
        )

    trial = data.get("aco_trial") or {}
    src = trial.get("source_vary") or data.get("source_vary")
    vary_path = resolve_vary(src, path)
    if vary_path is None:
        return f"skip {path.name}: missing vary ({src!r})"

    vary = load_json(vary_path)
    trials = vary.get("trials") or []

    # Prefer the trial compare-seeds actually selected (ants/run in aco_trial).
    winning = None
    ants = trial.get("ants")
    run = trial.get("run")
    if ants is not None and run is not None:
        for t in trials:
            if t.get("ants") == ants and t.get("run") == run:
                winning = t
                break
        if winning is None:
            winning = {"ants": ants, "run": run}

    if winning is None:
        winning = select_best_beating_trial(trials)
    if winning is None:
        return f"skip {path.name}: could not identify winning trial"

    new_disc = aco_discovery_cost(trials, winning)
    if new_disc is None:
        return f"skip {path.name}: no same-ants wall times in {vary_path}"

    theta = (data.get("pivot_theta") or {}).get("wall_time_s")
    aco = (data.get("pivot_aco_seed") or {}).get("wall_time_s")
    if theta is None or aco is None:
        return f"skip {path.name}: missing pivot wall times"

    inclusive = float(theta) - float(new_disc) - float(aco)
    inclusive_pct = (
        100.0 * inclusive / float(theta) if float(theta) > 0 else 0.0
    )

    data["aco_discovery_s"] = new_disc
    data["inclusive_reduction_s"] = inclusive
    data["inclusive_reduction_pct"] = inclusive_pct
    if "aco_trial" in data and isinstance(data["aco_trial"], dict):
        data["aco_trial"]["aco_discovery_s"] = new_disc

    if not dry_run:
        write_json(path, data)

    return (
        f"{'would update' if dry_run else 'updated'} {path.name}: "
        f"discovery {old_disc} → {new_disc:.6g}  "
        f"inclusive {data.get('inclusive_reduction_s'):.6g}s "
        f"({inclusive_pct:.4g}%)  "
        f"[ants={winning.get('ants')} run={winning.get('run')} "
        f"via {vary_path.relative_to(ROOT)}]"
    )


def collect_targets(args) -> list[Path]:
    paths: list[Path] = []
    for raw in args.targets:
        p = Path(raw)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if p.is_dir():
            paths.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"warn: not found: {raw}", file=sys.stderr)
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "targets",
        nargs="+",
        help="compare JSON file(s) or directories of them",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned updates without writing",
    )
    ap.add_argument(
        "--force-nobeat",
        action="store_true",
        help="also rewrite no-beat skip markers from their source vary",
    )
    args = ap.parse_args()

    targets = collect_targets(args)
    if not targets:
        print("no JSON targets found", file=sys.stderr)
        return 1

    for path in targets:
        print(patch_one(path, dry_run=args.dry_run, force_nobeat=args.force_nobeat))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
