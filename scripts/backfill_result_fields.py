#!/usr/bin/env python3
"""
Backfill paper-required fields into vary result JSON (one-time, not at build).

  - graph.reduced_max_degree / graph.reduced_avg_degree
  - top-level missing_at_size summary (from trials[].construction or Julia)

Usage:
  python3 scripts/backfill_result_fields.py RESULTS_DIR …
  python3 scripts/backfill_result_fields.py --ants=100 --size=5 results/vary_k2t5i_
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emit.result_fields import file_missing_at_size_weight  # noqa: E402


def _pool_missing_at_size_from_trials(trials):
    pooled_n: dict[str, dict[str, int]] = {}
    pooled_sum: dict[str, dict[str, float]] = {}
    for trial in trials:
        construction = trial.get("construction") or {}
        mas = construction.get("missing_at_size") or {}
        ants_key = str(trial.get("ants"))
        for size_str, entry in mas.items():
            if entry.get("samples"):
                n = len(entry["samples"])
                mean = sum(int(x) for x in entry["samples"]) / n
            elif entry.get("mean") is not None and entry.get("n"):
                n = int(entry["n"])
                mean = float(entry["mean"])
            else:
                continue
            pooled_n.setdefault(ants_key, {}).setdefault(size_str, 0)
            pooled_sum.setdefault(ants_key, {}).setdefault(size_str, 0.0)
            pooled_n[ants_key][size_str] += n
            pooled_sum[ants_key][size_str] += mean * n
    out: dict = {}
    for ants_key, sizes in pooled_n.items():
        size_out = {}
        for size_str, n in sizes.items():
            if n == 0:
                continue
            size_out[size_str] = {
                "mean": pooled_sum[ants_key][size_str] / n,
                "n": n,
            }
        if size_out:
            out[ants_key] = size_out
    return out


def _julia_missing_at_size(path: Path, *, ants: int, size: int) -> tuple[float, int] | None:
    script = ROOT / "scripts" / "missing-at-size.jl"
    proc = subprocess.run(
        ["julia", str(script), str(path), f"--ants={ants}", f"--size={size}"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError(f"{path.name}: julia failed: {' | '.join(err[-2:])}")
    mean = None
    for line in proc.stdout.splitlines():
        if line.startswith("MEAN,"):
            value = line.split(",", 1)[1].strip()
            mean = None if not value else float(value)
    if mean is None:
        return None
    return mean, ants


def backfill_missing_at_size_file(
    path: Path, *, ants: int, size: int, force: bool = False
) -> bool:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not force and file_missing_at_size_weight(data, ants=ants, target_size=size):
        return False

    summary = _pool_missing_at_size_from_trials(data.get("trials") or [])
    ants_key = str(ants)
    size_key = str(size)
    if summary.get(ants_key, {}).get(size_key):
        data["missing_at_size"] = summary
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return True

    result = _julia_missing_at_size(path, ants=ants, size=size)
    if result is None:
        print(f"#   {path.name}: no missing-at-{size} samples", file=sys.stderr)
        return False
    mean, n = result
    mas = data.setdefault("missing_at_size", {})
    mas.setdefault(ants_key, {})[size_key] = {"mean": mean, "n": n}
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return True


def backfill_directory(
    directory: Path,
    *,
    ants: int,
    size: int,
    max_degree: bool,
    missing_at_size: bool,
    force: bool,
) -> tuple[int, int]:
    deg_patched = 0
    mas_patched = 0

    if max_degree:
        script = ROOT / "scripts" / "backfill_reduced_max_degree.py"
        proc = subprocess.run(
            [sys.executable, str(script), str(directory)],
            cwd=str(ROOT),
        )
        if proc.returncode != 0:
            raise SystemExit(f"reduced_max_degree backfill failed for {directory}")

    if missing_at_size:
        files = sorted(directory.glob("*.json"))
        print(
            f"# {directory}: backfilling missing_at_size[{ants}][{size}] "
            f"for {len(files)} file(s)…",
            file=sys.stderr,
        )
        for path in files:
            if backfill_missing_at_size_file(path, ants=ants, size=size, force=force):
                mas_patched += 1
                print(f"#   patched {path.name}", file=sys.stderr)

    return deg_patched, mas_patched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", help="vary result directories")
    parser.add_argument("--ants", type=int, default=100)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument(
        "--no-max-degree",
        action="store_true",
        help="skip graph.reduced_max_degree backfill",
    )
    parser.add_argument(
        "--no-missing-at-size",
        action="store_true",
        help="skip missing_at_size backfill",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="recompute missing_at_size even when already present",
    )
    args = parser.parse_args(argv)

    total_mas = 0
    for arg in args.directories:
        directory = Path(arg).resolve()
        if not directory.is_dir():
            print(f"# skip (not a directory): {directory}", file=sys.stderr)
            continue
        _, mas = backfill_directory(
            directory,
            ants=args.ants,
            size=args.size,
            max_degree=not args.no_max_degree,
            missing_at_size=not args.no_missing_at_size,
            force=args.force,
        )
        total_mas += mas

    print(f"# done: missing_at_size patched in {total_mas} file(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
