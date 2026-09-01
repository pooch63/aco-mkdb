#!/usr/bin/env python3
"""Backfill graph.reduced_max_degree into vary result JSON files."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emit.graph_structure_cache import lookup_from_vary_json  # noqa: E402


def folder_uses_inject(dir_name: str) -> bool:
    return bool(re.search(r"k\d+t\d+i", dir_name))


def collect_jobs(directory: Path):
    jobs = {}
    paths_by_key = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        graph = data.get("graph") or {}
        key = (
            graph.get("reduced_nU"),
            graph.get("reduced_nV"),
            graph.get("reduced_edges"),
        )
        if key[0] is None or graph.get("reduced_max_degree") is not None:
            continue
        inject = folder_uses_inject(directory.name)
        cached = lookup_from_vary_json(data, inject=inject)
        if cached is not None:
            graph["reduced_max_degree"] = int(cached["reduced_max_degree"])
            if cached.get("reduced_avg_degree") is not None:
                graph["reduced_avg_degree"] = float(cached["reduced_avg_degree"])
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            continue
        key_str = f"{key[0]},{key[1]},{key[2]}"
        paths_by_key.setdefault(key, []).append(path)
        if key_str in jobs:
            continue
        dataset = data.get("dataset")
        if not dataset:
            continue
        seed = data.get("base_seed") or data.get("seed") or "1"
        jobs[key_str] = {
            "dataset": dataset,
            "seed": str(seed),
            "k": data.get("k"),
            "theta": data.get("theta"),
            "inject": inject,
            "key": key_str,
        }
    return jobs, paths_by_key


def run_batch(jobs: dict) -> dict[str, int]:
    if not jobs:
        return {}
    script = ROOT / "scripts" / "reduced_max_degree_batch.jl"
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tmp:
        json.dump(list(jobs.values()), tmp)
        jobs_path = tmp.name
    try:
        proc = subprocess.run(
            ["julia", str(script), jobs_path],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
    finally:
        os.unlink(jobs_path)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise RuntimeError("julia failed: " + " | ".join(err[-3:]))
    for line in reversed(proc.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return {k: int(v) for k, v in json.loads(line).items()}
    raise RuntimeError("julia produced no JSON object")


def patch_directory(directory: Path) -> int:
    inject = folder_uses_inject(directory.name)
    by_key: dict[tuple, int] = {}
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        graph = data.get("graph") or {}
        key = (
            graph.get("reduced_nU"),
            graph.get("reduced_nV"),
            graph.get("reduced_edges"),
        )
        max_deg = graph.get("reduced_max_degree")
        if key[0] is not None and max_deg is not None:
            by_key[key] = int(max_deg)

    updated = 0
    for path in sorted(directory.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        graph = data.setdefault("graph", {})
        if graph.get("reduced_max_degree") is not None:
            continue
        cached = lookup_from_vary_json(data, inject=inject)
        if cached is not None:
            graph["reduced_max_degree"] = int(cached["reduced_max_degree"])
            if cached.get("reduced_avg_degree") is not None:
                graph["reduced_avg_degree"] = float(cached["reduced_avg_degree"])
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            updated += 1
            continue
        key = (
            graph.get("reduced_nU"),
            graph.get("reduced_nV"),
            graph.get("reduced_edges"),
        )
        if key in by_key:
            graph["reduced_max_degree"] = by_key[key]
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            updated += 1

    jobs, paths_by_key = collect_jobs(directory)
    if not jobs:
        if updated:
            print(f"# {directory}: patched {updated} file(s)", file=sys.stderr)
        else:
            print(f"# {directory}: nothing to backfill", file=sys.stderr)
        return updated
    print(
        f"# {directory}: computing max degree for {len(jobs)} reduced graph(s)…",
        file=sys.stderr,
    )
    computed = run_batch(jobs)
    for key, paths in paths_by_key.items():
        key_str = f"{key[0]},{key[1]},{key[2]}"
        max_deg = computed.get(key_str)
        if max_deg is None:
            print(f"#   missing result for {key_str}", file=sys.stderr)
            continue
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = data.setdefault("graph", {})
            graph["reduced_max_degree"] = max_deg
            ru, rv, re = graph["reduced_nU"], graph["reduced_nV"], graph["reduced_edges"]
            n = int(ru) + int(rv)
            if n > 0:
                graph["reduced_avg_degree"] = (2.0 * float(re)) / float(n)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            updated += 1
    print(f"# {directory}: patched {updated} file(s)", file=sys.stderr)
    return updated


def main(argv: list[str] | None = None) -> int:
    if not argv or len(argv) < 2:
        print(f"Usage: {sys.argv[0]} RESULTS_DIR …", file=sys.stderr)
        return 1
    total = 0
    for arg in argv[1:]:
        total += patch_directory(Path(arg).resolve())
    return 0 if total >= 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
