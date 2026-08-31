#!/usr/bin/env python3
"""
TEMPORARY: backfill reduced_max_degree / reduced_avg_degree into results/vary*.

Average degree is exact from existing fields: 2|E_R|/(|U_R|+|V_R|).
Maximum degree requires reloading each indexed CSV, re-applying the same
inject+CNN reduction as vary.bash (seed from JSON, --inject --u=5 --v=5 when
the folder name ends in ``i`` / ``i_`` / ``i_P`` etc.), and reading degrees
off the frozen reduced graph.

Delete this file after a successful run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INJECT_U = 5
INJECT_V = 5


def vary_json_paths():
    paths = []
    for child in sorted(RESULTS.iterdir()):
        if not child.is_dir() or not child.name.startswith("vary"):
            continue
        for path in sorted(child.rglob("*.json")):
            paths.append(path)
    return paths


def folder_uses_inject(dir_name: str) -> bool:
    """vary_k2t5i_ / vary_k2t5i_PN → inject; vary_k2t5 without trailing i → not."""
    # Match k<digits>t<digits>i as in scripts/vary.bash OUT_DIR naming.
    return bool(re.search(r"k\d+t\d+i", dir_name))


def avg_degree(nU, nV, edges):
    n = int(nU) + int(nV)
    if n <= 0:
        return 0.0
    return (2.0 * float(edges)) / float(n)


def write_julia_helper(path: Path):
    path.write_text(
        r"""
# One-shot helper: load → optional inject → CNN reduce → print degree stats.
# Args: dataset seed k theta inject(0|1) inject_u inject_v
# Prints one CSV line: reduced_nU,reduced_nV,reduced_edges,max_deg,avg_deg
const ROOT = dirname(@__DIR__)
include(joinpath(ROOT, "bin", "load.jl"))

dataset = ARGS[1]
seed = parse(UInt64, ARGS[2])
k = parse(Int, ARGS[3])
θ = parse(Int, ARGS[4])
do_inject = ARGS[5] == "1"
inj_u = parse(Int, ARGS[6])
inj_v = parse(Int, ARGS[7])

Random.seed!(seed)
graph_path = resolve_graph_path(dataset)
isfile(graph_path) || error("missing graph: $graph_path")

inject = (; enabled=do_inject, nU=inj_u, nV=inj_v, attempts=20)
g, _edges, _plant = load_graph_maybe_inject(graph_path, inject, k, Random.default_rng())
g_red = deepcopy(g)
fg = apply_graph_reductions!(g_red, k, θ, nothing, nothing, true, ReductionMode.simple)
max_deg, avg_deg = reduced_degree_stats(fg)
println("STATS,", length(fg.u_ids), ",", length(fg.v_ids), ",",
        length(fg.v_adj), ",", max_deg, ",", avg_deg)
""",
        encoding="utf-8",
    )


def compute_max_via_julia(helper: Path, dataset, seed, k, theta, do_inject):
    cmd = [
        "julia",
        str(helper),
        str(dataset),
        str(seed),
        str(k),
        str(theta),
        "1" if do_inject else "0",
        str(INJECT_U),
        str(INJECT_V),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = err[-3:] if err else ["(no stderr)"]
        raise RuntimeError("julia failed: " + " | ".join(tail))
    for ln in reversed(proc.stdout.splitlines()):
        ln = ln.strip()
        if ln.startswith("STATS,"):
            parts = ln.split(",")
            return {
                "reduced_nU": int(parts[1]),
                "reduced_nV": int(parts[2]),
                "reduced_edges": int(parts[3]),
                "reduced_max_degree": int(parts[4]),
                "reduced_avg_degree": float(parts[5]),
            }
    raise RuntimeError("julia produced no STATS line")


def main():
    paths = vary_json_paths()
    if not paths:
        print("No vary JSON under results/", file=sys.stderr)
        return 1

    # Group by (dataset, k, theta, seed, inject?) for one Julia call each.
    groups = defaultdict(list)
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        graph = data.get("graph") or {}
        key = (
            data.get("dataset"),
            data.get("k"),
            data.get("theta"),
            data.get("seed"),
            folder_uses_inject(path.parent.name),
            graph.get("reduced_nU"),
            graph.get("reduced_nV"),
            graph.get("reduced_edges"),
        )
        groups[key].append(path)

    helper = Path(tempfile.mkstemp(prefix="patch_deg_", suffix=".jl", dir=str(ROOT / "scripts"))[1])
    write_julia_helper(helper)

    max_by_key = {}
    missing_max = 0
    try:
        for key, group_paths in sorted(groups.items(), key=lambda kv: str(kv[0][0])):
            dataset, k, theta, seed, do_inject, ru, rv, re = key
            print(
                f"# degrees for {dataset} (inject={do_inject}, {len(group_paths)} file(s))…",
                file=sys.stderr,
            )
            try:
                stats = compute_max_via_julia(
                    helper, dataset, seed, k, theta, do_inject
                )
            except Exception as exc:
                print(f"#   skip max: {exc}", file=sys.stderr)
                missing_max += 1
                max_by_key[key] = None
                continue
            if (
                int(stats["reduced_nU"]) != int(ru)
                or int(stats["reduced_nV"]) != int(rv)
                or int(stats["reduced_edges"]) != int(re)
            ):
                print(
                    f"#   skip max: reduced size mismatch "
                    f"got {stats['reduced_nU']}x{stats['reduced_nV']} "
                    f"E={stats['reduced_edges']} "
                    f"want {ru}x{rv} E={re}",
                    file=sys.stderr,
                )
                missing_max += 1
                max_by_key[key] = None
                continue
            max_by_key[key] = int(stats["reduced_max_degree"])
            print(
                f"#   max={max_by_key[key]} avg={stats['reduced_avg_degree']:.4g}",
                file=sys.stderr,
            )
    finally:
        helper.unlink(missing_ok=True)

    updated = 0
    for key, group_paths in groups.items():
        max_deg = max_by_key.get(key)
        for path in group_paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            graph = data.setdefault("graph", {})
            ru, rv, re = graph["reduced_nU"], graph["reduced_nV"], graph["reduced_edges"]
            graph["reduced_avg_degree"] = avg_degree(ru, rv, re)
            if max_deg is not None:
                graph["reduced_max_degree"] = max_deg
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            updated += 1

    print(
        f"# patched {updated} file(s); "
        f"max-degree groups missing={missing_max}/{len(groups)}",
        file=sys.stderr,
    )
    return 0 if missing_max == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
