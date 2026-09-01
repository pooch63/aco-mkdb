"""Read graph structure metrics from data/<dataset>/graph_structure.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CACHE_FILE = "graph_structure.json"
ROOT = Path(__file__).resolve().parents[1]


def cache_path(dataset: str, *, data_root: Path | None = None) -> Path:
    root = data_root or ROOT / "data"
    return root / dataset.replace("\\", "/") / CACHE_FILE


def cache_key(
    k: int,
    theta: int,
    *,
    inject: bool,
    seed: str | None = None,
    inject_u: int = 5,
    inject_v: int = 5,
    reduction: str = "simple",
) -> str:
    red = "none" if reduction == "none" else "simple"
    parts = [f"k={k}", f"theta={theta}", f"reduction={red}"]
    if inject:
        if seed is None:
            raise ValueError("seed required when inject=True")
        parts.extend(
            ["inject=1", f"u={inject_u}", f"v={inject_v}", f"seed={seed}"]
        )
    else:
        parts.append("inject=0")
    return ",".join(parts)


def load_cache(dataset: str, *, data_root: Path | None = None) -> dict[str, Any]:
    path = cache_path(dataset, data_root=data_root)
    if not path.is_file():
        return {"version": 1, "entries": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("entries") or {}
    return {"version": data.get("version", 1), "entries": entries}


def lookup_entry(
    dataset: str,
    key: str,
    *,
    nU: int,
    nV: int,
    edges: int,
    reduced_nU: int,
    reduced_nV: int,
    reduced_edges: int,
    data_root: Path | None = None,
) -> dict[str, Any] | None:
    entry = load_cache(dataset, data_root=data_root)["entries"].get(key)
    if not entry:
        return None
    checks = (
        ("nU", nU),
        ("nV", nV),
        ("edges", edges),
        ("reduced_nU", reduced_nU),
        ("reduced_nV", reduced_nV),
        ("reduced_edges", reduced_edges),
    )
    for field, want in checks:
        if entry.get(field) != want:
            return None
    if entry.get("reduced_max_degree") is None:
        return None
    return entry


def lookup_from_vary_json(
    data: dict[str, Any],
    *,
    inject: bool,
    inject_u: int = 5,
    inject_v: int = 5,
    data_root: Path | None = None,
) -> dict[str, Any] | None:
    dataset = data.get("dataset")
    k = data.get("k")
    theta = data.get("theta")
    graph = data.get("graph") or {}
    if not dataset or k is None or theta is None:
        return None
    ru = graph.get("reduced_nU")
    rv = graph.get("reduced_nV")
    re = graph.get("reduced_edges")
    if ru is None or rv is None or re is None:
        return None
    nU = graph.get("nU")
    nV = graph.get("nV")
    edges = data.get("edge_count") or graph.get("edges")
    if nU is None or nV is None or edges is None:
        return None
    seed = str(data.get("base_seed") or data.get("seed") or "1")
    reduction = data.get("reduction") or "simple"
    key = cache_key(
        int(k),
        int(theta),
        inject=inject,
        seed=seed if inject else None,
        inject_u=inject_u,
        inject_v=inject_v,
        reduction=str(reduction),
    )
    return lookup_entry(
        str(dataset),
        key,
        nU=int(nU),
        nV=int(nV),
        edges=int(edges),
        reduced_nU=int(ru),
        reduced_nV=int(rv),
        reduced_edges=int(re),
        data_root=data_root,
    )
