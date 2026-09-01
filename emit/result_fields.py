"""
Required vary.jl JSON fields for paper emit modes.

Build-time emit reads only pre-recorded JSON — no Julia re-simulation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import display_name, load_json

# graph.* fields used by table / compare / seed-compare
GRAPH_FIELDS = (
    "nU",
    "nV",
    "reduced_nU",
    "reduced_nV",
    "reduced_edges",
    "reduced_max_degree",
)

COMPARE_PLOT_REQUIREMENTS = {
    "theta-time": (),
    "density-size": ("edge_count",),
    "max-deg-time": ("reduced_max_degree",),
    "deg-size-time": ("reduced_max_degree",),
}


def _trial_missing_at_size_entry(trial, target_size):
    construction = trial.get("construction") or {}
    return (construction.get("missing_at_size") or {}).get(str(target_size))


def file_missing_at_size_weight(data, *, ants, target_size):
    """
    Return (mean, n) for one graph file, or None if unavailable.

  Reads top-level missing_at_size first, then trials[].construction.
    """
    if not data:
        return None

    entry = (
        (data.get("missing_at_size") or {}).get(str(ants)) or {}
    ).get(str(target_size))
    if entry and entry.get("mean") is not None and entry.get("n"):
        return float(entry["mean"]), int(entry["n"])

    total_n = 0
    total_sum = 0.0
    for trial in data.get("trials") or []:
        if trial.get("ants") != ants:
            continue
        trial_entry = _trial_missing_at_size_entry(trial, target_size)
        if not trial_entry:
            continue
        if trial_entry.get("samples"):
            samples = [int(x) for x in trial_entry["samples"]]
            total_n += len(samples)
            total_sum += sum(samples)
        elif trial_entry.get("mean") is not None and trial_entry.get("n"):
            n = int(trial_entry["n"])
            total_n += n
            total_sum += float(trial_entry["mean"]) * n
    if total_n == 0:
        return None
    return total_sum / total_n, total_n


def pool_missing_at_size_mean(directory, *, ants=100, target_size=5):
    """Pooled mean missing at |S|=target_size across all JSON in a directory."""
    from .common import list_json_paths

    json_paths = list_json_paths(directory)
    total_n = 0
    total_sum = 0.0
    n_graphs = 0
    missing_files: list[str] = []

    for path in json_paths:
        data = load_json(path)
        weight = file_missing_at_size_weight(
            data, ants=ants, target_size=target_size
        )
        if weight is None:
            missing_files.append(display_name(path, data or {}))
            continue
        mean, n = weight
        total_sum += mean * n
        total_n += n
        n_graphs += 1

    if total_n == 0:
        return None, n_graphs, len(json_paths), missing_files
    return total_sum / total_n, n_graphs, len(json_paths), missing_files


def validate_graph_fields(data, path, *, extra=()):
    """Return list of missing field descriptions for one vary JSON file."""
    issues = []
    graph = data.get("graph") or {}
    name = display_name(path, data)

    for field in GRAPH_FIELDS:
        if field in extra:
            continue
        if graph.get(field) is None and field not in ("reduced_max_degree",):
            issues.append(f"{name}: graph.{field}")
    for field in extra:
        if field == "reduced_max_degree":
            if graph.get("reduced_max_degree") is None:
                issues.append(f"{name}: graph.reduced_max_degree")
        elif field == "edge_count":
            if data.get("edge_count") is None and graph.get("edges") is None:
                issues.append(f"{name}: edge_count")

    trials = data.get("trials") or []
    if not trials:
        issues.append(f"{name}: trials (empty)")

    heuristic = data.get("heuristic") or {}
    if heuristic.get("final_edges") is None:
        issues.append(f"{name}: heuristic.final_edges")

    return issues


def validate_compare_directory(json_paths, plots):
    """Warn if compare plots need fields missing from JSON (non-fatal)."""
    required_extra = set()
    for plot in plots:
        for req in COMPARE_PLOT_REQUIREMENTS.get(plot, ()):
            required_extra.add(req)

    issues = []
    for path in json_paths:
        data = load_json(path)
        if data is None:
            issues.append(f"{path}: unreadable")
            continue
        issues.extend(validate_graph_fields(data, path, extra=tuple(required_extra)))

    if issues:
        sample = "\n  ".join(sorted(set(issues))[:12])
        more = f"\n  … and {len(issues) - 12} more" if len(issues) > 12 else ""
        print(
            "Warning: result JSON missing fields required for compare plots.\n"
            f"  {sample}{more}\n"
            "Re-run vary.jl or: python3 scripts/backfill_result_fields.py <dir>\n"
            "(graph degrees are cached in data/<dataset>/graph_structure.json)",
            file=sys.stderr,
        )


def validate_missing_at_size_dirs(aco_dir, aco_n_dir, *, ants=100, target_size=5):
    """Warn if missing-at-size means are not recorded in JSON (non-fatal)."""
    problems = []
    for label, directory in (("ACO", aco_dir), ("ACO-N", aco_n_dir)):
        mean, n_graphs, n_files, missing = pool_missing_at_size_mean(
            directory, ants=ants, target_size=target_size
        )
        if mean is None or n_graphs == 0:
            problems.append(
                f"{label} ({directory}): no missing_at_size[{ants}][{target_size}] "
                f"in any of {n_files} file(s)"
            )
            if missing:
                problems.append(
                    f"  e.g. missing: {', '.join(missing[:5])}"
                    + ("…" if len(missing) > 5 else "")
                )
    if problems:
        print(
            "Warning: result JSON missing construction missing-at-size statistics.\n"
            + "\n".join(f"  {p}" for p in problems)
            + "\nRe-run vary.jl or: python3 scripts/backfill_result_fields.py <dir>",
            file=sys.stderr,
        )


def warn_incomplete(directory: str | Path, *, label: str = "") -> None:
    """Print a one-line summary of missing graph fields (non-fatal)."""
    from .common import list_json_paths

    json_paths = list_json_paths(directory)
    no_max = no_mas = 0
    for path in json_paths:
        data = load_json(path)
        if data is None:
            continue
        graph = data.get("graph") or {}
        if graph.get("reduced_max_degree") is None:
            no_max += 1
        if file_missing_at_size_weight(data, ants=100, target_size=5) is None:
            no_mas += 1
    if no_max or no_mas:
        prefix = f"# {label}: " if label else "# "
        print(
            f"{prefix}{len(json_paths)} file(s); "
            f"{no_max} lack graph.reduced_max_degree; "
            f"{no_mas} lack missing_at_size[100][5]",
            file=sys.stderr,
        )
