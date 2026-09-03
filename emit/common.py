"""Shared helpers for emit modes (JSON I/O, dataset labels, output)."""

from __future__ import annotations

import glob
import json
import os
import sys


def list_json_paths(directory):
    return sorted(glob.glob(os.path.join(directory, "*.json")))


def load_json(path):
    """Load a JSON file, or None on decode/IO error."""
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def preamble_sidecar_path(output):
    """Map ``foo.tex`` → ``foo.preamble.tex`` (or ``<output>.preamble.tex``)."""
    if output.endswith(".tex"):
        return output[: -len(".tex")] + ".preamble.tex"
    return output + ".preamble.tex"


def write_tex(tex, output, *, preamble=None):
    """
    Write LaTeX to output path, or print to stdout when output is None.

    Optional ``preamble`` is written beside the fragment as
    ``<stem>.preamble.tex`` for ``%%PREAMBLE%%`` assembly. Passing
    ``preamble=None`` removes a stale sidecar when writing to a file.
    """
    if output:
        with open(output, "w") as f:
            f.write(tex + "\n")
        sidecar = preamble_sidecar_path(output)
        if preamble:
            with open(sidecar, "w") as f:
                f.write(preamble.rstrip() + "\n")
        elif os.path.isfile(sidecar):
            os.remove(sidecar)
    else:
        print(tex)
        if preamble:
            print("% --- preamble contribution ---", file=sys.stderr)
            print(preamble.rstrip(), file=sys.stderr)


def report_skipped(skipped):
    if not skipped:
        return
    print(f"# Skipped {len(skipped)} file(s):", file=sys.stderr)
    for path, reason in skipped:
        print(f"#   {os.path.basename(path)}: {reason}", file=sys.stderr)


def series_name(path, data):
    """pgfplots series / symbolic-x label (hyphenated leaf name)."""
    label = data.get("dataset") or os.path.splitext(os.path.basename(path))[0]
    label = str(label).replace("_", "-")
    # Drop collection prefixes like "konect-small/" or "amazon/".
    return label.rsplit("/", 1)[-1]


def display_name(path, data):
    """Human-readable table row label (title case, spaces)."""
    label = data.get("dataset") or os.path.splitext(os.path.basename(path))[0]
    leaf = str(label).replace("\\", "/").split("/")[-1]
    leaf = leaf.removesuffix("_ants")
    return leaf.replace("_", " ").replace("-", " ").title()


def resolve_path(path, relative_to=None):
    """Resolve path; if relative and missing, try relative_to's directory."""
    if path is None:
        return None
    if os.path.isfile(path):
        return path
    if relative_to is not None:
        alt = os.path.join(os.path.dirname(os.path.abspath(relative_to)), path)
        if os.path.isfile(alt):
            return alt
    # compare-seeds sometimes writes "vary_dir//file.json"
    cleaned = path.replace("//", "/")
    if cleaned != path and os.path.isfile(cleaned):
        return cleaned
    if relative_to is not None:
        alt = os.path.join(os.path.dirname(os.path.abspath(relative_to)), cleaned)
        if os.path.isfile(alt):
            return alt
    return path if os.path.isfile(path) else None


def infer_vary_dir(compare_dir):
    """Map compare_k2t5i_… → vary_k2t5i_… beside compare dir or under results/."""
    abs_compare = os.path.abspath(compare_dir.rstrip(os.sep))
    base = os.path.basename(abs_compare)
    if not base.startswith("compare_"):
        return None
    vary_name = "vary_" + base[len("compare_"):]
    parent = os.path.dirname(abs_compare)
    candidates = [
        os.path.join(parent, vary_name),
        os.path.join(parent, "results", vary_name),
        os.path.join(os.path.dirname(parent), "results", vary_name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


def aco_runs_per_ant(data):
    """Replicates per ant count (1 when vary.jl ran each count once)."""
    if data is None:
        return 1
    n = data.get("aco_runs")
    if n is not None:
        return int(n)
    runs = [
        int(t["run"])
        for t in data.get("trials") or []
        if t.get("run") is not None
    ]
    return max(runs) if runs else 1


def is_jit_warmup_trial(t, data=None, *, n_runs=None):
    """
    True for the first ACO replicate when multiple runs were recorded.

    That replicate pays Julia JIT on the real graph and must be excluded from
    quality, feasibility, timing, discovery, and seed selection. Prefer an
    explicit ``jit_warmup`` flag when present; otherwise treat ``run == 1`` as
    warmup whenever ``aco_runs > 1``.
    """
    if t.get("jit_warmup") in (True, 1):
        return True
    if t.get("jit_warmup") in (False, 0):
        return False
    if n_runs is None:
        n_runs = aco_runs_per_ant(data)
    if n_runs <= 1:
        return False
    return int(t.get("run", 1)) == 1


def counted_trials(trials, data=None, *, n_runs=None):
    """Trials excluding the per-ant-count JIT warmup replicate."""
    return [
        t
        for t in (trials or [])
        if not is_jit_warmup_trial(t, data, n_runs=n_runs)
    ]


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


def aco_discovery_cost(trials, winning_trial, until_found=False, data=None):
    """
    Full ACO wall time at the winning ant count.

    Sums wall_time_s over every counted same-ant replicate — the full search
    budget at that ant count, even if the seeded subgraph came from an earlier
    run. JIT warmup replicates (run 1 when aco_runs > 1) are omitted.

    until_found is kept for call-site compatibility and ignored: discovery is
    never truncated at the winning run / time-to-best.
    """
    del until_found  # API compat; always charge the full same-ants budget.
    if winning_trial is None:
        return None

    ants = winning_trial.get("ants")
    if ants is None:
        return None

    same = [
        t
        for t in counted_trials(trials, data)
        if t.get("ants") == ants
    ]
    if not same:
        return None

    total = 0.0
    any_time = False
    for t in same:
        wt = t.get("wall_time_s")
        if wt is None:
            continue
        total += float(wt)
        any_time = True
    return total if any_time else None
