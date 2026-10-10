"""
epoch-table mode — epoch traces → mean and standard deviation of |E|.

Reads ``epoch_trace`` from the budget-suite JSON (one file per graph).
Each column pair is one epoch budget: the mean edge count across counted
replicates, and the sample standard deviation of those edge counts.
Algorithm 1 has no schedule that depends on ``T``, so epoch ``t`` of a
longer trace is the edge count a run stopped there would have returned.

JIT warmup replicates are omitted. The process-wide ``max_replicates``
cap is ignored, matching the epoch-budget figure: a standard deviation
needs every counted replicate.
"""

from __future__ import annotations

import statistics
import sys

from .common import list_json_paths, report_skipped, write_tex
from .compare.budget_curves import load_suite

# Knots on the recorded trace. Same span as the epoch panel of the
# quality figure, thinned so each budget still has its own mean and SD.
EPOCHS = (1, 5, 20, 100)
DEFAULT_ANTS = 100


def _fmt(value):
    if value is None:
        return "--"
    return f"{value:.1f}"


def _stats(graph, epoch):
    edges = []
    for rep in graph["reps"]:
        pt = rep.get(epoch)
        if pt is None or pt.get("edges") is None:
            continue
        edges.append(int(pt["edges"]))
    if not edges:
        return None, None
    mean = statistics.mean(edges)
    if len(edges) < 2:
        return mean, None
    return mean, statistics.stdev(edges)


def render(graphs, epochs=EPOCHS):
    """One row per graph. Cells are mean |E| and its sample SD."""
    if not graphs:
        print("# epoch table: no graphs with an epoch trace", file=sys.stderr)
        return (
            r"\begin{center}" "\n"
            r"\scriptsize Epoch edge counts are not available." "\n"
            r"\end{center}"
        )

    header = ["Graph"]
    sub = [""]
    for epoch in epochs:
        header.append(rf"\multicolumn{{2}}{{c}}{{$T={epoch}$}}")
        sub.extend(["Mean", "SD"])

    rules = []
    for i in range(len(epochs)):
        left = 2 + 2 * i
        rules.append(rf"\cmidrule(lr){{{left}-{left + 1}}}")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Mean and standard deviation of $|E|$ across counted "
        r"replicates after $T$ epochs.}",
        r"\label{tab:epoch-edges}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "rr" * len(epochs) + "}",
        r"\toprule",
        " & ".join(header) + r" \\",
        "".join(rules),
        " & ".join(sub) + r" \\",
        r"\midrule",
    ]

    missing = []
    for graph in sorted(graphs, key=lambda g: g["name"].lower()):
        cells = [graph["name"].replace("_", r"\_")]
        for epoch in epochs:
            mean, sd = _stats(graph, epoch)
            if mean is None:
                missing.append((graph["name"], epoch))
            cells.append(_fmt(mean))
            cells.append(_fmt(sd))
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    if missing:
        bits = ", ".join(f"{name} at T={epoch}" for name, epoch in missing)
        print(
            f"# epoch table: no edge count for {bits}",
            file=sys.stderr,
        )
    return "\n".join(lines)


def run(directory, output, *, ants=DEFAULT_ANTS):
    paths = list_json_paths(directory)
    if not paths:
        print(f"# epoch table: no JSON in {directory}", file=sys.stderr)
        write_tex(render([]), output)
        return
    graphs, skipped = load_suite(paths, ants=ants if ants is not None else DEFAULT_ANTS)
    report_skipped(skipped)
    write_tex(render(graphs) + "\n", output)
    print(
        f"# epoch table: {len(graphs)} graph(s), epochs {','.join(str(t) for t in EPOCHS)}",
        file=sys.stderr,
    )
