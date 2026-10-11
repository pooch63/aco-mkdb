"""
reduce mode — super-reduce phase JSON → one compact full-width table.

Reads the array written by ``report_super_reduce`` (one object per graph).
Each row is the original ``nU``, ``nV``, and ``edges``, then the same counts
and wall time after:

- common-neighbor reduction (``cnn_*``)
- one ``super_reduce!`` pass (``once_*``, ``efficient=true``)
- the repeated reduction (``full_*``, ``efficient=false``)

Times in the JSON are seconds and stay in seconds. Each reduction block
is the percentage of nodes and of edges removed from the loaded graph.
On each graph the largest node and edge removal are bold; a tie bolds
every method that matches. Wall time is not bold. Missing fields become
``--`` and a stderr warning. This module only reads the JSON.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

from .common import begin_float, begin_tikz, end_float, end_tikz, load_json, series_name, write_tex

_PHASES = (
    ("cnn", "CNR"),
    ("once", "Once"),
    ("full", "Full"),
)


def _fmt_thousands(value):
    """Count in thousands, rounded to one decimal."""
    if value is None:
        return "--"
    return f"{float(value) / 1000:.1f}"


def _fmt_seconds(value):
    """Wall time already stored in seconds."""
    if value is None:
        return "--"
    seconds = float(value)
    if seconds >= 10:
        return f"{seconds:.1f}"
    if seconds >= 1:
        return f"{seconds:.2f}"
    if seconds >= 0.01:
        return f"{seconds:.3f}"
    return f"{seconds:.4f}"


def _removed_pct(before, after):
    """Percentage of ``before`` removed, or None when it cannot be scored."""
    if before is None or after is None or int(before) <= 0:
        return None
    return 100.0 * (int(before) - int(after)) / int(before)


def _fmt_pct(before, after):
    """Percentage of ``before`` removed. ``after`` is what remains."""
    removed = _removed_pct(before, after)
    if removed is None:
        return "--"
    return f"{removed:.1f}"


def _bold_winners(texts, scores, *, higher):
    """
    Bold every displayed value that ties the best score.

    Comparison uses the raw score, then any cell that prints the same
    text as a winner is bold too, so a rounding tie is not split.
    """
    ranked = [(i, score) for i, score in enumerate(scores) if score is not None]
    if not ranked:
        return list(texts)
    best = max(score for _, score in ranked) if higher else min(score for _, score in ranked)
    winner_text = {texts[i] for i, score in ranked if score == best and texts[i] != "--"}
    marked = []
    for text in texts:
        if text in winner_text:
            marked.append(rf"\textbf{{{text}}}")
        else:
            marked.append(text)
    return marked


def _cell(row, key, warned):
    if key not in row or row.get(key) is None:
        warned.add(key)
        return None
    return row[key]


def render(rows):
    """LaTeX for every recorded graph. ``rows`` is a list of JSON objects."""
    if not rows:
        print("# reduction table: no graphs in the results file", file=sys.stderr)
        return (
            r"\begin{center}" "\n"
            r"\scriptsize Reduction sizes are not available." "\n"
            r"\end{center}"
        )

    warned = set()
    body = []
    ks = {r.get("k") for r in rows if isinstance(r, dict) and r.get("k") is not None}
    thetas = {r.get("theta") for r in rows if isinstance(r, dict) and r.get("theta") is not None}
    injected = all(
        isinstance(r, dict) and r.get("injected") is True for r in rows
    )
    k_tex = str(int(next(iter(ks)))) if len(ks) == 1 else "varied"
    theta_tex = str(int(next(iter(thetas)))) if len(thetas) == 1 else "varied"
    for row in rows:
        if not isinstance(row, dict) or "dataset" not in row:
            continue
        name = series_name("", row).replace("_", r"\_")
        nU = _cell(row, "nU", warned)
        nV = _cell(row, "nV", warned)
        edges = _cell(row, "edges", warned)
        nodes = None if nU is None or nV is None else int(nU) + int(nV)
        node_text, edge_text, time_text = [], [], []
        node_score, edge_score = [], []
        for prefix, _label in _PHASES:
            left = _cell(row, f"{prefix}_nU", warned)
            right = _cell(row, f"{prefix}_nV", warned)
            kept_edges = _cell(row, f"{prefix}_edges", warned)
            seconds = _cell(row, f"{prefix}_s", warned)
            kept_nodes = None if left is None or right is None else int(left) + int(right)
            node_text.append(_fmt_pct(nodes, kept_nodes))
            edge_text.append(_fmt_pct(edges, kept_edges))
            time_text.append(_fmt_seconds(seconds))
            node_score.append(_removed_pct(nodes, kept_nodes))
            edge_score.append(_removed_pct(edges, kept_edges))
        cells = [
            name,
            _fmt_thousands(nU),
            _fmt_thousands(nV),
            _fmt_thousands(edges),
        ]
        for n, e, t in zip(
            _bold_winners(node_text, node_score, higher=True),
            _bold_winners(edge_text, edge_score, higher=True),
            time_text,
        ):
            cells.extend((n, e, t))
        body.append(" & ".join(cells) + r" \\")

    if warned:
        fields = ", ".join(sorted(warned))
        print(
            f"# reduction table: missing {fields} on at least one graph",
            file=sys.stderr,
        )
    if not body:
        print("# reduction table: no usable rows", file=sys.stderr)
        return (
            r"\begin{center}" "\n"
            r"\scriptsize Reduction sizes are not available." "\n"
            r"\end{center}"
        )

    # scriptsize, zero booktabs padding, and a short header: 40 graphs
    # should stay about one full-width block, not a loose float page.
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\abovecaptionskip}{1pt}",
        r"\setlength{\belowcaptionskip}{0pt}",
        r"\caption{Percentage of nodes ($|U|+|V|$) and edges removed, and wall "
        r"time $t$ in seconds, for common-neighbor reduction (CNR), one "
        r"pass (Once), and the repeated reduction (Full). "
        r"Bold is the largest removal on that graph. "
        r"Initial side sizes $U_k$ and $V_k$, and the edge count $E_k$, "
        r"are in thousands. "
        + (
            rf"A $\theta \times \theta$ biclique with $k$ missing edges was "
            rf"injected first. "
            if injected else ""
        )
        + rf"Experiments use $k={k_tex}$ and $\theta={theta_tex}$." + "}",
        r"\label{tab:reduction}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{1.6pt}",
        r"\setlength{\aboverulesep}{0.12ex}",
        r"\setlength{\belowrulesep}{0.12ex}",
        r"\setlength{\cmidrulesep}{0pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\begin{tabular}{@{}l rrr rrr rrr rrr@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Initial}"
        r"& \multicolumn{3}{c}{CNR}"
        r"& \multicolumn{3}{c}{Once}"
        r"& \multicolumn{3}{c}{Full} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}"
        r"\cmidrule(lr){8-10}\cmidrule(lr){11-13}",
        r"Graph & $U_k$ & $V_k$ & $E_k$"
        r" & $n\%$ & $e\%$ & $t$ (s)"
        r" & $n\%$ & $e\%$ & $t$ (s)"
        r" & $n\%$ & $e\%$ & $t$ (s) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


# Same file PHASE=kt reads. Duplicate pairs are ignored.
_SWEEP_FILE = Path(__file__).resolve().parents[1] / "scripts" / "kt-sweep.bash"
_PAIR_RE = re.compile(r"(\d+)\s*:\s*(\d+)")

# First three colors of the runtime sweep palette, one per reduction.
_SWEEP_SERIES = (
    ("cnn", "CNR", (31, 119, 180), "*"),
    ("once", "Once", (255, 127, 14), "triangle*"),
    ("full", "Full", (44, 160, 44), "square*"),
)
_SWEEP_ROWS = (
    ("nodes", "Nodes removed (\\%)"),
    ("edges", "Edges removed (\\%)"),
)


def load_kt_pairs(path=None):
    """Unique (k, θ) pairs from ``scripts/kt-sweep.bash``, in file order."""
    path = Path(path) if path is not None else _SWEEP_FILE
    pairs = []
    seen = set()
    if not path.is_file():
        print(f"# reduce sweep: missing pair list {path}", file=sys.stderr)
        return pairs
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for match in _PAIR_RE.finditer(stripped):
            pair = (int(match.group(1)), int(match.group(2)))
            if pair in seen:
                continue
            seen.add(pair)
            pairs.append(pair)
    return pairs


def sweep_axes(pairs):
    """
    One k-sweep per θ that has two or more k values, then one θ-sweep
    per k that has two or more θ values. Order follows the pair list.
    """
    k_by_theta = {}
    theta_by_k = {}
    theta_order = []
    k_order = []
    for k, theta in pairs:
        if theta not in k_by_theta:
            k_by_theta[theta] = []
            theta_order.append(theta)
        if k not in k_by_theta[theta]:
            k_by_theta[theta].append(k)
        if k not in theta_by_k:
            theta_by_k[k] = []
            k_order.append(k)
        if theta not in theta_by_k[k]:
            theta_by_k[k].append(theta)
    columns = []
    for theta in theta_order:
        values = k_by_theta[theta]
        if len(values) >= 2:
            columns.append(
                {"axis": "k", "fixed": theta, "values": sorted(values)}
            )
    for k in k_order:
        values = theta_by_k[k]
        if len(values) >= 2:
            columns.append(
                {"axis": "theta", "fixed": k, "values": sorted(values)}
            )
    return columns


def _pair_path(directory, k, theta):
    return Path(directory) / f"k{k}t{theta}.json"


def _pair_rows(directory, k, theta):
    data = load_json(_pair_path(directory, k, theta))
    if isinstance(data, dict):
        data = data.get("rows") or data.get("graphs") or []
    if not isinstance(data, list):
        return {}
    rows = {}
    for row in data:
        if isinstance(row, dict) and row.get("dataset"):
            rows[row["dataset"]] = row
    return rows


def _median(values):
    vals = sorted(values)
    n = len(vals)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return float(vals[mid])
    return 0.5 * (float(vals[mid - 1]) + float(vals[mid]))


def _pct_removed(row, prefix, kind):
    try:
        if kind == "nodes":
            before = int(row["nU"]) + int(row["nV"])
            after = int(row[f"{prefix}_nU"]) + int(row[f"{prefix}_nV"])
        else:
            before = int(row["edges"])
            after = int(row[f"{prefix}_edges"])
    except (KeyError, TypeError, ValueError):
        return None
    if before <= 0:
        return None
    return 100.0 * (before - after) / before


def _column_series(directory, column, kind):
    """
    Median percent removed at each swept value, over graphs present in
    every recorded pair of this column.
    """
    loaded = []
    for value in column["values"]:
        if column["axis"] == "k":
            k, theta = value, column["fixed"]
        else:
            k, theta = column["fixed"], value
        rows = _pair_rows(directory, k, theta)
        if not rows:
            print(
                f"# reduce sweep: no rows in {_pair_path(directory, k, theta)}",
                file=sys.stderr,
            )
            continue
        loaded.append((value, rows))
    if len(loaded) < 2:
        return None
    common = set.intersection(*(set(rows) for _, rows in loaded))
    if not common:
        return None
    series = []
    for prefix, label, rgb, mark in _SWEEP_SERIES:
        points = []
        for value, rows in loaded:
            samples = [
                pct
                for name in common
                if (pct := _pct_removed(rows[name], prefix, kind)) is not None
            ]
            med = _median(samples)
            if med is None:
                continue
            points.append((float(value), med))
        series.append(
            {
                "label": label,
                "rgb": rgb,
                "mark": mark,
                "points": points,
                "n": len(common),
            }
        )
    if not any(item["points"] for item in series):
        return None
    return series


def _pct_limits(values):
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1.0:
        span = 1.0
    pad = 0.08 * span
    lo = max(0.0, lo - pad)
    hi = min(100.0, hi + pad)
    if hi <= lo:
        hi = min(100.0, lo + 1.0)
    return lo, hi


def _pct_ticks(lo, hi):
    span = hi - lo
    step = 20.0
    for candidate in (1.0, 2.0, 5.0, 10.0, 20.0, 25.0):
        if span / candidate <= 5.0:
            step = candidate
            break
    start = math.ceil((lo - 1e-9) / step) * step
    ticks = []
    value = start
    while value <= hi + 1e-6:
        ticks.append(value)
        value += step
    if len(ticks) < 2:
        return [lo, hi]
    return ticks


def _tick_label(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def _column_title(column):
    if column["axis"] == "k":
        return rf"$\theta = {int(column['fixed'])}$", r"$k$"
    return rf"$k = {int(column['fixed'])}$", r"$\theta$"


def render_sweep(directory, pairs_path=None):
    """
    One-column figure: median percent removed versus the kt sweep.

    Columns are the k-sweeps and θ-sweeps implied by the pair list.
    Rows are nodes removed and edges removed. Each panel has one line
    per reduction.
    """
    directory = Path(directory)
    pairs = load_kt_pairs(pairs_path)
    columns = sweep_axes(pairs)
    if not columns:
        print("# reduce sweep: pair list has no sweep to plot", file=sys.stderr)
        return [r"% reduce-sweep: no (k, theta) sweep in scripts/kt-sweep.bash"]

    panels = []
    for kind, ylabel in _SWEEP_ROWS:
        row = []
        for column in columns:
            series = _column_series(directory, column, kind)
            title, xlabel = _column_title(column)
            row.append(
                {
                    "series": series,
                    "title": title,
                    "xlabel": xlabel,
                    "ylabel": ylabel,
                }
            )
        panels.append(row)

    drawable = [
        panel
        for row in panels
        for panel in row
        if panel["series"] is not None
    ]
    if not drawable:
        print(
            f"# reduce sweep: no recorded pairs under {directory}",
            file=sys.stderr,
        )
        return [r"% reduce-sweep: no recorded (k, theta) results"]

    for row in panels:
        for panel in row:
            series = panel["series"]
            if series is None:
                continue
            n = series[0]["n"]
            print(
                f"# reduce sweep {panel['ylabel']} {panel['title']}: "
                f"median over {n} graphs",
                file=sys.stderr,
            )

    n_cols = len(columns)
    env, lines = begin_float("figure", star=False)
    lines += begin_tikz(resize=False)
    for index, (_prefix, _label, rgb, _mark) in enumerate(_SWEEP_SERIES):
        r, g, b = rgb
        lines.append(
            rf"  \definecolor{{reduceS{index}}}{{RGB}}{{{r},{g},{b}}}"
        )
    # Two short rows in one column, matching the runtime sweep panels.
    hsep = "1.15cm" if n_cols <= 2 else "0.7cm"
    width = "0.30\\columnwidth" if n_cols <= 2 else "0.22\\columnwidth"
    lines += [
        r"  \begin{groupplot}[",
        rf"    group style={{group size={n_cols} by {len(panels)},",
        rf"      horizontal sep={hsep}, vertical sep=1.05cm}},",
        r"    scale only axis,",
        rf"    width={width},",
        r"    height=0.24\columnwidth,",
        r"    grid=major,",
        r"    major grid style={black!22, thin},",
        r"    tick align=outside,",
        r"    tick style={black, thin},",
        r"    tick label style={font=\scriptsize, inner sep=1pt},",
        r"    label style={font=\scriptsize},",
        r"    title style={font=\scriptsize, inner sep=1pt},",
        r"    ylabel near ticks,",
        r"  ]",
    ]
    legend_open = False
    for row_index, row in enumerate(panels):
        ys = [
            y
            for panel in row
            if panel["series"] is not None
            for item in panel["series"]
            for _x, y in item["points"]
        ]
        if ys:
            ymin, ymax = _pct_limits(ys)
        else:
            ymin, ymax = 0.0, 100.0
        yticks = ",".join(_tick_label(t) for t in _pct_ticks(ymin, ymax))
        for col_index, panel in enumerate(row):
            series = panel["series"]
            if series is None:
                xs = columns[col_index]["values"]
            else:
                xs = sorted(
                    {x for item in series for x, _y in item["points"]}
                )
            if not xs:
                xs = columns[col_index]["values"]
            xmin = min(xs) - 0.45
            xmax = max(xs) + 0.45
            xtick = ",".join(_tick_label(x) for x in xs)
            lines.append(r"  \nextgroupplot[")
            lines.append(rf"    xmin={xmin:g}, xmax={xmax:g},")
            lines.append(rf"    xtick={{{xtick}}},")
            lines.append(rf"    ymin={ymin:.4f}, ymax={ymax:.4f},")
            lines.append(rf"    ytick={{{yticks}}},")
            lines.append(rf"    xlabel={{{panel['xlabel']}}},")
            if row_index == 0:
                lines.append(rf"    title={{{panel['title']}}},")
            if col_index == 0:
                lines.append(rf"    ylabel={{{panel['ylabel']}}},")
            else:
                lines.append(r"    yticklabels={},")
            if not legend_open and series is not None:
                lines += [
                    r"    legend to name=reduceSweep,",
                    r"    legend columns=-1,",
                    r"    legend style={font=\scriptsize, draw=none, fill=none,",
                    r"      /tikz/every even column/.append style={column sep=0.5em}},",
                ]
            lines.append(r"  ]")
            if series is None:
                continue
            for index, item in enumerate(series):
                if not item["points"]:
                    continue
                coords = " ".join(
                    f"({x:g},{y:.4f})" for x, y in item["points"]
                )
                lines.append(
                    rf"  \addplot[semithick, mark={item['mark']}, "
                    rf"mark size=1.5pt, color=reduceS{index}, "
                    rf"mark options={{solid, fill=reduceS{index}, "
                    rf"draw=reduceS{index}}}] coordinates {{{coords}}};"
                )
                if not legend_open:
                    lines.append(rf"  \addlegendentry{{{item['label']}}}")
            if any(item["points"] for item in series):
                legend_open = True
    lines.append(r"  \end{groupplot}")
    if legend_open:
        lines += [
            r"  \node[anchor=north, inner sep=0pt] "
            r"at ([yshift=-2pt]current bounding box.south) "
            r"{\pgfplotslegendfromname{reduceSweep}};",
        ]
    lines += end_tikz(resize=False)
    lines += [
        r"  \caption{Median percentage of nodes and edges removed "
        r"as $k$ and $\theta$ vary.}",
        r"  \label{fig:reduce-sweep}",
    ]
    lines += end_float(env)
    return lines


def run(path, output):
    path = Path(path)
    if path.is_dir():
        write_tex("\n".join(render_sweep(path)) + "\n", output)
        return
    data = load_json(path)
    if data is None:
        print(f"# reduction table: could not read {path}", file=sys.stderr)
        write_tex(render([]), output)
        return
    if isinstance(data, dict):
        data = data.get("rows") or data.get("graphs") or []
    if not isinstance(data, list):
        print(f"# reduction table: expected a list in {path}", file=sys.stderr)
        data = []
    write_tex(render(data), output)
