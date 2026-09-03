"""
compare mode — vary.jl ant-count JSON → complexity figures.

Pass a single results directory (e.g. ``vary_k2t5i_PN``), like table or
statistics mode. Select plot groups with ``--plots`` (comma-separated):

  theta-time
    - θ-heuristic wall time vs θ(|U_R|+|V_R|)+|E_R|

  deg-size-time
    - ratio of discovery time to each bound vs |U_R|+|V_R| (matched graphs)

  density-size
    - edge density vs |E(D*)|

  max-deg-time
    - max reduced degree vs ACO-PN discovery time

  flag-ablation
    - 2-panel groupplot (quality | discovery time) across ACO, ACO-P,
      ACO-N, and ACO-PN at a fixed ant count. Requires --flag-dir for
      each variant (see build.json flag_dirs).

ACO points use the same best-trial / discovery rules as table mode.

Layout
------
  helpers.py       shared metrics and pgfplots primitives
  complexity.py    time / complexity scaling figures
  flag_ablation.py P/N flag ablation figure
"""

from __future__ import annotations

import sys

from ..common import display_name, load_json, report_skipped, write_tex
from ..result_fields import validate_compare_directory
from ..table import summarize_file
from . import complexity, flag_ablation

PLOT_GROUPS = complexity.PLOT_GROUPS + flag_ablation.PLOT_GROUPS


def _parse_plots(plots):
    """Validate and normalize a comma-separated plot-group list."""
    if plots is None:
        return list(PLOT_GROUPS)
    selected = []
    for raw in plots.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in PLOT_GROUPS:
            raise SystemExit(
                f"Unknown compare plot group {name!r}; "
                f"choose from: {', '.join(PLOT_GROUPS)}"
            )
        if name not in selected:
            selected.append(name)
    if not selected:
        raise SystemExit("No compare plot groups selected.")
    return selected


def build_compare_plots(named_rows, plots=None, flag_ablation_matched=None):
    """
    Build selected compare figures from [(name, row), ...].

    Plot groups are listed in ``PLOT_GROUPS``; pass ``plots`` as a
    comma-separated subset (e.g. ``theta-time,deg-size-time``).
    """
    selected = _parse_plots(plots)
    builders = {**complexity.BUILDERS, **flag_ablation.BUILDERS}

    parts = []
    for name in selected:
        if name in complexity.BUILDERS:
            block = builders[name](named_rows)
        else:
            block = builders[name](flag_ablation_matched or [])
        if not block:
            continue
        if parts:
            parts.append("")
        parts += block

    return "\n".join(parts)


def run(json_paths, output, ants=None, plots=None, flag_dirs=None):
    selected = _parse_plots(plots) if plots else list(PLOT_GROUPS)
    matched_ablation = None
    skipped = []

    if "flag-ablation" in selected:
        if not flag_dirs:
            raise SystemExit(
                "flag-ablation plot requires --flag-dir=LABEL=DIR for "
                "ACO, ACO-P, ACO-N, and ACO-PN"
            )
        matched_ablation, flag_skipped = (
            flag_ablation.load_flag_ablation_matched(flag_dirs, ants=ants)
        )
        skipped.extend(flag_skipped)
        if not matched_ablation:
            print(
                "Warning: flag-ablation: no graphs matched across all four "
                "variant directories.",
                file=sys.stderr,
            )

    named_rows = []
    if any(p != "flag-ablation" for p in selected):
        for path in json_paths:
            data = load_json(path)
            if data is None:
                skipped.append((path, "unreadable"))
                continue
            row = summarize_file(data, ants=ants)
            if row is None:
                skipped.append((path, "not a vary.jl ant-count result"))
                continue
            named_rows.append((display_name(path, data), row))

        if not named_rows:
            raise SystemExit("No usable vary JSON files -- nothing to plot.")

        validate_compare_directory(json_paths, selected)

    print(
        f"# compare: {len(named_rows)} dataset(s)"
        + (f"; ants={ants}" if ants is not None else "")
        + (
            f"; flag-ablation matched={len(matched_ablation or [])}"
            if "flag-ablation" in selected
            else ""
        ),
        file=sys.stderr,
    )

    write_tex(
        build_compare_plots(
            named_rows,
            plots=plots,
            flag_ablation_matched=matched_ablation,
        ),
        output,
    )
    report_skipped(skipped)
