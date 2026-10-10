"""
Vertically dense full-width tables.

Long per-graph tables are drawn as several side-by-side copies of the same
columns, so the row count falls by about the number of blocks. The block
count is the largest that still fits in ``WIDE_IN`` inches at a readable
font. That width is wider than the 6.5in text block (1in margins): the
fragment centers a box of ``WIDE_IN`` on ``\\textwidth`` so the extra width
uses the side margins.

``main.tex`` switches the appendix to one column before these fragments, and
redefines ``\\@makecaption`` to a single line. Each fragment is an in-flow
``\\captionof`` plus the tabular, not a float, so the tables stack with only
the space this module inserts.
"""

from __future__ import annotations

import math
import re
import sys

# Letter paper, body geometry margin=1in → textwidth 6.5in. 7.5in leaves
# about half an inch of physical margin once the box is centered.
WIDE_IN = 7.5
WIDE_PT = WIDE_IN * 72.27

MIN_FONT_PT = 4.5
FONT_CHOICES = (6.0, 5.5, 5.25, 5.0, 4.75, 4.5)
TABCOLSEP_PT = 0.85
PANEL_GAP_PT = 4.2

PREAMBLE = "\\usepackage{adjustbox}\n\\usepackage{capt-of}"

_CMD = re.compile(r"\\[a-zA-Z]+")


def cell_em(cell: str) -> float:
    """Approximate rendered width in em."""
    raw = cell or ""
    extra = 0.35 if "dagger" in raw else 0.0
    s = raw.replace(r"\textbf{", "")
    s = _CMD.sub("", s)
    s = s.replace("$", "").replace("{", "").replace("}", "").replace("^", "")
    width = 0.0
    for ch in s:
        if ch.isdigit() or ch in ".-+":
            width += 0.50
        elif ch == " ":
            width += 0.33
        elif ch.isupper():
            width += 0.72
        else:
            width += 0.48
    return width + extra


def _column_ems(headers, sections) -> list[float]:
    rows = [headers]
    for section in sections:
        rows.extend(section)
    n_cols = len(headers)
    ems = [0.0] * n_cols
    for row in rows:
        if len(row) != n_cols:
            raise ValueError(
                f"row has {len(row)} cells, header has {n_cols}"
            )
        for i, cell in enumerate(row):
            ems[i] = max(ems[i], cell_em(cell))
    return ems


def _panel_width_pt(col_ems, font_pt: float) -> float:
    # Side bearings plus the inter-column \\tabcolsep that @{} does not eat.
    content = sum((em + 0.08) * font_pt for em in col_ems)
    gaps = max(0, len(col_ems) - 1) * 2 * TABCOLSEP_PT
    return content + gaps


def choose_layout(headers, sections) -> tuple[int, float, float]:
    """
    Return (n_panels, font_pt, estimated_width_pt).

    Fewer rows wins. At the same row count, the larger font wins.
    """
    n_rows = sum(len(section) for section in sections)
    if n_rows <= 0:
        raise ValueError("dense table has no rows")
    col_ems = _column_ems(headers, sections)
    best = None
    max_panels = min(8, n_rows)
    for n_panels in range(1, max_panels + 1):
        rows_per = math.ceil(n_rows / n_panels)
        for font in FONT_CHOICES:
            if font < MIN_FONT_PT:
                continue
            panel = _panel_width_pt(col_ems, font)
            total = n_panels * panel + (n_panels - 1) * PANEL_GAP_PT
            # A few percent of shrink is fine; adjustbox does it. Below the
            # font floor, do not spend the slack.
            limit = WIDE_PT * (1.04 if font >= 5.0 else 1.0)
            if total > limit:
                continue
            key = (rows_per, -font, n_panels)
            if best is None or key < best[0]:
                best = (key, n_panels, font, total)
    if best is None:
        return 1, MIN_FONT_PT, _panel_width_pt(col_ems, MIN_FONT_PT)
    return best[1], best[2], best[3]


def _flatten(sections):
    """(rule_before, cells) in table order. The first row has no rule."""
    flat = []
    for si, section in enumerate(sections):
        for ri, cells in enumerate(section):
            flat.append((si > 0 and ri == 0, list(cells)))
    return flat


def _split_panels(flat, n_panels):
    """
    Nearly equal contiguous blocks.

    If a block would open on a section break, pull the previous row across
    so the rule has a row above it. That can make one block a row longer.
    """
    n = len(flat)
    base, extra = divmod(n, n_panels)
    panels = []
    i = 0
    for p in range(n_panels):
        take = base + (1 if p < extra else 0)
        panels.append(flat[i : i + take])
        i += take
    for p in range(1, n_panels):
        if not panels[p] or not panels[p][0][0]:
            continue
        if len(panels[p - 1]) <= 1:
            continue
        moved = panels[p - 1].pop()
        if moved[0]:
            panels[p - 1].append(moved)
            continue
        panels[p].insert(0, moved)
    return panels


def _group_cells(groups):
    cells = []
    rules = []
    col = 1
    for span, title in groups:
        if title:
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{{title}}}")
            rules.append((col, col + span - 1))
        elif span == 1:
            cells.append("")
        else:
            cells.append(rf"\multicolumn{{{span}}}{{c}}{{}}")
        col += span
    return cells, rules


def _panel_col_spec(n_cols: int) -> str:
    return "l" + "r" * (n_cols - 1)


def _full_col_spec(n_cols: int, n_panels: int) -> str:
    one = _panel_col_spec(n_cols)
    sep = r"@{\hspace{1.6pt}\vrule width 0.15pt\hspace{1.6pt}}"
    return "@{}" + sep.join([one] * n_panels) + "@{}"


def render_dense_table(
    *,
    caption: str,
    label: str,
    column_headers: list[str],
    sections: list[list[list[str]]],
    groups: list[tuple[int, str | None]] | None = None,
) -> str:
    """In-flow wide table: caption plus side-by-side row blocks."""
    sections = [s for s in sections if s]
    if not sections:
        raise ValueError(f"dense table {label} has no rows")
    n_cols = len(column_headers)
    if groups is not None and sum(span for span, _ in groups) != n_cols:
        raise ValueError(
            f"{label}: group spans {sum(span for span, _ in groups)} "
            f"!= {n_cols} columns"
        )

    n_panels, font, est = choose_layout(column_headers, sections)
    leading = round(font * 1.14, 2)
    flat = _flatten(sections)
    panels = _split_panels(flat, n_panels)
    rows_per = max(len(panel) for panel in panels)

    print(
        f"# dense {label}: {n_panels} block(s), {font:g}pt, "
        f"{rows_per} rows, ~{est:.0f}pt wide",
        file=sys.stderr,
    )

    group_cells = group_rules = None
    if groups is not None:
        group_cells, group_rules = _group_cells(groups)

    lines = [
        r"\par\vspace{2pt}",
        r"\noindent\makebox[\textwidth][c]{%",
        rf"\begin{{minipage}}{{{WIDE_IN}in}}",
        r"\centering",
        r"\captionof{table}{" + caption + "}",
        rf"\label{{{label}}}",
        r"\begin{adjustbox}{max width=\linewidth}%",
        rf"\begingroup\fontsize{{{font:g}}}{{{leading:g}}}\selectfont",
        rf"\setlength{{\tabcolsep}}{{{TABCOLSEP_PT}pt}}",
        r"\setlength{\aboverulesep}{0.25pt}",
        r"\setlength{\belowrulesep}{0.25pt}",
        r"\setlength{\abovetopsep}{0pt}",
        r"\setlength{\belowbottomsep}{0pt}",
        r"\setlength{\heavyrulewidth}{0.3pt}",
        r"\setlength{\lightrulewidth}{0.15pt}",
        r"\setlength{\cmidrulewidth}{0.15pt}",
        rf"\begin{{tabular}}{{{_full_col_spec(n_cols, n_panels)}}}",
        r"    \toprule",
    ]

    def join_panels(per_panel_cells):
        return "    " + " & ".join(
            " & ".join(cells) for cells in per_panel_cells
        ) + r" \\"

    if group_cells is not None:
        lines.append(join_panels([group_cells] * n_panels))
        rules = []
        for p in range(n_panels):
            offset = p * n_cols
            for start, end in group_rules:
                rules.append(
                    rf"\cmidrule(lr){{{offset + start}-{offset + end}}}"
                )
        lines.append("    " + " ".join(rules))

    lines.append(join_panels([column_headers] * n_panels))
    lines.append(r"    \midrule")

    for i in range(rows_per):
        if i > 0:
            rules = []
            for p, panel in enumerate(panels):
                if i < len(panel) and panel[i][0]:
                    offset = p * n_cols
                    rules.append(
                        rf"\cmidrule(lr){{{offset + 1}-{offset + n_cols}}}"
                    )
            if rules:
                lines.append("    " + " ".join(rules))
        row_cells = []
        for panel in panels:
            if i < len(panel):
                row_cells.extend(panel[i][1])
            else:
                row_cells.extend([""] * n_cols)
        lines.append("    " + " & ".join(row_cells) + r" \\")

    lines += [
        r"    \bottomrule",
        r"\end{tabular}%",
        r"\endgroup",
        r"\end{adjustbox}",
        r"\end{minipage}}",
        r"\par",
    ]
    return "\n".join(lines)
