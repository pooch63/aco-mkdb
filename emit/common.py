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


def write_tex(tex, output):
    """Write LaTeX to output path, or print to stdout when output is None."""
    if output:
        with open(output, "w") as f:
            f.write(tex + "\n")
    else:
        print(tex)


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
    """Map compare_k2t5i_… → vary_k2t5i_… next to the compare directory."""
    base = os.path.basename(os.path.abspath(compare_dir.rstrip(os.sep)))
    if not base.startswith("compare_"):
        return None
    sibling = os.path.join(
        os.path.dirname(os.path.abspath(compare_dir)),
        "vary_" + base[len("compare_"):],
    )
    return sibling if os.path.isdir(sibling) else None
