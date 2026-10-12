#!/usr/bin/env python3
"""
Build the paper from main.tex placeholders and emit-generated fragments.

Workflow:
  1. emit   — run `python -m emit` for each fragment in build.json
  2. assemble — substitute %%NAME%% placeholders → build.tex
  3. pdf    — compile build.tex with latexmk, then remove build.tex
  4. (default) all three steps

Emit is make-style and parallel: a fragment is regenerated only when its
output is missing or older than ``build.json``, any dependency results folder
(newest file mtime under that folder), or the emit Python sources that build
that fragment (mode module + COMPARE plot module when applicable).
Stale jobs run concurrently (``-j`` / ``--jobs``, default: CPU count).
Use ``--force`` to rebuild all.

Use --keep-tex to retain build.tex after compiling.
Use --tables to also assemble and compile tables.tex (landscape per-graph
tables) into tables.pdf. main.tex includes the same tables in a short
side-by-side form (``:compact``); the landscape pages stay opt-in.

Placeholders in main.tex and tables.tex use the form %%QUALITY%%,
%%COMPARE:theta-time%%, etc.
%%PREAMBLE%% aggregates emit sidecar snippets (*.preamble.tex) into the
document preamble — not a normal emit target.

Experiment paths in build.json are usually bare names under results_dir
(e.g. ``"input": "vary_k2t5i_PN"`` with ``"results_dir": "../results"``).
Change results_dir once to point at another tree (e.g. ``../results_old``).
Absolute paths and ``./`` / ``../`` paths are still resolved from paper/.

STATISTICS missing-at-* fields use missing_at_base (plain ACO dir; $F_N$ is
that path + "N"). Falls back to vary_base if missing_at_base is omitted.
STATISTICS pivot-tested / pivot-excluded fields use compare_dir (compare-seeds
JSON) together with the vary input directory.
COMPARE k-sweep / theta-sweep / density-wins / param-runtime use param_dirs
(labeled vary_* suites). method-density also reads method_results, a
compare-methods JSON, and the primary vary directory for ACO.
METHODS:<dir> reads load.jl --benchmark JSON from results_dir/<dir>; the
fragment's "methods" list sets column order (first = focal method).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
PLACEHOLDER_RE = re.compile(r"%%([A-Z][A-Z0-9_]*)(?::([^%]+))?%%")
AUX_SUFFIXES = (".aux", ".bbl", ".blg", ".log", ".fls", ".fdb_latexmk", ".out")
# Assembled from fragment sidecars (*.preamble.tex); not a normal emit target.
SPECIAL_PLACEHOLDERS = frozenset({"PREAMBLE"})
DEFAULT_RESULTS_DIR = "../results"
# COMPARE plots that read flag_dirs / param_dirs instead of (or besides) input.
COMPARE_FLAG_PLOTS = frozenset(
    {
        "flag-ablation",
        "flag-ablation-replicates",
        "flag-feasibility",
        "flag-feasibility-replicates",
    }
)
COMPARE_PARAM_PLOTS = frozenset(
    {
        "k-sweep",
        "theta-sweep",
        "density-wins",
        "param-runtime",
    }
)
# Win-rate and runtime sweeps that join extra methods from method_sweep_dir.
METHOD_SWEEP_PLOTS = frozenset(
    {
        "k-sweep",
        "theta-sweep",
        "param-runtime",
    }
)
# Epoch-trace figures. Input is budget_dir, not the ant-count sweep.
COMPARE_BUDGET_PLOTS = frozenset(
    {
        "budget",
        "pheromone-ablation",
    }
)
STATISTICS_MISSING_AT_FIELDS = frozenset(
    {
        "missing-at-5",
        "missing-at-10",
        "aco-missing-at-5",
        "aco-n-missing-at-5",
        "aco-missing-at-10",
        "aco-n-missing-at-10",
    }
)

EMIT_DIR = REPO_ROOT / "emit"
# Shared by every ``python -m emit`` invocation.
EMIT_SHARED_SOURCES = (
    EMIT_DIR / "__init__.py",
    EMIT_DIR / "__main__.py",
    EMIT_DIR / "common.py",
    EMIT_DIR / "result_fields.py",
)
# Mode entry modules (plus extras each mode imports for output).
EMIT_MODE_SOURCES: dict[str, tuple[Path, ...]] = {
    "quality": (
        EMIT_DIR / "quality.py",
        EMIT_DIR / "compare" / "budget_curves.py",
    ),
    "seed-compare": (
        EMIT_DIR / "seed_compare.py",
        EMIT_DIR / "dense_table.py",
    ),
    "table": (
        EMIT_DIR / "table.py",
        EMIT_DIR / "dense_table.py",
    ),
    "methods": (
        EMIT_DIR / "methods.py",
        EMIT_DIR / "dense_table.py",
    ),
    "ablation": (EMIT_DIR / "ablation.py",),
    "reduce": (EMIT_DIR / "reduce.py",),
    "heuristics": (EMIT_DIR / "heuristics.py",),
    "metrics": (EMIT_DIR / "metrics.py",),
    "epoch-table": (
        EMIT_DIR / "epoch_table.py",
        EMIT_DIR / "compare" / "budget_curves.py",
    ),
    "statistics": (
        EMIT_DIR / "statistics.py",
        EMIT_DIR / "table.py",
        EMIT_DIR / "seed_compare.py",
    ),
    "compare": (
        EMIT_DIR / "compare" / "__init__.py",
        EMIT_DIR / "compare" / "helpers.py",
        EMIT_DIR / "table.py",
    ),
}
# COMPARE ``--plots=`` group → implementing module.
COMPARE_PLOT_SOURCES: dict[str, Path] = {
    "theta-time": EMIT_DIR / "compare" / "complexity.py",
    "deg-size-time": EMIT_DIR / "compare" / "complexity.py",
    "bound-time": EMIT_DIR / "compare" / "complexity.py",
    "density-size": EMIT_DIR / "compare" / "complexity.py",
    "max-deg-time": EMIT_DIR / "compare" / "complexity.py",
    "flag-ablation": EMIT_DIR / "compare" / "flag_ablation.py",
    "flag-ablation-replicates": EMIT_DIR / "compare" / "flag_ablation.py",
    "flag-feasibility": EMIT_DIR / "compare" / "flag_ablation.py",
    "flag-feasibility-replicates": EMIT_DIR / "compare" / "flag_ablation.py",
    "budget": EMIT_DIR / "compare" / "budget_curves.py",
    "pheromone-ablation": EMIT_DIR / "compare" / "budget_curves.py",
    "k-sweep": EMIT_DIR / "compare" / "param_sweep.py",
    "theta-sweep": EMIT_DIR / "compare" / "param_sweep.py",
    "density-wins": EMIT_DIR / "compare" / "param_sweep.py",
    "method-density": EMIT_DIR / "compare" / "param_sweep.py",
    "method-size": EMIT_DIR / "compare" / "param_sweep.py",
    "param-runtime": EMIT_DIR / "compare" / "param_sweep.py",
}


def fragment_output_name(name: str, args: str | None) -> str:
    """Map placeholder (name, args) → generated/*.tex basename."""
    if not args:
        return name.lower()
    safe = (
        args.strip()
        .replace(",", "__")
        .replace(":", "__")
        .replace(" ", "")
    )
    return f"{name.lower()}__{safe}"


def parse_table_args(args: str | None) -> tuple[str, str]:
    """
    Split %%TABLE:…%% args into (path_suffix, subset).

    Examples: ``k2t5i_PN`` → (``k2t5i_PN``, ``full``);
    ``k2t5i_PN:highlights`` → (``k2t5i_PN``, ``highlights``);
    ``k2t5i_PN:compact`` → (``k2t5i_PN``, ``compact``).
    METHODS args use the same suffix (``dir:compact``).
    """
    if not args:
        return "", "full"
    raw = args.strip()
    for subset in ("highlights", "compact", "sizes", "full"):
        marker = f":{subset}"
        if raw.endswith(marker):
            return raw[: -len(marker)], subset
    return raw, "full"


def placeholders_in_source(source_text: str) -> list[tuple[str, str | None]]:
    """Unique (name, args) pairs in document order."""
    seen: set[tuple[str, str | None]] = set()
    ordered: list[tuple[str, str | None]] = []
    for name, args in PLACEHOLDER_RE.findall(source_text):
        key = (name, args or None)
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def load_config() -> dict:
    path = PAPER_DIR / "build.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve(path_str: str) -> Path:
    return (PAPER_DIR / path_str).resolve()


def results_root(cfg: dict) -> Path:
    """Absolute path to the experiment results tree (default: ../results)."""
    return resolve(cfg.get("results_dir", DEFAULT_RESULTS_DIR))


def resolve_results_path(cfg: dict, path_str: str) -> Path:
    """
    Resolve an experiment path from build.json.

    Bare names (``vary_k2t5i_PN``) join under results_dir. Absolute paths and
    paths starting with ``./`` or ``../`` stay paper-relative for compatibility.
    """
    path = Path(path_str)
    if path.is_absolute() or path_str.startswith(("./", "../")):
        return resolve(path_str)
    return (results_root(cfg) / path_str).resolve()


def fragment_input_path(cfg: dict, name: str, frag: dict, args: str | None) -> Path:
    """
    Resolve emit input directory; TABLE uses vary_base + k/θ/flags suffix,
    METHODS uses its placeholder arg as the results directory name.
    """
    if name == "COMPARE" and args in COMPARE_BUDGET_PLOTS:
        budget = frag.get("budget_dir")
        if not budget:
            raise SystemExit(
                "COMPARE budget plots require budget_dir in build.json"
            )
        return resolve_results_path(cfg, budget)
    if name == "METHODS" and args:
        directory, _subset = parse_table_args(args)
        return resolve_results_path(cfg, directory)
    if name == "REDUCE" and (not args or args == "sweep"):
        sweep = frag.get("sweep") or "reduce_sweep"
        return resolve_results_path(cfg, sweep)
    if name == "TABLE":
        base = cfg.get("vary_base")
        if not base:
            raise SystemExit("TABLE fragment requires vary_base in build.json")
        suffix, _subset = parse_table_args(args)
        return resolve_results_path(cfg, base + suffix)
    input_path = frag.get("input")
    if not input_path:
        raise SystemExit(f"Fragment {name} has no input path in build.json")
    return resolve_results_path(cfg, input_path)


def newest_mtime(path: Path, _cache: dict[Path, float | None] | None = None) -> float | None:
    """
    Newest mtime of ``path`` (file) or any file under it (directory).

    Returns None when the path does not exist. Directory mtime alone is not
    enough: overwriting a JSON file updates the file but often not the folder.
    """
    if _cache is not None and path in _cache:
        return _cache[path]
    try:
        st = path.stat()
    except OSError:
        if _cache is not None:
            _cache[path] = None
        return None
    if path.is_file():
        if _cache is not None:
            _cache[path] = st.st_mtime
        return st.st_mtime
    if not path.is_dir():
        if _cache is not None:
            _cache[path] = st.st_mtime
        return st.st_mtime
    newest = st.st_mtime
    for child in path.rglob("*"):
        try:
            newest = max(newest, child.stat().st_mtime)
        except OSError:
            continue
    if _cache is not None:
        _cache[path] = newest
    return newest


def is_stale(
    out_path: Path,
    dep_paths: list[Path],
    *,
    mtime_cache: dict[Path, float | None] | None = None,
) -> bool:
    """True when output is missing or older than any dependency (make-style)."""
    if not out_path.is_file():
        return True
    out_mtime = out_path.stat().st_mtime
    for dep in dep_paths:
        dep_mtime = newest_mtime(dep, mtime_cache)
        if dep_mtime is None:
            # Missing input: re-run so emit can warn / write fallbacks.
            return True
        if dep_mtime > out_mtime:
            return True
    return False


def fragment_emit_source_deps(
    name: str,
    frag: dict,
    args: str | None,
) -> list[Path]:
    """
    Emit Python sources whose mtime invalidates this fragment.

    Ties each figure/table to the mode (and COMPARE plot module) that
    builds it, plus shared emit entrypoints and ``build.py`` itself.
    """
    deps: list[Path] = [
        PAPER_DIR / "build.py",
        *EMIT_SHARED_SOURCES,
    ]
    mode = frag.get("emit")
    if mode in EMIT_MODE_SOURCES:
        deps.extend(EMIT_MODE_SOURCES[mode])

    if name == "COMPARE" or mode == "compare":
        plots = args or frag.get("plots") or ""
        plot_names = {p.strip() for p in str(plots).split(",") if p.strip()}
        if not plot_names:
            plot_names = set(COMPARE_PLOT_SOURCES)
        for plot in plot_names:
            src = COMPARE_PLOT_SOURCES.get(plot)
            if src is not None:
                deps.append(src)
        # param_sweep imports pgfplots helpers from flag_ablation.
        if plot_names & COMPARE_PARAM_PLOTS:
            deps.append(EMIT_DIR / "compare" / "flag_ablation.py")

    return deps


def heuristics_param_dirs(cfg: dict, frag: dict) -> dict:
    """Vary suites joined onto each method-sweep table.

    A fragment-level ``param_dirs`` wins. Otherwise use the compare
    sweep list, which is the paper's $(k, \\theta)$ suite map.
    """
    if frag.get("param_dirs"):
        return frag["param_dirs"]
    compare = (cfg.get("fragments") or {}).get("COMPARE") or {}
    return compare.get("param_dirs") or {}


def fragment_dep_paths(
    cfg: dict,
    name: str,
    frag: dict,
    args: str | None,
) -> list[Path]:
    """Results folders, emit sources, and build.json that invalidate a fragment."""
    deps: list[Path] = [PAPER_DIR / "build.json"]
    deps.extend(fragment_emit_source_deps(name, frag, args))
    plots = args or frag.get("plots") or ""
    plot_names = {p.strip() for p in str(plots).split(",") if p.strip()}

    if name == "COMPARE" and plot_names:
        if plot_names & COMPARE_FLAG_PLOTS and frag.get("flag_dirs"):
            for path in frag["flag_dirs"].values():
                deps.append(resolve_results_path(cfg, path))
        if plot_names & COMPARE_PARAM_PLOTS and frag.get("param_dirs"):
            for path in frag["param_dirs"].values():
                deps.append(resolve_results_path(cfg, path))
        if frag.get("method_sweep_dir") and plot_names & METHOD_SWEEP_PLOTS:
            sweep_path = resolve_results_path(cfg, frag["method_sweep_dir"])
            if sweep_path.exists():
                deps.append(sweep_path)
        if plot_names & COMPARE_BUDGET_PLOTS and frag.get("ablation_dirs"):
            for path in frag["ablation_dirs"].values():
                deps.append(resolve_results_path(cfg, path))
        if plot_names - (COMPARE_FLAG_PLOTS | COMPARE_PARAM_PLOTS):
            deps.append(fragment_input_path(cfg, name, frag, args))
        if {"method-density", "method-size"} & plot_names and frag.get("method_results"):
            deps.append(resolve_results_path(cfg, frag["method_results"]))
    else:
        deps.append(fragment_input_path(cfg, name, frag, args))
        if name == "REDUCE":
            deps.append(REPO_ROOT / "scripts" / "kt-sweep.bash")
            if not args and frag.get("input"):
                deps.append(resolve_results_path(cfg, frag["input"]))
        if name == "QUALITY" and args == "sweeps" and frag.get("ablation_dirs"):
            for path in frag["ablation_dirs"].values():
                deps.append(resolve_results_path(cfg, path))
        elif name == "QUALITY" and frag.get("budget_dir"):
            deps.append(resolve_results_path(cfg, frag["budget_dir"]))
        if frag.get("flag_dirs"):
            for path in frag["flag_dirs"].values():
                deps.append(resolve_results_path(cfg, path))
        if frag.get("param_dirs"):
            for path in frag["param_dirs"].values():
                deps.append(resolve_results_path(cfg, path))

    if frag.get("vary_dir"):
        deps.append(resolve_results_path(cfg, frag["vary_dir"]))
    if name == "HEURISTICS":
        for path in heuristics_param_dirs(cfg, frag).values():
            deps.append(resolve_results_path(cfg, path))

    if name == "STATISTICS" and args in STATISTICS_MISSING_AT_FIELDS:
        missing_at_base = (
            frag.get("missing_at_base")
            or frag.get("vary_base")
            or cfg.get("missing_at_base")
            or cfg.get("vary_base")
        )
        if missing_at_base:
            base = resolve_results_path(cfg, missing_at_base)
            deps.append(base)
            deps.append(Path(str(base) + "N"))

    if name == "STATISTICS" and args and args.startswith("pivot-"):
        compare_dir = frag.get("compare_dir") or cfg.get("compare_dir")
        if compare_dir:
            deps.append(resolve_results_path(cfg, compare_dir))
    if name == "METRICS" and frag.get("compare_dir"):
        deps.append(resolve_results_path(cfg, frag["compare_dir"]))

    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in deps:
        if path in seen:
            continue
        seen.add(path)
        unique.append(path)
    return unique


@dataclass(frozen=True)
class EmitJob:
    label: str
    cmd: list[str]
    out_path: Path
    dep_paths: list[Path]


def build_emit_job(
    cfg: dict,
    name: str,
    frag: dict,
    args: str | None,
    generated: Path,
) -> EmitJob:
    out_name = fragment_output_name(name, args)
    out_path = generated / f"{out_name}.tex"
    input_path = fragment_input_path(cfg, name, frag, args)

    cmd = [
        sys.executable,
        "-m",
        "emit",
        frag["emit"],
        str(input_path),
        "-o",
        str(out_path),
    ]
    if frag.get("vary_dir"):
        cmd.append(
            f"--vary-dir={resolve_results_path(cfg, frag['vary_dir'])}"
        )
    plots = args or frag.get("plots")
    if plots and name not in ("SEED_COMPARE", "STATISTICS", "TABLE", "METHODS"):
        cmd.append(f"--plots={plots}")
    if frag.get("methods"):
        cmd.append(f"--methods={','.join(frag['methods'])}")
    if name == "SEED_COMPARE" and args:
        cmd.append(f"--subset={args}")
    if name == "TABLE":
        _suffix, subset = parse_table_args(args)
        if subset != "full":
            cmd.append(f"--subset={subset}")
    if name == "METHODS":
        _directory, subset = parse_table_args(args)
        if subset != "full":
            cmd.append(f"--subset={subset}")
    if frag.get("ants") is not None:
        cmd.append(f"--ants={frag['ants']}")
    # Per-fragment override, else top-level build.json max_replicates.
    max_repl = frag.get("max_replicates", cfg.get("max_replicates"))
    if max_repl is not None:
        cmd.append(f"--max-replicates={int(max_repl)}")
    if name == "QUALITY" and args == "sweeps" and frag.get("ablation_dirs"):
        for label, path in frag["ablation_dirs"].items():
            cmd.append(
                f"--ablation-dir={label}={resolve_results_path(cfg, path)}"
            )
    if name == "QUALITY" and args != "sweeps" and frag.get("budget_dir"):
        cmd.append(
            f"--budget-dir={resolve_results_path(cfg, frag['budget_dir'])}"
        )
    if frag.get("flag_dirs"):
        for label, path in frag["flag_dirs"].items():
            cmd.append(
                f"--flag-dir={label}={resolve_results_path(cfg, path)}"
            )
    param_dirs = frag.get("param_dirs")
    if name == "HEURISTICS" and not param_dirs:
        param_dirs = heuristics_param_dirs(cfg, frag)
    if param_dirs:
        for label, path in param_dirs.items():
            cmd.append(
                f"--param-dir={label}={resolve_results_path(cfg, path)}"
            )
    if name == "COMPARE" and frag.get("method_results") and (
        args is None
        or {"method-density", "method-size"} & {p.strip() for p in str(args).split(",")}
    ):
        cmd.append(
            f"--method-results={resolve_results_path(cfg, frag['method_results'])}"
        )
    if name == "COMPARE" and frag.get("method_sweep_dir"):
        sweep_plots = {p.strip() for p in str(args or "").split(",") if p.strip()}
        if not sweep_plots or sweep_plots & METHOD_SWEEP_PLOTS:
            cmd.append(
                "--method-sweep-dir="
                + str(resolve_results_path(cfg, frag["method_sweep_dir"]))
            )
    if name == "COMPARE" and frag.get("runtime_sweeps"):
        for spec in frag["runtime_sweeps"].get("k") or []:
            vals = ",".join(str(int(v)) for v in spec["values"])
            cmd.append(f"--runtime-sweep=k:{int(spec['theta'])}:{vals}")
        for spec in frag["runtime_sweeps"].get("theta") or []:
            vals = ",".join(str(int(v)) for v in spec["values"])
            cmd.append(f"--runtime-sweep=theta:{int(spec['k'])}:{vals}")
    if name == "COMPARE" and args == "pheromone-ablation":
        for label, path in (frag.get("ablation_dirs") or {}).items():
            cmd.append(
                f"--trace-dir={label}={resolve_results_path(cfg, path)}"
            )
    if name == "REDUCE" and not args and frag.get("input"):
        extra = resolve_results_path(cfg, frag["input"])
        if extra.is_file() and extra != input_path:
            cmd.append(f"--also={extra}")
    if name == "STATISTICS" and args:
        cmd.append(f"--field={args}")
    if name == "ABLATION" and args:
        cmd.append(f"--field={args}")
    if name == "STATISTICS" and args in STATISTICS_MISSING_AT_FIELDS:
        missing_at_base = (
            frag.get("missing_at_base")
            or frag.get("vary_base")
            or cfg.get("missing_at_base")
            or cfg.get("vary_base")
        )
        if missing_at_base:
            cmd.append(
                f"--vary-base={resolve_results_path(cfg, missing_at_base)}"
            )
    if name == "STATISTICS" and args and args.startswith("pivot-"):
        compare_dir = frag.get("compare_dir") or cfg.get("compare_dir")
        if compare_dir:
            cmd.append(
                f"--compare-dir={resolve_results_path(cfg, compare_dir)}"
            )
    if name == "METRICS" and frag.get("compare_dir"):
        cmd.append(
            f"--compare-dir={resolve_results_path(cfg, frag['compare_dir'])}"
        )

    label = name if not args else f"{name}:{args}"
    return EmitJob(
        label=label,
        cmd=cmd,
        out_path=out_path,
        dep_paths=fragment_dep_paths(cfg, name, frag, args),
    )


def _run_emit_job(job: EmitJob) -> str:
    """Run one emit subprocess; return a short status line for stderr."""
    rel = job.out_path.relative_to(PAPER_DIR)
    result = subprocess.run(
        job.cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        sys.stderr.write(result.stdout)
        if not result.stdout.endswith("\n"):
            sys.stderr.write("\n")
    if result.stderr:
        sys.stderr.write(result.stderr)
        if not result.stderr.endswith("\n"):
            sys.stderr.write("\n")
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            job.cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return f"# emit {job.label} → {rel}"


def source_names(cfg: dict, *, tables: bool) -> list[str]:
    """LaTeX sources whose placeholders should be emitted and assembled."""
    names = [cfg["source"]]
    if tables:
        names.append(cfg.get("tables_source", "tables.tex"))
    return names


def read_source(name: str) -> str:
    path = PAPER_DIR / name
    if not path.is_file():
        raise SystemExit(f"Missing LaTeX source: {name}")
    return path.read_text(encoding="utf-8")


def run_emit(
    cfg: dict,
    *,
    force: bool = False,
    jobs: int | None = None,
    sources: list[str] | None = None,
) -> None:
    generated = resolve(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)
    source_text = "\n".join(read_source(name) for name in (sources or [cfg["source"]]))

    planned: list[EmitJob] = []
    for name, args in placeholders_in_source(source_text):
        if name in SPECIAL_PLACEHOLDERS:
            continue
        frag = cfg["fragments"].get(name)
        if frag is None:
            continue
        planned.append(build_emit_job(cfg, name, frag, args, generated))

    stale: list[EmitJob] = []
    skipped = 0
    mtime_cache: dict[Path, float | None] = {}
    for job in planned:
        if force or is_stale(job.out_path, job.dep_paths, mtime_cache=mtime_cache):
            stale.append(job)
        else:
            skipped += 1

    if not stale:
        print(
            f"# emit: {skipped} fragment(s) up to date, nothing to rebuild",
            file=sys.stderr,
        )
        return

    workers = jobs if jobs is not None else (os.cpu_count() or 4)
    workers = max(1, workers)
    print(
        f"# emit: rebuilding {len(stale)}/{len(planned)} "
        f"fragment(s) with {workers} worker(s)"
        + (f", skipped {skipped} up to date" if skipped else ""),
        file=sys.stderr,
    )

    if workers == 1 or len(stale) == 1:
        for job in stale:
            print(_run_emit_job(job), file=sys.stderr)
        return

    errors: list[tuple[str, BaseException]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_emit_job, job): job for job in stale}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                print(fut.result(), file=sys.stderr)
            except Exception as exc:
                errors.append((job.label, exc))
                print(f"# emit FAILED {job.label}: {exc}", file=sys.stderr)

    if errors:
        labels = ", ".join(label for label, _ in errors)
        raise SystemExit(f"emit failed for: {labels}")


def collect_preamble(generated: Path, source_text: str) -> str:
    """
    Aggregate unique ``*.preamble.tex`` sidecars for fragments used in the
    document. Duplicate snippets (same body) are included once.
    """
    snippets: list[str] = []
    seen: set[str] = set()
    for name, args in placeholders_in_source(source_text):
        if name in SPECIAL_PLACEHOLDERS:
            continue
        out_name = fragment_output_name(name, args)
        path = generated / f"{out_name}.preamble.tex"
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8").strip()
        if not body or body in seen:
            continue
        seen.add(body)
        snippets.append(body)
    if not snippets:
        return "% (no preamble contributions from emit)\n"
    return "\n\n".join(snippets) + "\n"


def assemble(
    cfg: dict,
    *,
    source_name: str | None = None,
    output_name: str | None = None,
) -> Path:
    source = PAPER_DIR / (source_name or cfg["source"])
    output = PAPER_DIR / (output_name or cfg["output"])
    generated = resolve(cfg["generated_dir"])
    text = read_source(source.name)

    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        args = match.group(2) or None
        if name == "PREAMBLE":
            return collect_preamble(generated, text)
        frag = cfg["fragments"].get(name)
        if frag is None:
            missing.append(name if not args else f"{name}:{args}")
            return match.group(0)
        out_name = fragment_output_name(name, args)
        fragment_path = generated / f"{out_name}.tex"
        if not fragment_path.is_file():
            missing.append(name if not args else f"{name}:{args}")
            return match.group(0)
        body = fragment_path.read_text(encoding="utf-8").strip()
        if not body:
            missing.append(name if not args else f"{name}:{args}")
            return match.group(0)
        # STATISTICS fragments are inline numbers/phrases; a trailing
        # newline becomes a space in LaTeX and opens a gap before
        # punctuation (e.g. "23 ," / "456 ;"). Block figures/tables
        # keep the newline for clean line separation.
        if name == "STATISTICS":
            return body
        return body + "\n"

    built = PLACEHOLDER_RE.sub(replace, text)
    if missing:
        unique = sorted(set(missing))
        raise SystemExit(
            "Missing generated fragment(s): "
            + ", ".join(unique)
            + ". Run: python paper/build.py emit"
        )

    unknown = {
        name
        for name, _args in placeholders_in_source(built)
    } - set(cfg["fragments"]) - SPECIAL_PLACEHOLDERS
    if unknown:
        raise SystemExit(
            f"Placeholders in {source.name} with no build.json entry: "
            + ", ".join(sorted(unknown))
        )

    output.write_text(built, encoding="utf-8")
    print(f"# assembled {output.relative_to(PAPER_DIR)}", file=sys.stderr)
    return output


def job_stem(
    cfg: dict,
    tex_path: Path | None = None,
    *,
    pdf_name: str | None = None,
) -> str:
    tex = tex_path or (PAPER_DIR / cfg["output"])
    name = pdf_name or cfg.get("pdf", tex.with_suffix(".pdf").name)
    return Path(name).stem


def run_quiet(
    cmd: list[str],
    *,
    cwd: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command; print its output only when it fails."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        if result.stdout:
            sys.stderr.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result


def clean_aux_files(
    cfg: dict,
    tex_path: Path | None = None,
    *,
    pdf_name: str | None = None,
) -> None:
    """Remove LaTeX auxiliary files; keeps the PDF."""
    tex = tex_path or (PAPER_DIR / cfg["output"])
    stem = job_stem(cfg, tex, pdf_name=pdf_name)

    if subprocess.run(["which", "latexmk"], capture_output=True).returncode == 0:
        run_quiet(
            ["latexmk", "-silent", "-c", f"-jobname={stem}", tex.name],
            cwd=PAPER_DIR,
            check=False,
        )
    else:
        for suffix in AUX_SUFFIXES:
            (PAPER_DIR / f"{stem}{suffix}").unlink(missing_ok=True)

    print(f"# cleaned aux files for {stem}.pdf", file=sys.stderr)


def remove_build_tex(cfg: dict, tex_path: Path | None = None) -> None:
    """Remove the assembled build.tex after a successful PDF build."""
    tex = tex_path or (PAPER_DIR / cfg["output"])
    if tex.is_file():
        tex.unlink()
        print(f"# removed {tex.relative_to(PAPER_DIR)}", file=sys.stderr)


def compile_pdf(
    cfg: dict,
    tex_path: Path | None = None,
    *,
    keep_aux: bool = False,
    pdf_name: str | None = None,
) -> None:
    tex = tex_path or (PAPER_DIR / cfg["output"])
    pdf_name = pdf_name or cfg.get("pdf", tex.with_suffix(".pdf").name)
    tex.touch()
    for tool in ("latexmk", "pdflatex"):
        if subprocess.run(["which", tool], capture_output=True).returncode == 0:
            break
    else:
        raise SystemExit("Neither latexmk nor pdflatex found on PATH")

    if tool == "latexmk":
        cmd = [
            "latexmk",
            "-silent",
            "-g",
            "-pdf",
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-jobname={job_stem(cfg, tex, pdf_name=pdf_name)}",
            tex.name,
        ]
    else:
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-jobname={job_stem(cfg, tex, pdf_name=pdf_name)}",
            tex.name,
        ]
        run_quiet(cmd, cwd=PAPER_DIR)
        if subprocess.run(["which", "bibtex"], capture_output=True).returncode == 0:
            run_quiet(
                ["bibtex", job_stem(cfg, tex, pdf_name=pdf_name)],
                cwd=PAPER_DIR,
                check=False,
            )
        run_quiet(cmd, cwd=PAPER_DIR)
        run_quiet(cmd, cwd=PAPER_DIR)
        print(f"# wrote {pdf_name}", file=sys.stderr)
        if not keep_aux:
            clean_aux_files(cfg, tex, pdf_name=pdf_name)
        return

    run_quiet(cmd, cwd=PAPER_DIR)
    print(f"# wrote {pdf_name}", file=sys.stderr)
    if not keep_aux:
        clean_aux_files(cfg, tex, pdf_name=pdf_name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        nargs="?",
        choices=("all", "emit", "assemble", "pdf"),
        default="all",
        help="emit fragments, assemble build.tex, compile PDF, or all (default)",
    )
    parser.add_argument(
        "--keep-aux",
        action="store_true",
        help="keep .aux, .log, .fls, .fdb_latexmk, .out after compiling",
    )
    parser.add_argument(
        "--keep-tex",
        action="store_true",
        help="keep build.tex after compiling (removed by default)",
    )
    parser.add_argument(
        "--force",
        "-B",
        action="store_true",
        help="rebuild all emit fragments even when outputs look up to date",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="parallel emit workers (default: CPU count)",
    )
    parser.add_argument(
        "--tables",
        action="store_true",
        help="also assemble and compile tables.tex into tables.pdf",
    )
    args = parser.parse_args(argv)
    cfg = load_config()
    sources = source_names(cfg, tables=args.tables)

    if args.step in ("all", "emit"):
        run_emit(cfg, force=args.force, jobs=args.jobs, sources=sources)

    documents: list[tuple[str, str, str | None]] = [
        (cfg["source"], cfg["output"], cfg.get("pdf")),
    ]
    if args.tables:
        documents.append(
            (
                cfg.get("tables_source", "tables.tex"),
                cfg.get("tables_output", "tables-build.tex"),
                cfg.get("tables_pdf", "tables.pdf"),
            )
        )

    if args.step == "emit":
        return

    for source_name, output_name, pdf_name in documents:
        tex_path = assemble(
            cfg, source_name=source_name, output_name=output_name
        )
        if args.step == "assemble":
            continue
        compile_pdf(
            cfg, tex_path, keep_aux=args.keep_aux, pdf_name=pdf_name
        )
        if not args.keep_tex:
            remove_build_tex(cfg, tex_path)


if __name__ == "__main__":
    main()
