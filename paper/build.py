#!/usr/bin/env python3
"""
Build the paper from main.tex placeholders and emit-generated fragments.

Workflow:
  1. emit   — run `python -m emit` for each fragment in build.json
  2. assemble — substitute %%NAME%% placeholders → build.tex
  3. pdf    — compile build.tex with latexmk, then remove build.tex
  4. (default) all three steps

Use --keep-tex to retain build.tex after compiling.

Placeholders in main.tex use the form %%QUALITY%%, %%COMPARE:theta-time%%, etc.
%%PREAMBLE%% aggregates emit sidecar snippets (*.preamble.tex) into the
document preamble — not a normal emit target.

Experiment paths in build.json are usually bare names under results_dir
(e.g. ``"input": "vary_k2t5i_PN"`` with ``"results_dir": "../results"``).
Change results_dir once to point at another tree (e.g. ``../results_old``).
Absolute paths and ``./`` / ``../`` paths are still resolved from paper/.

STATISTICS missing-at-5 fields use missing_at_base (plain ACO dir; ACO-N is
that path + "N"). Falls back to vary_base if missing_at_base is omitted.
COMPARE k-sweep / theta-sweep use param_dirs (labeled vary_* suites).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent
PLACEHOLDER_RE = re.compile(r"%%([A-Z][A-Z0-9_]*)(?::([^%]+))?%%")
AUX_SUFFIXES = (".aux", ".log", ".fls", ".fdb_latexmk", ".out")
# Assembled from fragment sidecars (*.preamble.tex); not a normal emit target.
SPECIAL_PLACEHOLDERS = frozenset({"PREAMBLE"})
DEFAULT_RESULTS_DIR = "../results"


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
    ``k2t5i_PN:highlights`` → (``k2t5i_PN``, ``highlights``).
    """
    if not args:
        return "", "full"
    raw = args.strip()
    for subset in ("highlights", "full"):
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
    """Resolve emit input directory; TABLE uses vary_base + k/θ/flags suffix."""
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


def run_emit(cfg: dict) -> None:
    generated = resolve(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)
    source_text = (PAPER_DIR / cfg["source"]).read_text(encoding="utf-8")

    for name, args in placeholders_in_source(source_text):
        if name in SPECIAL_PLACEHOLDERS:
            continue
        frag = cfg["fragments"].get(name)
        if frag is None:
            continue
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
        if plots and name not in ("SEED_COMPARE", "STATISTICS", "TABLE"):
            cmd.append(f"--plots={plots}")
        if name == "SEED_COMPARE" and args:
            cmd.append(f"--subset={args}")
        if name == "TABLE":
            _suffix, subset = parse_table_args(args)
            if subset != "full":
                cmd.append(f"--subset={subset}")
        if frag.get("ants") is not None:
            cmd.append(f"--ants={frag['ants']}")
        if frag.get("flag_dirs"):
            for label, path in frag["flag_dirs"].items():
                cmd.append(
                    f"--flag-dir={label}={resolve_results_path(cfg, path)}"
                )
        if frag.get("param_dirs"):
            for label, path in frag["param_dirs"].items():
                cmd.append(
                    f"--param-dir={label}={resolve_results_path(cfg, path)}"
                )
        if name == "STATISTICS" and args:
            cmd.append(f"--field={args}")
        if name == "STATISTICS" and args in (
            "missing-at-5",
            "aco-missing-at-5",
            "aco-n-missing-at-5",
        ):
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

        label = name if not args else f"{name}:{args}"
        print(f"# emit {label} → {out_path.relative_to(PAPER_DIR)}", file=sys.stderr)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


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


def assemble(cfg: dict) -> Path:
    source = PAPER_DIR / cfg["source"]
    output = PAPER_DIR / cfg["output"]
    generated = resolve(cfg["generated_dir"])
    text = source.read_text(encoding="utf-8")

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
            "Placeholders in main.tex with no build.json entry: "
            + ", ".join(sorted(unknown))
        )

    output.write_text(built, encoding="utf-8")
    if "fig:compare-bound-ratio" not in built and "%%COMPARE:deg-size-time%%" in text:
        raise SystemExit(
            "assemble: bound-ratio figure missing from build.tex; "
            "run: python3 paper/build.py emit"
        )
    print(f"# assembled {output.relative_to(PAPER_DIR)}", file=sys.stderr)
    return output


def job_stem(cfg: dict, tex_path: Path | None = None) -> str:
    tex = tex_path or (PAPER_DIR / cfg["output"])
    pdf_name = cfg.get("pdf", tex.with_suffix(".pdf").name)
    return Path(pdf_name).stem


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


def clean_aux_files(cfg: dict, tex_path: Path | None = None) -> None:
    """Remove LaTeX auxiliary files; keeps the PDF."""
    tex = tex_path or (PAPER_DIR / cfg["output"])
    stem = job_stem(cfg, tex)

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


def verify_pdf(cfg: dict, tex_path: Path | None = None) -> None:
    """Fail loudly if the bound-ratio complexity figure did not make it into the PDF."""
    tex = tex_path or (PAPER_DIR / cfg["output"])
    pdf_path = PAPER_DIR / cfg.get("pdf", tex.with_suffix(".pdf").name)
    if not pdf_path.is_file():
        raise SystemExit(f"PDF not found after compile: {pdf_path}")

    probe = subprocess.run(
        ["pdftotext", str(pdf_path), "-"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        print(
            "# warning: pdftotext unavailable; skipped bound-ratio PDF check",
            file=sys.stderr,
        )
        return

    text = probe.stdout
    required = (
        "Ratio of ACO-PN discovery time to each candidate complexity bound",
        "Time / bound",
    )
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        raise SystemExit(
            "PDF is missing the bound-ratio complexity figure "
            f"({', '.join(missing)}). "
            "Close and reopen paper/build.pdf in your viewer, or run: "
            "python3 paper/build.py all"
        )

    print(
        "# verified bound-ratio figure in "
        f"{pdf_path.relative_to(PAPER_DIR)} (section 3.3, fig:compare-bound-ratio)",
        file=sys.stderr,
    )


def compile_pdf(
    cfg: dict,
    tex_path: Path | None = None,
    *,
    keep_aux: bool = False,
) -> None:
    tex = tex_path or (PAPER_DIR / cfg["output"])
    pdf_name = cfg.get("pdf", tex.with_suffix(".pdf").name)
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
            f"-jobname={job_stem(cfg, tex)}",
            tex.name,
        ]
    else:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", tex.name]
        run_quiet(cmd, cwd=PAPER_DIR)
        run_quiet(cmd, cwd=PAPER_DIR)
        print(f"# wrote {pdf_name}", file=sys.stderr)
        if not keep_aux:
            clean_aux_files(cfg, tex)
        return

    run_quiet(cmd, cwd=PAPER_DIR)
    print(f"# wrote {pdf_name}", file=sys.stderr)
    if not keep_aux:
        clean_aux_files(cfg, tex)


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
    args = parser.parse_args(argv)
    cfg = load_config()

    tex_path: Path | None = None
    if args.step in ("all", "emit"):
        run_emit(cfg)
    if args.step in ("all", "assemble"):
        tex_path = assemble(cfg)
    if args.step in ("all", "pdf"):
        if tex_path is None:
            tex_path = assemble(cfg)
        compile_pdf(cfg, tex_path, keep_aux=args.keep_aux)
        verify_pdf(cfg, tex_path)
        if not args.keep_tex:
            remove_build_tex(cfg, tex_path)


if __name__ == "__main__":
    main()
