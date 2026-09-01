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
Paths in build.json are relative to this directory (paper/).
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


def fragment_output_name(name: str, args: str | None) -> str:
    """Map placeholder (name, args) → generated/*.tex basename."""
    if not args:
        return name.lower()
    safe = args.strip().replace(",", "__").replace(" ", "")
    return f"{name.lower()}__{safe}"


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


def fragment_input_path(cfg: dict, name: str, frag: dict, args: str | None) -> Path:
    """Resolve emit input directory; TABLE uses vary_base + optional prefix suffix."""
    if name == "TABLE":
        base = cfg.get("vary_base")
        if not base:
            raise SystemExit("TABLE fragment requires vary_base in build.json")
        return resolve(base + (args or ""))
    input_path = frag.get("input")
    if not input_path:
        raise SystemExit(f"Fragment {name} has no input path in build.json")
    return resolve(input_path)


def run_emit(cfg: dict) -> None:
    generated = resolve(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)
    source_text = (PAPER_DIR / cfg["source"]).read_text(encoding="utf-8")

    for name, args in placeholders_in_source(source_text):
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
            cmd.append(f"--vary-dir={resolve(frag['vary_dir'])}")
        plots = args or frag.get("plots")
        if plots:
            cmd.append(f"--plots={plots}")
        if frag.get("ants") is not None:
            cmd.append(f"--ants={frag['ants']}")
        if name == "STATISTICS" and args:
            cmd.append(f"--field={args}")
        if name == "STATISTICS" and args in (
            "missing-at-5",
            "aco-missing-at-5",
            "aco-n-missing-at-5",
        ):
            vary_base = frag.get("vary_base") or cfg.get("vary_base")
            if vary_base:
                cmd.append(f"--vary-base={resolve(vary_base)}")

        label = name if not args else f"{name}:{args}"
        print(f"# emit {label} → {out_path.relative_to(PAPER_DIR)}", file=sys.stderr)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def assemble(cfg: dict) -> Path:
    source = PAPER_DIR / cfg["source"]
    output = PAPER_DIR / cfg["output"]
    generated = resolve(cfg["generated_dir"])
    text = source.read_text(encoding="utf-8")

    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        args = match.group(2) or None
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
    } - set(cfg["fragments"])
    if unknown:
        raise SystemExit(
            "Placeholders in main.tex with no build.json entry: "
            + ", ".join(sorted(unknown))
        )

    output.write_text(built, encoding="utf-8")
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


def compile_pdf(
    cfg: dict,
    tex_path: Path | None = None,
    *,
    keep_aux: bool = False,
) -> None:
    tex = tex_path or (PAPER_DIR / cfg["output"])
    pdf_name = cfg.get("pdf", tex.with_suffix(".pdf").name)
    for tool in ("latexmk", "pdflatex"):
        if subprocess.run(["which", tool], capture_output=True).returncode == 0:
            break
    else:
        raise SystemExit("Neither latexmk nor pdflatex found on PATH")

    if tool == "latexmk":
        cmd = [
            "latexmk",
            "-silent",
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
            tex_path = PAPER_DIR / cfg["output"]
            if not tex_path.is_file():
                tex_path = assemble(cfg)
        compile_pdf(cfg, tex_path, keep_aux=args.keep_aux)
        if not args.keep_tex:
            remove_build_tex(cfg, tex_path)


if __name__ == "__main__":
    main()
