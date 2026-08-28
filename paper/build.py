#!/usr/bin/env python3
"""
Build the paper from main.tex placeholders and emit-generated fragments.

Workflow:
  1. emit   — run `python -m emit` for each fragment in build.json
  2. assemble — substitute %%NAME%% placeholders → build.tex
  3. pdf    — compile build.tex with latexmk
  4. (default) all three steps

Placeholders in main.tex use the form %%QUALITY%%, %%TABLE%%, etc.
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
PLACEHOLDER_RE = re.compile(r"%%([A-Z][A-Z0-9_]*)%%")
AUX_SUFFIXES = (".aux", ".log", ".fls", ".fdb_latexmk", ".out")


def load_config() -> dict:
    path = PAPER_DIR / "build.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve(path_str: str) -> Path:
    return (PAPER_DIR / path_str).resolve()


def run_emit(cfg: dict) -> None:
    generated = resolve(cfg["generated_dir"])
    generated.mkdir(parents=True, exist_ok=True)

    for name, frag in cfg["fragments"].items():
        mode = frag["emit"]
        out_path = generated / frag["file"]
        input_path = resolve(frag["input"])

        cmd = [
            sys.executable,
            "-m",
            "emit",
            mode,
            str(input_path),
            "-o",
            str(out_path),
        ]
        if frag.get("vary_dir"):
            cmd.append(f"--vary-dir={resolve(frag['vary_dir'])}")
        if frag.get("ants") is not None:
            cmd.append(f"--ants={frag['ants']}")

        print(f"# emit {name} → {out_path.relative_to(PAPER_DIR)}", file=sys.stderr)
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def assemble(cfg: dict) -> Path:
    source = PAPER_DIR / cfg["source"]
    output = PAPER_DIR / cfg["output"]
    generated = resolve(cfg["generated_dir"])
    text = source.read_text(encoding="utf-8")

    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        frag = cfg["fragments"].get(name)
        if frag is None:
            missing.append(name)
            return match.group(0)
        fragment_path = generated / frag["file"]
        if not fragment_path.is_file():
            missing.append(name)
            return match.group(0)
        body = fragment_path.read_text(encoding="utf-8").strip()
        if not body:
            missing.append(name)
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

    unknown = set(PLACEHOLDER_RE.findall(built)) - set(cfg["fragments"])
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


def clean_aux_files(cfg: dict, tex_path: Path | None = None) -> None:
    """Remove LaTeX auxiliary files; keeps the PDF."""
    tex = tex_path or (PAPER_DIR / cfg["output"])
    stem = job_stem(cfg, tex)

    if subprocess.run(["which", "latexmk"], capture_output=True).returncode == 0:
        subprocess.run(
            ["latexmk", "-c", f"-jobname={stem}", tex.name],
            cwd=PAPER_DIR,
            check=False,
        )
    else:
        for suffix in AUX_SUFFIXES:
            (PAPER_DIR / f"{stem}{suffix}").unlink(missing_ok=True)

    print(f"# cleaned aux files for {stem}.pdf", file=sys.stderr)


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
            "-pdf",
            "-interaction=nonstopmode",
            "-file-line-error",
            f"-jobname={job_stem(cfg, tex)}",
            tex.name,
        ]
    else:
        cmd = ["pdflatex", "-interaction=nonstopmode", "-file-line-error", tex.name]
        subprocess.run(cmd, cwd=PAPER_DIR, check=True)
        subprocess.run(cmd, cwd=PAPER_DIR, check=True)
        print(f"# wrote {pdf_name}", file=sys.stderr)
        if not keep_aux:
            clean_aux_files(cfg, tex)
        return

    subprocess.run(cmd, cwd=PAPER_DIR, check=True)
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


if __name__ == "__main__":
    main()
