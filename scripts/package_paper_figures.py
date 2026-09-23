"""Build or verify the manuscript PDF and portable Prism reference package."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/paper"
OUT = BASE
MANIFEST = BASE / "manifest.json"
PRISM = BASE / "prism"

# Kept for local gallery builders that import the published selection.
_FIGURES = json.loads(MANIFEST.read_text())["figures"]
MAIN_COUNT = sum(row["section"] == "main" for row in _FIGURES)
SELECTED = [("figures", Path(row["source"]).stem) for row in _FIGURES]
TITLES = [row["title"] for row in _FIGURES]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pdf_text(path):
    return subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)


def sources(metadata):
    rows = metadata["figures"]
    paths = [ROOT / row["source"] for row in rows]
    assert len(paths) == len(set(paths)) and len(paths) > 0
    assert [row["page"] for row in rows] == list(range(1, len(rows) + 1))
    assert len({row["latex_label"] for row in rows}) == len(rows)
    for path in paths:
        assert path.is_relative_to(BASE / "figures")
        info = subprocess.check_output(["pdfinfo", str(path)], text=True)
        assert re.search(r"^Pages:\s+1\s*$", info, re.M), path
    return paths


def bundle_captions(metadata):
    text = (BASE / "FIGURE_CAPTIONS.md").read_text()
    for row in metadata["figures"]:
        relative = (ROOT / row["source"]).relative_to(BASE).as_posix()
        text = text.replace(f"]({relative})", f"]({row['bundle_file']})")
    return text.replace("manifest.json", "captions.json")


def bundle_manifest(metadata):
    portable = json.loads(json.dumps(metadata))
    portable["pdf"] = "paper_figures.pdf"
    portable["repository_numerical_inputs"] = portable.pop("numerical_inputs", [])
    if "render_command" in portable:
        portable["repository_render_command"] = portable.pop("render_command")
    for row in portable["figures"]:
        row["repository_source"] = row["source"]
        row["source"] = row["bundle_file"]
    return portable


def write_bundle(metadata):
    target = PRISM / "prism_figure_reference.zip"
    temporary = target.with_suffix(".build.zip")
    readme = f"""# Manuscript figure reference

Working materials for writing the EFF-Dock manuscript, not a published paper.

- [Combined PDF](paper_figures.pdf)
- [English captions and author notes](FIGURE_CAPTIONS.md)
- `methods.tex`: editable equations and current methods.
- `main.tex` and `figure_captions.tex`: reference LaTeX document and figure blocks.
- `figures/`: {len(metadata["figures"])} individual PDFs in manuscript page order.
- `captions.json`: file, caption, label and checksum mapping. The
  `repository_numerical_inputs` entries refer to the GitHub source checkout.

Use XeLaTeX or LuaLaTeX for the reference document. Adapt numbering, placement
and citation keys to the manuscript. Methods are working manuscript text;
the repository methods and parameter tables provide implementation detail.
"""
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        archive.writestr("README.md", readme)
        archive.writestr("FIGURE_CAPTIONS.md", bundle_captions(metadata))
        archive.writestr(
            "captions.json",
            json.dumps(bundle_manifest(metadata), ensure_ascii=False, indent=2) + "\n",
        )
        for name in ("main.tex", "figure_captions.tex", "methods.tex"):
            archive.write(PRISM / name, name)
        archive.write(ROOT / metadata["pdf"], "paper_figures.pdf")
        for row in metadata["figures"]:
            archive.write(ROOT / row["source"], row["bundle_file"])
    temporary.replace(target)


def verify(metadata):
    paths = sources(metadata)
    combined = ROOT / metadata["pdf"]
    assert digest(combined) == metadata["pdf_sha256"], combined
    assert pdf_text(combined) == "".join(pdf_text(path) for path in paths)
    for row, path in zip(metadata["figures"], paths, strict=True):
        assert digest(path) == row["sha256"], path
        assert path.with_suffix(".png").is_file(), path
    tex = (PRISM / "figure_captions.tex").read_text()
    inclusions = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    assert inclusions == [row["bundle_file"] for row in metadata["figures"]]
    assert re.findall(r"\\label\{([^}]+)\}", tex) == [
        row["latex_label"] for row in metadata["figures"]
    ]
    with ZipFile(PRISM / "prism_figure_reference.zip") as archive:
        assert archive.testzip() is None
        assert json.loads(archive.read("captions.json")) == bundle_manifest(metadata)
        assert archive.read("FIGURE_CAPTIONS.md").decode() == bundle_captions(metadata)
        assert archive.read("paper_figures.pdf") == combined.read_bytes()
        for name in ("main.tex", "figure_captions.tex", "methods.tex"):
            assert archive.read(name) == (PRISM / name).read_bytes()
        for row, path in zip(metadata["figures"], paths, strict=True):
            assert archive.read(row["bundle_file"]) == path.read_bytes()
        for text_name in ("README.md", "FIGURE_CAPTIONS.md"):
            for link in re.findall(r"\]\(([^)]+)\)", archive.read(text_name).decode()):
                if "://" not in link and not link.startswith("#"):
                    assert link.split("#")[0] in archive.namelist(), link


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="Verify existing files without rewriting"
    )
    mode.add_argument("--bundle-only", action="store_true", help="Refresh ZIP without merging PDFs")
    args = parser.parse_args()
    required = ["pdfinfo", "pdftotext"]
    if not (args.check or args.bundle_only):
        required.append("pdfunite")
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        parser.error("Poppler utilities required on PATH: " + ", ".join(missing))
    metadata = json.loads(MANIFEST.read_text())
    paths = sources(metadata)
    if not args.check:
        if not args.bundle_only:
            combined = ROOT / metadata["pdf"]
            temporary = combined.with_suffix(".build.pdf")
            subprocess.run(["pdfunite", *map(str, paths), str(temporary)], check=True)
            assert pdf_text(temporary) == "".join(pdf_text(path) for path in paths)
            temporary.replace(combined)
            metadata["pdf_sha256"] = digest(combined)
            for row, path in zip(metadata["figures"], paths, strict=True):
                row["sha256"] = digest(path)
            MANIFEST.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
            caption_path = BASE / "FIGURE_CAPTIONS.md"
            caption_path.write_text(
                re.sub(
                    r"PDF SHA-256: `[0-9a-f]{64}`",
                    f"PDF SHA-256: `{metadata['pdf_sha256']}`",
                    caption_path.read_text(),
                )
            )
        write_bundle(metadata)
    verify(metadata)
    print(
        f"Verified {len(paths)} source PDFs, merged page order, captions, methods, manifest and Prism ZIP"
    )


if __name__ == "__main__":
    main()
