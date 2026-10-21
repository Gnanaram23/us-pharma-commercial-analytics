"""Turn each project's analysis.py into an executed analysis.ipynb.

The .py file (jupytext 'percent' format) is the source of truth because it diffs
cleanly in git. The notebook is generated from it and executed, so GitHub shows
every table and chart inline.

Run from the repo root:  python -m common.build_notebooks [project-folder ...]
"""
import sys
from pathlib import Path

import jupytext
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]


def build(folder: Path) -> None:
    src = folder / "analysis.py"
    nb = jupytext.read(src)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(folder)}})
    out = folder / "analysis.ipynb"
    nbformat.write(nb, out)
    print(f"built {out.relative_to(ROOT)}")


if __name__ == "__main__":
    targets = [ROOT / a for a in sys.argv[1:]] or sorted(p for p in ROOT.glob("0*-*") if (p / "analysis.py").exists())
    for t in targets:
        build(t)
