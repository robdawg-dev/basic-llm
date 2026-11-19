"""
Where the NumPy scripts read data from and write results to.

Paths are resolved relative to this file, so every script works no matter
which directory you run it from (e.g. `python numpy/shakespeare.py` from
the repo root, or `python shakespeare.py` from inside numpy/).

  inputs  -> ../pytorch/   shake_ids.npy, shake_merges.json (shared with
                           the PyTorch port, which trains on the same data)
  outputs -> ./out/        checkpoints, loss curves (git-ignored)
"""

from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "pytorch"
OUT_DIR = HERE / "out"


def data_path(name: str) -> str:
    """Path to a shared input file, e.g. data_path("shake_ids.npy")."""
    return str(DATA_DIR / name)


def out_path(name: str) -> str:
    """Path for a generated file under numpy/out/, creating the dir if needed."""
    OUT_DIR.mkdir(exist_ok=True)
    return str(OUT_DIR / name)
