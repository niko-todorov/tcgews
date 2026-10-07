"""
paths.py  (src/py/paths.py)
===========================
Repo-root-relative path resolution. Nothing is hard-coded to D:\\ — the root is
discovered by walking up from this file, so the tree works unchanged if the repo
moves or runs on Linux (e.g. Cloud Run).

Layout assumed:
    <root>/
      data/{basin}/final/    -> {model}_final.csv, extremes.csv, *.pkl
      data/{basin}/ews/       -> EWS NetCDF output (created on demand)
      src/py/                 -> shared/common python (basins.py, paths.py, ...)
      src/py/nc/              -> model code
      src/py/ews/             -> dashboard + real-time inference

Author: Nikolay Todorov
Dissertation: "Predictive Tropical Cyclogenesis Using Machine Learning:
               A Novel Approach to Early Warning Systems"
"""

from __future__ import annotations

from pathlib import Path


def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data").is_dir() and (p / "src").is_dir():
            return p
        if (p / ".git").exists():
            return p
    # Fallback: <root>/src/py/paths.py -> parents[2] is <root>.
    return start.parents[2]


REPO_ROOT = _find_root(Path(__file__).resolve())
SRC_PY = REPO_ROOT / "src" / "py"     # shared/common python source lives directly here
DATA = REPO_ROOT / "data"

NC = SRC_PY / "nc"       # model code
EWS = SRC_PY / "ews"     # dashboard + inference


def basin_dir(folder: str) -> Path:
    return DATA / folder


def final_dir(folder: str) -> Path:
    """Where {model}_final.csv, extremes.csv, and *.pkl live."""
    return DATA / folder / "final"


def shared_cache_dir(kind: str, create: bool = False) -> Path:
    """Basin-agnostic cache for global downloads shared across basins — e.g.
    IFS Open Data GRIBs, which are whole-world files. Lives at data/_cache/{kind}."""
    d = DATA / "_cache" / kind
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def ews_cache_dir(folder: str, create: bool = False) -> Path:
    """Raw input download cache — ERA5 (ncp/ncs) and IFS Open Data (grib +
    cropped nc), per basin, off the data tree (data/{basin}/ews/cache)."""
    d = DATA / folder / "ews" / "cache"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def ews_output_dir(folder: str, create: bool = False) -> Path:
    """Where EWS NetCDF outputs are written / read from."""
    d = DATA / folder / "ews"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d
