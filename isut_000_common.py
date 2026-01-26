# -*- coding: utf-8 -*-
"""
isuit_000_common.py
==============
Small, dependency-light helper layer to reduce repeated boilerplate across:

- 03.Advanced_Validation
- 04.Data_Pipeline
- 05.Visualization_Suite

Design goals
------------
1) Keep existing numerical behavior intact (result-preserving refactor).
2) Standardize *relative* output paths (script-local folders).
3) Make SPARC data discovery / optional download consistent and auditable.
4) Provide a tiny, reusable "run metadata" writer for reviewer defense.

This file intentionally avoids any heavy dependencies. It uses:
- pathlib / json / hashlib / platform / time
- (optional) requests
- (optional) pandas
"""

from __future__ import annotations

import json
import os
import sys
import time
import platform
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Optional dependencies
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------

REPO_MARKERS = ("01.Core_Engine", "02.Theoretical_Framework", "03.Advanced_Validation", "04.Data_Pipeline", "05.Visualization_Suite")


def find_repo_root(start: Path) -> Path:
    """
    Walk upwards to find a directory that looks like the ISUT repo root.
    """
    start = start.resolve()
    for p in [start] + list(start.parents):
        if all((p / m).exists() for m in REPO_MARKERS):
            return p
    # Fallback: assume parent of script folder is the root
    return start.parent


@dataclass(frozen=True)
class ScriptPaths:
    """
    Standard output layout:

    <script_dir>/<script_stem>/
        data/
        figures/
        logs/
        runs/   (optional, used when you want timestamped runs)
    """
    script_dir: Path
    script_stem: str
    out_root: Path
    data_dir: Path
    fig_dir: Path
    log_dir: Path

    @staticmethod
    def for_script(script_file: str | Path) -> "ScriptPaths":
        sf = Path(script_file).resolve()
        script_dir = sf.parent
        stem = sf.stem
        out_root = script_dir / stem
        data_dir = out_root / "data"
        fig_dir = out_root / "figures"
        log_dir = out_root / "logs"
        for d in (out_root, data_dir, fig_dir, log_dir):
            d.mkdir(parents=True, exist_ok=True)
        return ScriptPaths(
            script_dir=script_dir,
            script_stem=stem,
            out_root=out_root,
            data_dir=data_dir,
            fig_dir=fig_dir,
            log_dir=log_dir,
        )


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------------------
# Environment / metadata
# ------------------------------------------------------------------------------

def env_info(extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
    }
    if pd is not None:
        info["pandas"] = getattr(pd, "__version__", "unknown")
    if extra:
        info.update(extra)
    return info


def write_json(path: Path, obj: Any) -> Path:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def write_run_metadata(
    out_dir: Path,
    *,
    args: Optional[Dict[str, Any]] = None,
    notes: Optional[Dict[str, Any]] = None,
    name: str = "run_metadata.json",
) -> Path:
    payload: Dict[str, Any] = {
        "env": env_info(),
        "args": args or {},
        "notes": notes or {},
    }
    return write_json(out_dir / name, payload)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------
# SPARC data discovery & optional download
# ------------------------------------------------------------------------------

SPARC_URL_TEMPLATE = "https://raw.githubusercontent.com/jobovy/sparc-rotation-curves/master/data/{gal}_rotmod.dat"


def find_sparc_data_dir(
    start_dir: Path,
    *,
    env_keys: Tuple[str, ...] = ("ISUT_SPARC_DIR", "SPARC_DATA_DIR", "SPARC_DIR"),
    allow_empty: bool = True,
) -> Path:
    """
    Find a directory that contains SPARC *_rotmod.dat files.

    Priority:
    1) Environment variable (ISUT_SPARC_DIR / SPARC_DATA_DIR / SPARC_DIR)
    2) Common relative locations (repo-root/data, repo-root/sparc_data, etc.)
    3) Fallback to <start_dir>/sparc_data
    """
    # (1) env var override
    for k in env_keys:
        v = os.environ.get(k, "").strip()
        if v:
            p = Path(v).expanduser().resolve()
            if p.exists() and p.is_dir():
                return p

    repo_root = find_repo_root(start_dir)

    # (2) common candidates (mirrors your existing scripts)
    candidates = [
        start_dir / "sparc_data" / "Rotmod_LTG",
        start_dir / "sparc_data",
        start_dir / "galaxies" / "65_galaxies",
        start_dir.parent / "sparc_data" / "Rotmod_LTG",
        start_dir.parent / "sparc_data",
        start_dir.parent / "data" / "galaxies" / "65_galaxies",
        repo_root / "sparc_data" / "Rotmod_LTG",
        repo_root / "sparc_data",
        repo_root / "data" / "sparc_data" / "Rotmod_LTG",
        repo_root / "data" / "sparc_data",
    ]

    def has_dat_files(d: Path) -> bool:
        try:
            return any(x.name.endswith(".dat") for x in d.iterdir() if x.is_file())
        except Exception:
            return False

    for c in candidates:
        if c.exists() and c.is_dir():
            if has_dat_files(c) or allow_empty:
                return c

    # (3) fallback
    return start_dir / "sparc_data"


def ensure_rotmod_file(
    gal_name: str,
    data_dir: Path,
    *,
    allow_download: bool = True,
    timeout: float = 5.0,
    log: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """
    Ensure <data_dir>/<gal>_rotmod.dat exists. Optionally download from SPARC github mirror.

    Raises FileNotFoundError with a helpful message when missing and download fails/disabled.
    """
    data_dir = Path(data_dir).expanduser().resolve()
    ensure_dir(data_dir)

    fp = data_dir / f"{gal_name}_rotmod.dat"
    if fp.exists():
        return fp

    if not allow_download:
        raise FileNotFoundError(
            f"Missing SPARC file: {fp}. Set ISUT_SPARC_DIR or run with download enabled."
        )

    try:
        import requests  # type: ignore
    except Exception as e:  # pragma: no cover
        raise FileNotFoundError(
            f"Missing SPARC file: {fp}. 'requests' is not installed, cannot download."
        ) from e

    url = SPARC_URL_TEMPLATE.format(gal=gal_name)
    status = "unknown"
    try:
        r = requests.get(url, timeout=timeout)
        status = str(r.status_code)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}")
        fp.write_text(r.text, encoding="utf-8")
        if log is not None:
            log.append({"galaxy": gal_name, "url": url, "status": status, "saved_to": str(fp)})
        return fp
    except Exception as e:
        if log is not None:
            log.append({"galaxy": gal_name, "url": url, "status": status, "error": str(e)})
        raise FileNotFoundError(
            f"Failed to obtain SPARC file for {gal_name}. Tried {fp} and download {url} (status={status}). "
            f"Either (a) place the file locally, or (b) set ISUT_SPARC_DIR to a valid dataset folder."
        ) from e


def load_rotmod_dataframe(
    gal_name: str,
    data_dir: Path,
    *,
    allow_download: bool = True,
    timeout: float = 5.0,
    log: Optional[List[Dict[str, Any]]] = None,
):
    """
    Load SPARC rotmod file into a pandas DataFrame (if available).
    Falls back to a numpy array dict if pandas is unavailable.
    """
    fp = ensure_rotmod_file(gal_name, data_dir, allow_download=allow_download, timeout=timeout, log=log)

    if pd is None:
        import numpy as np  # type: ignore
        arr = np.loadtxt(fp)
        return {"raw": arr, "path": str(fp)}

    df = pd.read_csv(fp, sep=r"\s+", comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df