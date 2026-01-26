# -*- coding: utf-8 -*-
"""
Reviewer Artifact Manifest Generator
===================================

Purpose
-------
This script is reviewer-facing glue.

It walks the repository (or a chosen subfolder) and produces:
- a CSV manifest with SHA256 checksums + file sizes + mtimes
- a JSON copy of the same manifest for programmatic audits

Why reviewers like this
-----------------------
If you claim "open evidence pack", a manifest makes it easy to:
- verify nothing silently changed between runs
- cite exact artifact hashes in a rebuttal
- run diff audits across branches/releases

Usage
-----
python 03.Advanced_Validation/9.reviewer_artifact_manifest__.py
python 03.Advanced_Validation/9.reviewer_artifact_manifest__.py --root . --glob "*.png" "*.csv"

Notes
-----
- Defaults to scanning common evidence artifacts (csv/png/json/pdf/tex).
- Skips very large/binary folders by default (e.g., __pycache__).
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import os
from pathlib import Path
from typing import Dict, List

import sys

# repo-root import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isut_000_common import ScriptPaths, sha256_file, write_json, find_repo_root


DEFAULT_EXTS = {".csv", ".png", ".json", ".pdf", ".tex"}


def should_skip_dir(p: Path) -> bool:
    parts = set(p.parts)
    return any(x in parts for x in {"__pycache__", ".git", ".venv", "venv", "site-packages"})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="", help="Root folder to scan (default: repo root).")
    ap.add_argument(
        "--glob",
        nargs="*",
        default=[],
        help='Optional glob patterns (e.g. "*.png" "*.csv"). If omitted, uses default extensions.',
    )
    ap.add_argument("--max-mb", type=float, default=50.0, help="Skip files larger than this (MB).")
    args = ap.parse_args()

    sp = ScriptPaths.for_script(__file__)
    repo_root = find_repo_root(Path(__file__).resolve())
    scan_root = Path(args.root).resolve() if args.root else repo_root

    patterns: List[str] = [g for g in args.glob if g.strip()]
    max_bytes = int(args.max_mb * 1024 * 1024)

    rows: List[Dict[str, str]] = []
    for dirpath, dirnames, filenames in os.walk(scan_root):
        dp = Path(dirpath)
        if should_skip_dir(dp):
            dirnames[:] = []  # don't recurse
            continue

        for fn in filenames:
            fp = dp / fn
            try:
                if fp.is_symlink() or not fp.is_file():
                    continue
                sz = fp.stat().st_size
                if sz > max_bytes:
                    continue

                if patterns:
                    ok = any(fnmatch.fnmatch(fp.name, pat) or fnmatch.fnmatch(str(fp.relative_to(scan_root)), pat) for pat in patterns)
                    if not ok:
                        continue
                else:
                    if fp.suffix.lower() not in DEFAULT_EXTS:
                        continue

                rows.append(
                    {
                        "relpath": str(fp.relative_to(scan_root)).replace("\\", "/"),
                        "bytes": str(sz),
                        "mtime": str(int(fp.stat().st_mtime)),
                        "sha256": sha256_file(fp),
                    }
                )
            except Exception:
                continue

    rows.sort(key=lambda r: r["relpath"])

    csv_path = sp.data_dir / "artifact_manifest.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["relpath", "bytes", "mtime", "sha256"])
        w.writeheader()
        w.writerows(rows)

    json_path = sp.data_dir / "artifact_manifest.json"
    write_json(json_path, {"scan_root": str(scan_root), "n_files": len(rows), "files": rows})

    print(f"[OK] Wrote {len(rows)} entries")
    print(f" - CSV : {csv_path}")
    print(f" - JSON: {json_path}")


if __name__ == "__main__":
    main()
