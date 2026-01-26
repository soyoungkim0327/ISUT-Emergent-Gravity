# -*- coding: utf-8 -*-
"""isut_111_qumond_reviewer_pack_runner.py

One-Command "Hostile Reviewer" Pack Runner
===========================================

This script runs a curated subset of QUMOND diagnostics and produces a single
summary JSON pointing to all generated artifacts.

Why this is useful
------------------
Reviewers tend to ask for *repeatable* and *quantitative* checks:
- spherical symmetry reduction (algebraic MOND limit)
- PDE residuals for both Poisson solves
- mesh/boundary sensitivity (N, padding)
- curl-free conservative field check
- "Dark Matter Reverse" effective phantom density

This runner lets you regenerate all of them with one command.

Outputs (relative path)
-----------------------
  ./isut_111_qumond_reviewer_pack_runner/
      logs/run_metadata.json

Usage
-----
  # quick smoke run
  python 01.Core_Engine/isut_111_qumond_reviewer_pack_runner.py --fast

  # more serious run
  python 01.Core_Engine/isut_111_qumond_reviewer_pack_runner.py --N 96 --L 200 --pad 2
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
LOG_DIR = os.path.join(OUT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def _run(cmd: list[str]) -> int:
    # flush=True keeps ordering sane when subprocess writes to stdout.
    print(f"[Run] {' '.join(cmd)}", flush=True)
    return subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run QUMOND reviewer-defense scripts in one shot")
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=200.0)
    ap.add_argument("--pad", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument("--a0", type=float, default=0.12)
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"])
    ap.add_argument("--fast", action="store_true", help="Use smaller defaults for a quick smoke run")
    args = ap.parse_args()

    # Fast profile: smaller N and strict periodic checks where possible
    if args.fast:
        args.N = 48
        args.L = 160.0
        args.pad = 1

    py = sys.executable
    scripts: list[list[str]] = []

    # 1) Spherical exactness check
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_102_qumond_spherical_sanity.py"),
        "--N", str(args.N),
        "--L", str(max(args.L, 200.0)),  # give the sphere a bit more room
        "--pad", str(args.pad),
        "--a0", str(args.a0),
        "--nu", args.nu,
        "--rmax", "50",
        "--nr", "160",
    ])

    # 2) Poisson residuals (best with pad=1 / periodic spectral operators)
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_104_qumond_poisson_residuals.py"),
        "--N", str(args.N),
        "--L", str(args.L),
        "--pad", "1",
        "--a0", str(args.a0),
        "--nu", args.nu,
    ])

    # 3) Convergence sweep (small sweep)
    pad_list = ["1"]
    if int(args.pad) != 1:
        pad_list.append(str(args.pad))
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_105_qumond_convergence_suite.py"),
        "--N-list", str(max(32, args.N // 2)), str(args.N),
        "--pad-list", *pad_list,
        "--L", str(args.L),
        "--a0", str(args.a0),
        "--nu", args.nu,
        "--nr", "120",
    ])

    # 4) Curl-free check (strict with pad=1)
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_106_qumond_curl_free_check.py"),
        "--N", str(args.N),
        "--L", str(args.L),
        "--pad", "1",
        "--a0", str(args.a0),
        "--nu", args.nu,
    ])

    # 5) Phantom density (strict with pad=1)
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_107_qumond_phantom_density.py"),
        "--N", str(args.N),
        "--L", str(args.L),
        "--pad", "1",
        "--a0", str(args.a0),
        "--nu", args.nu,
    ])

    # 6) Solar-system high-g limit sanity check (analytic spherical regime check; SI units)
    # Note: this script uses an SI a0 default (1.2e-10 m/s^2) and does not share the engine's internal unit choices.
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_108_qumond_solar_system_sanity.py"),
        "--nu", args.nu,
    ])

    # 7) Solar-system EFE quadrupole sanity check (field-level, order-of-magnitude)
    scripts.append([
        py,
        os.path.join(CURRENT_DIR, "isut_110_qumond_solar_system_efe_quadrupole.py"),
        "--N", str(max(args.N, 128)),
        "--L", "10.0",
        "--pad", "2",
        "--nu", "nu_beta",
        "--beta", "12.0",
        "--eta", "1.6",
        "--a0", "1.0",
    ])

    t0 = time.time()
    results = []
    for cmd in scripts:
        rc = _run(cmd)
        results.append({"cmd": cmd, "returncode": int(rc)})
        if rc != 0:
            print(f"[Warn] Nonzero return code: {rc}", flush=True)

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "args": vars(args),
        "runtime_sec": float(time.time() - t0),
        "runs": results,
        "artifact_folders": {
            "spherical": "isut_102_qumond_spherical_sanity/",
            "poisson": "10.qumond_poisson_residuals__/",
            "convergence": "isut_105_qumond_convergence_suite/",
            "curl": "12.qumond_curl_free_check__/",
            "phantom": "13.qumond_phantom_density__/",
            "solar_system": "isut_108_qumond_solar_system_sanity/",
            "efe_field": "isut_109_qumond_efe_field_bc_demo/",
            "efe_quadrupole": "isut_110_qumond_solar_system_efe_quadrupole/",
        },
        "note": "Each script writes data/figures/logs under its own folder (relative to 01.Core_Engine).",
    }

    out = os.path.join(LOG_DIR, "run_metadata.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Reviewer pack runner complete")
    print(f"  [META] {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()