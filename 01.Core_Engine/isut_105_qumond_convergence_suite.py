# -*- coding: utf-8 -*-
"""11.qumond_convergence_suite__.py

QUMOND Convergence & Boundary Sensitivity Suite
==============================================

Hostile-reviewer questions this addresses
----------------------------------------
1) "FFT is periodic. How sensitive are your results to boundary artifacts?"
2) "Are your results stable under grid refinement?"

This script runs small sweeps over:
  - grid resolution N
  - padding factor pad (proxy for isolated BC)

and compares midplane rotation curves.

Outputs (relative path)
-----------------------
  ./11.qumond_convergence_suite__/
      data/    (CSV)
      figures/ (PNG)
      logs/    (JSON)

Usage
-----
  python 01.Core_Engine/11.qumond_convergence_suite__.py \
      --N-list 64 96 --pad-list 1 2 3 --L 200 --nu nu_standard

Notes
-----
- This is intended to be *fast* on a laptop by default.
- Increase N-list / r-samples for stronger convergence evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, List, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    StaticFieldSampler,
    exponential_disk_density,
    nu_simple,
    nu_standard,
)


# =============================================================================
# Output dirs (match repo conventions)
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")
LOG_DIR = os.path.join(OUT_ROOT, "logs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def _rotation_curve_midplane(sampler: StaticFieldSampler, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.zeros((len(r), 3), dtype=np.float64)
    pos[:, 0] = r
    a = sampler.accel_at(pos)
    g = np.maximum(-a[:, 0], 0.0)
    v = np.sqrt(np.maximum(r, 0.0) * g)
    return g, v


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND convergence & boundary sensitivity suite")
    ap.add_argument("--N-list", type=int, nargs="+", default=[64, 96], help="Grid sizes")
    ap.add_argument("--pad-list", type=int, nargs="+", default=[1, 2, 3], help="Padding factors")
    ap.add_argument("--L", type=float, default=200.0, help="Box size")
    ap.add_argument("--G", type=float, default=1.0, help="Gravitational constant")
    ap.add_argument("--a0", type=float, default=0.12, help="Acceleration scale")
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"], help="nu(y)")
    ap.add_argument("--M", type=float, default=1000.0, help="Disk total mass")
    ap.add_argument("--Rd", type=float, default=3.0, help="Disk scale length")
    ap.add_argument("--zd", type=float, default=0.4, help="Disk scale height")
    ap.add_argument("--rmin", type=float, default=0.5, help="min radius")
    ap.add_argument("--rmax", type=float, default=40.0, help="max radius")
    ap.add_argument("--nr", type=int, default=220, help="# radii")
    args = ap.parse_args()

    print(f"[System] {SCRIPT_NAME} initialized")
    print(f"[Info] Output Root : {OUT_ROOT}")

    nu_func: Callable[[np.ndarray], np.ndarray] = nu_standard if args.nu == "nu_standard" else nu_simple
    r = np.linspace(float(args.rmin), float(args.rmax), int(args.nr))

    # Store curves keyed by (N,pad)
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    runtimes: List[Dict[str, float]] = []

    for N in args.N_list:
        for pad in args.pad_list:
            key = f"N{N}_pad{pad}"
            t0 = time.time()

            grid = PMGrid3D(N=int(N), boxsize=float(args.L), G=float(args.G))
            rho = exponential_disk_density(grid, M_total=float(args.M), R_d=float(args.Rd), z_d=float(args.zd), renormalize=True)
            solver = QUMONDSolverFFT(grid)
            res = solver.solve(rho=rho, a0=float(args.a0), nu_func=nu_func, pad_factor=int(pad))

            sampler = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)
            g, v = _rotation_curve_midplane(sampler, r)

            dt = float(time.time() - t0)
            runtimes.append({"N": int(N), "pad": int(pad), "runtime_sec": dt})
            curves[key] = {"g": g, "v": v}
            print(f"  [{key}] runtime={dt:.2f}s")

    # Choose baseline as max N and max pad provided
    N_base = max(args.N_list)
    pad_base = max(args.pad_list)
    base_key = f"N{N_base}_pad{pad_base}"
    v_base = curves[base_key]["v"]

    # Save CSV (v curves)
    csv_path = os.path.join(DATA_DIR, "data_convergence_rotation_curves.csv")
    header_cols = ["r"] + [f"v_{k}" for k in curves.keys()] + [f"relerr_{k}_vs_{base_key}" for k in curves.keys()]
    arr_cols = [r]
    for k in curves.keys():
        arr_cols.append(curves[k]["v"])
    for k in curves.keys():
        arr_cols.append((curves[k]["v"] - v_base) / np.maximum(v_base, 1e-30))
    out_arr = np.column_stack(arr_cols)
    np.savetxt(csv_path, out_arr, delimiter=",", header=",".join(header_cols), comments="")

    # Plot curves
    fig1 = plt.figure(figsize=(10, 6))
    ax = fig1.add_subplot(111)
    for k in curves.keys():
        ax.plot(r, curves[k]["v"], linewidth=2.0, alpha=0.85, label=k)
    ax.set_xlabel("r")
    ax.set_ylabel("v(r)")
    ax.set_title("QUMOND rotation curve: resolution/padding sweep")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(fontsize=9)
    fig1.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_convergence_rotation_curves.png")
    fig1.savefig(fig_path, dpi=220)
    plt.close(fig1)

    # Plot relative errors vs baseline
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    for k in curves.keys():
        rel = np.abs((curves[k]["v"] - v_base) / np.maximum(v_base, 1e-30))
        ax2.plot(r, rel, linewidth=2.0, alpha=0.85, label=f"{k} vs {base_key}")
    ax2.set_yscale("log")
    ax2.set_xlabel("r")
    ax2.set_ylabel("|Δv|/v (vs baseline)")
    ax2.set_title("Relative difference to baseline (grid/pad sensitivity)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    fig_err = os.path.join(FIG_DIR, "fig_convergence_relerr.png")
    fig2.savefig(fig_err, dpi=220)
    plt.close(fig2)

    # Metadata
    meta: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "args": vars(args),
        "baseline": {"N": int(N_base), "pad": int(pad_base), "key": base_key},
        "runtimes": runtimes,
        "outputs": {
            "csv": os.path.relpath(csv_path, OUT_ROOT),
            "fig_curves": os.path.relpath(fig_path, OUT_ROOT),
            "fig_relerr": os.path.relpath(fig_err, OUT_ROOT),
        },
        "notes": {
            "boundary": "FFT is periodic; padding approximates isolated BC by pushing periodic images away.",
            "interpretation": "Convergence is indicated by decreasing |Δv|/v as N increases and pad increases.",
        },
    }

    meta_path = os.path.join(LOG_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Convergence suite complete")
    print(f"  [CSV]  {csv_path}")
    print(f"  [PNG]  {fig_path}")
    print(f"  [PNG]  {fig_err}")
    print(f"  [META] {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
