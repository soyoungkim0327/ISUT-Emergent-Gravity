# -*- coding: utf-8 -*-
"""9.qumond_spherical_sanity__.py

QUMOND Spherical Symmetry Sanity Check
=====================================

Reviewer-facing goal
--------------------
In spherical symmetry, the QUMOND field solve reduces (up to discretization
and boundary effects) to the familiar algebraic relation:

    g(r) = g_N(r) * nu( g_N(r) / a0 ).

This script constructs a spherical Plummer density on a 3D mesh, solves QUMOND
via the 2-Poisson FFT method, samples the radial acceleration, and compares it
to the algebraic MOND prediction.

Outputs (relative path)
-----------------------
Creates artifacts under:

  ./9.qumond_spherical_sanity__/
      figures/  (PNG)
      data/     (CSV)
      logs/     (JSON)

Usage
-----
  python 01.Core_Engine/9.qumond_spherical_sanity__.py --N 96 --L 240 --pad 2 --nu nu_standard

Notes
-----
- FFT Poisson is periodic by default. Use pad>1 and/or larger L to push
  periodic images away.
- Compare errors primarily in the interior region (e.g. r ≤ L/4).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None

from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    StaticFieldSampler,
    nu_simple,
    nu_standard,
)


# ============================================================================
# Output dirs (match repo conventions)
# ============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")
LOG_DIR = os.path.join(OUT_ROOT, "logs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def plummer_density(grid: PMGrid3D, M_total: float, a: float, renormalize: bool = True) -> np.ndarray:
    """Plummer density on the mesh.

    rho(r) = (3M / (4 pi a^3)) * (1 + r^2/a^2)^(-5/2)
    """
    x, y, z = grid.centered_coords()
    r2 = x * x + y * y + z * z
    rho = (3.0 * M_total / (4.0 * np.pi * a**3)) * (1.0 + r2 / (a**2)) ** (-2.5)
    if renormalize:
        m_grid = float(np.sum(rho) * (grid.dx**3))
        if m_grid > 0:
            rho = rho * (M_total / m_grid)
    return rho.astype(np.float64, copy=False)


def plummer_M_enc(r: np.ndarray, M_total: float, a: float) -> np.ndarray:
    """Enclosed mass for Plummer: M(r)=M r^3 / (r^2+a^2)^{3/2}."""
    r2 = r * r
    return M_total * (r**3) / np.power(r2 + a**2, 1.5)


def sample_radial_accel_xaxis(sampler: StaticFieldSampler, r: np.ndarray) -> np.ndarray:
    pos = np.zeros((len(r), 3), dtype=np.float64)
    pos[:, 0] = r
    a = sampler.accel_at(pos)
    # along +x axis, inward radial accel magnitude is -a_x
    return np.maximum(-a[:, 0], 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND spherical symmetry sanity check")
    ap.add_argument("--N", type=int, default=96, help="Grid size (N^3)")
    ap.add_argument("--L", type=float, default=240.0, help="Box size")
    ap.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="Padding factor")
    ap.add_argument("--G", type=float, default=1.0, help="Gravitational constant (code units)")

    ap.add_argument("--M", type=float, default=1000.0, help="Total mass")
    ap.add_argument("--a_plummer", type=float, default=4.0, help="Plummer scale length")

    ap.add_argument("--a0", type=float, default=0.12, help="Acceleration scale a0")
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"], help="nu(y) choice")

    ap.add_argument("--rmin", type=float, default=0.5, help="Min radius for sampling")
    ap.add_argument("--rmax", type=float, default=60.0, help="Max radius for sampling")
    ap.add_argument("--nr", type=int, default=300, help="Number of radii")
    args = ap.parse_args()

    print(f"[System] {SCRIPT_NAME} initialized")
    print(f"[Info] Output Root : {OUT_ROOT}")

    nu_func: Callable[[np.ndarray], np.ndarray] = nu_standard if args.nu == "nu_standard" else nu_simple

    grid = PMGrid3D(N=int(args.N), boxsize=float(args.L), G=float(args.G))
    rho = plummer_density(grid, M_total=float(args.M), a=float(args.a_plummer), renormalize=True)

    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho, a0=float(args.a0), nu_func=nu_func, pad_factor=int(args.pad))

    sampler_Q = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)

    r = np.linspace(float(args.rmin), float(args.rmax), int(args.nr))
    gQ = sample_radial_accel_xaxis(sampler_Q, r)

    # Analytic Newtonian g_N for Plummer
    M_enc = plummer_M_enc(r, M_total=float(args.M), a=float(args.a_plummer))
    r_safe = np.maximum(r, 1e-12)
    gN = float(args.G) * M_enc / (r_safe**2)

    # Algebraic MOND prediction (spherical)
    y = gN / float(args.a0)
    g_alg = gN * nu_func(y)

    rel_err = (gQ - g_alg) / np.maximum(g_alg, 1e-30)

    # Save CSV
    csv_path = os.path.join(DATA_DIR, "data_spherical_sanity.csv")
    if pd is not None:
        pd.DataFrame({
            "r": r,
            "gN": gN,
            "g_alg": g_alg,
            "g_qumond": gQ,
            "rel_err": rel_err,
        }).to_csv(csv_path, index=False)
    else:
        arr = np.column_stack([r, gN, g_alg, gQ, rel_err])
        np.savetxt(csv_path, arr, delimiter=",", header="r,gN,g_alg,g_qumond,rel_err", comments="")

    # Plot g(r)
    fig1 = plt.figure(figsize=(9.2, 5.4))
    ax = fig1.add_subplot(111)
    ax.plot(r, gN, "--", linewidth=1.6, label="Newton g_N (analytic)")
    ax.plot(r, g_alg, linewidth=2.0, label=f"Algebraic MOND: g_N*nu({args.nu})")
    ax.plot(r, gQ, linewidth=2.2, alpha=0.9, label=f"QUMOND field solve (pad={args.pad})")
    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("g(r)")
    ax.set_title("Spherical sanity: QUMOND vs algebraic relation")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()
    fig1.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_spherical_g_compare.png")
    fig1.savefig(fig_path, dpi=220)
    plt.close(fig1)

    # Plot relative error
    fig2 = plt.figure(figsize=(9.2, 4.8))
    ax2 = fig2.add_subplot(111)
    ax2.plot(r, np.abs(rel_err), linewidth=2.0)
    ax2.set_yscale("log")
    ax2.set_xlabel("r")
    ax2.set_ylabel("|g_QUMOND - g_alg| / g_alg")
    ax2.set_title("Relative error (expect small in interior; improves with larger L/pad/N)")
    ax2.grid(True, which="both", linestyle=":", alpha=0.4)
    fig2.tight_layout()
    fig_err_path = os.path.join(FIG_DIR, "fig_spherical_rel_error.png")
    fig2.savefig(fig_err_path, dpi=220)
    plt.close(fig2)

    # Metadata
    meta: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "args": vars(args),
        "grid": {"dx": grid.dx},
        "notes": {
            "interpretation": "In spherical symmetry, QUMOND reduces to the algebraic relation (modulo discretization/BC).",
            "boundary": "FFT is periodic; pad>1 approximates isolated BC by pushing images away.",
        },
        "outputs": {
            "csv": os.path.relpath(csv_path, OUT_ROOT),
            "fig_g": os.path.relpath(fig_path, OUT_ROOT),
            "fig_err": os.path.relpath(fig_err_path, OUT_ROOT),
        },
        "summary": {
            "median_abs_rel_err": float(np.median(np.abs(rel_err))),
            "p95_abs_rel_err": float(np.quantile(np.abs(rel_err), 0.95)),
        },
    }

    meta_path = os.path.join(LOG_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Spherical sanity complete")
    print(f"  [CSV] {csv_path}")
    print(f"  [PNG] {fig_path}")
    print(f"  [PNG] {fig_err_path}")
    print(f"  [META] {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
