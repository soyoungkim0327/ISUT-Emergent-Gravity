#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""isut_109_qumond_efe_field_bc_demo.py

Field-level EFE demo (QUMOND + constant external field proxy)
=============================================================

Why this exists (reviewer-facing)
--------------------------------
In AQUAL/QUMOND, the External Field Effect (EFE) is formally a *boundary condition* effect.
A periodic FFT solver cannot represent the linear potential corresponding to a uniform external
field; however, one can still emulate the EFE influence on the *internal* solution by evaluating
ν(|g_N + g_ext|/a0) in the QUMOND source term, where g_ext is a constant vector.

This captures the leading dependence of the QUMOND source on the external field.

What this script does
---------------------
* Solves QUMOND for a reproducible baryonic exponential disk model.
* Compares midplane rotation curves with and without an imposed constant g_ext.
* Writes CSV + PNG + run metadata *relative to this script path*.

Outputs (relative)
------------------
  ./isut_109_qumond_efe_field_bc_demo/
      data/    (CSV + JSON)
      figures/ (PNG)

Usage
-----
  python 01.Core_Engine/isut_109_qumond_efe_field_bc_demo.py --N 96 --L 200 --pad 2 --gext 0.02*a0@x
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    nu_simple,
    nu_standard,
    make_nu_from_mu_beta,
)


# =============================================================================
# Output dirs (match repo conventions)
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# Simple baryonic disk model (reproducible, reviewer-friendly)
# =============================================================================

def build_exponential_disk_density(
    grid: PMGrid3D,
    M_d: float = 1.0,
    R_d: float = 20.0,
    z_h: float = 2.0,
) -> np.ndarray:
    """Exponential disk with sech^2 vertical profile.

    rho(R,z) = (M_d / (4π R_d^2 z_h)) exp(-R/R_d) sech^2(z/z_h)

    This normalizes to M_d in the infinite-volume limit; within a finite box it is approximately M_d.
    """
    x, y, z = grid.centered_coords()
    R = np.sqrt(x**2 + y**2)
    sech = 1.0 / np.cosh(z / z_h)
    rho = (M_d / (4.0 * np.pi * R_d**2 * z_h)) * np.exp(-R / R_d) * (sech**2)
    return rho.astype(np.float64, copy=False)


def rotation_curve_midplane_xaxis(
    grid: PMGrid3D,
    accel: np.ndarray,
    r_max: float | None = None,
    n_samples: int = 250,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute v_c(r) along +x axis in the midplane (y=0,z=0)."""
    if r_max is None:
        r_max = 0.45 * grid.boxsize
    r = np.linspace(grid.dx, float(r_max), int(n_samples))
    v = np.zeros_like(r)

    N = grid.N
    ic = N // 2
    for i, rr in enumerate(r):
        ix = ic + int(round(rr / grid.dx))
        ix = max(0, min(N - 1, ix))
        # radial acceleration along +x: a_r = -a_x at (x>0,y=0,z=0)
        a_x = float(accel[ix, ic, ic, 0])
        a_r = -a_x
        v[i] = np.sqrt(max(0.0, rr * a_r))
    return r, v


# =============================================================================
# CLI helpers
# =============================================================================

def _parse_gext(arg: str, a0: float) -> Tuple[float, float, float]:
    """
    Parse g_ext from a small DSL.

    Supported forms:
      - "0" / "none"      -> (0,0,0)
      - "0.02*a0@x"       -> magnitude in units of a0, direction axis (x/y/z)
      - "gx,gy,gz"        -> explicit components in code units
    """
    s = arg.strip().lower()
    if s in ("0", "0.0", "none", ""):
        return (0.0, 0.0, 0.0)

    if "*a0@" in s:
        mag_s, axis = s.split("*a0@")
        mag = float(mag_s) * float(a0)
        axis = axis.strip()
        if axis not in ("x", "y", "z"):
            raise ValueError("axis must be one of x,y,z in form '0.02*a0@x'")
        if axis == "x":
            return (mag, 0.0, 0.0)
        if axis == "y":
            return (0.0, mag, 0.0)
        return (0.0, 0.0, mag)

    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("gext must be '0.02*a0@x' or 'gx,gy,gz'")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=96, help="Grid resolution per axis")
    ap.add_argument("--L", type=float, default=200.0, help="Box size (code length units)")
    ap.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="FFT padding factor")
    ap.add_argument("--G", type=float, default=1.0, help="Gravitational constant in code units")
    ap.add_argument("--a0", type=float, default=0.12, help="Acceleration scale a0 in code units")
    ap.add_argument(
        "--nu",
        type=str,
        default="nu_standard",
        choices=["nu_standard", "nu_simple", "nu_beta"],
        help="Interpolation ν(y) choice",
    )
    ap.add_argument("--beta", type=float, default=4.0, help="Only used if --nu nu_beta")
    ap.add_argument(
        "--gext",
        type=str,
        default="0.02*a0@x",
        help="External field: '0.02*a0@x' or 'gx,gy,gz' (code units)",
    )

    # disk parameters
    ap.add_argument("--Md", type=float, default=1.0, help="Disk mass in code units")
    ap.add_argument("--Rd", type=float, default=20.0, help="Disk scale length")
    ap.add_argument("--zh", type=float, default=2.0, help="Disk scale height")
    args = ap.parse_args()

    grid = PMGrid3D(N=args.N, boxsize=args.L, G=args.G)

    rho = build_exponential_disk_density(grid, M_d=args.Md, R_d=args.Rd, z_h=args.zh)

    solver = QUMONDSolverFFT(grid)

    if args.nu == "nu_standard":
        nu_func = nu_standard
    elif args.nu == "nu_simple":
        nu_func = nu_simple
    else:
        nu_func = make_nu_from_mu_beta(beta=args.beta)

    gext_vec = _parse_gext(args.gext, a0=args.a0)
    gext_is_zero = np.allclose(gext_vec, (0.0, 0.0, 0.0))

    # Baseline (no EFE)
    res0 = solver.solve(rho=rho, a0=args.a0, nu_func=nu_func, pad_factor=args.pad, g_ext=None)

    # With external field
    res1 = solver.solve(
        rho=rho,
        a0=args.a0,
        nu_func=nu_func,
        pad_factor=args.pad,
        g_ext=None if gext_is_zero else gext_vec,
    )

    # Rotation curves sampled along midplane x-axis
    r0, v0 = rotation_curve_midplane_xaxis(grid, res0.accel, n_samples=250)
    r1, v1 = rotation_curve_midplane_xaxis(grid, res1.accel, n_samples=250)

    # Save CSV
    csv_path = os.path.join(DATA_DIR, "rotation_curve_efe_field.csv")
    header = "r,v_no_efe,v_with_efe"
    arr = np.column_stack([r0, v0, v1])
    np.savetxt(csv_path, arr, delimiter=",", header=header, comments="")

    # Plot
    fig = plt.figure()
    plt.plot(r0, v0, label="QUMOND (g_ext = 0)")
    plt.plot(r0, v1, label=f"QUMOND + EFE (g_ext={args.gext})")
    plt.xlabel("r (code units)")
    plt.ylabel("v_c(r) (code units)")
    plt.title("Field-level EFE sensitivity (periodic FFT + padding)")
    plt.legend()
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "rotation_curve_efe_field.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    meta: Dict[str, object] = {
        "script": os.path.basename(__file__),
        "grid": {"N": args.N, "L": args.L, "G": args.G, "pad": args.pad, "dx": grid.dx},
        "a0": args.a0,
        "nu": args.nu,
        "beta": args.beta if args.nu == "nu_beta" else None,
        "g_ext": {"parsed": gext_vec, "raw": args.gext},
        "disk": {"Md": args.Md, "Rd": args.Rd, "zh": args.zh},
        "notes": (
            "EFE implemented as ν(|g_N + g_ext|/a0) in the QUMOND source term. "
            "This approximates an external-field boundary condition in a periodic FFT solver."
        ),
        "outputs": {"csv": csv_path, "figure": fig_path},
    }
    with open(os.path.join(DATA_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {fig_path}")
    print(f"[OK] Wrote: {os.path.join(DATA_DIR, 'run_metadata.json')}")


if __name__ == "__main__":
    main()