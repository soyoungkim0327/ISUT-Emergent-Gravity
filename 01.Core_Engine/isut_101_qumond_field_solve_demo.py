# -*- coding: utf-8 -*-
"""7.qumond_field_solve_demo__.py

QUMOND Field-Solve Demo (Conservative Completion)
=================================================

This is a *working* demo that implements the paper-level conservative
completion:

  - solve for potentials (Φ_N, Φ) on a mesh
  - compute acceleration as a = -∇Φ
  - compare Newton vs QUMOND rotation curve in the disk midplane

Outputs
-------
Creates structured artifacts under:

  ./7.qumond_field_solve_demo__/figures/
  ./7.qumond_field_solve_demo__/data/
  ./7.qumond_field_solve_demo__/logs/

including:
  - rotation_curve_qumond_midplane.csv
  - rotation_curve_qumond_midplane.png
  - rho_phi_slices.png
  - run_metadata.json

Usage
-----
  python 7.qumond_field_solve_demo__.py --N 96 --L 200 --pad 2 --nu nu_standard

Notes
-----
- FFT Poisson is periodic by default. pad>1 uses a padded solve and crops
  back, which reduces periodic-image contamination.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# Headless plotting by default
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None

# Local import (module shipped in this repo)
from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    StaticFieldSampler,
    cic_gather_vector_field,
    exponential_disk_density,
    nu_simple,
    nu_standard,
)


def _setup_out_dirs(script_path: str) -> dict[str, Path]:
    current_dir = Path(os.path.dirname(os.path.abspath(script_path)))
    script_name = Path(script_path).stem
    out_root = current_dir / script_name

    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    log_dir = out_root / "logs"

    for d in (fig_dir, data_dir, log_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "out_root": out_root,
        "fig_dir": fig_dir,
        "data_dir": data_dir,
        "log_dir": log_dir,
    }


def _finite_diff_grad(phi: np.ndarray, dx: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Central-difference gradient via np.gradient.

    We use this instead of spectral derivatives so that the returned
    phi from a padded+cropped solve still yields a reasonable interior gradient.
    """
    dphi_dx, dphi_dy, dphi_dz = np.gradient(phi, dx, dx, dx, edge_order=2)
    return dphi_dx, dphi_dy, dphi_dz


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND conservative completion demo (mesh Poisson solves)")

    ap.add_argument("--N", type=int, default=96, help="Grid size (N^3). Use 64~160 for laptop.")
    ap.add_argument("--L", type=float, default=200.0, help="Box size (same units as positions)")
    ap.add_argument("--G", type=float, default=1.0, help="Gravitational constant in code units")

    ap.add_argument("--M", type=float, default=1000.0, help="Disk total mass (code units)")
    ap.add_argument("--R_d", type=float, default=3.0, help="Exponential disk scale length")
    ap.add_argument("--z_d", type=float, default=0.3, help="Exponential vertical scale height")

    ap.add_argument("--a0", type=float, default=0.12, help="Acceleration scale")
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"], help="Interpolation ν(y)")
    ap.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="Padding factor (reduces periodic artifacts)")

    ap.add_argument("--r_min", type=float, default=0.2, help="Min radius for rotation curve sampling")
    ap.add_argument("--r_max", type=float, default=40.0, help="Max radius for rotation curve sampling")
    ap.add_argument("--nr", type=int, default=300, help="Number of radius samples")

    args = ap.parse_args()

    out = _setup_out_dirs(__file__)
    print(f"[Info] Output root: {out['out_root']}")

    # ----------------------------
    # Build grid and baryon density
    # ----------------------------
    grid = PMGrid3D(N=int(args.N), boxsize=float(args.L), G=float(args.G))

    t0 = time.time()
    rho = exponential_disk_density(
        grid=grid,
        M_total=float(args.M),
        R_d=float(args.R_d),
        z_d=float(args.z_d),
        renormalize=True,
    )

    # ----------------------------
    # Solve QUMOND
    # ----------------------------
    nu_func = nu_standard if args.nu == "nu_standard" else nu_simple

    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho, a0=float(args.a0), nu_func=nu_func, pad_factor=int(args.pad))

    # Newtonian field for comparison (computed from phi_N)
    dphiN_dx, dphiN_dy, dphiN_dz = _finite_diff_grad(res.phi_N, grid.dx)
    gN_field = np.stack([-dphiN_dx, -dphiN_dy, -dphiN_dz], axis=-1)  # g_N = -∇Φ_N

    # QUMOND sampler
    sampler = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)

    # ----------------------------
    # Rotation curve sample (midplane, x-axis)
    # ----------------------------
    r = np.linspace(float(args.r_min), float(args.r_max), int(args.nr))
    pos = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=1)

    aQ_vec = sampler.accel_at(pos)
    aN_vec = cic_gather_vector_field(gN_field, pos, grid)

    # along x-axis: radial unit vector is +x
    aQ_r = -aQ_vec[:, 0]
    aN_r = -aN_vec[:, 0]

    vQ = np.sqrt(np.maximum(r, 1e-12) * np.maximum(aQ_r, 0.0))
    vN = np.sqrt(np.maximum(r, 1e-12) * np.maximum(aN_r, 0.0))

    # Save CSV
    csv_path = out["data_dir"] / "rotation_curve_qumond_midplane.csv"
    if pd is not None:
        df = pd.DataFrame({
            "r": r,
            "aN": aN_r,
            "aQ": aQ_r,
            "vN": vN,
            "vQ": vQ,
        })
        df.to_csv(csv_path, index=False)
    else:
        arr = np.stack([r, aN_r, aQ_r, vN, vQ], axis=1)
        np.savetxt(csv_path, arr, delimiter=",", header="r,aN,aQ,vN,vQ", comments="")

    # Plot rotation curve
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(r, vN, linestyle="--", linewidth=1.8, label="Newton (Φ_N)")
    ax.plot(r, vQ, linewidth=2.4, alpha=0.9, label=f"QUMOND (ν={args.nu}, pad={args.pad})")
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Circular speed v")
    ax.set_title("Midplane rotation curve from conservative QUMOND field solve")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend()
    fig.tight_layout()

    png_rc = out["fig_dir"] / "rotation_curve_qumond_midplane.png"
    fig.savefig(png_rc, dpi=250)
    plt.close(fig)

    # ----------------------------
    # Slice plots
    # ----------------------------
    mid = grid.N // 2

    rho_sl = rho[:, :, mid]
    phiN_sl = res.phi_N[:, :, mid]
    phi_sl = res.phi[:, :, mid]
    a_mag_sl = np.sqrt(np.sum(res.accel[:, :, mid, :] ** 2, axis=-1))

    fig2 = plt.figure(figsize=(12, 10))
    axs = [fig2.add_subplot(2, 2, i + 1) for i in range(4)]

    im0 = axs[0].imshow(rho_sl.T, origin="lower")
    axs[0].set_title(r"$\rho$ (midplane)")
    plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

    im1 = axs[1].imshow(phiN_sl.T, origin="lower")
    axs[1].set_title(r"$\Phi_N$ (midplane)")
    plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

    im2 = axs[2].imshow(phi_sl.T, origin="lower")
    axs[2].set_title(r"$\Phi$ (QUMOND) (midplane)")
    plt.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    im3 = axs[3].imshow(a_mag_sl.T, origin="lower")
    axs[3].set_title(r"$|\mathbf{a}|$ (midplane)")
    plt.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

    for ax_ in axs:
        ax_.set_xticks([])
        ax_.set_yticks([])

    fig2.tight_layout()
    png_slice = out["fig_dir"] / "rho_phi_slices.png"
    fig2.savefig(png_slice, dpi=250)
    plt.close(fig2)

    # ----------------------------
    # Metadata
    # ----------------------------
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "runtime_sec": float(time.time() - t0),
        "grid": {"N": int(args.N), "L": float(args.L), "dx": float(grid.dx), "G": float(args.G)},
        "disk": {"M": float(args.M), "R_d": float(args.R_d), "z_d": float(args.z_d)},
        "qumond": {"a0": float(args.a0), "nu": str(args.nu), "pad": int(args.pad)},
        "rotation_curve": {"r_min": float(args.r_min), "r_max": float(args.r_max), "nr": int(args.nr)},
        "artifacts": {
            "csv": str(csv_path),
            "png_rotation_curve": str(png_rc),
            "png_slices": str(png_slice),
        },
    }

    meta_path = out["log_dir"] / "run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[Saved] {csv_path}")
    print(f"[Saved] {png_rc}")
    print(f"[Saved] {png_slice}")
    print(f"[Saved] {meta_path}")


if __name__ == "__main__":
    main()
