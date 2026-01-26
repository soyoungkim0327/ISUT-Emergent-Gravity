# -*- coding: utf-8 -*-
"""isut_108_qumond_phantom_density_3d.py

QUMOND "Phantom Density" Diagnostic (Dark-Matter-Reverse)
=========================================================

Why this exists (reviewer defense)
---------------------------------
Even if the theory does not introduce particle dark matter, an equivalent
Newtonian reinterpretation exists:

  ∇²Φ = 4πG (ρ_bar + ρ_ph)

where ρ_ph is an *effective* ("phantom") density that would reproduce the
same potential in ordinary Poisson gravity. Reviewers often ask:

  - "What is the implied extra density profile?"
  - "Can you compute it and show it's not pathological?"

In QUMOND, the solved field satisfies:

  ∇²Φ = -∇·[ ν(|g_N|/a0) g_N ]

Thus we define an effective density on the mesh:

  ρ_eff := -(1/(4πG)) ∇·[ ν g_N ]
  ρ_bar_eff := ρ_bar - ⟨ρ_bar⟩   (periodic Poisson uses mean-subtracted source)
  ρ_ph := ρ_eff - ρ_bar_eff

Outputs (relative path)
-----------------------
  ./isut_108_qumond_phantom_density_3d/
      data/    (CSV)
      figures/ (PNG)
      logs/    (JSON)

Usage
-----
  python 01.Core_Engine/isut_108_qumond_phantom_density_3d.py --N 96 --L 200 --pad 1

Notes
-----
This extends the original slice diagnostic with an additional 3D scatter
visualization (PNG) and an optional interactive HTML export (if plotly is
installed). For large meshes, the 3D plot sub-samples the highest-|rho_ph|
voxels to stay lightweight.

Notes
-----
- For strict spectral operators (best for this diagnostic), use pad=1.
- For pad>1 cropped fields, a finite-difference approximation would be needed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
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


def _grad_periodic(phi: np.ndarray, grid: PMGrid3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi_k = np.fft.fftn(phi)
    dphi_dx = np.fft.ifftn(1j * grid.kx * phi_k).real
    dphi_dy = np.fft.ifftn(1j * grid.ky * phi_k).real
    dphi_dz = np.fft.ifftn(1j * grid.kz * phi_k).real
    return dphi_dx, dphi_dy, dphi_dz


def _div_periodic(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, grid: PMGrid3D) -> np.ndarray:
    Fx_k = np.fft.fftn(Fx)
    Fy_k = np.fft.fftn(Fy)
    Fz_k = np.fft.fftn(Fz)
    div_k = 1j * (grid.kx * Fx_k + grid.ky * Fy_k + grid.kz * Fz_k)
    return np.fft.ifftn(div_k).real


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND phantom density diagnostic")
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=200.0)
    ap.add_argument("--pad", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--a0", type=float, default=0.12)
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"])
    ap.add_argument("--M", type=float, default=1000.0)
    ap.add_argument("--Rd", type=float, default=3.0)
    ap.add_argument("--zd", type=float, default=0.4)
    ap.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="3D scatter threshold: keep voxels with |rho_ph| above this quantile.",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=12000,
        help="Maximum number of voxels plotted in the 3D scatter (randomly subsampled if needed).",
    )
    ap.add_argument(
        "--html",
        action="store_true",
        help="Also export an interactive HTML (requires plotly).",
    )
    args = ap.parse_args()

    print(f"[System] {SCRIPT_NAME} initialized")
    print(f"[Info] Output Root : {OUT_ROOT}")
    if args.pad != 1:
        print("[Warn] pad>1 returns cropped fields; this diagnostic is most meaningful for pad=1 spectral operators.")

    nu_func: Callable[[np.ndarray], np.ndarray] = nu_standard if args.nu == "nu_standard" else nu_simple

    grid = PMGrid3D(N=int(args.N), boxsize=float(args.L), G=float(args.G))
    rho_bar = exponential_disk_density(grid, M_total=float(args.M), R_d=float(args.Rd), z_d=float(args.zd), renormalize=True)

    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho_bar, a0=float(args.a0), nu_func=nu_func, pad_factor=int(args.pad))

    # Compute Newtonian field g_N = -∇Φ_N
    dphiN_dx, dphiN_dy, dphiN_dz = _grad_periodic(res.phi_N, grid)
    gNx, gNy, gNz = -dphiN_dx, -dphiN_dy, -dphiN_dz

    gN_mag = np.sqrt(gNx**2 + gNy**2 + gNz**2)
    nu = nu_func(gN_mag / max(float(args.a0), 1e-30))

    # Effective density source for Φ
    divF = _div_periodic(nu * gNx, nu * gNy, nu * gNz, grid)
    rho_eff = -(1.0 / (4.0 * np.pi * float(args.G))) * divF

    # Periodic Newtonian Poisson uses mean-subtracted baryon source
    rho_bar_eff = rho_bar - float(np.mean(rho_bar))

    rho_ph = rho_eff - rho_bar_eff

    # Basic stats
    cell_vol = grid.dx**3
    M_bar = float(np.sum(rho_bar) * cell_vol)
    M_ph_pos = float(np.sum(np.clip(rho_ph, 0.0, None)) * cell_vol)
    M_ph_neg = float(np.sum(np.clip(rho_ph, None, 0.0)) * cell_vol)
    M_eff = float(np.sum(rho_eff) * cell_vol)

    stats = {
        "M_bar_grid": M_bar,
        "M_eff_grid": M_eff,
        "M_ph_pos": M_ph_pos,
        "M_ph_neg": M_ph_neg,
        "rho_ph_min": float(np.min(rho_ph)),
        "rho_ph_max": float(np.max(rho_ph)),
        "rho_eff_min": float(np.min(rho_eff)),
        "rho_eff_max": float(np.max(rho_eff)),
    }

    # Save CSV summary
    csv_path = os.path.join(DATA_DIR, "data_phantom_density_summary.csv")
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(stats.keys()))
        w.writeheader()
        w.writerow(stats)

    # Slice plots (midplane)
    mid = int(args.N) // 2
    rho_bar_sl = rho_bar[:, :, mid]
    rho_ph_sl = rho_ph[:, :, mid]
    rho_eff_sl = rho_eff[:, :, mid]

    fig = plt.figure(figsize=(12, 9))
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(rho_bar_sl.T, origin="lower")
    ax1.set_title(r"$\rho_{\rm bar}$ (midplane)")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(rho_eff_sl.T, origin="lower")
    ax2.set_title(r"$\rho_{\rm eff}$ (from QUMOND source)")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    ax3 = fig.add_subplot(2, 2, 3)
    # signed phantom density can be visualized with symmetric scaling
    vmax = np.quantile(np.abs(rho_ph_sl), 0.995)
    im3 = ax3.imshow(rho_ph_sl.T, origin="lower", vmin=-vmax, vmax=vmax)
    ax3.set_title(r"$\rho_{\rm ph} = \rho_{\rm eff} - (\rho_{\rm bar}-\langle\rho\rangle)$")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    txt = (
        f"M_bar(grid) = {M_bar:.3g}\n"
        f"M_eff(grid) = {M_eff:.3g}\n"
        f"M_ph(+) = {M_ph_pos:.3g}\n"
        f"M_ph(-) = {M_ph_neg:.3g}\n"
        f"rho_ph min/max = {stats['rho_ph_min']:.3g} / {stats['rho_ph_max']:.3g}\n"
        f"pad={args.pad}, nu={args.nu}"
    )
    ax4.text(0.02, 0.98, txt, va="top", ha="left", fontsize=11)

    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_phantom_density_slices.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3D paper visualization: high-|rho_ph| voxel scatter
    # ------------------------------------------------------------------
    abs_rho = np.abs(rho_ph)
    q = float(np.clip(args.quantile, 0.0, 1.0))
    thr = np.quantile(abs_rho, q)
    mask = abs_rho >= thr
    idx = np.argwhere(mask)
    if idx.size == 0:
        print(f"[Warn] 3D scatter: no voxels above quantile={q}. Skipping.")
    else:
        if idx.shape[0] > args.max_points:
            rng = np.random.default_rng(0)
            sel = rng.choice(idx.shape[0], size=args.max_points, replace=False)
            idx = idx[sel]

        # Convert voxel indices -> centered coordinates in box units
        N = grid.N
        dx = grid.dx
        ix, iy, iz = idx[:, 0], idx[:, 1], idx[:, 2]
        x = (ix - (N // 2)) * dx
        y = (iy - (N // 2)) * dx
        z = (iz - (N // 2)) * dx
        vals = rho_ph[ix, iy, iz]

        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection="3d")
        sc = ax3.scatter(x, y, z, c=vals, s=2)
        ax3.set_title(f"Phantom density (QUMOND): high-|rho_ph| voxels (q={q:.3f})")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("z")
        fig3.colorbar(sc, ax=ax3, shrink=0.6, pad=0.02)
        fig3.tight_layout()

        fig3_path = os.path.join(FIG_DIR, "fig_phantom_density_3d_scatter.png")
        fig3.savefig(fig3_path, dpi=220, bbox_inches="tight")
        plt.close(fig3)
        print(f"[Saved] {fig3_path}")

        if getattr(args, "html", False):
            html_path = os.path.join(FIG_DIR, "fig_phantom_density_3d_scatter.html")
            try:
                import plotly.graph_objects as go

                fig_html = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=x,
                            y=y,
                            z=z,
                            mode="markers",
                            marker=dict(size=2, color=vals, colorscale="RdBu", opacity=0.7),
                        )
                    ]
                )
                fig_html.update_layout(
                    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                    title=f"Phantom density: high-|rho_ph| voxels (q={q:.3f})",
                )
                fig_html.write_html(html_path, include_plotlyjs="cdn")
                print(f"[Saved] {html_path}")
            except Exception as e:
                print(f"[Warn] HTML export requested but failed: {e}")

    # Metadata
    meta: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "args": vars(args),
        "stats": stats,
        "outputs": {
            "csv": os.path.relpath(csv_path, OUT_ROOT),
            "fig": os.path.relpath(fig_path, OUT_ROOT),
        },
        "notes": {
            "interpretation": "rho_ph is the Newtonian-equivalent extra density that reproduces the QUMOND potential.",
            "periodic": "For periodic FFT Poisson, baryon source is mean-subtracted; rho_bar_eff := rho_bar-<rho_bar>.",
        },
    }
    meta_path = os.path.join(LOG_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Phantom density diagnostic complete")
    print(f"  [CSV]  {csv_path}")
    print(f"  [PNG]  {fig_path}")
    print(f"  [META] {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
