# -*- coding: utf-8 -*-
"""10.qumond_poisson_residuals__.py

QUMOND Poisson Residual Diagnostics
==================================

This script is designed for "hostile reviewer" questions like:

  - "Do your potentials actually satisfy the stated PDEs?"
  - "Are you just plotting something, or is the solver consistent?"

We compute discrete residuals for:

  (A) Newtonian solve (periodic FFT):
        ∇²Φ_N = 4πG(ρ - ⟨ρ⟩)

      Note: periodic Poisson requires zero-mean source. The FFT solve sets
      the k=0 mode to zero, which is equivalent to subtracting the volume mean.
      This has no effect on forces (gradients).

  (B) QUMOND completion:
        ∇²Φ = -∇·[ ν(|g_N|/a0) g_N ]

Residuals are reported as RMS/L∞ and visualized with histograms.

Outputs (relative path)
-----------------------
  ./10.qumond_poisson_residuals__/
      data/    (CSV)
      figures/ (PNG)
      logs/    (JSON)

Usage
-----
  python 01.Core_Engine/10.qumond_poisson_residuals__.py --N 96 --L 200 --pad 1

Tips
----
- For a strict spectral residual check, use pad=1 (pure periodic domain).
- For pad>1, the returned fields are cropped; the interior should still be
  consistent, but boundary points can show larger finite-difference residual.
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


# =============================================================================
# Spectral helpers (periodic)
# =============================================================================

def _laplacian_periodic(phi: np.ndarray, grid: PMGrid3D) -> np.ndarray:
    """Spectral Laplacian on a periodic grid."""
    phi_k = np.fft.fftn(phi)
    lap_k = -(grid.k2 * phi_k)
    return np.fft.ifftn(lap_k).real


def _grad_periodic(phi: np.ndarray, grid: PMGrid3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spectral gradient on a periodic grid."""
    phi_k = np.fft.fftn(phi)
    dphi_dx = np.fft.ifftn(1j * grid.kx * phi_k).real
    dphi_dy = np.fft.ifftn(1j * grid.ky * phi_k).real
    dphi_dz = np.fft.ifftn(1j * grid.kz * phi_k).real
    return dphi_dx, dphi_dy, dphi_dz


def _div_periodic(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, grid: PMGrid3D) -> np.ndarray:
    """Spectral divergence on a periodic grid."""
    Fx_k = np.fft.fftn(Fx)
    Fy_k = np.fft.fftn(Fy)
    Fz_k = np.fft.fftn(Fz)
    div_k = 1j * (grid.kx * Fx_k + grid.ky * Fy_k + grid.kz * Fz_k)
    return np.fft.ifftn(div_k).real


def _fd_laplacian(phi: np.ndarray, dx: float) -> np.ndarray:
    """Second-order finite-difference Laplacian (non-periodic edges).

    Edges use one-sided differences implicitly via np.gradient.
    """
    dphi_dx, dphi_dy, dphi_dz = np.gradient(phi, dx, dx, dx, edge_order=2)
    d2phi_dx2 = np.gradient(dphi_dx, dx, axis=0, edge_order=2)
    d2phi_dy2 = np.gradient(dphi_dy, dx, axis=1, edge_order=2)
    d2phi_dz2 = np.gradient(dphi_dz, dx, axis=2, edge_order=2)
    return d2phi_dx2 + d2phi_dy2 + d2phi_dz2


def _fd_div(Fx: np.ndarray, Fy: np.ndarray, Fz: np.ndarray, dx: float) -> np.ndarray:
    dFx_dx = np.gradient(Fx, dx, axis=0, edge_order=2)
    dFy_dy = np.gradient(Fy, dx, axis=1, edge_order=2)
    dFz_dz = np.gradient(Fz, dx, axis=2, edge_order=2)
    return dFx_dx + dFy_dy + dFz_dz


def _mask_interior(shape: Tuple[int, int, int], margin: int) -> np.ndarray:
    m = margin
    mask = np.zeros(shape, dtype=bool)
    mask[m:-m, m:-m, m:-m] = True
    return mask


def _summarize_residual(res: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    r = res[mask]
    rr = ref[mask]

    rms = float(np.sqrt(np.mean(r**2)))
    linf = float(np.max(np.abs(r)))
    ref_rms = float(np.sqrt(np.mean(rr**2)))
    ref_linf = float(np.max(np.abs(rr)))

    return {
        "rms": rms,
        "linf": linf,
        "ref_rms": ref_rms,
        "ref_linf": ref_linf,
        "rel_rms": float(rms / (ref_rms if ref_rms > 0 else 1.0)),
        "rel_linf": float(linf / (ref_linf if ref_linf > 0 else 1.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND Poisson residual diagnostics")
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=200.0)
    ap.add_argument("--pad", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--a0", type=float, default=0.12)
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"])
    ap.add_argument("--M", type=float, default=1000.0)
    ap.add_argument("--Rd", type=float, default=3.0)
    ap.add_argument("--zd", type=float, default=0.4)
    ap.add_argument("--margin", type=int, default=4, help="Interior margin (cells) for pad>1 finite-diff check")
    args = ap.parse_args()

    print(f"[System] {SCRIPT_NAME} initialized")
    print(f"[Info] Output Root : {OUT_ROOT}")

    grid = PMGrid3D(N=int(args.N), boxsize=float(args.L), G=float(args.G))
    rho = exponential_disk_density(grid, M_total=float(args.M), R_d=float(args.Rd), z_d=float(args.zd), renormalize=True)

    nu_func: Callable[[np.ndarray], np.ndarray] = nu_standard if args.nu == "nu_standard" else nu_simple

    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho, a0=float(args.a0), nu_func=nu_func, pad_factor=int(args.pad))

    # Sources
    rho_mean = float(np.mean(rho))
    rhs_N = 4.0 * np.pi * float(args.G) * (rho - rho_mean)

    # Compute g_N from phi_N (derive with same operator used for residual)
    if args.pad == 1:
        # Strict spectral residual
        lap_phiN = _laplacian_periodic(res.phi_N, grid)
        dphiN_dx, dphiN_dy, dphiN_dz = _grad_periodic(res.phi_N, grid)
        gNx, gNy, gNz = -dphiN_dx, -dphiN_dy, -dphiN_dz

        gN_mag = np.sqrt(gNx**2 + gNy**2 + gNz**2)
        nu = nu_func(gN_mag / max(float(args.a0), 1e-30))
        divF = _div_periodic(nu * gNx, nu * gNy, nu * gNz, grid)
        # Periodic FFT Poisson solver sets k=0 mode to 0.
        # Numerically, div(F) should have zero mean, but subtract any tiny
        # floating-point mean to make the residual diagnostic reflect the
        # *shape* mismatch rather than a constant offset.
        rhs_Q = (-divF) - float(np.mean(-divF))

        lap_phi = _laplacian_periodic(res.phi, grid)

        mask = np.ones_like(res.phi, dtype=bool)
        mode = "spectral(periodic)"
    else:
        # Cropped padded field: do an interior finite-difference residual
        lap_phiN = _fd_laplacian(res.phi_N, grid.dx)
        dphiN_dx, dphiN_dy, dphiN_dz = np.gradient(res.phi_N, grid.dx, grid.dx, grid.dx, edge_order=2)
        gNx, gNy, gNz = -dphiN_dx, -dphiN_dy, -dphiN_dz

        gN_mag = np.sqrt(gNx**2 + gNy**2 + gNz**2)
        nu = nu_func(gN_mag / max(float(args.a0), 1e-30))
        rhs_Q_raw = -_fd_div(nu * gNx, nu * gNy, nu * gNz, grid.dx)
        rhs_Q = rhs_Q_raw - float(np.mean(rhs_Q_raw))
        lap_phi = _fd_laplacian(res.phi, grid.dx)

        mask = _mask_interior(res.phi.shape, int(args.margin))
        mode = "finite-diff(interior)"

    # Residuals
    resN = lap_phiN - rhs_N
    resQ = lap_phi - rhs_Q

    sumN = _summarize_residual(resN, rhs_N, mask)
    sumQ = _summarize_residual(resQ, rhs_Q, mask)

    # Save summary JSON
    summary: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "mode": mode,
        "args": vars(args),
        "notes": {
            "periodic_mean": "FFT periodic Poisson sets k=0 to zero => equivalent to subtracting mean density.",
            "interpretation": "Residual norms should be small relative to RHS norms if solve is consistent.",
        },
        "newton_residual": sumN,
        "qumond_residual": sumQ,
    }

    meta_path = os.path.join(LOG_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Save CSV
    csv_path = os.path.join(DATA_DIR, "data_poisson_residual_summary.csv")
    rows = [
        {"eq": "newton", **sumN},
        {"eq": "qumond", **sumQ},
    ]
    # Minimal CSV writer (no pandas requirement)
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Histograms
    fig = plt.figure(figsize=(11, 4.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.hist(np.log10(np.abs(resN[mask]) + 1e-60), bins=80, alpha=0.9)
    ax1.set_title("log10 |Newton residual|")
    ax1.set_xlabel("log10(|∇²Φ_N - 4πG(ρ-⟨ρ⟩)|)")
    ax1.set_ylabel("count")

    ax2.hist(np.log10(np.abs(resQ[mask]) + 1e-60), bins=80, alpha=0.9)
    ax2.set_title("log10 |QUMOND residual|")
    ax2.set_xlabel("log10(|∇²Φ + ∇·(ν g_N)|)")
    ax2.set_ylabel("count")

    fig.suptitle(f"Poisson residual diagnostics ({mode})")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_poisson_residual_hist.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print("=" * 72)
    print("Residual summary")
    print(f"  Newton  rel_rms={sumN['rel_rms']:.3e}  rel_linf={sumN['rel_linf']:.3e}")
    print(f"  QUMOND  rel_rms={sumQ['rel_rms']:.3e}  rel_linf={sumQ['rel_linf']:.3e}")
    print(f"  [CSV]  {csv_path}")
    print(f"  [PNG]  {fig_path}")
    print(f"  [META] {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
