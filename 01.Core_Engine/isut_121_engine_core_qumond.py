# -*- coding: utf-8 -*-
"""1.isut_engine_core_qumond.py

ISUT Universe Engine (QUMOND Conservative Completion)
====================================================

This script adds an explicit *potential-based* gravity kernel using a QUMOND
(Quasi-linear MOND) style field solve.

Why this exists
---------------
The paper's conservative completion requires accelerations to be obtained from a
scalar potential:

    a(x) = -∇Φ(x)

QUMOND makes this practical by using *two* Poisson solves:

  1) Newtonian potential
        ∇² Φ_N = 4π G ρ
        g_N = -∇Φ_N

  2) QUMOND potential
        ∇² Φ = -∇·[ ν(|g_N|/a0) g_N ]

Once Φ is solved, we return a = -∇Φ.

This script is designed to be runnable by users as part of an "Open Evidence Pack":
- Produces CSV/PNG artifacts + JSON metadata
- Uses a simple analytic exponential disk density as a demo baryonic source

Usage
-----
From repository root:

  python 01.Core_Engine/1.isut_engine_core_qumond.py --headless

Outputs are written to:
  01.Core_Engine/1.isut_engine_core_qumond/runs/<timestamp>/

"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

# Local import (same folder)
from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    StaticFieldSampler,
    cic_gather_vector_field,
    exponential_disk_density,
    nu_simple,
    nu_standard,
)


# =============================================================================
# Output manager (structured artifacts)
# =============================================================================

class OutputManager:
    """Create a run folder and save CSV/PNG/JSON reproducibly."""

    def __init__(self, base_dir: str, run_prefix: str = "core"):
        self.out_root = Path(base_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.run_dir = self.out_root / "runs" / f"{run_prefix}_{self.run_id}"
        self.fig_dir = self.run_dir / "figures"
        self.data_dir = self.run_dir / "data"
        self.log_dir = self.run_dir / "logs"
        for d in (self.fig_dir, self.data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

    def env(self) -> Dict[str, str]:
        import numpy as _np
        env = {
            "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": getattr(_np, "__version__", "unknown"),
        }
        try:
            import matplotlib
            env["matplotlib"] = getattr(matplotlib, "__version__", "unknown")
        except Exception:
            env["matplotlib"] = "missing"
        return env

    def save_json(self, name: str, obj: Dict) -> Path:
        p = self.log_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        return p

    def save_csv(self, name: str, arr: np.ndarray, header: str) -> Path:
        p = self.data_dir / name
        np.savetxt(p, arr, delimiter=",", header=header, comments="")
        return p

    def save_fig(self, name: str, fig, dpi: int = 150) -> Path:
        p = self.fig_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return p


# =============================================================================
# Helpers: spectral gradient, rotation curve, integrator
# =============================================================================


def spectral_accel_from_phi(phi: np.ndarray, grid: PMGrid3D) -> np.ndarray:
    """Compute acceleration field a=-∇phi using spectral derivatives."""
    phi_k = np.fft.fftn(phi)
    dphi_dx = np.fft.ifftn(1j * grid.kx * phi_k).real
    dphi_dy = np.fft.ifftn(1j * grid.ky * phi_k).real
    dphi_dz = np.fft.ifftn(1j * grid.kz * phi_k).real
    ax = -dphi_dx
    ay = -dphi_dy
    az = -dphi_dz
    return np.stack([ax, ay, az], axis=-1)


def radial_accel_at_positions(accel_xyz: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Return inward radial acceleration magnitude g(r) = -a·rhat (>=0 for inward)."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.maximum(r, 1e-12)
    rhat = np.stack([x / r_safe, y / r_safe, z / r_safe], axis=1)
    a_dot = np.sum(accel_xyz * rhat, axis=1)  # signed
    g_in = np.maximum(-a_dot, 0.0)
    return g_in


def compute_rotation_curve(
    sampler: StaticFieldSampler,
    radii: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute v(r)=sqrt(r*g(r)) in the midplane along x-axis (y=z=0)."""
    pos = np.zeros((len(radii), 3), dtype=np.float64)
    pos[:, 0] = radii
    a = sampler.accel_at(pos)
    g = radial_accel_at_positions(a, pos)
    v = np.sqrt(np.maximum(radii, 0.0) * g)
    return g, v


def leapfrog_kdk_orbit(
    sampler: StaticFieldSampler,
    r0: float,
    dt: float,
    steps: int,
) -> Dict[str, np.ndarray]:
    """Integrate a single tracer in the fixed potential using leapfrog KDK."""

    # initial position on x-axis, midplane
    pos = np.array([[r0, 0.0, 0.0]], dtype=np.float64)

    # circular speed from local radial accel
    a0 = sampler.accel_at(pos)
    g0 = radial_accel_at_positions(a0, pos)[0]
    v_circ = float(np.sqrt(max(r0 * g0, 0.0)))

    vel = np.array([[0.0, v_circ, 0.0]], dtype=np.float64)

    t_hist = np.zeros((steps,), dtype=np.float64)
    r_hist = np.zeros((steps,), dtype=np.float64)
    E_hist = np.zeros((steps,), dtype=np.float64)

    for i in range(steps):
        # record energy BEFORE step
        phi = sampler.phi_at(pos)[0]
        v2 = float(np.sum(vel[0] ** 2))
        E = 0.5 * v2 + float(phi)

        t_hist[i] = i * dt
        r_hist[i] = float(np.linalg.norm(pos[0]))
        E_hist[i] = E

        # KDK
        acc = sampler.accel_at(pos)
        vel_half = vel + 0.5 * dt * acc
        pos = pos + dt * vel_half
        acc2 = sampler.accel_at(pos)
        vel = vel_half + 0.5 * dt * acc2

    return {
        "t": t_hist,
        "r": r_hist,
        "E": E_hist,
        "v_circ_init": np.array([v_circ], dtype=np.float64),
    }


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ISUT QUMOND core demo")
    p.add_argument("--N", type=int, default=96, help="Grid size (N^3)")
    p.add_argument("--L", type=float, default=200.0, help="Box size")
    p.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="Zero-padding factor")

    p.add_argument("--M", type=float, default=1000.0, help="Disk total mass")
    p.add_argument("--Rd", type=float, default=3.0, help="Disk radial scale length")
    p.add_argument("--zd", type=float, default=0.5, help="Disk vertical scale height")

    p.add_argument("--a0", type=float, default=0.12, help="Acceleration scale a0")
    p.add_argument(
        "--nu",
        type=str,
        default="nu_standard",
        choices=["nu_standard", "nu_simple"],
        help="Interpolation ν(y)",
    )

    p.add_argument("--rmax", type=float, default=40.0, help="Max radius for rotation curve")
    p.add_argument("--nr", type=int, default=240, help="Number of radius samples")

    p.add_argument("--r0", type=float, default=10.0, help="Initial radius for orbit demo")
    p.add_argument("--dt", type=float, default=0.05, help="Integrator timestep")
    p.add_argument("--steps", type=int, default=2000, help="Orbit integration steps")

    p.add_argument("--save_slices", action="store_true", help="Save rho/phi slices figure")
    p.add_argument("--headless", action="store_true", help="Do not show plots (save only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Matplotlib is imported lazily so headless runs don't require a GUI backend.
    import matplotlib

    if args.headless:
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    out_base = os.path.join(current_dir, script_name)
    out = OutputManager(out_base, run_prefix="core")

    # Choose ν
    nu_func: Callable[[np.ndarray], np.ndarray] = nu_standard if args.nu == "nu_standard" else nu_simple

    # Build grid + density
    grid = PMGrid3D(N=args.N, boxsize=args.L, G=1.0)
    rho = exponential_disk_density(grid, M_total=args.M, R_d=args.Rd, z_d=args.zd, renormalize=True)

    # Solve QUMOND
    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho, a0=args.a0, nu_func=nu_func, pad_factor=args.pad)

    # Newtonian accel field from phi_N
    accel_N = spectral_accel_from_phi(res.phi_N, grid)

    sampler_N = StaticFieldSampler(grid=grid, phi=res.phi_N, accel=accel_N)
    sampler_Q = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)

    # Rotation curve
    radii = np.linspace(0.5, args.rmax, args.nr)
    gN, vN = compute_rotation_curve(sampler_N, radii)
    gQ, vQ = compute_rotation_curve(sampler_Q, radii)

    rc_arr = np.column_stack([radii, gN, vN, gQ, vQ])
    out.save_csv(
        "rotation_curve_midplane.csv",
        rc_arr,
        header="r,gN,vN,gQ,vQ",
    )

    fig1 = plt.figure(figsize=(9, 5))
    ax = fig1.add_subplot(111)
    ax.plot(radii, vN, "--", label="Newtonian (from Φ_N)")
    ax.plot(radii, vQ, "-", label=f"QUMOND (ν={args.nu})")
    ax.set_xlabel("Radius r")
    ax.set_ylabel("Circular speed v")
    ax.set_title("Midplane rotation curve (field solve)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out.save_fig("rotation_curve_midplane.png", fig1)

    # Orbit energy audit
    orb = leapfrog_kdk_orbit(sampler_Q, r0=args.r0, dt=args.dt, steps=args.steps)

    E0 = orb["E"][0]
    drift = float((orb["E"][-1] - E0) / (E0 if abs(E0) > 1e-30 else 1.0))

    orb_arr = np.column_stack([orb["t"], orb["r"], orb["E"]])
    out.save_csv("orbit_energy_audit.csv", orb_arr, header="t,r,E")

    fig2 = plt.figure(figsize=(11, 4.5))
    ax1 = fig2.add_subplot(121)
    ax1.plot(orb["t"], orb["r"], linewidth=1.5)
    ax1.set_title("Orbit radius")
    ax1.set_xlabel("t")
    ax1.set_ylabel("r")
    ax1.grid(True, alpha=0.3)

    ax2 = fig2.add_subplot(122)
    ax2.plot(orb["t"], orb["E"], linewidth=1.5)
    ax2.set_title(f"Energy (drift={drift:.3e})")
    ax2.set_xlabel("t")
    ax2.set_ylabel("E")
    ax2.grid(True, alpha=0.3)
    out.save_fig("orbit_energy_audit.png", fig2)

    # Optional diagnostic slices
    if args.save_slices:
        mid = args.N // 2
        fig3 = plt.figure(figsize=(12, 4))
        axa = fig3.add_subplot(131)
        im0 = axa.imshow(rho[:, :, mid].T, origin="lower")
        axa.set_title("rho (midplane)")
        plt.colorbar(im0, ax=axa, fraction=0.046)

        axb = fig3.add_subplot(132)
        im1 = axb.imshow(res.phi_N[:, :, mid].T, origin="lower")
        axb.set_title("Φ_N (midplane)")
        plt.colorbar(im1, ax=axb, fraction=0.046)

        axc = fig3.add_subplot(133)
        im2 = axc.imshow(res.phi[:, :, mid].T, origin="lower")
        axc.set_title("Φ (QUMOND, midplane)")
        plt.colorbar(im2, ax=axc, fraction=0.046)

        out.save_fig("rho_phi_slices.png", fig3)

    # Metadata
    meta = {
        "args": vars(args),
        "env": out.env(),
        "outputs": {
            "rotation_curve_csv": "data/rotation_curve_midplane.csv",
            "rotation_curve_png": "figures/rotation_curve_midplane.png",
            "orbit_csv": "data/orbit_energy_audit.csv",
            "orbit_png": "figures/orbit_energy_audit.png",
        },
        "notes": {
            "boundary": "FFT periodic; optional zero-padding approximates isolated BC by moving images away.",
            "conservative": "Acceleration computed as a=-∇Φ from the solved potential.",
        },
    }
    out.save_json("run_metadata.json", meta)

    print("=" * 72)
    print("ISUT QUMOND core demo complete")
    print(f"Run directory: {out.run_dir}")
    print("Artifacts:")
    print(f"  - {out.data_dir / 'rotation_curve_midplane.csv'}")
    print(f"  - {out.fig_dir / 'rotation_curve_midplane.png'}")
    print(f"  - {out.data_dir / 'orbit_energy_audit.csv'}")
    print(f"  - {out.fig_dir / 'orbit_energy_audit.png'}")
    print("=" * 72)

    if not args.headless:
        plt.show()


if __name__ == "__main__":
    main()
