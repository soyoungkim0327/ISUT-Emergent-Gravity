# -*- coding: utf-8 -*-
"""8.qumond_energy_audit__.py

QUMOND Energy-Drift Audit (Conservative Field)
==============================================

This audit complements the QUMOND field-solve demo by running a tracer
integration in the *same* conservative acceleration field a=-∇Φ.

What it checks
--------------
- With a fixed potential field (no back-reaction), a symplectic leapfrog
  (KDK) integrator should keep energy drift bounded.
- The reported drift is a numerical stability diagnostic (mesh resolution,
  CIC interpolation, dt), not a proof of continuum conservation.

Outputs
-------
Creates structured artifacts under:

  ./8.qumond_energy_audit__/runs/audit_<timestamp>/{figures,data,logs}

including:
  - energy_drift.csv
  - energy_drift.png
  - snapshot.png
  - run_metadata.json

Usage
-----
  python 8.qumond_energy_audit__.py --N 96 --L 200 --pad 2 --steps 2000

"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

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
    exponential_disk_density,
    nu_simple,
    nu_standard,
)


# -----------------------------------------------------------------------------
# Output manager (lightweight)
# -----------------------------------------------------------------------------

class OutputManager:
    def __init__(self, script_path: str):
        current_dir = Path(os.path.dirname(os.path.abspath(script_path)))
        script_name = Path(script_path).stem
        out_root = current_dir / script_name
        out_root.mkdir(parents=True, exist_ok=True)

        run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.run_dir = out_root / "runs" / f"audit_{run_id}"
        self.fig_dir = self.run_dir / "figures"
        self.data_dir = self.run_dir / "data"
        self.log_dir = self.run_dir / "logs"
        for d in (self.fig_dir, self.data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"[Info] Run directory: {self.run_dir}")

    def save_json(self, name: str, obj: dict) -> Path:
        p = self.log_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"[Saved] {p}")
        return p

    def save_csv(self, name: str, rows: list[dict]) -> Path:
        p = self.data_dir / name
        if pd is not None:
            pd.DataFrame(rows).to_csv(p, index=False)
        else:
            # very simple fallback
            import csv
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        print(f"[Saved] {p}")
        return p

    def save_fig(self, fig, name: str, dpi: int = 200) -> Path:
        p = self.fig_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Saved] {p}")
        return p


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _sample_midplane_radial_accel(sampler: StaticFieldSampler, r: np.ndarray) -> np.ndarray:
    """Sample radial accel at positions (r,0,0) and return inward magnitude."""
    pos = np.stack([r, np.zeros_like(r), np.zeros_like(r)], axis=1)
    a = sampler.accel_at(pos)
    # along +x axis, inward accel magnitude is -a_x if field points to center
    return np.maximum(-a[:, 0], 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND conservative-field energy audit")
    ap.add_argument("--N", type=int, default=96, help="Grid size per dimension")
    ap.add_argument("--L", type=float, default=200.0, help="Boxsize")
    ap.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="Padding factor")
    ap.add_argument("--G", type=float, default=1.0, help="Gravitational constant in code units")

    ap.add_argument("--M", type=float, default=1000.0, help="Disk mass")
    ap.add_argument("--Rd", type=float, default=3.0, help="Disk radial scale")
    ap.add_argument("--zd", type=float, default=0.5, help="Disk vertical scale")

    ap.add_argument("--a0", type=float, default=0.12, help="MOND/ISUT a0")
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"], help="nu(y)")

    ap.add_argument("--npart", type=int, default=2000, help="Number of tracer particles")
    ap.add_argument("--rmax", type=float, default=40.0, help="Max initial radius")
    ap.add_argument("--rmin", type=float, default=1.0, help="Min initial radius")
    ap.add_argument("--steps", type=int, default=2000, help="Number of integration steps")
    ap.add_argument("--dt", type=float, default=0.03, help="Fixed timestep")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out = OutputManager(__file__)

    # Build QUMOND field
    grid = PMGrid3D(N=args.N, boxsize=args.L, G=args.G)
    rho = exponential_disk_density(grid, M_total=args.M, R_d=args.Rd, z_d=args.zd, renormalize=True)

    solver = QUMONDSolverFFT(grid)
    nu_func = nu_standard if args.nu == "nu_standard" else nu_simple
    res = solver.solve(rho=rho, a0=args.a0, nu_func=nu_func, pad_factor=args.pad)

    sampler = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)

    # Init tracer disk in midplane
    rng = np.random.default_rng(args.seed)
    r = rng.uniform(args.rmin, args.rmax, size=args.npart)
    th = rng.uniform(0.0, 2.0 * np.pi, size=args.npart)
    x = r * np.cos(th)
    y = r * np.sin(th)

    # Circular speed from midplane accel
    a_r = _sample_midplane_radial_accel(sampler, r)
    v_circ = np.sqrt(np.maximum(r, 1e-12) * np.maximum(a_r, 0.0))

    vx = -v_circ * np.sin(th)
    vy =  v_circ * np.cos(th)

    dt = float(args.dt)

    def accel_xy(xy: np.ndarray) -> np.ndarray:
        pos = np.stack([xy[:, 0], xy[:, 1], np.zeros_like(xy[:, 0])], axis=1)
        a = sampler.accel_at(pos)
        return a[:, :2]

    def phi_xy(xy: np.ndarray) -> np.ndarray:
        pos = np.stack([xy[:, 0], xy[:, 1], np.zeros_like(xy[:, 0])], axis=1)
        return sampler.phi_at(pos)

    # Energy function for tracers in static potential
    def total_energy(x: np.ndarray, y: np.ndarray, vx: np.ndarray, vy: np.ndarray) -> float:
        K = 0.5 * np.sum(vx**2 + vy**2)
        U = np.sum(phi_xy(np.stack([x, y], axis=1)))
        return float(K + U)

    xy = np.stack([x, y], axis=1)
    v = np.stack([vx, vy], axis=1)

    E0 = total_energy(xy[:, 0], xy[:, 1], v[:, 0], v[:, 1])

    rows: list[dict] = []

    # Integrate with leapfrog KDK
    for s in range(args.steps):
        a = accel_xy(xy)
        v += 0.5 * dt * a
        xy += dt * v
        a2 = accel_xy(xy)
        v += 0.5 * dt * a2

        if s % max(1, args.steps // 500) == 0 or s == args.steps - 1:
            E = total_energy(xy[:, 0], xy[:, 1], v[:, 0], v[:, 1])
            drift = (E - E0) / (E0 if E0 != 0 else 1.0)
            rows.append({"step": s, "time": (s + 1) * dt, "E_total": E, "Drift_Rel": drift})

    out.save_csv("energy_drift.csv", rows)

    # Plot drift
    steps = np.array([r["step"] for r in rows], dtype=float)
    drift = np.array([r["Drift_Rel"] for r in rows], dtype=float)

    fig = plt.figure(figsize=(9, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(steps, drift)
    ax.set_xlabel("Step")
    ax.set_ylabel("(E-E0)/E0")
    ax.set_title("Energy drift (QUMOND conservative field; tracer leapfrog)")
    ax.grid(True, linestyle=":")
    out.save_fig(fig, "energy_drift.png", dpi=220)

    # Snapshot
    fig2 = plt.figure(figsize=(5.2, 5.2))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.scatter(xy[:, 0], xy[:, 1], s=0.6, alpha=0.4)
    ax2.set_aspect("equal")
    ax2.set_title("Tracer snapshot")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True, linestyle=":")
    out.save_fig(fig2, "snapshot.png", dpi=220)

    meta = {
        "script": Path(__file__).name,
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "grid": {"N": args.N, "L": args.L, "pad": args.pad, "G": args.G},
        "disk": {"M": args.M, "Rd": args.Rd, "zd": args.zd},
        "model": {"a0": args.a0, "nu": args.nu},
        "integrator": {"scheme": "leapfrog_KDK", "dt": dt, "steps": args.steps, "npart": args.npart},
        "final": {"E0": E0, "E": rows[-1]["E_total"], "drift_rel": rows[-1]["Drift_Rel"]},
    }
    out.save_json("run_metadata.json", meta)

    print(f"[Done] Final drift: {rows[-1]['Drift_Rel']:.6%}")


if __name__ == "__main__":
    main()
