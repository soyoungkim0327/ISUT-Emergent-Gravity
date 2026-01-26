# -*- coding: utf-8 -*-
"""2.isut_engine_audit_qumond.py

ISUT Engine Audit (QUMOND Conservative Kernel)
==============================================

This audit suite validates numerical stability of a *potential-based* QUMOND
field-solve kernel by tracking the energy drift of a tracer orbit in a static
baryonic disk potential.

Compared to the legacy (algebraic) audit, this script:
- builds a 3D exponential disk density on a mesh
- performs a QUMOND 2-Poisson field solve
- integrates a tracer orbit with leapfrog KDK in the solved potential
- reports relative energy drift as a numerical stability diagnostic

Usage
-----
  python 01.Core_Engine/2.isut_engine_audit_qumond.py --mode all --headless

Outputs
-------
  01.Core_Engine/2.isut_engine_audit_qumond/runs/<timestamp>/{data,figures,logs}

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
from typing import Dict, Any, List, Tuple

import numpy as np

# Local solver module
from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    StaticFieldSampler,
    exponential_disk_density,
    nu_simple,
    nu_standard,
)


# ============================================================
# Output manager (structured artifacts)
# ============================================================

class OutputManager:
    def __init__(self, base_dir: str, prefix: str = "audit"):
        self.out_root = Path(base_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.run_dir = self.out_root / "runs" / f"{prefix}_{self.run_id}"
        self.fig_dir = self.run_dir / "figures"
        self.data_dir = self.run_dir / "data"
        self.log_dir = self.run_dir / "logs"

        for d in (self.fig_dir, self.data_dir, self.log_dir):
            d.mkdir(parents=True, exist_ok=True)

        print(f"[Info] Run directory: {self.run_dir}")

    def env(self) -> Dict[str, Any]:
        return {
            "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", "unknown"),
        }

    def save_json(self, name: str, obj: Dict[str, Any]) -> Path:
        p = self.log_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"   L log: {p.name}")
        return p

    def save_csv(self, name: str, rows: List[Dict[str, Any]]) -> Path:
        # lightweight CSV (no pandas dependency)
        import csv

        p = self.data_dir / name
        if len(rows) == 0:
            with open(p, "w", encoding="utf-8") as f:
                f.write("")
            return p

        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"   L data: {p.name}")
        return p

    def save_fig(self, name: str, fig) -> Path:
        p = self.fig_dir / name
        fig.savefig(p, dpi=160, bbox_inches="tight")
        print(f"   L fig : {p.name}")
        return p


# ============================================================
# Physics / numerics helpers
# ============================================================

@dataclass
class OrbitState:
    x: float
    y: float
    vx: float
    vy: float
    t: float = 0.0


def radial_accel_in_midplane(ax: float, ay: float, x: float, y: float) -> float:
    r = float(np.hypot(x, y))
    if r <= 0:
        return 0.0
    rhatx, rhaty = x / r, y / r
    a_rad = ax * rhatx + ay * rhaty
    return -a_rad  # centripetal magnitude


def kdk_step(state: OrbitState, dt: float, accel_fn) -> OrbitState:
    # Kick
    ax, ay = accel_fn(state.x, state.y)
    vxh = state.vx + 0.5 * dt * ax
    vyh = state.vy + 0.5 * dt * ay

    # Drift
    x2 = state.x + dt * vxh
    y2 = state.y + dt * vyh

    # Kick
    ax2, ay2 = accel_fn(x2, y2)
    vx2 = vxh + 0.5 * dt * ax2
    vy2 = vyh + 0.5 * dt * ay2

    return OrbitState(x=x2, y=y2, vx=vx2, vy=vy2, t=state.t + dt)


def build_static_field(args) -> Tuple[PMGrid3D, StaticFieldSampler]:
    grid = PMGrid3D(N=args.N, boxsize=args.L, G=args.G)
    rho = exponential_disk_density(grid, M_total=args.M, R_d=args.Rd, z_d=args.zd, renormalize=True)

    nu = nu_standard if args.nu == "nu_standard" else nu_simple

    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho, a0=args.a0, nu_func=nu, pad_factor=args.pad)

    sampler = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)
    return grid, sampler


def run_orbit(dt: float, steps: int, r0: float, sampler: StaticFieldSampler) -> Dict[str, Any]:
    # init on x-axis
    x0, y0 = r0, 0.0
    pos0 = np.array([[x0, y0, 0.0]], dtype=np.float64)
    a0 = sampler.accel_at(pos0)[0]
    g0 = radial_accel_in_midplane(a0[0], a0[1], x0, y0)
    v0 = float(np.sqrt(max(r0 * g0, 0.0)))

    st = OrbitState(x=x0, y=y0, vx=0.0, vy=v0, t=0.0)

    def accel_fn(x: float, y: float) -> Tuple[float, float]:
        p = np.array([[x, y, 0.0]], dtype=np.float64)
        a = sampler.accel_at(p)[0]
        return float(a[0]), float(a[1])

    history: List[Dict[str, Any]] = []

    # energy per unit mass: E = 0.5 v^2 + Phi
    for _ in range(steps):
        p = np.array([[st.x, st.y, 0.0]], dtype=np.float64)
        phi = float(sampler.phi_at(p)[0])
        v2 = st.vx * st.vx + st.vy * st.vy
        E = 0.5 * v2 + phi
        r = float(np.hypot(st.x, st.y))

        history.append({"t": st.t, "x": st.x, "y": st.y, "r": r, "vx": st.vx, "vy": st.vy, "phi": phi, "E": E})
        st = kdk_step(st, dt, accel_fn)

    E0 = history[0]["E"]
    E1 = history[-1]["E"]
    rel_drift = float(abs((E1 - E0) / (E0 if E0 != 0 else 1.0)))

    return {
        "dt": dt,
        "steps": steps,
        "r0": r0,
        "v0": v0,
        "rel_drift": rel_drift,
        "history": history,
    }


# ============================================================
# Modes
# ============================================================

def mode_smoke(out: OutputManager, args) -> None:
    print("\n>>> [QUMOND Audit] Smoke test")
    _, sampler = build_static_field(args)
    res = run_orbit(dt=args.dt, steps=args.steps_smoke, r0=args.r0, sampler=sampler)

    out.save_csv("audit_smoke.csv", res["history"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    t = [row["t"] for row in res["history"]]
    E = [row["E"] for row in res["history"]]
    ax.plot(t, E)
    ax.set_title(f"Smoke: rel. drift = {res['rel_drift']:.3%}")
    ax.set_xlabel("t")
    ax.set_ylabel("E")
    ax.grid(True, alpha=0.3)
    out.save_fig("audit_smoke.png", fig)
    plt.close(fig)

    print(f"   drift={res['rel_drift']:.4%}")


def mode_fiducial(out: OutputManager, args) -> None:
    print("\n>>> [QUMOND Audit] Fiducial run")
    _, sampler = build_static_field(args)
    res = run_orbit(dt=args.dt, steps=args.steps_fid, r0=args.r0, sampler=sampler)

    out.save_csv("audit_fiducial.csv", res["history"])

    import matplotlib.pyplot as plt

    t = np.array([row["t"] for row in res["history"]])
    r = np.array([row["r"] for row in res["history"]])
    E = np.array([row["E"] for row in res["history"]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.plot(t, r)
    ax1.set_title("Orbital radius")
    ax1.set_xlabel("t")
    ax1.set_ylabel("r")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, E)
    ax2.set_title(f"Energy (rel drift={res['rel_drift']:.3%})")
    ax2.set_xlabel("t")
    ax2.set_ylabel("E")
    ax2.grid(True, alpha=0.3)

    out.save_fig("audit_fiducial.png", fig)
    plt.close(fig)

    print(f"   drift={res['rel_drift']:.4%}")


def mode_sensitivity(out: OutputManager, args) -> None:
    print("\n>>> [QUMOND Audit] Sensitivity: drift vs dt")
    _, sampler = build_static_field(args)

    dt_list = args.dt_list
    rows = []

    for dt in dt_list:
        steps = int(args.T_total / dt)
        res = run_orbit(dt=float(dt), steps=steps, r0=args.r0, sampler=sampler)
        rows.append({"dt": float(dt), "steps": int(steps), "rel_drift": res["rel_drift"]})
        print(f"   dt={dt:>8g}  steps={steps:>6d}  drift={res['rel_drift']:.4%}")

    out.save_csv("audit_sensitivity.csv", rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot([r["dt"] for r in rows], [r["rel_drift"] for r in rows], "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("dt")
    ax.set_ylabel("relative energy drift")
    ax.set_title("QUMOND audit: drift vs dt")
    ax.grid(True, which="both", alpha=0.3)
    out.save_fig("audit_sensitivity.png", fig)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ISUT QUMOND Audit Suite")

    p.add_argument("--mode", default="all", choices=["all", "smoke", "fiducial", "sensitivity"], help="audit mode")
    p.add_argument("--headless", action="store_true", help="do not call plt.show()")

    # Field parameters
    p.add_argument("--N", type=int, default=96, help="grid resolution")
    p.add_argument("--L", type=float, default=200.0, help="box size")
    p.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="zero-padding factor")
    p.add_argument("--G", type=float, default=1.0, help="gravitational constant")
    p.add_argument("--a0", type=float, default=0.12, help="acceleration scale")
    p.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"], help="nu(y) choice")

    # Disk model
    p.add_argument("--M", type=float, default=1000.0, help="disk total mass")
    p.add_argument("--Rd", type=float, default=3.0, help="disk radial scale")
    p.add_argument("--zd", type=float, default=0.4, help="disk vertical scale")

    # Orbit/integration
    p.add_argument("--r0", type=float, default=10.0, help="initial orbit radius")
    p.add_argument("--dt", type=float, default=0.05, help="time step")
    p.add_argument("--steps_smoke", type=int, default=300, help="smoke steps")
    p.add_argument("--steps_fid", type=int, default=3000, help="fiducial steps")

    p.add_argument("--T_total", type=float, default=50.0, help="total time for sensitivity runs")
    p.add_argument("--dt_list", type=float, nargs="+", default=[0.01, 0.02, 0.05, 0.1, 0.2], help="dt grid")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Output directory mirrors naming convention in repo
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    target_dir = os.path.join(current_dir, script_name)

    out = OutputManager(target_dir, prefix="audit_qumond")

    out.save_json(
        "00_meta.json",
        {
            "args": vars(args),
            "env": out.env(),
            "notes": {
                "boundary": "FFT periodic; padding approximates isolated BC.",
                "energy": "E=0.5 v^2 + Phi sampled from solved potential.",
            },
        },
    )

    if args.mode in ("all", "smoke"):
        mode_smoke(out, args)

    if args.mode in ("all", "fiducial"):
        mode_fiducial(out, args)

    if args.mode in ("all", "sensitivity"):
        mode_sensitivity(out, args)

    out.save_json("99_complete.json", {"status": "SUCCESS"})

    print("\n" + "=" * 72)
    print("QUMOND audit complete")
    print(f"Artifacts in: {out.run_dir}")
    print("=" * 72)

    if not args.headless:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
