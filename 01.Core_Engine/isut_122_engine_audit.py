# -*- coding: utf-8 -*-
"""ISUT Engine Audit (Numerical Validation Suite)
===================================================

File: 2.isut_engine_audit.py

This script generates reviewer-ready artifacts (CSV/PNG/JSON) for numerical
stability checks.

Outputs
-------
Artifacts are saved under:

  ./2.isut_engine_audit/runs/audit_<timestamp>/{figures,data,logs}

Kernels
-------
This audit supports two acceleration kernels:

  - algebraic (legacy):  a = g_N * nu(g_N/a0)
  - qumond (conservative): solve two Poisson equations on a particle--mesh grid
      1) ∇²Φ_N = 4πGρ
      2) ∇²Φ   = -∇·[ν(|g_N|/a0) g_N], with g_N ≡ -∇Φ_N
     and return a = -∇Φ.

Notes
-----
- The QUMOND kernel provides an explicit potential-based (conservative) field
  completion. The energy-drift metrics reported here should be interpreted as
  numerical stability diagnostics of the discretized solver + integrator
  (mesh resolution, CIC interpolation, timestep control), not as a stand-alone
  "proof" of conservation.

Usage
-----
  python 2.isut_engine_audit.py --mode all --kernel qumond

  # Legacy algebraic check
  python 2.isut_engine_audit.py --mode all --kernel algebraic
"""

from __future__ import annotations

import argparse
import sys
import os
import time
import json
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any, List

import numpy as np

try:
    import pandas as pd
except Exception:
    pd = None

# Use a non-interactive backend by default so this runs on CI/servers.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# [Configuration] Output Directory Setup (Relative Path)
# ============================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
TARGET_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)


# ============================================================
# [Optional] QUMOND backend
# ============================================================
QUMOND_AVAILABLE = False
QUMOND_IMPORT_ERROR: Optional[str] = None

try:
    # Local module shipped with the repo
    from isut_100_qumond_pm import (
        PMGrid3D,
        QUMONDSolverFFT,
        StaticFieldSampler,
        exponential_disk_density,
        nu_simple,
        nu_standard,
    )
    QUMOND_AVAILABLE = True
except Exception as e:
    QUMOND_AVAILABLE = False
    QUMOND_IMPORT_ERROR = f"{type(e).__name__}: {e}"


# ============================================================
# [Output Manager] Reproducibility & Artifacts
# ============================================================
class OutputManager:
    """Structured output helper.

    Creates:
      <TARGET_DIR>/runs/<timestamp>/{figures, data, logs}
    """

    def __init__(self, base_dir: str):
        self.out_root = Path(base_dir)
        self.out_root.mkdir(parents=True, exist_ok=True)

        self.run_id = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.run_dir = self.out_root / "runs" / f"audit_{self.run_id}"

        self.fig_dir = self.run_dir / "figures"
        self.data_dir = self.run_dir / "data"
        self.log_dir = self.run_dir / "logs"

        for d in [self.fig_dir, self.data_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)

        print(f"[Info] Audit Run Directory Created: {self.run_dir}")

    def env(self) -> Dict[str, str]:
        return {
            "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "numpy": getattr(np, "__version__", "unknown"),
            "matplotlib": getattr(plt, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "missing") if pd else "missing",
        }

    def save_json(self, name: str, obj: dict) -> Path:
        p = self.log_dir / name
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        print(f"   L Log saved: {p.name}")
        return p

    def save_csv(self, name: str, rows_or_df) -> Path:
        p = self.data_dir / name
        if pd is not None:
            if isinstance(rows_or_df, pd.DataFrame):
                rows_or_df.to_csv(p, index=False)
            else:
                pd.DataFrame(rows_or_df).to_csv(p, index=False)
        else:
            # Fallback: write a simple CSV without pandas
            import csv

            if isinstance(rows_or_df, list) and rows_or_df:
                keys = list(rows_or_df[0].keys())
                with open(p, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=keys)
                    w.writeheader()
                    w.writerows(rows_or_df)
            else:
                with open(p, "w", encoding="utf-8") as f:
                    f.write(str(rows_or_df))

        print(f"   L Data saved: {p.name}")
        return p

    def save_fig(self, fig, name: str, dpi: int = 150) -> Path:
        p = self.fig_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        print(f"   L Figure saved: {p.name}")
        return p


# ============================================================
# [Core Physics] Units & Parameters (mirrored from engine)
# ============================================================
@dataclass(frozen=True)
class Units:
    G: float = 1.0
    c_light: float = 500.0


@dataclass
class GalaxyParams:
    """Disk mass model parameters (used by both kernels).

    For the QUMOND kernel, we build an analytic 3D exponential disk density:

      ρ(R,z) ∝ exp(-R/R_d) exp(-|z|/z_d)

    with R_d=r_scale and z_d=z_scale.
    """

    M_total: float = 1000.0
    r_scale: float = 3.0
    z_scale: float = 0.30
    r_soft: float = 0.1


@dataclass
class ISUTParams:
    a0: float = 0.12
    accel_form: str = "nu_simple"  # legacy default
    use_relational_time: bool = True
    time_dilation_scale: float = 100.0
    dt_base: float = 0.05


# ============================================================
# [Legacy Algebraic Kernel]
# ============================================================
class BaryonicGalaxy:
    def __init__(self, units: Units, params: GalaxyParams):
        self.u = units
        self.p = params

    def mass_enclosed(self, r: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, self.p.r_soft)
        h = self.p.r_scale
        return self.p.M_total * (1.0 - (1.0 + r_safe / h) * np.exp(-r_safe / h))

    def newtonian_accel(self, r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, self.p.r_soft)
        return (self.u.G * M_enc) / (r_safe**2)

    def potential_depth(self, r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, self.p.r_soft)
        return (self.u.G * M_enc) / r_safe


class ISUTGravity:
    def __init__(self, params: ISUTParams):
        self.p = params

    @staticmethod
    def nu_simple(y: np.ndarray) -> np.ndarray:
        y_safe = np.maximum(y, 1e-12)
        return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y_safe))

    def accel_isut(self, a_newton: np.ndarray) -> np.ndarray:
        y = a_newton / np.maximum(self.p.a0, 1e-12)
        if self.p.accel_form == "nu_simple":
            return a_newton * self.nu_simple(y)
        return a_newton

    @staticmethod
    def v_circular(r: np.ndarray, a_total: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, 1e-12)
        return np.sqrt(r_safe * np.maximum(a_total, 0.0))


class RelationalTime:
    def __init__(self, units: Units, params: ISUTParams):
        self.u = units
        self.p = params

    def gamma(self, potential_depth: np.ndarray) -> np.ndarray:
        return 1.0 + (potential_depth / (self.u.c_light**2)) * self.p.time_dilation_scale

    def dt_local(self, dt_base: float, potential_depth: np.ndarray) -> np.ndarray:
        g = self.gamma(potential_depth)
        return dt_base / np.maximum(g, 1e-12)


@dataclass
class Particles2D:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    t: float = 0.0


class LeapfrogKDK:
    """Legacy KDK integrator for the algebraic kernel."""

    def __init__(self, galaxy: BaryonicGalaxy, gravity: ISUTGravity, rtime: Optional[RelationalTime] = None):
        self.galaxy = galaxy
        self.gravity = gravity
        self.rtime = rtime

    def _accel_xy(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        r = np.sqrt(x**2 + y**2)
        r_safe = np.maximum(r, self.galaxy.p.r_soft)
        M = self.galaxy.mass_enclosed(r)
        aN = self.galaxy.newtonian_accel(r, M)
        a = self.gravity.accel_isut(aN)
        ax = -a * (x / r_safe)
        ay = -a * (y / r_safe)
        phi_depth = self.galaxy.potential_depth(r, M)
        return ax, ay, r, phi_depth

    def step(self, P: Particles2D, dt_base: float) -> Particles2D:
        ax, ay, r, phi_depth = self._accel_xy(P.x, P.y)
        dt1 = self.rtime.dt_local(dt_base, phi_depth) if self.rtime else dt_base * np.ones_like(r)

        P.vx += ax * dt1 * 0.5
        P.vy += ay * dt1 * 0.5

        P.x += P.vx * dt1
        P.y += P.vy * dt1

        ax2, ay2, r2, phi_depth2 = self._accel_xy(P.x, P.y)
        dt2 = self.rtime.dt_local(dt_base, phi_depth2) if self.rtime else dt_base * np.ones_like(r2)

        P.vx += ax2 * dt2 * 0.5
        P.vy += ay2 * dt2 * 0.5

        P.t += dt_base
        return P


# ============================================================
# [QUMOND Kernel] Field + Integrator
# ============================================================
@dataclass
class QUMONDField:
    grid: "PMGrid3D"
    sampler: "StaticFieldSampler"
    nu_name: str
    pad_factor: int


def build_qumond_field(
    units: Units,
    gp: GalaxyParams,
    a0: float,
    Ngrid: int,
    boxsize: float,
    pad_factor: int,
    nu_name: Literal["nu_simple", "nu_standard"],
) -> QUMONDField:
    if not QUMOND_AVAILABLE:
        raise RuntimeError(f"QUMOND backend not available: {QUMOND_IMPORT_ERROR}")

    grid = PMGrid3D(N=int(Ngrid), boxsize=float(boxsize), G=float(units.G))

    rho = exponential_disk_density(
        grid=grid,
        M_total=float(gp.M_total),
        R_d=float(gp.r_scale),
        z_d=float(gp.z_scale),
        renormalize=True,
    )

    solver = QUMONDSolverFFT(grid)

    nu_func = nu_simple if nu_name == "nu_simple" else nu_standard
    res = solver.solve(rho=rho, a0=float(a0), nu_func=nu_func, pad_factor=int(pad_factor))

    sampler = StaticFieldSampler(grid=grid, phi=res.phi, accel=res.accel)
    return QUMONDField(grid=grid, sampler=sampler, nu_name=nu_name, pad_factor=int(pad_factor))


class LeapfrogKDK_Field2D:
    """KDK integrator using a static external potential/acceleration field.

    The field is sampled via CIC from a mesh.
    """

    def __init__(self, field: QUMONDField, rtime: Optional[RelationalTime] = None):
        self.field = field
        self.rtime = rtime

    def _accel_phi_xy(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Embed into z=0 midplane
        pos = np.stack([x, y, np.zeros_like(x)], axis=-1)
        a = self.field.sampler.accel_at(pos)  # (n,3)
        phi = self.field.sampler.phi_at(pos)  # (n,)
        return a[:, 0], a[:, 1], phi

    def step(self, P: Particles2D, dt_base: float) -> Particles2D:
        ax, ay, phi = self._accel_phi_xy(P.x, P.y)
        r = np.sqrt(P.x**2 + P.y**2)

        # For relational time scaling, use a positive potential depth proxy.
        phi_depth = np.abs(phi)
        dt1 = self.rtime.dt_local(dt_base, phi_depth) if self.rtime else dt_base * np.ones_like(r)

        P.vx += ax * dt1 * 0.5
        P.vy += ay * dt1 * 0.5

        P.x += P.vx * dt1
        P.y += P.vy * dt1

        ax2, ay2, phi2 = self._accel_phi_xy(P.x, P.y)
        phi_depth2 = np.abs(phi2)
        dt2 = self.rtime.dt_local(dt_base, phi_depth2) if self.rtime else dt_base * np.ones_like(r)

        P.vx += ax2 * dt2 * 0.5
        P.vy += ay2 * dt2 * 0.5

        P.t += dt_base
        return P


# ============================================================
# [Audit Logic]
# ============================================================

def init_one_particle_algebraic(r0: float, galaxy: BaryonicGalaxy, gravity: ISUTGravity) -> Particles2D:
    r = np.array([float(r0)])
    M = galaxy.mass_enclosed(r)
    aN = galaxy.newtonian_accel(r, M)
    a = gravity.accel_isut(aN)
    v = gravity.v_circular(r, a)

    x = r
    y = np.zeros_like(r)
    vx = np.zeros_like(r)
    vy = v
    return Particles2D(x=x, y=y, vx=vx, vy=vy, t=0.0)


def init_one_particle_qumond(r0: float, field: QUMONDField) -> Particles2D:
    r = float(r0)
    x = np.array([r], dtype=np.float64)
    y = np.array([0.0], dtype=np.float64)
    pos = np.array([[r, 0.0, 0.0]], dtype=np.float64)

    a = field.sampler.accel_at(pos)[0]

    # Radial inward acceleration magnitude
    # At (r,0), inward is -x.
    a_rad = max(0.0, -float(a[0]))
    v = np.sqrt(max(0.0, r * a_rad))

    vx = np.array([0.0], dtype=np.float64)
    vy = np.array([v], dtype=np.float64)
    return Particles2D(x=x, y=y, vx=vx, vy=vy, t=0.0)


def run_energy_audit_algebraic(
    dt_val: float,
    steps: int,
    galaxy: BaryonicGalaxy,
    gravity: ISUTGravity,
    rtime: Optional[RelationalTime],
    r0: float = 10.0,
) -> Dict[str, Any]:

    P = init_one_particle_algebraic(r0, galaxy, gravity)
    integrator = LeapfrogKDK(galaxy, gravity, rtime)

    history: List[Dict[str, Any]] = []

    for _ in range(int(steps)):
        r = np.sqrt(P.x**2 + P.y**2)
        v2 = P.vx**2 + P.vy**2
        M_enc = galaxy.mass_enclosed(r)

        # Potential proxy: -GM/r (legacy)
        V_pot = -galaxy.u.G * M_enc / np.maximum(r, galaxy.p.r_soft)
        T_kin = 0.5 * v2
        E_tot = T_kin + V_pot

        history.append({
            "t": float(P.t),
            "r": float(r[0]),
            "E": float(E_tot[0]),
            "T": float(T_kin[0]),
            "V": float(V_pot[0]),
        })

        P = integrator.step(P, float(dt_val))

    E0 = history[0]["E"]
    E_final = history[-1]["E"]
    rel_drift = abs((E_final - E0) / (E0 if E0 != 0 else 1.0))

    return {
        "kernel": "algebraic",
        "dt": float(dt_val),
        "steps": int(steps),
        "history": history,
        "rel_drift": float(rel_drift),
        "passed": bool(rel_drift < 0.01),
    }


def run_energy_audit_qumond(
    dt_val: float,
    steps: int,
    field: QUMONDField,
    rtime: Optional[RelationalTime],
    r0: float = 10.0,
) -> Dict[str, Any]:

    P = init_one_particle_qumond(r0, field)
    integrator = LeapfrogKDK_Field2D(field, rtime)

    history: List[Dict[str, Any]] = []

    for _ in range(int(steps)):
        pos = np.array([[float(P.x[0]), float(P.y[0]), 0.0]], dtype=np.float64)
        phi = float(field.sampler.phi_at(pos)[0])

        v2 = float(P.vx[0] ** 2 + P.vy[0] ** 2)
        T = 0.5 * v2
        V = phi
        E = T + V

        r = float(np.sqrt(P.x[0] ** 2 + P.y[0] ** 2))
        history.append({
            "t": float(P.t),
            "r": r,
            "E": float(E),
            "T": float(T),
            "V": float(V),
        })

        P = integrator.step(P, float(dt_val))

    E0 = history[0]["E"]
    E_final = history[-1]["E"]
    rel_drift = abs((E_final - E0) / (E0 if E0 != 0 else 1.0))

    return {
        "kernel": "qumond",
        "nu": field.nu_name,
        "pad_factor": field.pad_factor,
        "dt": float(dt_val),
        "steps": int(steps),
        "history": history,
        "rel_drift": float(rel_drift),
        "passed": bool(rel_drift < 0.01),
    }


# ============================================================
# [Modes]
# ============================================================

def _df(rows: List[Dict[str, Any]]):
    if pd is None:
        return rows
    return pd.DataFrame(rows)


def mode_smoke(out: OutputManager, args, field: Optional[QUMONDField]):
    print("\n>>> [Audit] Smoke Test: quick functionality check...")

    units = Units()
    gp = GalaxyParams(M_total=args.M_total, r_scale=args.Rd, z_scale=args.zd)
    ip = ISUTParams(a0=args.a0, use_relational_time=False)

    if args.kernel == "qumond":
        if field is None:
            field = build_qumond_field(units, gp, a0=args.a0, Ngrid=args.Ngrid, boxsize=args.box, pad_factor=args.pad, nu_name=args.nu)
        res = run_energy_audit_qumond(dt_val=args.dt, steps=200, field=field, rtime=None, r0=args.r0)
    else:
        galaxy = BaryonicGalaxy(units, gp)
        gravity = ISUTGravity(ip)
        res = run_energy_audit_algebraic(dt_val=args.dt, steps=200, galaxy=galaxy, gravity=gravity, rtime=None, r0=args.r0)

    df = _df(res["history"])
    out.save_csv("audit_smoke.csv", df)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([h["t"] for h in res["history"]], [h["E"] for h in res["history"]], label="Total Energy")
    ax.set_title(f"Smoke Test ({res['kernel']})  Drift: {res['rel_drift']:.2%}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend()
    out.save_fig(fig, "audit_smoke.png")
    plt.close(fig)

    print(f"   > Drift: {res['rel_drift']:.2%} | Passed: {res['passed']}")


def mode_fiducial(out: OutputManager, args, field: Optional[QUMONDField]):
    print("\n>>> [Audit] Fiducial Test: standard parameters...")

    units = Units()
    gp = GalaxyParams(M_total=args.M_total, r_scale=args.Rd, z_scale=args.zd)
    ip = ISUTParams(a0=args.a0, use_relational_time=True)

    rtime = RelationalTime(units, ip) if ip.use_relational_time else None

    if args.kernel == "qumond":
        if field is None:
            field = build_qumond_field(units, gp, a0=args.a0, Ngrid=args.Ngrid, boxsize=args.box, pad_factor=args.pad, nu_name=args.nu)
        res = run_energy_audit_qumond(dt_val=args.dt, steps=args.steps, field=field, rtime=rtime, r0=args.r0)
    else:
        galaxy = BaryonicGalaxy(units, gp)
        gravity = ISUTGravity(ip)
        res = run_energy_audit_algebraic(dt_val=args.dt, steps=args.steps, galaxy=galaxy, gravity=gravity, rtime=rtime, r0=args.r0)

    df = _df(res["history"])
    out.save_csv("audit_fiducial.csv", df)

    # Plot radius + energy
    t = [h["t"] for h in res["history"]]
    r = [h["r"] for h in res["history"]]
    E = [h["E"] for h in res["history"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(t, r)
    ax1.set_title("Orbital Stability (radius)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("r")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t, E)
    ax2.set_title(f"Energy Drift ({res['kernel']}): {res['rel_drift']:.4%}")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("E")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out.save_fig(fig, "audit_fiducial.png")
    plt.close(fig)

    print(f"   > Drift: {res['rel_drift']:.4%} | Passed: {res['passed']}")


def mode_sensitivity(out: OutputManager, args, field: Optional[QUMONDField]):
    print("\n>>> [Audit] Sensitivity: drift vs timestep...")

    units = Units()
    gp = GalaxyParams(M_total=args.M_total, r_scale=args.Rd, z_scale=args.zd)
    ip = ISUTParams(a0=args.a0, use_relational_time=False)

    if args.kernel == "qumond" and field is None:
        field = build_qumond_field(units, gp, a0=args.a0, Ngrid=args.Ngrid, boxsize=args.box, pad_factor=args.pad, nu_name=args.nu)

    dt_list = [0.01, 0.02, 0.05, 0.1, 0.2]
    results: List[Dict[str, Any]] = []

    for dt in dt_list:
        # Keep total simulation time ~ constant
        total_time = 20.0
        steps = int(total_time / dt)

        if args.kernel == "qumond":
            res = run_energy_audit_qumond(dt_val=dt, steps=steps, field=field, rtime=None, r0=args.r0)
        else:
            galaxy = BaryonicGalaxy(units, gp)
            gravity = ISUTGravity(ip)
            res = run_energy_audit_algebraic(dt_val=dt, steps=steps, galaxy=galaxy, gravity=gravity, rtime=None, r0=args.r0)

        results.append({"dt": dt, "drift": res["rel_drift"], "passed": res["passed"]})
        print(f"   dt={dt}: drift={res['rel_drift']:.4%}")

    df = _df(results)
    out.save_csv("audit_sensitivity.csv", df)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([r["dt"] for r in results], [r["drift"] for r in results], "o-")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(f"Energy drift vs dt ({args.kernel})")
    ax.set_xlabel("dt")
    ax.set_ylabel("relative drift")
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0.01, color="k", linestyle="--", label="1% threshold")
    ax.legend()

    out.save_fig(fig, "audit_sensitivity.png")
    plt.close(fig)


# ============================================================
# [Main]
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ISUT Engine Audit")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "smoke", "fiducial", "sensitivity"], help="Audit mode")
    parser.add_argument("--kernel", type=str, default="qumond", choices=["qumond", "algebraic"], help="Acceleration kernel")

    # Shared physical params
    parser.add_argument("--a0", type=float, default=0.12, help="Acceleration scale a0 (code units)")
    parser.add_argument("--M-total", dest="M_total", type=float, default=1000.0, help="Disk mass (code units)")
    parser.add_argument("--Rd", type=float, default=3.0, help="Disk radial scale length")
    parser.add_argument("--zd", type=float, default=0.30, help="Disk vertical scale length")

    # Orbit / integration params
    parser.add_argument("--r0", type=float, default=10.0, help="Initial orbit radius")
    parser.add_argument("--dt", type=float, default=0.05, help="Base dt")
    parser.add_argument("--steps", type=int, default=1000, help="Steps for fiducial")

    # QUMOND mesh params
    parser.add_argument("--Ngrid", type=int, default=64, help="PM grid resolution (per axis)")
    parser.add_argument("--box", type=float, default=200.0, help="PM box size")
    parser.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="Padding factor for approximate isolation")
    parser.add_argument("--nu", type=str, default="nu_standard", choices=["nu_simple", "nu_standard"], help="nu(y) choice")

    args = parser.parse_args()

    if args.kernel == "qumond" and not QUMOND_AVAILABLE:
        print("[ERROR] QUMOND backend not available.")
        print(f"        Import error: {QUMOND_IMPORT_ERROR}")
        sys.exit(2)

    out = OutputManager(TARGET_DIR)

    # Save run metadata first
    meta = {
        "mode": args.mode,
        "kernel": args.kernel,
        "a0": args.a0,
        "disk": {"M_total": args.M_total, "Rd": args.Rd, "zd": args.zd},
        "orbit": {"r0": args.r0, "dt": args.dt, "steps": args.steps},
        "qumond": {"Ngrid": args.Ngrid, "box": args.box, "pad": args.pad, "nu": args.nu},
        "env": out.env(),
    }
    out.save_json("00_audit_meta.json", meta)

    field: Optional[QUMONDField] = None
    if args.kernel == "qumond":
        # Build field once for the run (shared across modes)
        units = Units()
        gp = GalaxyParams(M_total=args.M_total, r_scale=args.Rd, z_scale=args.zd)
        print("[QUMOND] Building static field (2 Poisson solves)...")
        field = build_qumond_field(units, gp, a0=args.a0, Ngrid=args.Ngrid, boxsize=args.box, pad_factor=args.pad, nu_name=args.nu)
        out.save_json(
            "01_qumond_field.json",
            {
                "grid": {"N": field.grid.N, "boxsize": field.grid.boxsize, "dx": field.grid.dx, "G": field.grid.G},
                "nu": field.nu_name,
                "pad_factor": field.pad_factor,
            },
        )

    print("=" * 60)
    print("ISUT Engine Audit: Numerical Validation Suite")
    print(f"   Kernel: {args.kernel}")
    print(f"   Target: {out.run_dir}")
    print("=" * 60)

    if args.mode == "all":
        mode_smoke(out, args, field)
        mode_fiducial(out, args, field)
        mode_sensitivity(out, args, field)
    elif args.mode == "smoke":
        mode_smoke(out, args, field)
    elif args.mode == "fiducial":
        mode_fiducial(out, args, field)
    elif args.mode == "sensitivity":
        mode_sensitivity(out, args, field)

    out.save_json("99_audit_complete.json", {"status": "SUCCESS"})
    print("\n" + "=" * 60)
    print("Audit Complete. Artifacts generated in:")
    print(f"   {out.run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
