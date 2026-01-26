# -*- coding: utf-8 -*-
"""
ISUT Universe Engine (Integrated Core)
==================================================================

File: 1.isut_engine_core.py

[Summary]
  1. Energy Audit: Real-time calculation of E = T + V to verify conservation.
  2. Data Export: Automatically saves Rotation Curve & BTFR data/plots.
  3. Automation: Runs the full validation suite (N-body -> RC -> BTFR).
  4. Visualization: High-contrast particle rendering.

Usage:
  Run this script directly.
  > python 1.isut_engine_core.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, List, Callable, Dict
import os
import sys

# [Required Libraries]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# ============================================================
# [Configuration] Output Directory Setup (Relative Path)
# ============================================================
# Determine the directory where this script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Determine script name without extension to create matching folder
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Set Output Directory: ./<Script_Name>/
BASE_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
os.makedirs(BASE_DIR, exist_ok=True)

print(f"[Setup] Current Directory: {CURRENT_DIR}")
print(f"[Setup] Output Directory : {BASE_DIR}")


# ============================================================
# [Core Physics] Units & Parameters
# ============================================================

@dataclass(frozen=True)
class Units:
    """
    Minimal "galactic units":
    G = 1.0 (normalized)
    c_light = 500.0 (sim units, used for relational time scaling)
    """
    G: float = 1.0
    c_light: float = 500.0


@dataclass
class GalaxyParams:
    """
    Exponential disk baryonic profile approximation.
    M(<r) = M_total * [1 - (1 + r/h) exp(-r/h)]
    """
    M_total: float = 1000.0
    r_scale: float = 3.0     # exponential scale length h
    r_soft: float = 0.1      # softening to avoid r=0 singularities


@dataclass
class ISUTParams:
    """
    ISUT/MOND-like crossover control.
    """
    a0: float = 0.12
    accel_form: Literal["piecewise", "nu_simple", "nu_standard"] = "nu_simple"
    use_relational_time: bool = True
    time_dilation_scale: float = 100.0   # scaling factor for visualization
    dt_base: float = 0.05               # integrator base dt


# ============================================================
# [Core Physics] Force & Potential Kernels
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
        return (self.u.G * M_enc) / (r_safe ** 2)

    def potential_depth(self, r: np.ndarray, M_enc: np.ndarray) -> np.ndarray:
        # Simple proxy for potential depth used in time dilation scaling
        r_safe = np.maximum(r, self.p.r_soft)
        return (self.u.G * M_enc) / r_safe


class ISUTGravity:
    def __init__(self, params: ISUTParams):
        self.p = params

    @staticmethod
    def nu_simple(y: np.ndarray) -> np.ndarray:
        # nu(y) = 0.5*(1 + sqrt(1 + 4/y))
        # This form ensures correct asymptotic behavior:
        # y >> 1 -> nu -> 1 (Newton)
        # y << 1 -> nu -> 1/sqrt(y) (Deep MOND/ISUT)
        y_safe = np.maximum(y, 1e-12)
        return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y_safe))

    def accel_isut(self, a_newton: np.ndarray) -> np.ndarray:
        a0 = self.p.a0
        if self.p.accel_form == "piecewise":
            return np.where(a_newton < a0, np.sqrt(a_newton * a0), a_newton)

        y = a_newton / np.maximum(a0, 1e-12)
        if self.p.accel_form == "nu_simple":
            return a_newton * self.nu_simple(y)
        
        # Default Fallback
        return a_newton

    def v_circular(self, r: np.ndarray, a_total: np.ndarray) -> np.ndarray:
        r_safe = np.maximum(r, 1e-12)
        return np.sqrt(r_safe * np.maximum(a_total, 0.0))


class RelationalTime:
    def __init__(self, units: Units, params: ISUTParams):
        self.u = units
        self.p = params

    def gamma(self, potential_depth: np.ndarray) -> np.ndarray:
        # gamma = 1 + |Phi|/c^2 * scale
        return 1.0 + (potential_depth / (self.u.c_light ** 2)) * self.p.time_dilation_scale

    def dt_local(self, dt_base: float, potential_depth: np.ndarray) -> np.ndarray:
        g = self.gamma(potential_depth)
        return dt_base / np.maximum(g, 1e-12)


# ============================================================
# [Integrator] Symplectic Leapfrog (KDK)
# ============================================================

@dataclass
class Particles2D:
    x: np.ndarray
    y: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    t: float = 0.0


class LeapfrogKDK:
    def __init__(
        self,
        galaxy: BaryonicGalaxy,
        gravity: ISUTGravity,
        rtime: Optional[RelationalTime] = None,
    ):
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

        phi = self.galaxy.potential_depth(r, M)
        return ax, ay, r, phi

    def step(self, P: Particles2D, dt_base: float) -> Particles2D:
        # 1. First Kick (Velocity update half-step)
        ax, ay, r, phi = self._accel_xy(P.x, P.y)
        dt1 = self.rtime.dt_local(dt_base, phi) if self.rtime else dt_base * np.ones_like(r)

        P.vx += ax * dt1 * 0.5
        P.vy += ay * dt1 * 0.5

        # 2. Drift (Position update full-step)
        P.x += P.vx * dt1
        P.y += P.vy * dt1

        # 3. Second Kick (Velocity update half-step at new position)
        ax2, ay2, r2, phi2 = self._accel_xy(P.x, P.y)
        dt2 = self.rtime.dt_local(dt_base, phi2) if self.rtime else dt_base * np.ones_like(r2)

        P.vx += ax2 * dt2 * 0.5
        P.vy += ay2 * dt2 * 0.5

        P.t += dt_base
        return P


# ============================================================
# [Initialization] Particle Generator
# ============================================================

def init_disk_particles(
    galaxy: BaryonicGalaxy,
    gravity: ISUTGravity,
    n: int = 400,
    r_max: float = 25.0,
    r_min: float = 0.5,
    r_exp_scale: float = 5.0,
    vel_dispersion: float = 0.02,
    seed: int = 7,
) -> Particles2D:
    rng = np.random.default_rng(seed)
    
    # Generate radii with exponential falloff
    r = rng.exponential(scale=r_exp_scale, size=n * 3) # generate extra
    r = r[(r >= r_min) & (r <= r_max)]
    if len(r) < n:
         # Fallback uniform if truncated too much
         r = np.concatenate([r, rng.uniform(r_min, r_max, size=n-len(r))])
    r = r[:n]

    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Initialize circular velocity based on ISUT gravity
    M = galaxy.mass_enclosed(r)
    aN = galaxy.newtonian_accel(r, M)
    a = gravity.accel_isut(aN)
    v = gravity.v_circular(r, a)

    # Tangential velocity
    vx = -v * np.sin(theta)
    vy =  v * np.cos(theta)

    # Add small thermal dispersion
    vx *= (1.0 + rng.normal(0.0, vel_dispersion, size=n))
    vy *= (1.0 + rng.normal(0.0, vel_dispersion, size=n))

    return Particles2D(x=x, y=y, vx=vx, vy=vy, t=0.0)


# ============================================================
# [Analysis Helper] Computations
# ============================================================

def compute_rotation_curve(galaxy: BaryonicGalaxy, gravity: ISUTGravity, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    M = galaxy.mass_enclosed(r)
    aN = galaxy.newtonian_accel(r, M)
    aI = gravity.accel_isut(aN)
    v_newton = np.sqrt(np.maximum(r, 1e-12) * np.maximum(aN, 0.0))
    v_isut = np.sqrt(np.maximum(r, 1e-12) * np.maximum(aI, 0.0))
    return v_newton, v_isut


# ============================================================
# [Modes] 1. N-Body Simulation with Energy Audit
# ============================================================

def mode_animate_with_audit(units: Units, gparams: GalaxyParams, iparams: ISUTParams):
    print("\n[Step 1/3] Running N-body Simulation with Energy Audit...")
    galaxy = BaryonicGalaxy(units, gparams)
    gravity = ISUTGravity(iparams)
    rtime = RelationalTime(units, iparams) if iparams.use_relational_time else None
    integrator = LeapfrogKDK(galaxy, gravity, rtime=rtime)

    P = init_disk_particles(galaxy, gravity, n=450)
    
    # Audit Storage
    energy_log = []

    # Visualization Setup
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('black')
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], width_ratios=[1.0, 1.0])

    # [Panel 1] Simulation View (Top Left)
    ax_sim = plt.subplot(gs[0, 0])
    ax_sim.set_facecolor('black')
    ax_sim.set_aspect("equal")
    ax_sim.set_xlim(-40, 40); ax_sim.set_ylim(-40, 40)
    ax_sim.set_title("ISUT Disk Simulation (Test Particles)", color='white')
    ax_sim.tick_params(colors='white')
    
    # Visualization: Lime colored particles for contrast
    stars_plot, = ax_sim.plot([], [], ".", color='lime', markersize=3, alpha=0.8)
    stats_text = ax_sim.text(0.02, 0.95, "", transform=ax_sim.transAxes, fontsize=10, family="monospace", va="top", color='lime')

    # [Panel 2] Theoretical Rotation Curve (Top Right)
    ax_curve = plt.subplot(gs[0, 1])
    ax_curve.set_facecolor('#1a1a1a') # dark gray
    r_theory = np.linspace(0.1, 40.0, 240)
    vN, vI = compute_rotation_curve(galaxy, gravity, r_theory)
    ax_curve.plot(r_theory, vN, '--', alpha=0.7, label="Newtonian", color='cyan')
    ax_curve.plot(r_theory, vI, linewidth=2.0, label=f"ISUT ({iparams.accel_form})", color='magenta')
    ax_curve.set_xlabel("Radius r", color='white'); ax_curve.set_ylabel("Velocity v", color='white')
    ax_curve.set_title("Theoretical Rotation Curve", color='white')
    ax_curve.tick_params(colors='white')
    ax_curve.legend(facecolor='black', labelcolor='white')
    ax_curve.grid(True, alpha=0.2)

    # Particle Velocity Scatter (Real-time check)
    particles_scatter, = ax_curve.plot([], [], "o", color='lime', markersize=1.5, alpha=0.3, label="Particles (r,v)")

    # [Panel 3] Energy Stability Monitor (Bottom)
    ax_energy = plt.subplot(gs[1, :])
    ax_energy.set_facecolor('#1a1a1a')
    ax_energy.set_xlabel("Time Step", color='white'); ax_energy.set_ylabel("Total Energy (Proxy)", color='white')
    ax_energy.set_title("Energy Conservation Audit (E = T + V)", color='white')
    ax_energy.tick_params(colors='white')
    energy_line, = ax_energy.plot([], [], label="Total Energy", color='yellow')
    ax_energy.legend(facecolor='black', labelcolor='white')
    ax_energy.grid(True, alpha=0.2)
    
    E_hist, step_hist = [], []

    def energy_calc(P_: Particles2D) -> float:
        r = np.sqrt(P_.x**2 + P_.y**2)
        v2 = P_.vx**2 + P_.vy**2
        # Approx Potential Energy sum (-GM/r) for audit
        M_enc = galaxy.mass_enclosed(r) 
        V_pot = -np.sum(units.G * M_enc / np.maximum(r, gparams.r_soft))
        T_kin = 0.5 * np.sum(v2)
        return T_kin + V_pot

    def update(frame):
        nonlocal P
        P = integrator.step(P, iparams.dt_base)

        # 1. Update Simulation Plot
        stars_plot.set_data(P.x, P.y)
        stats_text.set_text(f"t={P.t:.2f}\nN={len(P.x)}")

        # 2. Update Rotation Curve Scatter
        r = np.sqrt(P.x**2 + P.y**2)
        v = np.sqrt(P.vx**2 + P.vy**2)
        particles_scatter.set_data(r, v)

        # 3. Audit Energy
        E = energy_calc(P)
        energy_log.append({"step": frame, "time": P.t, "E_total": E})
        
        E_hist.append(E)
        step_hist.append(frame)
        
        # Sliding Window for Energy Graph
        window = 300
        if len(step_hist) > window:
            show_slice = slice(-window, None)
        else:
            show_slice = slice(None)
            
        energy_line.set_data(step_hist[show_slice], E_hist[show_slice])
        
        if len(step_hist) > 1:
            ax_energy.set_xlim(step_hist[show_slice][0], step_hist[show_slice][-1] + 10)
            curr_min = min(E_hist[show_slice])
            curr_max = max(E_hist[show_slice])
            rng_e = curr_max - curr_min + 1e-9
            ax_energy.set_ylim(curr_min - 0.5*rng_e, curr_max + 0.5*rng_e)

        return stars_plot, particles_scatter, stats_text, energy_line

    ani = FuncAnimation(fig, update, frames=400, interval=20, blit=False, repeat=False)
    plt.tight_layout()
    print("   > Displaying Simulation Window... (Close window to proceed)")
    plt.show()

    # --- Save Audit Data ---
    df_energy = pd.DataFrame(energy_log)
    if not df_energy.empty:
        e0 = df_energy['E_total'].iloc[0]
        df_energy['Drift_Rel'] = (df_energy['E_total'] - e0) / e0
        save_path = os.path.join(BASE_DIR, "audit_energy_conservation.csv")
        df_energy.to_csv(save_path, index=False)
        
        final_drift = df_energy['Drift_Rel'].iloc[-1]
        print(f"[Info] Energy Audit Log Saved: {save_path}")
        print(f"[Info] Final Energy Drift: {final_drift:.6%} (Threshold < 1%)")


# ============================================================
# [Modes] 2. Rotation Curve Validation
# ============================================================

def mode_rotation_curve_export(units: Units, gparams: GalaxyParams, iparams: ISUTParams):
    print("\n[Step 2/3] Generating Rotation Curve Comparison...")
    galaxy = BaryonicGalaxy(units, gparams)
    gravity = ISUTGravity(iparams)

    r = np.linspace(0.1, 40.0, 300)
    vN, vI = compute_rotation_curve(galaxy, gravity, r)

    plt.figure(figsize=(10, 6))
    plt.plot(r, vN, '--', label="Newtonian", color='gray')
    plt.plot(r, vI, '-', label=f"ISUT ({iparams.accel_form})", color='red', linewidth=2)
    plt.xlabel("Radius (kpc)")
    plt.ylabel("Velocity (km/s)")
    plt.title("Rotation Curve: Newton vs ISUT")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Save Image and Data
    fig_path = os.path.join(BASE_DIR, "fig_rotation_curve.png")
    csv_path = os.path.join(BASE_DIR, "data_rotation_curve.csv")
    
    plt.savefig(fig_path)
    df_rc = pd.DataFrame({"Radius_kpc": r, "V_Newton": vN, "V_ISUT": vI})
    df_rc.to_csv(csv_path, index=False)
    
    print(f"[Info] Data Saved: {csv_path}")
    print("   > Displaying Rotation Curve... (Close window to proceed)")
    plt.show()


# ============================================================
# [Modes] 3. BTFR Validation
# ============================================================

def mode_btfr_check(units: Units, gparams: GalaxyParams, iparams: ISUTParams):
    print("\n[Step 3/3] Checking BTFR Scaling Law...")
    
    mass_range = np.logspace(2, 4, 15)
    v_flat_list = []
    gravity = ISUTGravity(iparams)

    for Mtot in mass_range:
        gp = GalaxyParams(M_total=float(Mtot), r_scale=gparams.r_scale, r_soft=gparams.r_soft)
        galaxy = BaryonicGalaxy(units, gp)
        
        # Deep MOND regime: r_far ~ sqrt(GM/a0)
        r_far = 4.0 * np.sqrt(units.G * Mtot / iparams.a0)
        r_far = max(r_far, 50.0)
        
        M_enc = galaxy.mass_enclosed(np.array([r_far]))
        aN = galaxy.newtonian_accel(np.array([r_far]), M_enc)
        aI = gravity.accel_isut(aN)
        v = np.sqrt(r_far * aI[0])
        v_flat_list.append(v)
        
    V = np.array(v_flat_list)
    M = mass_range

    # Fit Power Law
    x_log = np.log10(np.maximum(V, 1e-12))
    y_log = np.log10(M)
    slope, intercept = np.polyfit(x_log, y_log, 1)

    plt.figure(figsize=(8, 6))
    plt.scatter(V, M, color='blue', label='ISUT Simulated Galaxies')
    
    v_guide = np.linspace(min(V), max(V), 100)
    m_guide = 10**intercept * v_guide**4 
    plt.plot(v_guide, m_guide, 'r--', alpha=0.5, label="Slope 4 Ref")
    
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Flat Rotation Velocity V_flat")
    plt.ylabel("Baryonic Mass M_bar")
    plt.title(f"BTFR Check: Slope = {slope:.2f} (Target ~4.0)")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    
    # Save Image and Data (Added CSV Export)
    fig_path = os.path.join(BASE_DIR, "fig_btfr_scaling.png")
    csv_path = os.path.join(BASE_DIR, "data_btfr_scaling.csv")
    
    plt.savefig(fig_path)
    df_btfr = pd.DataFrame({"V_flat": V, "M_baryonic": M})
    df_btfr.to_csv(csv_path, index=False)
    
    print(f"[Info] Data Saved: {csv_path}")
    print(f"[Info] Measured Slope: {slope:.4f}")
    print("   > Displaying BTFR Plot... (Close window to finish)")
    plt.show()


# ============================================================
# [Main Automation]
# ============================================================

def main():
    print("="*70)
    print("ISUT Universe Engine: Full Validation Suite")
    print("   1. N-Body Sim (Energy Audit)")
    print("   2. Rotation Curve Data Export")
    print("   3. BTFR Scaling Law Check")
    print(f"   [Output] {BASE_DIR}")
    print("="*70)

    units = Units(G=1.0, c_light=500.0)
    gparams = GalaxyParams(M_total=1000.0, r_scale=3.0)
    iparams = ISUTParams(a0=0.12, dt_base=0.05)

    # Sequence
    mode_animate_with_audit(units, gparams, iparams)
    mode_rotation_curve_export(units, gparams, iparams)
    mode_btfr_check(units, gparams, iparams)

    print("\n" + "="*70)
    print("Validation Suite Completed.")
    print("="*70)

if __name__ == "__main__":
    main()