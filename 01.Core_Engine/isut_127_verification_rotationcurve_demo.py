# -*- coding: utf-8 -*-
"""
ISUT Rotation-Curve Demo (Toy Illustration)
===========================================

Purpose
-------
This script produces a simple rotation-curve illustration comparing:
  - Newtonian (baryons only)
  - MOND (simple nu)
  - Clock-rate reinterpretation (alpha := sqrt(nu), a = alpha^2 * gN)

This is NOT an independent empirical validation. It is a visualization of
standard MOND-like phenomenology and confirms that the clock-rate
reinterpretation reproduces identical kinematics when alpha is defined as
sqrt(nu).

Outputs
-------
1) Structured outputs under:
   ./<SCRIPT_NAME>/
      figures/  (PNG)
      data/     (CSV)

2) Paper-ready exports copied to the script directory:
   - rotation_curve_demo.png
   - rotation_curve_demo.csv
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# [1] Configuration (same convention as your ISUT scripts)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"[System] {SCRIPT_NAME} initialized")
print(f"[Info] Output Root : {OUT_ROOT}")
print(f"[Info] Figures Dir : {FIG_DIR}")
print(f"[Info] Data Dir    : {DATA_DIR}")


# ==============================================================================
# [2] Constants (SI)
# ==============================================================================
G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
MSUN = 1.98847e30           # kg
KPC  = 3.085677581e19       # m


def nu_simple(y: np.ndarray, y_floor: float = 1e-300) -> np.ndarray:
    y_safe = np.maximum(y, y_floor)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y_safe))


def main() -> None:
    parser = argparse.ArgumentParser(description="Toy rotation-curve demo (Newton vs MOND vs clock-rate).")
    parser.add_argument("--r-min-kpc", type=float, default=0.1, help="Min radius (kpc).")
    parser.add_argument("--r-max-kpc", type=float, default=50.0, help="Max radius (kpc).")
    parser.add_argument("--n", type=int, default=800, help="Number of radius samples.")
    parser.add_argument("--Mbar-msun", type=float, default=5.0e10, help="Baryonic mass (Msun).")
    parser.add_argument("--a0", type=float, default=1.2e-10, help="MOND a0 (m/s^2).")
    args = parser.parse_args()

    # Radius grid
    r_kpc = np.linspace(args.r_min_kpc, args.r_max_kpc, args.n)
    r_m = r_kpc * KPC

    # Newtonian baryonic acceleration
    Mbar_kg = args.Mbar_msun * MSUN
    gN = G_SI * Mbar_kg / (r_m ** 2)

    # Newtonian circular speed
    v_newton = np.sqrt(r_m * gN) / 1e3  # km/s

    # MOND
    y = gN / args.a0
    nu = nu_simple(y)
    a_mond = gN * nu
    v_mond = np.sqrt(r_m * a_mond) / 1e3  # km/s

    # Clock-rate reinterpretation
    alpha = np.sqrt(nu)
    a_clock = gN * (alpha ** 2)
    v_clock = np.sqrt(r_m * a_clock) / 1e3  # km/s

    # Save CSV
    df = pd.DataFrame({
        "r_kpc": r_kpc,
        "r_m": r_m,
        "Mbar_msun": np.full_like(r_kpc, args.Mbar_msun),
        "a0_m_s2": np.full_like(r_kpc, args.a0),
        "gN_m_s2": gN,
        "y_gN_over_a0": y,
        "nu_simple": nu,
        "alpha_sqrt_nu": alpha,
        "a_mond_m_s2": a_mond,
        "a_clock_m_s2": a_clock,
        "v_newton_km_s": v_newton,
        "v_mond_km_s": v_mond,
        "v_clock_km_s": v_clock,
    })

    csv_path = os.path.join(DATA_DIR, "rotation_curve_demo.csv")
    df.to_csv(csv_path, index=False)

    # Save PNG
    png_path = os.path.join(FIG_DIR, "rotation_curve_demo.png")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(r_kpc, v_newton, linestyle="--", linewidth=1.6, label="Newtonian (baryons only)")
    ax.plot(r_kpc, v_mond, linewidth=2.4, alpha=0.85, label="MOND (simple nu)")
    ax.plot(r_kpc, v_clock, linestyle=":", linewidth=2.2, label="Clock-rate reinterpretation")

    ax.set_xlabel("Radius (kpc)")
    ax.set_ylabel("Circular speed (km/s)")
    ax.set_title("Rotation-curve demo (toy): Newton vs MOND vs clock-rate")
    ax.grid(True, linestyle=":")
    ax.legend()

    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {csv_path}")

    # Export fixed filenames to CURRENT_DIR for TeX includegraphics convenience
    export_png = os.path.join(CURRENT_DIR, "rotation_curve_demo.png")
    export_csv = os.path.join(CURRENT_DIR, "rotation_curve_demo.csv")

    with open(png_path, "rb") as fsrc:
        with open(export_png, "wb") as fdst:
            fdst.write(fsrc.read())

    df.to_csv(export_csv, index=False)

    print(f"[Export] {export_png}")
    print(f"[Export] {export_csv}")


if __name__ == "__main__":
    main()
