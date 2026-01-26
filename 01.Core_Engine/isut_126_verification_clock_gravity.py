# -*- coding: utf-8 -*-
"""
ISUT Equivalence Artifact (Documentation Only)
==============================================

Purpose
-------
This script documents the algebraic consistency between:
  (A) g_obs = g_N * nu(y)
  (B) define alpha := sqrt(nu(y)) and compute g_obs = g_N * alpha^2

This is an identity check (equivalence by definition), NOT an independent
physical validation. It is included as an evidence-pack artifact and can be
referenced in an Appendix.

Outputs
-------
1) Structured outputs under:
   ./<SCRIPT_NAME>/
      figures/  (PNG)
      data/     (CSV)

2) Paper-ready exports (fixed filenames) copied to the script directory:
   - figure_clockrate_equivalence.png
   - figure_clockrate_equivalence.csv

Notes
-----
- Uses a non-interactive Matplotlib backend for headless reproducibility.
- Axis labels are in English by default.
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
# [1] Configuration: Output directory setup (same convention as your ISUT scripts)
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
# [2] Physics helper (matches your nu_simple definition in engine)
#   nu(y) = 0.5*(1 + sqrt(1 + 4/y))  :contentReference[oaicite:3]{index=3}
# ==============================================================================
def nu_simple(y: np.ndarray, y_floor: float = 1e-300) -> np.ndarray:
    y_safe = np.maximum(y, y_floor)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y_safe))


# ==============================================================================
# [3] Main
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identity check: gN*nu(y) vs gN*alpha^2 with alpha:=sqrt(nu)."
    )
    parser.add_argument("--n", type=int, default=2000, help="Number of radius samples.")
    parser.add_argument("--r-min", type=float, default=0.1, help="Min radius (toy units).")
    parser.add_argument("--r-max", type=float, default=50.0, help="Max radius (toy units).")
    parser.add_argument("--G", type=float, default=1.0, help="Toy gravitational constant.")
    parser.add_argument("--M", type=float, default=1000.0, help="Toy baryonic mass.")
    parser.add_argument("--a0", type=float, default=0.5, help="Toy a0.")
    parser.add_argument("--eps", type=float, default=1e-30, help="Denominator floor for relative error.")
    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Synthetic toy galaxy
    # --------------------------------------------------------------------------
    r = np.linspace(args.r_min, args.r_max, args.n)
    gN = args.G * args.M / (r ** 2)

    y = gN / args.a0
    nu = nu_simple(y)

    # Method A
    gA = gN * nu

    # Method B (clock-rate reinterpretation)
    alpha = np.sqrt(nu)
    gB = gN * (alpha ** 2)

    abs_err = np.abs(gA - gB)
    rel_err = abs_err / np.maximum(np.abs(gA), args.eps)

    max_abs = float(np.max(abs_err))
    max_rel = float(np.max(rel_err))

    print("--- [ISUT Equivalence Check: MOND vs Clock-rate] ---")
    print("Note: Identity check (equivalence by definition), not physical validation.")
    print(f"max_abs_error = {max_abs:.3e}")
    print(f"max_rel_error = {max_rel:.3e}")

    # --------------------------------------------------------------------------
    # Save CSV (data artifact)
    # --------------------------------------------------------------------------
    df = pd.DataFrame({
        "r": r,
        "gN": gN,
        "a0": np.full_like(r, args.a0),
        "y_gN_over_a0": y,
        "nu_simple": nu,
        "alpha_sqrt_nu": alpha,
        "gA_gN_times_nu": gA,
        "gB_gN_times_alpha2": gB,
        "abs_err": abs_err,
        "rel_err": rel_err,
    })

    csv_path = os.path.join(DATA_DIR, "figure_clockrate_equivalence.csv")
    df.to_csv(csv_path, index=False)

    # --------------------------------------------------------------------------
    # Save PNG (paper-ready figure)
    # --------------------------------------------------------------------------
    png_path = os.path.join(FIG_DIR, "figure_clockrate_equivalence.png")

    fig = plt.figure(figsize=(9, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(r, gA, linewidth=3.5, alpha=0.35, label=r"Method A: $g_N\,\nu(y)$")
    ax1.plot(r, gB, linestyle="--", linewidth=1.8, label=r"Method B: $g_N\,\alpha^2$  ($\alpha:=\sqrt{\nu}$)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Radius")
    ax1.set_ylabel("Acceleration")
    ax1.set_title("Equivalence check (toy model): MOND vs clock-rate formulation")
    ax1.grid(True, which="both", linestyle=":")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(r, rel_err)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Radius")
    ax2.set_ylabel("Relative error")
    ax2.set_title(r"Relative difference $|g_A-g_B|/|g_A|$")
    ax2.grid(True, which="both", linestyle=":")

    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {csv_path}")

    # --------------------------------------------------------------------------
    # Export fixed filenames to CURRENT_DIR for TeX includegraphics convenience
    # --------------------------------------------------------------------------
    export_png = os.path.join(CURRENT_DIR, "figure_clockrate_equivalence.png")
    export_csv = os.path.join(CURRENT_DIR, "figure_clockrate_equivalence.csv")

    # Copy-by-read/write to avoid platform-specific shutil issues in locked envs
    with open(png_path, "rb") as fsrc:
        with open(export_png, "wb") as fdst:
            fdst.write(fsrc.read())

    df.to_csv(export_csv, index=False)

    print(f"[Export] {export_png}")
    print(f"[Export] {export_csv}")

    # Strict check: should be near machine precision
    if max_rel < 1e-12:
        print("[Result] PASSED (machine-precision level).")
    else:
        print("[Result] WARNING (unexpected discrepancy; inspect numerics).")


if __name__ == "__main__":
    main()
