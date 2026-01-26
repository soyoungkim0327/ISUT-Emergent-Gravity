# -*- coding: utf-8 -*-
"""isut_108_qumond_solar_system_sanity.py

Solar-System High-Acceleration Sanity Check
==========================================

Purpose
-------
A hostile reviewer can (correctly) ask:

  "If your galaxy-scale modification removes the need for dark matter,
   does it reduce to Newtonian/GR behavior in the Solar System?"

This script provides a *quantitative* answer in the simplest clean setting:
- isolated Sun
- high-acceleration regime (g >> a0)

We evaluate the fractional deviation implied by the adopted MOND/ISUT response
function \nu(y) with y = g_N / a0.

In spherical symmetry, QUMOND reduces to the algebraic relation:

  g = g_N * nu(g_N/a0)

So this test is an analytic, regime-check sanity test (not a claim about
external-field effects or full relativistic phenomenology).

Outputs (relative path)
-----------------------
Creates a folder next to this script:

  ./isut_108_qumond_solar_system_sanity/
      data/solar_system_deviation_table.csv
      figures/solar_system_fractional_deviation.png
      figures/solar_system_delta_acceleration.png
      logs/run_metadata.json

Usage
-----
  python 01.Core_Engine/isut_108_qumond_solar_system_sanity.py
  python 01.Core_Engine/isut_108_qumond_solar_system_sanity.py --nu nu_simple
  python 01.Core_Engine/isut_108_qumond_solar_system_sanity.py --a0 1.2e-10

"""

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import List, Tuple

import numpy as np

# Matplotlib is already used widely across the repo.
import matplotlib.pyplot as plt

from isut_100_qumond_pm import nu_simple, nu_standard


# -------------------------
# Physical constants (SI)
# -------------------------
G_SI = 6.67430e-11  # m^3 kg^-1 s^-2
M_SUN = 1.98847e30  # kg
AU = 1.495978707e11  # m


PLANETS_AU: List[Tuple[str, float]] = [
    ("Mercury", 0.38709893),
    ("Venus", 0.72333199),
    ("Earth", 1.00000011),
    ("Mars", 1.52366231),
    ("Jupiter", 5.20336301),
    ("Saturn", 9.53707032),
    ("Uranus", 19.19126393),
    ("Neptune", 30.06896348),
]


def _ensure_dirs(base: str) -> Tuple[str, str, str]:
    data_dir = os.path.join(base, "data")
    fig_dir = os.path.join(base, "figures")
    log_dir = os.path.join(base, "logs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    return data_dir, fig_dir, log_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Solar System high-g MOND/ISUT sanity test")
    ap.add_argument(
        "--a0",
        type=float,
        default=1.2e-10,
        help="Acceleration scale a0 in m/s^2 (default: 1.2e-10).",
    )
    ap.add_argument(
        "--nu",
        type=str,
        default="nu_standard",
        choices=["nu_standard", "nu_simple"],
        help="Which response function nu(y) to use.",
    )
    ap.add_argument(
        "--rmin-au",
        type=float,
        default=0.1,
        help="Min radius for continuous curve (AU).",
    )
    ap.add_argument(
        "--rmax-au",
        type=float,
        default=50.0,
        help="Max radius for continuous curve (AU).",
    )
    ap.add_argument(
        "--nr",
        type=int,
        default=800,
        help="Number of radius samples for continuous curve.",
    )
    args = ap.parse_args()

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
    OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
    data_dir, fig_dir, log_dir = _ensure_dirs(OUT_ROOT)

    nu_fn = nu_standard if args.nu == "nu_standard" else nu_simple

    # Planet table
    rows = []
    max_abs_frac = 0.0
    max_abs_delta = 0.0

    for name, a_au in PLANETS_AU:
        r = a_au * AU
        gN = G_SI * M_SUN / (r * r)
        y = gN / args.a0
        nu = float(nu_fn(np.array([y], dtype=np.float64))[0])
        g = gN * nu
        delta = g - gN
        frac = delta / gN

        max_abs_frac = max(max_abs_frac, abs(frac))
        max_abs_delta = max(max_abs_delta, abs(delta))

        rows.append(
            {
                "body": name,
                "a_AU": f"{a_au:.10g}",
                "r_m": f"{r:.6e}",
                "gN_m_s2": f"{gN:.12e}",
                "y=gN/a0": f"{y:.12e}",
                "nu(y)": f"{nu:.16e}",
                "delta_g_m_s2": f"{delta:.12e}",
                "frac_delta_g": f"{frac:.12e}",
            }
        )

    csv_path = os.path.join(data_dir, "solar_system_deviation_table.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "body",
                "a_AU",
                "r_m",
                "gN_m_s2",
                "y=gN/a0",
                "nu(y)",
                "delta_g_m_s2",
                "frac_delta_g",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # Continuous curve for plots
    r_au = np.geomspace(max(args.rmin_au, 1e-6), max(args.rmax_au, args.rmin_au * 1.01), args.nr)
    r = r_au * AU
    gN = G_SI * M_SUN / (r * r)
    y = gN / args.a0
    nu = nu_fn(y)
    g = gN * nu
    delta = g - gN
    frac = delta / gN

    # Plot 1: fractional deviation
    plt.figure()
    plt.loglog(r_au, np.abs(frac) + 1e-300)  # avoid log(0)
    plt.xlabel("r [AU]")
    plt.ylabel("|Δg / g_N|")
    plt.title(f"Solar-System high-g limit: {args.nu}, a0={args.a0:.3e} m/s²")
    plt.grid(True, which="both")
    frac_fig = os.path.join(fig_dir, "solar_system_fractional_deviation.png")
    plt.savefig(frac_fig, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot 2: absolute delta acceleration
    plt.figure()
    plt.loglog(r_au, np.abs(delta) + 1e-300)
    plt.xlabel("r [AU]")
    plt.ylabel("|Δg| [m/s²]")
    plt.title(f"Solar-System |Δg|: {args.nu}, a0={args.a0:.3e} m/s²")
    plt.grid(True, which="both")
    delta_fig = os.path.join(fig_dir, "solar_system_delta_acceleration.png")
    plt.savefig(delta_fig, dpi=200, bbox_inches="tight")
    plt.close()

    # Metadata (reviewer-facing)
    meta = {
        "args": {
            "a0_m_s2": args.a0,
            "nu": args.nu,
            "rmin_au": args.rmin_au,
            "rmax_au": args.rmax_au,
            "nr": args.nr,
        },
        "constants": {
            "G_SI": G_SI,
            "M_SUN": M_SUN,
            "AU": AU,
        },
        "summary": {
            "max_abs_frac_delta_g": max_abs_frac,
            "max_abs_delta_g_m_s2": max_abs_delta,
            "note": "This is an isolated Sun, spherical-limit regime check. External-field effects are not modeled here.",
        },
        "outputs": {
            "table_csv": os.path.relpath(csv_path, CURRENT_DIR).replace("\\", "/"),
            "frac_fig": os.path.relpath(frac_fig, CURRENT_DIR).replace("\\", "/"),
            "delta_fig": os.path.relpath(delta_fig, CURRENT_DIR).replace("\\", "/"),
        },
    }

    meta_path = os.path.join(log_dir, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("[OK] Solar-system sanity artifacts written to:")
    print(f" - {OUT_ROOT}")
    print(f" - max |Δg/gN| = {max_abs_frac:.3e}")
    print(f" - max |Δg|     = {max_abs_delta:.3e} m/s^2")


if __name__ == "__main__":
    main()
