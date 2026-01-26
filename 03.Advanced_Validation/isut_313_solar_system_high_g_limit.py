# -*- coding: utf-8 -*-
"""10.solar_system_high_g_limit__.py

Solar-System High-Acceleration Validation (Reviewer Defense)
===========================================================

What reviewers will ask
-----------------------
If a galaxy-scale modification removes the need for dark matter, it must:
1) reduce to Newtonian behavior in the Solar System (high acceleration), and
2) not introduce measurable anomalous accelerations at planetary scales.

This script quantifies the high-acceleration limit by evaluating

    g = g_N * nu(g_N/a0)

for an isolated Sun and reporting the implied fractional deviation
|Δg/g_N| and absolute |Δg| at planetary semi-major axes.

Why this belongs in Advanced_Validation
--------------------------------------
It is not a simulation-kernel test; it is a regime/constraint sanity check that
helps preempt a common hostile-reviewer line of attack.

Outputs (relative path)
-----------------------
Creates a folder next to this script:

  ./10.solar_system_high_g_limit__/
      data/solar_system_deviation_table.csv
      figures/solar_system_fractional_deviation.png
      figures/solar_system_delta_acceleration.png
      logs/run_metadata.json

Usage
-----
  python 03.Advanced_Validation/10.solar_system_high_g_limit__.py
  python 03.Advanced_Validation/10.solar_system_high_g_limit__.py --a0 1.2e-10

Notes
-----
- This is an isolated Sun spherical-limit check.
- External-field effects (EFE) are not modeled here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# repo-root import
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isut_000_common import ScriptPaths, write_run_metadata

# import the canonical nu(y) implementations used by the engine
CORE_DIR = REPO_ROOT / "01.Core_Engine"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from isut_100_qumond_pm import nu_standard, nu_simple  # noqa: E402


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


def _as_float(x: np.ndarray) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def main() -> None:
    ap = argparse.ArgumentParser(description="Solar-system high-g regime validation")
    ap.add_argument(
        "--a0",
        type=float,
        default=1.2e-10,
        help="Acceleration scale a0 in m/s^2 (default: 1.2e-10).",
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
    ap.add_argument("--nr", type=int, default=800, help="Number of samples for continuous curve.")
    args = ap.parse_args()

    sp = ScriptPaths.for_script(__file__)

    # Planet table (both nu forms)
    rows: List[Dict[str, str]] = []

    def compute(nu_fn, r_m: float) -> Tuple[float, float, float, float]:
        gN = G_SI * M_SUN / (r_m * r_m)
        y = gN / args.a0
        nu = _as_float(nu_fn(np.array([y], dtype=np.float64)))
        g = gN * nu
        delta = g - gN
        frac = delta / gN
        return gN, y, nu, frac

    # Summary trackers
    max_abs_frac_standard = 0.0
    max_abs_frac_simple = 0.0
    max_abs_delta_standard = 0.0
    max_abs_delta_simple = 0.0

    for name, a_au in PLANETS_AU:
        r_m = a_au * AU
        gN, y, nu_std, frac_std = compute(nu_standard, r_m)
        _, _, nu_sim, frac_sim = compute(nu_simple, r_m)
        delta_std = gN * (nu_std - 1.0)
        delta_sim = gN * (nu_sim - 1.0)

        max_abs_frac_standard = max(max_abs_frac_standard, abs(frac_std))
        max_abs_frac_simple = max(max_abs_frac_simple, abs(frac_sim))
        max_abs_delta_standard = max(max_abs_delta_standard, abs(delta_std))
        max_abs_delta_simple = max(max_abs_delta_simple, abs(delta_sim))

        rows.append(
            {
                "body": name,
                "a_AU": f"{a_au:.10g}",
                "r_m": f"{r_m:.6e}",
                "gN_m_s2": f"{gN:.12e}",
                "y=gN/a0": f"{y:.12e}",
                "nu_standard": f"{nu_std:.16e}",
                "frac_delta_standard": f"{frac_std:.12e}",
                "nu_simple": f"{nu_sim:.16e}",
                "frac_delta_simple": f"{frac_sim:.12e}",
            }
        )

    csv_path = sp.data_dir / "solar_system_deviation_table.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "body",
                "a_AU",
                "r_m",
                "gN_m_s2",
                "y=gN/a0",
                "nu_standard",
                "frac_delta_standard",
                "nu_simple",
                "frac_delta_simple",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    # Continuous curves
    r_au = np.geomspace(max(args.rmin_au, 1e-6), max(args.rmax_au, args.rmin_au * 1.01), args.nr)
    r_m = r_au * AU
    gN = G_SI * M_SUN / (r_m * r_m)
    y = gN / args.a0

    nu_std = nu_standard(y)
    nu_sim = nu_simple(y)

    frac_std = nu_std - 1.0
    frac_sim = nu_sim - 1.0

    # Plot: fractional deviation (both)
    plt.figure()
    plt.loglog(r_au, np.abs(frac_std) + 1e-300, label="nu_standard")
    plt.loglog(r_au, np.abs(frac_sim) + 1e-300, label="nu_simple")
    plt.xlabel("r [AU]")
    plt.ylabel("|Δg / g_N|")
    plt.title(f"Solar-system high-g limit (a0={args.a0:.3e} m/s²)")
    plt.grid(True, which="both")
    plt.legend()
    fig_frac = sp.fig_dir / "solar_system_fractional_deviation.png"
    plt.savefig(fig_frac, dpi=200, bbox_inches="tight")
    plt.close()

    # Plot: absolute delta acceleration (both)
    delta_std = gN * (nu_std - 1.0)
    delta_sim = gN * (nu_sim - 1.0)

    plt.figure()
    plt.loglog(r_au, np.abs(delta_std) + 1e-300, label="nu_standard")
    plt.loglog(r_au, np.abs(delta_sim) + 1e-300, label="nu_simple")
    plt.xlabel("r [AU]")
    plt.ylabel("|Δg| [m/s²]")
    plt.title(f"Solar-system |Δg| (a0={args.a0:.3e} m/s²)")
    plt.grid(True, which="both")
    plt.legend()
    fig_delta = sp.fig_dir / "solar_system_delta_acceleration.png"
    plt.savefig(fig_delta, dpi=200, bbox_inches="tight")
    plt.close()

    # Metadata
    write_run_metadata(
        sp.log_dir,
        args={"a0_m_s2": args.a0, "rmin_au": args.rmin_au, "rmax_au": args.rmax_au, "nr": args.nr},
        notes={
            "max_abs_frac_standard": max_abs_frac_standard,
            "max_abs_frac_simple": max_abs_frac_simple,
            "max_abs_delta_standard_m_s2": max_abs_delta_standard,
            "max_abs_delta_simple_m_s2": max_abs_delta_simple,
            "note": "Isolated Sun spherical-limit check; EFE not modeled.",
        },
    )

    print("[OK] Solar-system high-g validation complete")
    print(f" - {sp.out_root}")
    print(f" - max |Δg/gN| (standard): {max_abs_frac_standard:.3e}")
    print(f" - max |Δg/gN| (simple)  : {max_abs_frac_simple:.3e}")


if __name__ == "__main__":
    main()
