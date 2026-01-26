#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""isut_110_qumond_solar_system_efe_quadrupole.py

Solar-system EFE sanity check: QUMOND quadrupole + perihelion scale
===================================================================

Context (reviewer-facing)
-------------------------
In MOND/QUMOND, the External Field Effect (EFE) can induce a small *quadrupolar* distortion
of the internal potential of an otherwise isolated point mass (e.g., the Sun) when the system
is embedded in an approximately uniform external field (Milky Way).

A standard parametrization (e.g., Hees et al. 2014, arXiv:1402.6950) writes the anomalous
internal potential as:

  δu(x) = (1/2) x_i x_j Q_ij,   with  Q_ij = Q2 (e_i e_j - δ_ij/3)

where e is the (unit) direction of the external field.

This script does two things:

1) Numerically estimates Q2 in *dimensionless MOND units* (r_M = sqrt(GM/a0) = 1) using the
   repository's FFT QUMOND solver with the EFE proxy ν(|g_N + g_ext|/a0) in the source term.

2) Converts that into an SI-scale Q2 for the Sun and reports the *maximum* perihelion-precession
   rate scale using Eq. (15) in Hees et al. (2014):

  \dot{\tilde{\omega}} = Q2 * sqrt(1-e^2) / (4 n) * [1 + 5 cos(2 \tilde{\omega})]

We report the amplitude envelope (cos term = ±1), i.e. the max |.| factor 6.

Outputs (relative)
------------------
  ./isut_110_qumond_solar_system_efe_quadrupole/
      data/    (CSV + JSON)
      figures/ (PNG)

Important limitations
---------------------
* The FFT solver uses periodic boundary conditions; padding reduces but does not eliminate
  image effects. This is a *sanity check / order-of-magnitude* validation, not a precision
  ephemerides model.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    nu_simple,
    nu_standard,
    make_nu_from_mu_beta,
)

# =============================================================================
# Output dirs (match repo conventions)
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# =============================================================================
# Helpers
# =============================================================================

def _make_point_mass_density(grid: PMGrid3D, M: float, sigma: float) -> np.ndarray:
    """Gaussian-smoothed point mass at the box center."""
    x, y, z = grid.centered_coords()
    r2 = x**2 + y**2 + z**2
    rho = np.exp(-0.5 * r2 / (sigma**2))
    rho_sum = float(np.sum(rho))
    if rho_sum <= 0:
        raise RuntimeError("rho_sum <= 0 in point mass builder")
    rho *= (M / rho_sum) / (grid.dx**3)
    return rho


def _estimate_Q2_from_axis_potential(phi: np.ndarray, grid: PMGrid3D, m_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Q2 from axis-potential differences:

    For e along +x, the quadrupole form implies:
      δu_x(r) - δu_y(r) = (Q2/2) r^2   (Newtonian radial part cancels)
    so:
      Q2(r) ≈ 2 [phi(r,0,0) - phi(0,r,0)] / r^2

    We compute this at r = m*dx for integer m values.
    """
    N = grid.N
    ic = N // 2
    dx = grid.dx
    Q2_list = []
    r_list = []
    for m in m_list:
        m = int(m)
        if m <= 0:
            continue
        if ic + m >= N or ic - m < 0:
            continue
        r = m * dx
        # Sample along +x and +y axis
        phi_x = float(phi[ic + m, ic, ic])
        phi_y = float(phi[ic, ic + m, ic])
        Q2 = 2.0 * (phi_x - phi_y) / (r**2)
        Q2_list.append(Q2)
        r_list.append(r)
    return np.array(r_list, dtype=float), np.array(Q2_list, dtype=float)


def _solar_constants(a0_si: float) -> Dict[str, float]:
    # GM_sun in m^3/s^2
    GM_sun = 1.32712440018e20
    rM = np.sqrt(GM_sun / a0_si)  # MOND radius of the Sun
    return {"GM_sun": GM_sun, "rM_sun_m": float(rM)}


def _max_precession_rate(Q2_si: float, e: float, period_s: float) -> float:
    """
    Maximum |dot(omega)| from Hees et al. (2014) Eq. (15) envelope:

      dot(omega) = Q2 * sqrt(1-e^2)/(4n) * [1 + 5 cos(2 ω̃)]
    -> |.| max when cos=+1 -> factor 6.
    """
    n = 2.0 * np.pi / period_s
    return abs(Q2_si) * np.sqrt(max(0.0, 1.0 - e**2)) * 6.0 / (4.0 * n)


def _rad_s_to_mas_century(rate_rad_s: float) -> float:
    sec_per_century = 36525.0 * 86400.0
    arcsec_per_rad = 206264.80624709636
    return rate_rad_s * arcsec_per_rad * 1000.0 * sec_per_century


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128, help="Grid resolution per axis (dimensionless run)")
    ap.add_argument("--L", type=float, default=10.0, help="Box size in units of r_M (dimensionless)")
    ap.add_argument("--pad", type=int, default=2, choices=[1, 2, 3], help="FFT padding factor")
    ap.add_argument("--G", type=float, default=1.0, help="Dimensionless G")
    ap.add_argument("--M", type=float, default=1.0, help="Point mass M in dimensionless units")
    ap.add_argument("--a0", type=float, default=1.0, help="Dimensionless a0 (set to 1 for r_M=1 scaling)")
    ap.add_argument("--eta", type=float, default=1.6, help="External field magnitude in units of a0 (|g_ext| = eta*a0)")
    ap.add_argument("--sigma", type=float, default=None, help="Gaussian smoothing sigma (defaults to 1.5*dx)")
    ap.add_argument(
        "--nu",
        type=str,
        default="nu_beta",
        choices=["nu_standard", "nu_simple", "nu_beta"],
        help="Interpolation ν(y) choice for the *field solve*",
    )
    ap.add_argument("--beta", type=float, default=12.0, help="Only used if --nu nu_beta (sharp transition helps Solar System)")
    ap.add_argument("--mmin", type=int, default=2, help="Min integer radius m for Q2 sampling (r=m*dx)")
    ap.add_argument("--mmax", type=int, default=6, help="Max integer radius m for Q2 sampling (r=m*dx)")
    ap.add_argument("--a0_si", type=float, default=1.2e-10, help="Physical a0 in m/s^2 for Solar conversion")
    args = ap.parse_args()

    # Dimensionless grid: coordinates in units of r_M
    grid = PMGrid3D(N=args.N, boxsize=args.L, G=args.G)

    sigma = float(args.sigma) if args.sigma is not None else 1.5 * grid.dx
    rho = _make_point_mass_density(grid=grid, M=args.M, sigma=sigma)

    solver = QUMONDSolverFFT(grid)

    if args.nu == "nu_standard":
        nu_func = nu_standard
    elif args.nu == "nu_simple":
        nu_func = nu_simple
    else:
        nu_func = make_nu_from_mu_beta(beta=args.beta)

    # External field direction: +x
    g_ext = (args.eta * args.a0, 0.0, 0.0)

    res = solver.solve(
        rho=rho,
        a0=args.a0,
        nu_func=nu_func,
        pad_factor=args.pad,
        g_ext=g_ext,
    )

    # Estimate Q2 by axis sampling in the inner region (r << r_M)
    m_list = np.arange(int(args.mmin), int(args.mmax) + 1)
    r_list, Q2_list = _estimate_Q2_from_axis_potential(res.phi, grid, m_list)

    Q2_dimless = float(np.median(Q2_list)) if Q2_list.size else float("nan")
    Q2_dimless_std = float(np.std(Q2_list)) if Q2_list.size else float("nan")

    # Convert to q parameter in the Hees et al. convention:
    #   q = -2 Q2 r_M / (3 a0). In our dimensionless units r_M=1, a0=1.
    q_dimless = -2.0 * Q2_dimless / 3.0

    # Convert to SI Q2 for the Sun:
    #   Q2 = - (3 a0 q)/(2 r_M)
    consts = _solar_constants(a0_si=float(args.a0_si))
    rM_sun = consts["rM_sun_m"]
    Q2_si = - (3.0 * float(args.a0_si) * q_dimless) / (2.0 * rM_sun)

    # Cassini-scale benchmark from Hees et al. (2014): Q2 ~ (3 ± 3)×10^{-27} s^{-2}
    cassini_Q2_mean = 3.0e-27
    cassini_Q2_sigma = 3.0e-27
    cassini_2sigma = abs(cassini_Q2_mean) + 2.0 * cassini_Q2_sigma

    # Planet precession scales (max envelope, orientation-free)
    planets = [
        {"name": "Mercury", "e": 0.2056, "P_days": 87.969},
        {"name": "Earth",   "e": 0.0167, "P_days": 365.256},
        {"name": "Saturn",  "e": 0.0565, "P_days": 10759.22},
    ]
    for p in planets:
        rate = _max_precession_rate(Q2_si, e=p["e"], period_s=p["P_days"] * 86400.0)
        p["omega_dot_rad_s_max"] = rate
        p["omega_dot_mas_century_max"] = _rad_s_to_mas_century(rate)

    # Save CSV
    csv_path = os.path.join(DATA_DIR, "q2_estimates_dimless.csv")
    header = "r_over_rM,Q2_dimless"
    np.savetxt(csv_path, np.column_stack([r_list, Q2_list]), delimiter=",", header=header, comments="")

    csv_planets = os.path.join(DATA_DIR, "perihelion_scale.csv")
    header2 = "planet,e,P_days,omega_dot_rad_s_max,omega_dot_mas_century_max"
    rows = [[p["name"], p["e"], p["P_days"], p["omega_dot_rad_s_max"], p["omega_dot_mas_century_max"]] for p in planets]
    # write with numpy as string-safe
    with open(csv_planets, "w", encoding="utf-8") as f:
        f.write(header2 + "\n")
        for row in rows:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]}\n")

    # Plot Q2(r)
    fig = plt.figure()
    if Q2_list.size:
        plt.plot(r_list, Q2_list, marker="o")
        plt.axhline(Q2_dimless, linestyle="--", label=f"median={Q2_dimless:.3e}")
        plt.xlabel(r"$r/r_M$ (dimensionless)")
        plt.ylabel(r"$Q_2$ (dimensionless, in code units)")
        plt.title("EFE quadrupole estimate from axis-potential differences")
        plt.legend()
    else:
        plt.text(0.1, 0.5, "No Q2 samples (increase N or adjust mmin/mmax)", transform=plt.gca().transAxes)
        plt.axis("off")
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "q2_dimless_vs_r.png")
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    # Plot perihelion scales
    fig2 = plt.figure()
    names = [p["name"] for p in planets]
    vals = [p["omega_dot_mas_century_max"] for p in planets]
    plt.bar(names, vals)
    plt.ylabel("max |perihelion drift| (mas/century)")
    plt.title("Solar-system EFE perihelion scale (envelope, Hees+2014 Eq.15)")
    fig2.tight_layout()
    fig2_path = os.path.join(FIG_DIR, "perihelion_scale_mas_century.png")
    fig2.savefig(fig2_path, dpi=180)
    plt.close(fig2)

    meta = {
        "script": os.path.basename(__file__),
        "dimensionless_setup": {"N": args.N, "L": args.L, "pad": args.pad, "G": args.G, "M": args.M, "a0": args.a0},
        "nu": args.nu,
        "beta": args.beta if args.nu == "nu_beta" else None,
        "eta": args.eta,
        "g_ext_vec": g_ext,
        "sigma": sigma,
        "Q2_dimless_median": Q2_dimless,
        "Q2_dimless_std": Q2_dimless_std,
        "q_dimless": q_dimless,
        "a0_si": float(args.a0_si),
        "rM_sun_m": rM_sun,
        "Q2_sun_si": Q2_si,
        "cassini_benchmark": {
            "Q2_mean": cassini_Q2_mean,
            "Q2_sigma": cassini_Q2_sigma,
            "two_sigma_abs_bound": cassini_2sigma,
        },
        "cassini_pass": bool(abs(Q2_si) < cassini_2sigma),
        "planets": planets,
        "notes": (
            "This is an order-of-magnitude sanity check using a periodic FFT QUMOND solver with padding. "
            "It estimates Q2 from axis-potential differences and converts to SI using the Sun's MOND radius."
        ),
        "outputs": {"q2_csv": csv_path, "perihelion_csv": csv_planets, "q2_plot": fig_path, "perihelion_plot": fig2_path},
    }
    with open(os.path.join(DATA_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[RESULT] Q2_dimless (median) =", Q2_dimless)
    print("[RESULT] q_dimless =", q_dimless)
    print("[RESULT] Q2_sun (SI) =", Q2_si, "s^-2")
    print("[CHECK] Cassini 2σ bound ~", cassini_2sigma, "s^-2  -> PASS =", abs(Q2_si) < cassini_2sigma)
    for p in planets:
        print(f"  {p['name']}: max drift ~ {p['omega_dot_mas_century_max']:.3g} mas/century")
    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote: {csv_planets}")
    print(f"[OK] Wrote: {fig_path}")
    print(f"[OK] Wrote: {fig2_path}")


if __name__ == "__main__":
    main()