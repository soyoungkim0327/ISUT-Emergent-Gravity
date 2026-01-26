# -*- coding: utf-8 -*-
"""12.qumond_curl_free_check__.py

QUMOND Conservative-Field Check: curl(a) ≈ 0
================================================

If acceleration is computed from a scalar potential,

    a = -∇Φ,

then it must satisfy

    ∇×a = 0,

in the continuum. Numerically, we expect the curl magnitude to be near machine
precision (for pad=1 periodic spectral derivatives) or small in the interior
for padded/cropped fields.

Hostile-reviewer angle
----------------------
"You claim a conservative completion. Show me an explicit conservative-ness
diagnostic." This is that diagnostic.

Outputs (relative path)
-----------------------
  ./12.qumond_curl_free_check__/
      data/    (CSV)
      figures/ (PNG)
      logs/    (JSON)

Usage
-----
  python 01.Core_Engine/12.qumond_curl_free_check__.py --N 96 --L 200 --pad 1

Notes
-----
- For strict curl=0 evidence, use pad=1 (periodic spectral derivatives).
- For pad>1, we compute a finite-difference curl and report interior stats.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isut_100_qumond_pm import (
    PMGrid3D,
    QUMONDSolverFFT,
    exponential_disk_density,
    nu_simple,
    nu_standard,
)


# =============================================================================
# Output dirs (match repo conventions)
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
DATA_DIR = os.path.join(OUT_ROOT, "data")
LOG_DIR = os.path.join(OUT_ROOT, "logs")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def _grad_periodic(f: np.ndarray, grid: PMGrid3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fk = np.fft.fftn(f)
    dfdx = np.fft.ifftn(1j * grid.kx * fk).real
    dfdy = np.fft.ifftn(1j * grid.ky * fk).real
    dfdz = np.fft.ifftn(1j * grid.kz * fk).real
    return dfdx, dfdy, dfdz


def _curl_periodic(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, grid: PMGrid3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    daz_dy = _grad_periodic(az, grid)[1]
    day_dz = _grad_periodic(ay, grid)[2]

    dax_dz = _grad_periodic(ax, grid)[2]
    daz_dx = _grad_periodic(az, grid)[0]

    day_dx = _grad_periodic(ay, grid)[0]
    dax_dy = _grad_periodic(ax, grid)[1]

    cx = daz_dy - day_dz
    cy = dax_dz - daz_dx
    cz = day_dx - dax_dy
    return cx, cy, cz


def _curl_fd(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    daz_dy = np.gradient(az, dx, axis=1, edge_order=2)
    day_dz = np.gradient(ay, dx, axis=2, edge_order=2)

    dax_dz = np.gradient(ax, dx, axis=2, edge_order=2)
    daz_dx = np.gradient(az, dx, axis=0, edge_order=2)

    day_dx = np.gradient(ay, dx, axis=0, edge_order=2)
    dax_dy = np.gradient(ax, dx, axis=1, edge_order=2)

    cx = daz_dy - day_dz
    cy = dax_dz - daz_dx
    cz = day_dx - dax_dy
    return cx, cy, cz


def _mask_interior(shape: Tuple[int, int, int], margin: int) -> np.ndarray:
    m = margin
    mask = np.zeros(shape, dtype=bool)
    mask[m:-m, m:-m, m:-m] = True
    return mask


def _stats(field: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    x = field[mask]
    return {
        "rms": float(np.sqrt(np.mean(x**2))),
        "linf": float(np.max(np.abs(x))),
        "median": float(np.median(np.abs(x))),
        "p95": float(np.quantile(np.abs(x), 0.95)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="QUMOND curl-free diagnostic")
    ap.add_argument("--N", type=int, default=96)
    ap.add_argument("--L", type=float, default=200.0)
    ap.add_argument("--pad", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--G", type=float, default=1.0)
    ap.add_argument("--a0", type=float, default=0.12)
    ap.add_argument("--nu", type=str, default="nu_standard", choices=["nu_standard", "nu_simple"])
    ap.add_argument("--M", type=float, default=1000.0)
    ap.add_argument("--Rd", type=float, default=3.0)
    ap.add_argument("--zd", type=float, default=0.4)
    ap.add_argument("--margin", type=int, default=4, help="interior margin (cells) for pad>1 FD check")
    args = ap.parse_args()

    print(f"[System] {SCRIPT_NAME} initialized")
    print(f"[Info] Output Root : {OUT_ROOT}")

    nu_func: Callable[[np.ndarray], np.ndarray] = nu_standard if args.nu == "nu_standard" else nu_simple

    grid = PMGrid3D(N=int(args.N), boxsize=float(args.L), G=float(args.G))
    rho = exponential_disk_density(grid, M_total=float(args.M), R_d=float(args.Rd), z_d=float(args.zd), renormalize=True)

    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho, a0=float(args.a0), nu_func=nu_func, pad_factor=int(args.pad))

    ax = res.accel[..., 0]
    ay = res.accel[..., 1]
    az = res.accel[..., 2]
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)

    if args.pad == 1:
        cx, cy, cz = _curl_periodic(ax, ay, az, grid)
        mode = "spectral(periodic)"
        mask = np.ones_like(ax, dtype=bool)
    else:
        cx, cy, cz = _curl_fd(ax, ay, az, grid.dx)
        mode = "finite-diff(interior)"
        mask = _mask_interior(ax.shape, int(args.margin))

    curl_mag = np.sqrt(cx**2 + cy**2 + cz**2)

    # Stats
    curl_stats = _stats(curl_mag, mask)
    a_stats = _stats(a_mag, mask)

    ratio = curl_mag / np.maximum(a_mag, 1e-30)
    ratio_stats = _stats(ratio, mask)

    # Save CSV summary
    csv_path = os.path.join(DATA_DIR, "data_curl_free_summary.csv")
    import csv
    rows = [
        {"metric": "|curl a|", **curl_stats},
        {"metric": "|a|", **a_stats},
        {"metric": "|curl a|/|a|", **ratio_stats},
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Plot histogram of ratio
    fig = plt.figure(figsize=(8.6, 5.0))
    axp = fig.add_subplot(111)
    axp.hist(np.log10(ratio[mask] + 1e-60), bins=100, alpha=0.9)
    axp.set_xlabel("log10(|curl a|/|a|)")
    axp.set_ylabel("count")
    axp.set_title(f"Curl-free diagnostic ({mode})")
    axp.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "fig_curl_ratio_hist.png")
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    # Metadata
    meta: Dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "mode": mode,
        "args": vars(args),
        "curl_stats": curl_stats,
        "a_stats": a_stats,
        "ratio_stats": ratio_stats,
        "outputs": {
            "csv": os.path.relpath(csv_path, OUT_ROOT),
            "fig": os.path.relpath(fig_path, OUT_ROOT),
        },
        "notes": {
            "interpretation": "For a conservative potential-derived acceleration, curl(a) should be ~0 (up to numerical error).",
        },
    }
    meta_path = os.path.join(LOG_DIR, "run_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("Curl-free check complete")
    print(f"  mode={mode}")
    print(f"  median log10(|curl a|/|a|) = {float(np.median(np.log10(ratio[mask] + 1e-60))):.2f}")
    print(f"  [CSV]  {csv_path}")
    print(f"  [PNG]  {fig_path}")
    print(f"  [META] {meta_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
