""" 
Residual Whiteness / Structure Test (Reviewer Defense)
======================================================

Reviewer attack addressed
-------------------------
"Your fits look good, but are the residuals hiding a systematic trend?"

This script compiles standardized residuals across All65 using the saved
All65 best-fit parameters (Ups_Disk, Ups_Bulge, Best_a0_SI).
It then reports:
  - Global residual mean/std
  - Per-galaxy lag-1 correlation (adjacent radial bins)
  - Mean ACF across galaxies (lags 1..10)
  - A histogram + mean ACF plot

This is intentionally lightweight (no statsmodels). It is not a definitive
statistical test, but it provides a transparent check that the pipeline is
not leaving an obvious radius-structured residual signature.

Outputs (relative)
------------------
  03.Advanced_Validation/14.residual_whiteness_test__/
      data/residuals_all_points.csv
      data/residual_acf_summary.csv
      figures/residual_hist.png
      figures/residual_acf.png
      logs/run_metadata.json
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required for this script") from e

import matplotlib.pyplot as plt

# --- shared helpers ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isut_000_common import ScriptPaths, find_sparc_data_dir, ensure_rotmod_file, write_run_metadata


# Units: 1 (km/s)^2 / kpc -> m/s^2
ACCEL_CONV = 3.24078e-14


def nu_simple(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-30)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y))


def load_rotmod(path: Path) -> Tuple[np.ndarray, ...]:
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected rotmod format: {path} (ncol={df.shape[1]})")
    R = df.iloc[:, 0].to_numpy(float)
    Vobs = df.iloc[:, 1].to_numpy(float)
    Verr = df.iloc[:, 2].to_numpy(float)
    Vgas = df.iloc[:, 3].to_numpy(float)
    Vdisk = df.iloc[:, 4].to_numpy(float)
    Vbul = df.iloc[:, 5].to_numpy(float)
    return R, Vobs, Verr, Vgas, Vdisk, Vbul


def compute_v_bary2(Vgas: np.ndarray, Vdisk: np.ndarray, Vbul: np.ndarray, ups_d: float, ups_b: float) -> np.ndarray:
    Vb2 = (np.abs(Vgas) * Vgas) + (ups_d * np.abs(Vdisk) * Vdisk) + (ups_b * np.abs(Vbul) * Vbul)
    return np.maximum(Vb2, 0.0)


def predict_isut_velocity(R: np.ndarray, Vb2: np.ndarray, a0_si: float) -> np.ndarray:
    a0_code = a0_si / ACCEL_CONV
    gN = Vb2 / np.maximum(R, 0.01)
    y = gN / a0_code
    g = gN * nu_simple(y)
    return np.sqrt(np.maximum(g * R, 0.0))


def autocorr(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - np.mean(x)
    denom = np.sum(x * x)
    if denom <= 0:
        return np.full(max_lag + 1, np.nan)
    acf = np.empty(max_lag + 1, float)
    acf[0] = 1.0
    for lag in range(1, max_lag + 1):
        if lag >= len(x):
            acf[lag] = np.nan
        else:
            acf[lag] = float(np.sum(x[:-lag] * x[lag:]) / denom)
    return acf


def find_latest_all65_results(repo_root: Path) -> Path:
    candidates = list(repo_root.rglob("All65_Full_Results.csv")) + list(repo_root.rglob("All65_Full_Results_USED.csv"))
    if not candidates:
        raise FileNotFoundError("Could not locate All65_Full_Results.csv (run the All65 pipeline first).")
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Residual whiteness/structure test on All65")
    ap.add_argument("--max_gal", type=int, default=65, help="Limit number of galaxies (debug)")
    ap.add_argument("--max_lag", type=int, default=10)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args(argv)

    sp = ScriptPaths.for_script(__file__)
    allow_download = (not args.no_download) and (os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1")

    all65_path = find_latest_all65_results(REPO_ROOT)
    df_all = pd.read_csv(all65_path)
    df_all = df_all.dropna(subset=["Galaxy", "Best_a0_SI", "Ups_Disk", "Ups_Bulge"])
    df_all = df_all.head(int(args.max_gal))

    sparc_dir = find_sparc_data_dir(Path(__file__).resolve().parent)

    rows = []
    acfs = []

    for _, row in df_all.iterrows():
        gal = str(row["Galaxy"]).strip()
        a0_si = float(row["Best_a0_SI"])
        ups_d = float(row["Ups_Disk"])
        ups_b = float(row["Ups_Bulge"])

        try:
            rotmod = ensure_rotmod_file(gal, sparc_dir, allow_download=allow_download)
            R, Vobs, Verr, Vgas, Vdisk, Vbul = load_rotmod(rotmod)
        except Exception:
            continue

        Vb2 = compute_v_bary2(Vgas, Vdisk, Vbul, ups_d, ups_b)
        Vpred = predict_isut_velocity(R, Vb2, a0_si)

        res = (Vobs - Vpred) / np.maximum(Verr, 1e-6)
        # per-point table
        for i in range(len(R)):
            rows.append({
                "Galaxy": gal,
                "R_kpc": float(R[i]),
                "Vobs_kms": float(Vobs[i]),
                "Vpred_kms": float(Vpred[i]),
                "Verr_kms": float(Verr[i]),
                "residual_sigma": float(res[i]),
            })

        # per-galaxy ACF
        acf = autocorr(res, max_lag=int(args.max_lag))
        acfs.append(acf)
        # lag-1 correlation as quick scalar
        lag1 = float(acf[1]) if len(acf) > 1 and np.isfinite(acf[1]) else float("nan")
        acfs_row = {"Galaxy": gal, "lag1": lag1}
        for k in range(0, int(args.max_lag) + 1):
            acfs_row[f"acf_{k}"] = float(acf[k]) if np.isfinite(acf[k]) else float("nan")
        acfs.append(acf)

    df_pts = pd.DataFrame(rows)
    pts_path = sp.data_dir / "residuals_all_points.csv"
    df_pts.to_csv(pts_path, index=False)

    # Build ACF summary
    # Recompute per-galaxy ACF from point table to avoid double append confusion
    acf_rows = []
    for gal, gdf in df_pts.groupby("Galaxy"):
        res = gdf.sort_values("R_kpc")["residual_sigma"].to_numpy(float)
        acf = autocorr(res, max_lag=int(args.max_lag))
        r = {"Galaxy": gal}
        for k in range(0, int(args.max_lag) + 1):
            r[f"acf_{k}"] = float(acf[k]) if np.isfinite(acf[k]) else float("nan")
        r["lag1"] = r.get("acf_1", float("nan"))
        acf_rows.append(r)
    df_acf = pd.DataFrame(acf_rows)
    acf_path = sp.data_dir / "residual_acf_summary.csv"
    df_acf.to_csv(acf_path, index=False)

    # Plots
    res_all = df_pts["residual_sigma"].to_numpy(float)
    res_all = res_all[np.isfinite(res_all)]

    plt.figure(figsize=(8, 5))
    plt.hist(res_all, bins=60, alpha=0.9)
    plt.axvline(0, color="k", lw=1)
    plt.title("All65 Standardized Residual Histogram")
    plt.xlabel("(V_obs - V_pred) / sigma")
    plt.ylabel("count")
    plt.grid(alpha=0.2)
    hist_path = sp.fig_dir / "residual_hist.png"
    plt.tight_layout()
    plt.savefig(hist_path, dpi=160)
    plt.close()

    # Mean ACF
    acf_cols = [f"acf_{k}" for k in range(0, int(args.max_lag) + 1)]
    mean_acf = df_acf[acf_cols].mean(axis=0, skipna=True).to_numpy(float)
    lags = np.arange(0, len(mean_acf))
    plt.figure(figsize=(8, 5))
    plt.plot(lags, mean_acf, marker="o")
    plt.title("Mean Residual Autocorrelation (All65)")
    plt.xlabel("lag")
    plt.ylabel("ACF")
    plt.grid(alpha=0.25)
    acf_fig_path = sp.fig_dir / "residual_acf.png"
    plt.tight_layout()
    plt.savefig(acf_fig_path, dpi=160)
    plt.close()

    # Metadata
    write_run_metadata(
        sp.log_dir,
        args={
            "all65_csv": str(all65_path),
            "max_gal": int(args.max_gal),
            "max_lag": int(args.max_lag),
            "allow_download": allow_download,
            "sparc_dir": str(sparc_dir),
        },
        notes={
            "files": {
                "points": str(pts_path.relative_to(sp.out_root)),
                "acf_summary": str(acf_path.relative_to(sp.out_root)),
                "hist": str(hist_path.relative_to(sp.out_root)),
                "acf_fig": str(acf_fig_path.relative_to(sp.out_root)),
            },
        },
    )

    print(f"[OK] Wrote: {pts_path}")
    print(f"[OK] Wrote: {acf_path}")
    print(f"[OK] Wrote: {hist_path}")
    print(f"[OK] Wrote: {acf_fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
