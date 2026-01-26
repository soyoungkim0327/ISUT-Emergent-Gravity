# -*- coding: utf-8 -*-
"""
Clock-rate proxy prediction (All65): alpha(proxy) -> nu_pred = alpha^2
======================================================================

Goal (reviewer-facing)
----------------------
This script tests a stronger claim than an identity check:
- It constructs nu_obs(r) from observed rotation curves and baryonic Newtonian proxy gN(r),
  using SPARC rotmod data + previously fitted mass-to-light parameters (Ups_Disk/Ups_Bulge).
- It predicts nu_pred from a proxy function of gN alone (no defining alpha := sqrt(nu_obs)),
  using a global transition scale g_star and a fixed transition sharpness p.
- It outputs PNG+CSV artifacts suitable for an evidence pack.

Important scope note
--------------------
- This is still not a full relativistic metric completion (lensing/PPN are separate).
- It is a kinematic/dynamical-level test for rotation-curve data consistency.

Inputs
------
1) Validator script (data loader):
   - isut_300_valid_a0_constancy.py (loaded dynamically)
   - get_data(gal) expected to return:
       R[kpc], V_obs[km/s], V_err[km/s], V_gas[km/s], V_disk[km/s], V_bul[km/s]

2) All65 fitted parameters CSV:
   - All65_Full_Results.csv or All65_Full_Results_USED.csv
   Required columns:
     Galaxy, Ups_Disk, Ups_Bulge
   Optional:
     Best_a0_SI, Quality (for filtering)

Outputs
-------
./<SCRIPT_NAME>/
  All65/
    figures/
      clockrate_proxy_fit.png
      clockrate_proxy_residual_hist.png
    data/
      All65_clockrate_proxy_points.csv
      All65_clockrate_proxy_summary.csv
      gstar_grid_search.csv
  run_metadata.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import importlib.util
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# [0] Path conventions (matches your project style)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)

ALL65_OUT_DIR = os.path.join(BASE_OUT_DIR, "All65")
FIG_DIR = os.path.join(ALL65_OUT_DIR, "figures")
DATA_DIR = os.path.join(ALL65_OUT_DIR, "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"[System] {SCRIPT_NAME} initialized")
print(f"[Info] Current Dir : {CURRENT_DIR}")
print(f"[Info] Output Base : {BASE_OUT_DIR}")


# ==============================================================================
# [1] Constants
# ==============================================================================
# Conversion: (km/s)^2 / kpc  ->  m/s^2
# 1 (km/s)^2 = 1e6 (m^2/s^2), 1 kpc = 3.085677581e19 m
ACCEL_CONV = 1e6 / 3.085677581e19  # ~3.24078e-14


# ==============================================================================
# [2] Dynamic import of validator script (file name has dots, cannot normal-import)
# ==============================================================================
def load_validator_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Validator script not found: {path}")

    spec = importlib.util.spec_from_file_location("isut_a0_validator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load spec for: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


# ==============================================================================
# [3] Model: nu_pred from proxy gN (global g_star, fixed p)
#     nu_pred(gN) = (1 + (g_star/gN)^p)^(1/(2p))
#     alpha_pred  = sqrt(nu_pred)
# ==============================================================================
def nu_pred_from_gN(gN_si: np.ndarray, g_star: float, p: float, g_floor: float = 1e-30) -> np.ndarray:
    g = np.maximum(gN_si, g_floor)
    return (1.0 + (g_star / g) ** p) ** (1.0 / (2.0 * p))


def robust_loss_log(nu_obs: np.ndarray, nu_pred: np.ndarray, eps: float = 1e-30) -> float:
    """
    Robust loss on log10 ratio:
      res = log10(nu_pred/nu_obs)
      loss = median(|res|)
    """
    x = np.maximum(nu_obs, eps)
    y = np.maximum(nu_pred, eps)
    res = np.log10(y / x)
    return float(np.median(np.abs(res)))


# ==============================================================================
# [4] I/O helpers
# ==============================================================================
def find_all65_results_csv(explicit: str | None) -> str:
    if explicit and os.path.exists(explicit):
        return explicit

    # Common filenames in your workflow
    candidates = [
        os.path.join(CURRENT_DIR, "All65_Full_Results_USED.csv"),
        os.path.join(CURRENT_DIR, "All65_Full_Results.csv"),
        os.path.join(CURRENT_DIR, "All65_Holdout_Results.csv"),
        os.path.join(CURRENT_DIR, "All65", "data", "All65_Full_Results.csv"),
        os.path.join(CURRENT_DIR, "isut_300_valid_a0_constancy", "All65", "data", "All65_Full_Results.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not locate All65 results CSV. Provide --results-csv explicitly."
    )


# ==============================================================================
# [5] Main
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Clock-rate proxy prediction across All65 (no per-galaxy refit).")
    parser.add_argument("--validator-py", type=str, default=os.path.join(CURRENT_DIR, "isut_300_valid_a0_constancy.py"),
                        help="Path to isut_300_valid_a0_constancy.py")
    parser.add_argument("--results-csv", type=str, default=None,
                        help="Path to All65_Full_Results.csv (contains Ups_Disk/Ups_Bulge).")
    parser.add_argument("--allow-download", action="store_true",
                        help="Allow downloading missing rotmod files (NOT recommended for strict reproducibility).")

    # Filtering
    parser.add_argument("--quality", type=str, default="Good",
                        help="Use only rows with Quality == this value. Use 'ALL' to disable.")
    parser.add_argument("--require-opt-success", action="store_true",
                        help="If set, require Opt_Success == True for included galaxies.")

    # Model options
    parser.add_argument("--p", type=float, default=1.0,
                        help="Transition sharpness p in nu_pred(gN). Keep fixed for reviewer simplicity.")
    parser.add_argument("--gstar-min", type=float, default=1e-12,
                        help="Min g_star for grid search (SI m/s^2).")
    parser.add_argument("--gstar-max", type=float, default=1e-8,
                        help="Max g_star for grid search (SI m/s^2).")
    parser.add_argument("--grid-n", type=int, default=240,
                        help="Grid resolution in logspace for g_star search.")

    # Point filters
    parser.add_argument("--r-min-kpc", type=float, default=0.2,
                        help="Minimum radius (kpc) to avoid tiny-r blowups.")
    parser.add_argument("--min-vobs", type=float, default=5.0,
                        help="Minimum observed velocity (km/s) to include a point.")
    args = parser.parse_args()

    # Load validator module
    validator = load_validator_module(args.validator_py)
    if not hasattr(validator, "get_data"):
        raise AttributeError("Validator module missing get_data().")

    get_data = validator.get_data  # type: ignore

    # Load All65 results
    results_csv = find_all65_results_csv(args.results_csv)
    df_res = pd.read_csv(results_csv)

    required_cols = {"Galaxy", "Ups_Disk", "Ups_Bulge"}
    missing = required_cols - set(df_res.columns)
    if missing:
        raise ValueError(f"Missing required columns in results CSV: {missing}")

    # Apply galaxy-level filters
    df_use = df_res.copy()

    if args.quality.upper() != "ALL" and "Quality" in df_use.columns:
        df_use = df_use[df_use["Quality"].astype(str) == args.quality].copy()

    if args.require_opt_success and "Opt_Success" in df_use.columns:
        df_use = df_use[df_use["Opt_Success"] == True].copy()

    if df_use.empty:
        raise ValueError("No galaxies left after filtering. Relax --quality or --require-opt-success.")

    print(f"[Input] results_csv = {results_csv}")
    print(f"[Input] galaxies used = {len(df_use)}")

    # Build point-level dataset
    point_rows = []
    skipped = []

    for _, row in df_use.iterrows():
        gal = str(row["Galaxy"])
        ups_d = float(row["Ups_Disk"])
        ups_b = float(row["Ups_Bulge"])

        # Load rotmod data (R, Vobs, Verr, Vgas, Vdisk, Vbul)
        data, reason = get_data(gal, return_reason=True, allow_download=bool(args.allow_download), download_log=None)
        if data is None:
            skipped.append({"Galaxy": gal, "Reason": reason})
            continue

        R_kpc, Vobs, Verr, Vgas, Vdisk, Vbul = data

        # Point filters
        m = np.isfinite(R_kpc) & np.isfinite(Vobs) & (R_kpc >= args.r_min_kpc) & (Vobs >= args.min_vobs)
        R_kpc = R_kpc[m]
        Vobs = Vobs[m]
        Verr = Verr[m] if Verr is not None else np.zeros_like(Vobs)
        Vgas = Vgas[m] if Vgas is not None else np.zeros_like(Vobs)
        Vdisk = Vdisk[m] if Vdisk is not None else np.zeros_like(Vobs)
        Vbul = Vbul[m] if Vbul is not None else np.zeros_like(Vobs)

        if len(R_kpc) < 6:
            skipped.append({"Galaxy": gal, "Reason": "too_few_points_after_filter"})
            continue

        # Baryonic velocity-squared (km/s)^2
        Vbar2 = (Vgas ** 2) + (ups_d * (Vdisk ** 2)) + (ups_b * (Vbul ** 2))

        # gN in (km/s)^2/kpc  then convert to SI
        R_safe = np.maximum(R_kpc, 1e-3)
        gN_unit = np.abs(Vbar2) / R_safe
        gN_si = gN_unit * ACCEL_CONV

        # a_obs in SI
        aobs_unit = (Vobs ** 2) / R_safe
        aobs_si = aobs_unit * ACCEL_CONV

        # nu_obs
        valid = (gN_si > 0) & np.isfinite(gN_si) & np.isfinite(aobs_si)
        gN_si = gN_si[valid]
        aobs_si = aobs_si[valid]
        R_kpc = R_kpc[valid]

        if len(gN_si) < 6:
            skipped.append({"Galaxy": gal, "Reason": "invalid_gN_or_aobs"})
            continue

        nu_obs = aobs_si / gN_si

        # Store rows
        for i in range(len(gN_si)):
            point_rows.append({
                "Galaxy": gal,
                "r_kpc": float(R_kpc[i]),
                "gN_m_s2": float(gN_si[i]),
                "aobs_m_s2": float(aobs_si[i]),
                "nu_obs": float(nu_obs[i]),
                "Ups_Disk": ups_d,
                "Ups_Bulge": ups_b,
                "Best_a0_SI": float(row["Best_a0_SI"]) if "Best_a0_SI" in row else np.nan,
            })

    if not point_rows:
        raise RuntimeError("No usable data points. Check data paths, filters, or allow_download.")

    df_pts = pd.DataFrame(point_rows)
    df_skip = pd.DataFrame(skipped)

    print(f"[Data] points={len(df_pts)} galaxies_with_points={df_pts['Galaxy'].nunique()} skipped_galaxies={len(df_skip)}")

    # ----------------------------------------------------------------------
    # Global grid search for g_star (single parameter, no per-galaxy refit)
    # ----------------------------------------------------------------------
    p = float(args.p)
    g_grid = np.logspace(np.log10(args.gstar_min), np.log10(args.gstar_max), int(args.grid_n))

    nu_obs_all = df_pts["nu_obs"].to_numpy()
    gN_all = df_pts["gN_m_s2"].to_numpy()

    losses = []
    for g_star in g_grid:
        nu_pred = nu_pred_from_gN(gN_all, g_star=g_star, p=p)
        loss = robust_loss_log(nu_obs_all, nu_pred)
        losses.append(loss)

    losses = np.array(losses, dtype=float)
    best_idx = int(np.nanargmin(losses))
    g_star_best = float(g_grid[best_idx])
    best_loss = float(losses[best_idx])

    print(f"[Fit] p={p:.3g}")
    print(f"[Fit] best g_star = {g_star_best:.3e} m/s^2")
    print(f"[Fit] best median |log10(nu_pred/nu_obs)| = {best_loss:.4f}")

    # Save grid search table (CSV)
    df_grid = pd.DataFrame({
        "g_star_m_s2": g_grid,
        "median_abs_log10_residual": losses,
    })
    grid_csv = os.path.join(DATA_DIR, "gstar_grid_search.csv")
    df_grid.to_csv(grid_csv, index=False)

    # Compute predictions + residuals
    nu_pred_best = nu_pred_from_gN(gN_all, g_star=g_star_best, p=p)
    alpha_pred_best = np.sqrt(nu_pred_best)

    res_log10 = np.log10(np.maximum(nu_pred_best, 1e-30) / np.maximum(nu_obs_all, 1e-30))
    df_pts["g_star_best_m_s2"] = g_star_best
    df_pts["p_fixed"] = p
    df_pts["nu_pred"] = nu_pred_best
    df_pts["alpha_pred"] = alpha_pred_best
    df_pts["residual_log10_nu_pred_over_obs"] = res_log10

    # Save point-level CSV
    pts_csv = os.path.join(DATA_DIR, "All65_clockrate_proxy_points.csv")
    df_pts.to_csv(pts_csv, index=False)

    # Galaxy-level summary
    summary_rows = []
    for gal, sub in df_pts.groupby("Galaxy"):
        rr = sub["residual_log10_nu_pred_over_obs"].to_numpy()
        summary_rows.append({
            "Galaxy": gal,
            "N_points": int(len(sub)),
            "median_abs_log10_residual": float(np.median(np.abs(rr))),
            "mean_log10_residual": float(np.mean(rr)),
            "std_log10_residual": float(np.std(rr)),
        })
    df_sum = pd.DataFrame(summary_rows).sort_values("median_abs_log10_residual")
    sum_csv = os.path.join(DATA_DIR, "All65_clockrate_proxy_summary.csv")
    df_sum.to_csv(sum_csv, index=False)

    if not df_skip.empty:
        skip_csv = os.path.join(DATA_DIR, "All65_clockrate_proxy_skipped.csv")
        df_skip.to_csv(skip_csv, index=False)
        print(f"[Saved] {skip_csv}")

    # ----------------------------------------------------------------------
    # Figures (PNG)
    # ----------------------------------------------------------------------
    # (1) nu_obs vs nu_pred scatter (log-log) with y=x
    fig1_png = os.path.join(FIG_DIR, "clockrate_proxy_fit.png")
    x = np.maximum(nu_obs_all, 1e-30)
    y = np.maximum(nu_pred_best, 1e-30)

    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu_{\mathrm{obs}} = a_{\mathrm{obs}}/g_N$")
    ax.set_ylabel(r"$\nu_{\mathrm{pred}}(g_N)$ from clock-rate proxy")
    ax.set_title("All65 point-level test: proxy-predicted ν vs observed ν")

    # y=x reference
    lim_min = min(np.min(x), np.min(y))
    lim_max = max(np.max(x), np.max(y))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.2)

    ax.grid(True, which="both", linestyle=":")
    ax.text(
        0.02, 0.98,
        f"p={p:.3g}\n"
        f"g*={g_star_best:.3e} m/s^2\n"
        f"median|log10(pred/obs)|={best_loss:.4f}\n"
        f"points={len(df_pts)}",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(fig1_png, dpi=300)
    plt.close(fig)

    # (2) residual histogram (log10)
    fig2_png = os.path.join(FIG_DIR, "clockrate_proxy_residual_hist.png")
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(res_log10, bins=40)
    ax.set_xlabel(r"$\log_{10}(\nu_{\mathrm{pred}}/\nu_{\mathrm{obs}})$")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution (All65 points)")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(fig2_png, dpi=300)
    plt.close(fig)

    print(f"[Saved] {fig1_png}")
    print(f"[Saved] {fig2_png}")
    print(f"[Saved] {pts_csv}")
    print(f"[Saved] {sum_csv}")
    print(f"[Saved] {grid_csv}")

    # ----------------------------------------------------------------------
    # Run metadata (audit)
    # ----------------------------------------------------------------------
    meta = {
        "script": SCRIPT_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "validator_py": args.validator_py,
        "results_csv": results_csv,
        "filters": {
            "quality": args.quality,
            "require_opt_success": bool(args.require_opt_success),
            "r_min_kpc": args.r_min_kpc,
            "min_vobs_km_s": args.min_vobs,
        },
        "model": {
            "nu_pred_form": "nu = (1 + (g_star/gN)^p)^(1/(2p))",
            "p_fixed": p,
            "gstar_grid_min": args.gstar_min,
            "gstar_grid_max": args.gstar_max,
            "grid_n": args.grid_n,
            "g_star_best_m_s2": g_star_best,
            "median_abs_log10_residual_best": best_loss,
        },
        "counts": {
            "galaxies_used": int(df_use.shape[0]),
            "galaxies_with_points": int(df_pts["Galaxy"].nunique()),
            "points": int(len(df_pts)),
            "skipped_galaxies": int(len(df_skip)),
        },
    }
    with open(os.path.join(BASE_OUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Saved] {os.path.join(BASE_OUT_DIR, 'run_metadata.json')}")
    print("[Done] No per-galaxy refit performed. Only global g_star calibrated.")


if __name__ == "__main__":
    main()
