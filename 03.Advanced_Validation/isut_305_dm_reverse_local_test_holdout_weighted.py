# -*- coding: utf-8 -*-
"""
DM-reverse vs ISUT local verification (All65) - Robust + Holdout + Weighted
==========================================================================
Reviewer-facing upgrades over 1-5
--------------------------------
(1) Near-zero protection:
    - Extra acceleration residuals can blow up when g_DM,req is small.
    - We apply a physical floor (extra_floor) and compute log-residuals only
      where both required and predicted extra exceed this threshold.

(2) Weighted metrics:
    - Use velocity uncertainty Verr to propagate a_obs uncertainty:
        a_obs = V_obs^2 / r
        sigma_a_obs ~ 2 * V_obs * Verr / r
      (in SI after unit conversion)
    - Provide weighted median(|log10(pred/req)|) in addition to unweighted.

(3) Galaxy-level K-fold holdout:
    - Fit global g_star on TRAIN galaxies only (grid search)
    - Evaluate median |log10(pred/req)| on TEST galaxies
    - Report both unweighted and weighted metrics per fold

Scope note
----------
This remains a nonrelativistic rotation-curve (dynamical) test.
It is not a full relativistic metric/lensing completion.

Inputs
------
A) Validator data loader:
   isut_300_valid_a0_constancy.py (loaded dynamically)
   get_data(...) returns:
     R_kpc, Vobs, Verr, Vgas, Vdisk, Vbul

B) All65 fit summary:
   All65_Full_Results.csv (or _USED.csv)
   required columns: Galaxy, Ups_Disk, Ups_Bulge
   optional columns: Quality, Opt_Success, Best_a0_SI

C) Clock-rate proxy parameters:
   - g_star (SI m/s^2) and p
   If not provided, autodetect from latest 1-3 run_metadata.json containing model.g_star_best_m_s2

Outputs
-------
./<SCRIPT_NAME>/
  All65/
    figures/
      extra_scatter_floor.png
      extra_residual_hist_floor.png
      extra_residual_hist_floor_weighted.png
      holdout_boxplot_extra_unweighted.png
      holdout_boxplot_extra_weighted.png
      gstar_grid_extra_trainloss.png
    data/
      points_local_extra_floor.csv
      skipped_galaxies.csv
      holdout_metrics_extra.csv
      holdout_summary_extra.csv
      gstar_grid_search_extra.csv
  run_metadata.json
"""

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import importlib.util
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================================================================
# [0] Path conventions
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
# Conversion: (km/s)^2 / kpc -> m/s^2
# 1 (km/s)^2 = 1e6 (m^2/s^2)
# 1 kpc = 3.085677581e19 m
ACCEL_CONV = 1e6 / 3.085677581e19  # 3.24078e-14


# ==============================================================================
# [2] Robust loader for validator (filename contains dots)
# ==============================================================================
def load_validator_module(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Validator script not found: {path}")
    spec = importlib.util.spec_from_file_location("isut_validator_loader", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def call_get_data(get_data_func, gal: str, allow_download: bool):
    """
    Compatibility wrapper:
    - Preferred signature (your updated validator): get_data(gal, return_reason=True, allow_download=..., download_log=None)
    - Fallback: get_data(gal) -> data only
    """
    try:
        return get_data_func(gal, return_reason=True, allow_download=allow_download, download_log=None)
    except TypeError:
        data = get_data_func(gal)
        if data is None:
            return None, "get_data_returned_None"
        return data, "ok"


# ==============================================================================
# [3] Proxy model
# ==============================================================================
def nu_pred_from_gN(gN_si: np.ndarray, g_star: float, p: float, g_floor: float = 1e-30) -> np.ndarray:
    g = np.maximum(gN_si, g_floor)
    return (1.0 + (g_star / g) ** p) ** (1.0 / (2.0 * p))


# ==============================================================================
# [4] Auto-detect g_star from latest 1-3 run_metadata.json
# ==============================================================================
def _walk_find(root: str, filename: str) -> List[str]:
    hits = []
    for r, _, files in os.walk(root):
        if filename in files:
            hits.append(os.path.join(r, filename))
    return hits


def autodetect_gstar(search_root: str) -> Optional[float]:
    metas = _walk_find(search_root, "run_metadata.json")
    candidates = []
    for mpath in metas:
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                meta = json.load(f)
            g = meta.get("model", {}).get("g_star_best_m_s2", None)
            if g is not None and np.isfinite(float(g)):
                candidates.append((mpath, float(g), os.path.getmtime(mpath)))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[2], reverse=True)
    chosen = candidates[0]
    print(f"[Auto] g_star detected from: {chosen[0]}")
    return float(chosen[1])


# ==============================================================================
# [5] Utility: weighted median
# ==============================================================================
def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Weighted median for nonnegative weights.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)

    m = np.isfinite(v) & np.isfinite(w) & (w >= 0)
    v = v[m]
    w = w[m]
    if len(v) == 0:
        return float("nan")

    idx = np.argsort(v)
    v = v[idx]
    w = w[idx]
    cw = np.cumsum(w)
    if cw[-1] <= 0:
        return float("nan")
    cutoff = 0.5 * cw[-1]
    j = int(np.searchsorted(cw, cutoff, side="left"))
    return float(v[min(j, len(v) - 1)])


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v = v[m]
    w = w[m]
    if len(v) == 0:
        return float("nan")
    return float(np.sum(w * v) / np.sum(w))


# ==============================================================================
# [6] Metrics for extra residuals with floor
# ==============================================================================
def compute_extra_residuals(extra_req: np.ndarray,
                            extra_pred: np.ndarray,
                            floor: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute residual log10(extra_pred/extra_req) only for points where both are >= floor.
    Returns (residuals, mask_used).
    """
    eps = 1e-30
    req = np.asarray(extra_req, dtype=float)
    pred = np.asarray(extra_pred, dtype=float)

    used = np.isfinite(req) & np.isfinite(pred) & (req >= floor) & (pred >= floor)
    res = np.full_like(req, np.nan, dtype=float)
    res[used] = np.log10(np.maximum(pred[used], eps) / np.maximum(req[used], eps))
    return res, used


# ==============================================================================
# [7] Locate All65 results CSV
# ==============================================================================
def find_all65_results_csv(explicit: Optional[str]) -> str:
    if explicit and os.path.exists(explicit):
        return explicit
    candidates = [
        os.path.join(CURRENT_DIR, "All65_Full_Results_USED.csv"),
        os.path.join(CURRENT_DIR, "All65_Full_Results.csv"),
        os.path.join(CURRENT_DIR, "All65_Holdout_Results.csv"),
        os.path.join(CURRENT_DIR, "isut_300_valid_a0_constancy", "All65", "data", "All65_Full_Results.csv"),
        os.path.join(CURRENT_DIR, "isut_300_valid_a0_constancy", "All65", "data", "All65_Full_Results_USED.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("Could not locate All65 results CSV. Provide --results-csv explicitly.")


# ==============================================================================
# [8] K-fold split on galaxy IDs
# ==============================================================================
def kfold_split_galaxies(galaxies: List[str], k: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    gal = np.array(sorted(set(galaxies)))
    rng.shuffle(gal)
    folds = np.array_split(gal, k)
    return [f.tolist() for f in folds]


# ==============================================================================
# [9] Fit g_star on TRAIN using extra residual loss
# ==============================================================================
def fit_gstar_grid_extra(gN: np.ndarray,
                         aobs: np.ndarray,
                         floor: float,
                         p: float,
                         gstar_min: float,
                         gstar_max: float,
                         grid_n: int) -> Tuple[float, float]:
    """
    Fit g_star by minimizing median(|log10(extra_pred/extra_req)|) on train points.
    extra_req = aobs - gN
    extra_pred = gN*(nu_pred-1)
    We compute residuals only where both extra_req and extra_pred >= floor.
    """
    g_grid = np.logspace(np.log10(gstar_min), np.log10(gstar_max), int(grid_n))
    losses = []
    for g_star in g_grid:
        nu = nu_pred_from_gN(gN, g_star=g_star, p=p)
        extra_pred = gN * (nu - 1.0)
        extra_req = aobs - gN
        res, used = compute_extra_residuals(extra_req, extra_pred, floor=floor)
        if np.sum(used) < 50:
            losses.append(np.nan)
            continue
        losses.append(float(np.median(np.abs(res[used]))))
    losses = np.array(losses, dtype=float)
    if np.all(~np.isfinite(losses)):
        raise RuntimeError("Grid search failed: no usable points under the chosen floor.")
    best_idx = int(np.nanargmin(losses))
    return float(g_grid[best_idx]), float(losses[best_idx])


# ==============================================================================
# [10] Main
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="DM-reverse vs ISUT-extra: robust residuals + weighted + holdout.")
    parser.add_argument("--validator-py", type=str, default=os.path.join(CURRENT_DIR, "isut_300_valid_a0_constancy.py"))
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--allow-download", action="store_true")

    # filters
    parser.add_argument("--quality", type=str, default="Good", help='Quality filter; "ALL" disables.')
    parser.add_argument("--require-opt-success", action="store_true")

    # model params
    parser.add_argument("--g-star", type=float, default=None, help="If omitted, autodetect from 1-3 metadata.")
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--search-root", type=str, default=CURRENT_DIR)

    # point filters
    parser.add_argument("--r-min-kpc", type=float, default=0.5)
    parser.add_argument("--min-vobs", type=float, default=5.0)
    parser.add_argument("--min-points", type=int, default=8)

    # floor + weighting
    parser.add_argument("--extra-floor", type=float, default=1e-12,
                        help="Floor for extra accelerations (m/s^2) for log-ratio residuals.")
    parser.add_argument("--use-weighted-hist", action="store_true",
                        help="If set, also save weighted residual histogram.")

    # holdout
    parser.add_argument("--k", type=int, default=5, help="K-fold count (galaxy-level).")
    parser.add_argument("--seed", type=int, default=7, help="K-fold split seed.")
    parser.add_argument("--gstar-min", type=float, default=1e-12)
    parser.add_argument("--gstar-max", type=float, default=1e-8)
    parser.add_argument("--grid-n", type=int, default=240)
    args = parser.parse_args()

    # Load validator
    validator = load_validator_module(args.validator_py)
    if not hasattr(validator, "get_data"):
        raise AttributeError("Validator missing get_data().")
    get_data = validator.get_data  # type: ignore

    # Load All65 results
    results_csv = find_all65_results_csv(args.results_csv)
    df_res = pd.read_csv(results_csv)

    req_cols = {"Galaxy", "Ups_Disk", "Ups_Bulge"}
    missing = req_cols - set(df_res.columns)
    if missing:
        raise ValueError(f"Missing columns in results CSV: {missing}")

    df_use = df_res.copy()
    if args.quality.upper() != "ALL" and "Quality" in df_use.columns:
        df_use = df_use[df_use["Quality"].astype(str) == args.quality].copy()
    if args.require_opt_success and "Opt_Success" in df_use.columns:
        df_use = df_use[df_use["Opt_Success"] == True].copy()
    if df_use.empty:
        raise ValueError("No galaxies left after filtering. Relax --quality or --require-opt-success.")

    # Determine g_star
    g_star = args.g_star
    if g_star is None:
        g_star = autodetect_gstar(args.search_root)
    if g_star is None:
        raise ValueError("g_star not provided and could not be auto-detected. Provide --g-star.")
    p = float(args.p)
    extra_floor = float(args.extra_floor)

    print(f"[Model] g_star={g_star:.4e} m/s^2, p={p:.3g}, extra_floor={extra_floor:.2e}")
    print(f"[Input] results_csv={results_csv}, galaxies={len(df_use)}")

    point_rows = []
    skipped = []

    for _, row in df_use.iterrows():
        gal = str(row["Galaxy"])
        ups_d = float(row["Ups_Disk"])
        ups_b = float(row["Ups_Bulge"])

        data, reason = call_get_data(get_data, gal, allow_download=bool(args.allow_download))
        if data is None:
            skipped.append({"Galaxy": gal, "Reason": reason})
            continue

        R_kpc, Vobs, Verr, Vgas, Vdisk, Vbul = data

        m = (
            np.isfinite(R_kpc) & np.isfinite(Vobs)
            & (R_kpc >= args.r_min_kpc)
            & (Vobs >= args.min_vobs)
        )
        R_kpc = R_kpc[m]
        Vobs = Vobs[m]
        Verr = Verr[m] if Verr is not None else np.zeros_like(Vobs)
        Vgas = Vgas[m] if Vgas is not None else np.zeros_like(Vobs)
        Vdisk = Vdisk[m] if Vdisk is not None else np.zeros_like(Vobs)
        Vbul = Vbul[m] if Vbul is not None else np.zeros_like(Vobs)

        if len(R_kpc) < args.min_points:
            skipped.append({"Galaxy": gal, "Reason": "too_few_points_after_filter"})
            continue

        R_safe = np.maximum(R_kpc, 1e-3)

        # baryonic decomposition
        Vbar2 = (Vgas ** 2) + (ups_d * (Vdisk ** 2)) + (ups_b * (Vbul ** 2))

        # convert to SI accelerations
        gN_si = (np.abs(Vbar2) / R_safe) * ACCEL_CONV
        aobs_si = ((Vobs ** 2) / R_safe) * ACCEL_CONV

        # propagate aobs uncertainty: sigma_aobs ~ 2 Vobs Verr / r
        aobs_err_si = (2.0 * np.abs(Vobs) * np.abs(Verr) / R_safe) * ACCEL_CONV

        valid = np.isfinite(gN_si) & np.isfinite(aobs_si) & (gN_si > 0) & (aobs_si > 0)
        gN_si = gN_si[valid]
        aobs_si = aobs_si[valid]
        aobs_err_si = aobs_err_si[valid]
        R_kpc = R_kpc[valid]

        if len(gN_si) < args.min_points:
            skipped.append({"Galaxy": gal, "Reason": "invalid_gN_or_aobs"})
            continue

        # extra required and predicted
        extra_req = aobs_si - gN_si
        nu_pred = nu_pred_from_gN(gN_si, g_star=g_star, p=p)
        extra_pred = gN_si * (nu_pred - 1.0)

        # residual with floor
        res, used = compute_extra_residuals(extra_req, extra_pred, floor=extra_floor)
        if np.sum(used) < args.min_points:
            skipped.append({"Galaxy": gal, "Reason": "too_few_points_after_floor"})
            continue

        # weights: inverse-variance in aobs (relative)
        # w = 1 / (sigma_aobs/aobs)^2  (dimensionless, robust)
        rel = np.maximum(aobs_err_si / np.maximum(aobs_si, 1e-30), 1e-12)
        w = 1.0 / (rel ** 2)

        for i in range(len(R_kpc)):
            point_rows.append({
                "Galaxy": gal,
                "r_kpc": float(R_kpc[i]),
                "gN_m_s2": float(gN_si[i]),
                "aobs_m_s2": float(aobs_si[i]),
                "aobs_err_m_s2": float(aobs_err_si[i]),
                "extra_req_m_s2": float(extra_req[i]),
                "extra_pred_m_s2": float(extra_pred[i]),
                "nu_pred": float(nu_pred[i]),
                "residual_log10_extra_pred_over_req": float(res[i]) if np.isfinite(res[i]) else np.nan,
                "used_floor_mask": bool(used[i]),
                "weight_rel_invvar": float(w[i]),
                "Ups_Disk": ups_d,
                "Ups_Bulge": ups_b,
                "Best_a0_SI": float(row["Best_a0_SI"]) if "Best_a0_SI" in row else np.nan,
            })

    if not point_rows:
        raise RuntimeError("No usable galaxies/points after filtering; check inputs and floor.")

    df_pts = pd.DataFrame(point_rows)
    df_skip = pd.DataFrame(skipped)

    # save data
    skip_csv = os.path.join(DATA_DIR, "skipped_galaxies.csv")
    df_skip.to_csv(skip_csv, index=False)

    pts_csv = os.path.join(DATA_DIR, "points_local_extra_floor.csv")
    df_pts.to_csv(pts_csv, index=False)

    print(f"[Data] points={len(df_pts)} galaxies={df_pts['Galaxy'].nunique()} skipped={len(df_skip)}")
    print(f"[Saved] {pts_csv}")
    print(f"[Saved] {skip_csv}")

    # ----------------------------------------------------------------------
    # Global metrics (floor-applied subset)
    # ----------------------------------------------------------------------
    used = df_pts["used_floor_mask"].to_numpy(dtype=bool)
    res_used = df_pts.loc[used, "residual_log10_extra_pred_over_req"].to_numpy(dtype=float)
    w_used = df_pts.loc[used, "weight_rel_invvar"].to_numpy(dtype=float)

    med_abs = float(np.median(np.abs(res_used)))
    wmed_abs = weighted_median(np.abs(res_used), w_used)

    # ----------------------------------------------------------------------
    # Figure: scatter (floor-applied)
    # ----------------------------------------------------------------------
    x = np.maximum(df_pts.loc[used, "extra_req_m_s2"].to_numpy(dtype=float), 1e-30)
    y = np.maximum(df_pts.loc[used, "extra_pred_m_s2"].to_numpy(dtype=float), 1e-30)

    fig_scatter = os.path.join(FIG_DIR, "extra_scatter_floor.png")
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$g_{\rm DM,req}=a_{\rm obs}-g_{\rm bar}$ (m/s$^2$)")
    ax.set_ylabel(r"$g_{\rm ISUT,extra}=g_N(\nu_{\rm pred}-1)$ (m/s$^2$)")
    ax.set_title("All65 local test (floor-applied): DM-required extra vs ISUT-predicted extra")

    lim_min = min(float(np.min(x)), float(np.min(y)))
    lim_max = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.2)

    ax.text(
        0.02, 0.98,
        f"g*={g_star:.3e} m/s^2, p={p:.2g}\n"
        f"floor={extra_floor:.1e} m/s^2\n"
        f"median|log10|={med_abs:.4f}\n"
        f"w-median|log10|={wmed_abs:.4f}\n"
        f"points={int(np.sum(used))}, galaxies={df_pts.loc[used,'Galaxy'].nunique()}",
        transform=ax.transAxes, va="top"
    )
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(fig_scatter, dpi=300)
    plt.close(fig)

    # ----------------------------------------------------------------------
    # Figure: residual histogram (unweighted)
    # ----------------------------------------------------------------------
    fig_hist = os.path.join(FIG_DIR, "extra_residual_hist_floor.png")
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(res_used, bins=50)
    ax.set_xlabel(r"$\log_{10}(g_{\rm ISUT,extra}/g_{\rm DM,req})$")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution (floor-applied, unweighted)")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(fig_hist, dpi=300)
    plt.close(fig)

    # ----------------------------------------------------------------------
    # Figure: residual histogram (weighted)
    # ----------------------------------------------------------------------
    if bool(args.use_weighted_hist):
        fig_whist = os.path.join(FIG_DIR, "extra_residual_hist_floor_weighted.png")
        fig = plt.figure(figsize=(8.5, 5.5))
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(res_used, bins=50, weights=w_used)
        ax.set_xlabel(r"$\log_{10}(g_{\rm ISUT,extra}/g_{\rm DM,req})$")
        ax.set_ylabel("Weighted count")
        ax.set_title("Residual distribution (floor-applied, weighted by V_err)")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(fig_whist, dpi=300)
        plt.close(fig)
        print(f"[Saved] {fig_whist}")

    print(f"[Saved] {fig_scatter}")
    print(f"[Saved] {fig_hist}")

    # ----------------------------------------------------------------------
    # Holdout validation (galaxy-level K-fold) with refit g_star on TRAIN
    # ----------------------------------------------------------------------
    galaxies = sorted(set(df_pts.loc[used, "Galaxy"].astype(str).tolist()))
    folds = kfold_split_galaxies(galaxies, k=int(args.k), seed=int(args.seed))

    hold_rows = []
    grid_rows = []

    for fold_idx, test_gals in enumerate(folds):
        train_mask = used & (~df_pts["Galaxy"].isin(test_gals))
        test_mask = used & (df_pts["Galaxy"].isin(test_gals))

        train = df_pts.loc[train_mask].copy()
        test = df_pts.loc[test_mask].copy()

        if len(train) < 200 or len(test) < 80:
            continue

        # Fit g_star on TRAIN
        g_star_fit, loss_train = fit_gstar_grid_extra(
            gN=train["gN_m_s2"].to_numpy(dtype=float),
            aobs=train["aobs_m_s2"].to_numpy(dtype=float),
            floor=extra_floor,
            p=p,
            gstar_min=float(args.gstar_min),
            gstar_max=float(args.gstar_max),
            grid_n=int(args.grid_n),
        )

        # Evaluate TRAIN residuals
        extra_req_tr = train["aobs_m_s2"].to_numpy(dtype=float) - train["gN_m_s2"].to_numpy(dtype=float)
        nu_tr = nu_pred_from_gN(train["gN_m_s2"].to_numpy(dtype=float), g_star=g_star_fit, p=p)
        extra_pr_tr = train["gN_m_s2"].to_numpy(dtype=float) * (nu_tr - 1.0)
        res_tr, used_tr = compute_extra_residuals(extra_req_tr, extra_pr_tr, floor=extra_floor)
        res_tr = res_tr[used_tr]
        w_tr = train["weight_rel_invvar"].to_numpy(dtype=float)[used_tr]

        train_med = float(np.median(np.abs(res_tr)))
        train_wmed = weighted_median(np.abs(res_tr), w_tr)

        # Evaluate TEST residuals
        extra_req_te = test["aobs_m_s2"].to_numpy(dtype=float) - test["gN_m_s2"].to_numpy(dtype=float)
        nu_te = nu_pred_from_gN(test["gN_m_s2"].to_numpy(dtype=float), g_star=g_star_fit, p=p)
        extra_pr_te = test["gN_m_s2"].to_numpy(dtype=float) * (nu_te - 1.0)
        res_te, used_te = compute_extra_residuals(extra_req_te, extra_pr_te, floor=extra_floor)
        res_te = res_te[used_te]
        w_te = test["weight_rel_invvar"].to_numpy(dtype=float)[used_te]

        test_med = float(np.median(np.abs(res_te)))
        test_wmed = weighted_median(np.abs(res_te), w_te)

        hold_rows.append({
            "fold": fold_idx,
            "p_fixed": p,
            "extra_floor": extra_floor,
            "g_star_fit_m_s2": g_star_fit,
            "train_median_abs_log10": train_med,
            "train_weighted_median_abs_log10": train_wmed,
            "test_median_abs_log10": test_med,
            "test_weighted_median_abs_log10": test_wmed,
            "n_train_points": int(len(train)),
            "n_test_points": int(len(test)),
            "n_test_galaxies": int(len(test_gals)),
        })

        grid_rows.append({
            "fold": fold_idx,
            "g_star_fit_m_s2": g_star_fit,
            "grid_loss_train_median_abs": loss_train
        })

    hold_df = pd.DataFrame(hold_rows)
    hold_csv = os.path.join(DATA_DIR, "holdout_metrics_extra.csv")
    hold_df.to_csv(hold_csv, index=False)

    summary = pd.DataFrame([{
        "k": int(args.k),
        "seed": int(args.seed),
        "p_fixed": p,
        "extra_floor": extra_floor,
        "test_median_abs_log10_median": float(np.median(hold_df["test_median_abs_log10"].values)) if len(hold_df) else np.nan,
        "test_weighted_median_abs_log10_median": float(np.median(hold_df["test_weighted_median_abs_log10"].values)) if len(hold_df) else np.nan,
        "n_folds_used": int(len(hold_df)),
    }])
    sum_csv = os.path.join(DATA_DIR, "holdout_summary_extra.csv")
    summary.to_csv(sum_csv, index=False)

    # Boxplots
    if len(hold_df) > 0:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(hold_df["test_median_abs_log10"].values, vert=True)
        ax.set_title("Holdout (galaxy-level K-fold): test median |log10(pred/req)| (unweighted)")
        ax.set_ylabel("Median |log10(extra_pred/extra_req)| (test)")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "holdout_boxplot_extra_unweighted.png"), dpi=300)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.boxplot(hold_df["test_weighted_median_abs_log10"].values, vert=True)
        ax.set_title("Holdout (galaxy-level K-fold): test median |log10(pred/req)| (weighted)")
        ax.set_ylabel("Weighted median |log10(extra_pred/extra_req)| (test)")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "holdout_boxplot_extra_weighted.png"), dpi=300)
        plt.close(fig)

    # Plot the train grid-fit g_star values (sanity)
    if len(grid_rows) > 0:
        df_grid = pd.DataFrame(grid_rows)
        df_grid.to_csv(os.path.join(DATA_DIR, "gstar_grid_search_extra.csv"), index=False)

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df_grid["fold"].values, df_grid["g_star_fit_m_s2"].values, marker="o")
        ax.set_title("Fold-wise fitted g_star on TRAIN (extra-loss)")
        ax.set_xlabel("Fold")
        ax.set_ylabel("g_star_fit (m/s^2)")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "gstar_grid_extra_trainloss.png"), dpi=300)
        plt.close(fig)

    print(f"[Saved] {hold_csv}")
    print(f"[Saved] {sum_csv}")

    # ----------------------------------------------------------------------
    # Run metadata (audit)
    # ----------------------------------------------------------------------
    meta = {
        "script": SCRIPT_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": {
            "validator_py": args.validator_py,
            "results_csv": results_csv,
            "quality": args.quality,
            "require_opt_success": bool(args.require_opt_success),
        },
        "model": {"g_star_m_s2": float(g_star), "p_fixed": p},
        "filters": {
            "r_min_kpc": args.r_min_kpc,
            "min_vobs_km_s": args.min_vobs,
            "min_points": args.min_points,
        },
        "extra": {
            "extra_floor_m_s2": extra_floor,
            "weighted_hist": bool(args.use_weighted_hist),
        },
        "holdout": {
            "k": int(args.k),
            "seed": int(args.seed),
            "gstar_grid_min": float(args.gstar_min),
            "gstar_grid_max": float(args.gstar_max),
            "grid_n": int(args.grid_n),
        },
        "counts": {
            "galaxies_used_csv": int(len(df_use)),
            "galaxies_with_points": int(df_pts["Galaxy"].nunique()),
            "points_total": int(len(df_pts)),
            "points_used_floor": int(np.sum(used)),
            "skipped_galaxies": int(len(df_skip)),
        },
        "summary_metrics": {
            "median_abs_log10_unweighted": med_abs,
            "median_abs_log10_weighted": wmed_abs,
        },
    }
    with open(os.path.join(BASE_OUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Saved] {os.path.join(BASE_OUT_DIR, 'run_metadata.json')}")
    print("[Done] Robust DM-reverse extra test complete (floor + weighted + holdout).")


if __name__ == "__main__":
    main()
