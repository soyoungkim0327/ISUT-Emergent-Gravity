# -*- coding: utf-8 -*-
"""
DM-reverse vs ISUT local verification (All65): dual reconstruction test
======================================================================

 purpose
-----------------------
This script implements a "dual reconstruction" check suggested in the reverse-engineering plan:
1) Infer the required extra acceleration under a Newtonian+DM interpretation:
       g_DM_req(r) = a_obs(r) - g_bar(r)
   where a_obs = V_obs^2 / r and g_bar = V_bar^2 / r.

2) Predict the extra acceleration from an ISUT clock-rate proxy model:
       nu_pred(gN) = (1 + (g_star/gN)^p)^(1/(2p))
       g_ISUT_extra(r) = gN(r) * (nu_pred(gN(r)) - 1)

3) Test pointwise agreement (radius-by-radius), including "bump" structure:
   - global scatter and residual distribution
   - per-galaxy residual summaries
   - bump correlation after robust smoothing (rolling median)

This is a rotation-curve (nonrelativistic) dynamical test, not a full metric/lensing completion.

Inputs
------
A) Validator data loader:
   isut_300_valid_a0_constancy.py (loaded dynamically)
   get_data(gal_name, return_reason=True, allow_download=False, download_log=None)
   returns:
     R_kpc, Vobs, Verr, Vgas, Vdisk, Vbul

B) All65 fit summary (for baryonic decomposition via mass-to-light ratios):
   All65_Full_Results.csv (or _USED.csv)
   required columns: Galaxy, Ups_Disk, Ups_Bulge
   optional columns: Quality, Opt_Success, Best_a0_SI

C) Clock-rate proxy parameters:
   - g_star (SI m/s^2) and p
   If not provided, the script tries to auto-detect from the latest 1-3 run_metadata.json
   that contains "model.g_star_best_m_s2". Otherwise defaults to user-specified --g-star.

Outputs (audit-ready)
---------------------
./<SCRIPT_NAME>/
  All65/
    figures/
      extra_scatter.png
      extra_residual_hist.png
      bump_corr_hist.png
      bump_scatter.png
      overlay_top_bumpy_<rank>_<galaxy>.png  (top-K)
    data/
      points_local_extra.csv
      galaxy_summary_extra.csv
      galaxy_bumpiness_ranking.csv
      skipped_galaxies.csv
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
# [0] Path conventions (your style)
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
ACCEL_CONV = 1e6 / 3.085677581e19  # 3.24078e-14


# ==============================================================================
# [2] Dynamic import of validator (file has dots)
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


# ==============================================================================
# [3] Proxy model: nu_pred from gN, alpha_pred = sqrt(nu_pred)
# ==============================================================================
def nu_pred_from_gN(gN_si: np.ndarray, g_star: float, p: float, g_floor: float = 1e-30) -> np.ndarray:
    g = np.maximum(gN_si, g_floor)
    return (1.0 + (g_star / g) ** p) ** (1.0 / (2.0 * p))


def median_abs_log10_ratio(a: np.ndarray, b: np.ndarray, eps: float = 1e-30) -> float:
    x = np.maximum(a, eps)
    y = np.maximum(b, eps)
    return float(np.median(np.abs(np.log10(x / y))))


# ==============================================================================
# [4] Auto-detect g_star from latest 1-3 run metadata (optional)
# ==============================================================================
def _walk_find(root: str, filename: str) -> List[str]:
    hits = []
    for r, _, files in os.walk(root):
        if filename in files:
            hits.append(os.path.join(r, filename))
    return hits


def autodetect_gstar(search_root: str) -> Optional[float]:
    """
    Search for run_metadata.json containing model.g_star_best_m_s2 and pick newest.
    """
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
# [5] Smoothing + bump extraction (rolling median in log-space)
# ==============================================================================
def rolling_median_smooth(x: np.ndarray, window: int) -> np.ndarray:
    s = pd.Series(x)
    sm = s.rolling(window=window, center=True, min_periods=max(3, window // 3)).median()
    sm = sm.interpolate().bfill().ffill()
    return sm.to_numpy(dtype=float)


def bump_component_log10(pos_vals: np.ndarray, window: int, eps: float = 1e-30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute bump component in log10-space:
      logx = log10(max(pos_vals, eps))
      smooth = rolling_median(logx)
      bump = logx - smooth
    Returns (logx, bump).
    """
    logx = np.log10(np.maximum(pos_vals, eps))
    smooth = rolling_median_smooth(logx, window=window)
    bump = logx - smooth
    return logx, bump


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = np.sqrt(np.sum(x * x)) * np.sqrt(np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


# ==============================================================================
# [6] Locate All65 results CSV
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
# [7] Main
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="DM-reverse vs ISUT local (radius-by-radius) verification on All65.")
    parser.add_argument("--validator-py", type=str, default=os.path.join(CURRENT_DIR, "isut_300_valid_a0_constancy.py"))
    parser.add_argument("--results-csv", type=str, default=None)
    parser.add_argument("--allow-download", action="store_true", help="Allow downloading missing rotmod files (not recommended).")

    # filters
    parser.add_argument("--quality", type=str, default="Good", help='Use only rows with Quality == this value; "ALL" disables.')
    parser.add_argument("--require-opt-success", action="store_true")

    # model params (can be autodetected)
    parser.add_argument("--g-star", type=float, default=None, help="Global transition scale g_star (m/s^2). If omitted, auto-detect from latest 1-3 metadata.")
    parser.add_argument("--p", type=float, default=1.0, help="Transition sharpness p (fixed).")
    parser.add_argument("--search-root", type=str, default=CURRENT_DIR, help="Where to search for 1-3 run_metadata.json for g_star autodetect.")

    # point filters
    parser.add_argument("--r-min-kpc", type=float, default=0.5, help="Exclude inner radii below this (kpc).")
    parser.add_argument("--min-vobs", type=float, default=5.0, help="Minimum Vobs (km/s).")
    parser.add_argument("--min-points", type=int, default=8, help="Minimum points per galaxy after filtering.")

    # bump analysis
    parser.add_argument("--bump-window", type=int, default=11, help="Rolling median window (odd recommended).")
    parser.add_argument("--top-k-overlays", type=int, default=8, help="Save overlay plots for top-K bumpiest galaxies.")
    args = parser.parse_args()

    # Load validator
    validator = load_validator_module(args.validator_py)
    if not hasattr(validator, "get_data"):
        raise AttributeError("Validator missing get_data().")
    get_data = validator.get_data  # type: ignore

    # Load results CSV (ML parameters)
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

    print(f"[Model] g_star={g_star:.4e} m/s^2, p={p:.3g}")
    print(f"[Input] results_csv={results_csv}, galaxies={len(df_use)}")

    point_rows = []
    skipped = []

    for _, row in df_use.iterrows():
        gal = str(row["Galaxy"])
        ups_d = float(row["Ups_Disk"])
        ups_b = float(row["Ups_Bulge"])

        data, reason = get_data(
            gal,
            return_reason=True,
            allow_download=bool(args.allow_download),
            download_log=None
        )
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
        Vgas = Vgas[m] if Vgas is not None else np.zeros_like(Vobs)
        Vdisk = Vdisk[m] if Vdisk is not None else np.zeros_like(Vobs)
        Vbul = Vbul[m] if Vbul is not None else np.zeros_like(Vobs)

        if len(R_kpc) < args.min_points:
            skipped.append({"Galaxy": gal, "Reason": "too_few_points_after_filter"})
            continue

        # baryonic decomposition
        Vbar2 = (Vgas ** 2) + (ups_d * (Vdisk ** 2)) + (ups_b * (Vbul ** 2))
        R_safe = np.maximum(R_kpc, 1e-3)

        gN_si = (np.abs(Vbar2) / R_safe) * ACCEL_CONV
        aobs_si = ((Vobs ** 2) / R_safe) * ACCEL_CONV

        valid = np.isfinite(gN_si) & np.isfinite(aobs_si) & (gN_si > 0) & (aobs_si > 0)
        gN_si = gN_si[valid]
        aobs_si = aobs_si[valid]
        R_kpc = R_kpc[valid]

        if len(gN_si) < args.min_points:
            skipped.append({"Galaxy": gal, "Reason": "invalid_gN_or_aobs"})
            continue

        # DM reverse engineering: required extra acceleration
        extra_req = aobs_si - gN_si

        # Keep only positive extra_req for log-ratio tests
        pos = np.isfinite(extra_req) & (extra_req > 0)
        gN_si = gN_si[pos]
        aobs_si = aobs_si[pos]
        extra_req = extra_req[pos]
        R_kpc = R_kpc[pos]

        if len(extra_req) < args.min_points:
            skipped.append({"Galaxy": gal, "Reason": "nonpositive_extra_req"})
            continue

        nu_pred = nu_pred_from_gN(gN_si, g_star=g_star, p=p)
        extra_isut = gN_si * (nu_pred - 1.0)

        # residuals on extra acceleration
        eps = 1e-30
        res_log10 = np.log10(np.maximum(extra_isut, eps) / np.maximum(extra_req, eps))

        for i in range(len(R_kpc)):
            point_rows.append({
                "Galaxy": gal,
                "r_kpc": float(R_kpc[i]),
                "gN_m_s2": float(gN_si[i]),
                "aobs_m_s2": float(aobs_si[i]),
                "extra_req_m_s2": float(extra_req[i]),
                "extra_isut_m_s2": float(extra_isut[i]),
                "nu_pred": float(nu_pred[i]),
                "residual_log10_extra_isut_over_req": float(res_log10[i]),
                "Ups_Disk": ups_d,
                "Ups_Bulge": ups_b,
                "Best_a0_SI": float(row["Best_a0_SI"]) if "Best_a0_SI" in row else np.nan,
            })

    if not point_rows:
        raise RuntimeError("No usable galaxies/points after filtering; check inputs.")

    df_pts = pd.DataFrame(point_rows)
    df_skip = pd.DataFrame(skipped)

    # Save skipped list
    skip_csv = os.path.join(DATA_DIR, "skipped_galaxies.csv")
    df_skip.to_csv(skip_csv, index=False)

    # Save points CSV
    pts_csv = os.path.join(DATA_DIR, "points_local_extra.csv")
    df_pts.to_csv(pts_csv, index=False)

    print(f"[Data] points={len(df_pts)} galaxies={df_pts['Galaxy'].nunique()} skipped={len(df_skip)}")
    print(f"[Saved] {pts_csv}")
    print(f"[Saved] {skip_csv}")

    # Global figures: scatter and residual hist
    x = np.maximum(df_pts["extra_req_m_s2"].to_numpy(), 1e-30)
    y = np.maximum(df_pts["extra_isut_m_s2"].to_numpy(), 1e-30)
    res = df_pts["residual_log10_extra_isut_over_req"].to_numpy()

    # scatter
    fig_scatter = os.path.join(FIG_DIR, "extra_scatter.png")
    fig = plt.figure(figsize=(8.5, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=8, alpha=0.35)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$g_{\rm DM,req}=a_{\rm obs}-g_{\rm bar}$ (m/s$^2$)")
    ax.set_ylabel(r"$g_{\rm ISUT,extra}=g_N(\nu_{\rm pred}-1)$ (m/s$^2$)")
    ax.set_title("All65 local test: DM-required extra vs ISUT-predicted extra")

    lim_min = min(float(np.min(x)), float(np.min(y)))
    lim_max = max(float(np.max(x)), float(np.max(y)))
    ax.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--", linewidth=1.2)

    med_abs = float(np.median(np.abs(res)))
    ax.text(
        0.02, 0.98,
        f"g*={g_star:.3e} m/s^2, p={p:.2g}\n"
        f"median|log10(pred/req)|={med_abs:.4f}\n"
        f"points={len(df_pts)}, galaxies={df_pts['Galaxy'].nunique()}",
        transform=ax.transAxes, va="top"
    )
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout()
    fig.savefig(fig_scatter, dpi=300)
    plt.close(fig)

    # residual hist
    fig_hist = os.path.join(FIG_DIR, "extra_residual_hist.png")
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(res, bins=50)
    ax.set_xlabel(r"$\log_{10}(g_{\rm ISUT,extra}/g_{\rm DM,req})$")
    ax.set_ylabel("Count")
    ax.set_title("Residual distribution (All65 points): extra acceleration")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(fig_hist, dpi=300)
    plt.close(fig)

    print(f"[Saved] {fig_scatter}")
    print(f"[Saved] {fig_hist}")

    # ----------------------------------------------------------------------
    # Bump analysis per galaxy (log-space rolling median)
    # ----------------------------------------------------------------------
    bump_window = int(args.bump_window)
    if bump_window < 5:
        bump_window = 5
    if bump_window % 2 == 0:
        bump_window += 1  # odd recommended

    gal_rows = []
    bump_rows = []

    for gal, sub in df_pts.groupby("Galaxy"):
        sub = sub.sort_values("r_kpc").copy()

        req = sub["extra_req_m_s2"].to_numpy()
        pred = sub["extra_isut_m_s2"].to_numpy()

        log_req, bump_req = bump_component_log10(req, window=bump_window)
        log_pred, bump_pred = bump_component_log10(pred, window=bump_window)

        r_bump = pearson_r(bump_req, bump_pred)
        bump_amp = float(np.std(bump_req))  # "bumpiness" metric (req side)

        gal_rows.append({
            "Galaxy": gal,
            "N_points": int(len(sub)),
            "median_abs_log10_extra_ratio": float(np.median(np.abs(sub["residual_log10_extra_isut_over_req"].to_numpy()))),
            "bump_corr_pearson": r_bump,
            "bumpiness_std_log10_req": bump_amp,
        })

        for i in range(len(sub)):
            bump_rows.append({
                "Galaxy": gal,
                "r_kpc": float(sub["r_kpc"].iloc[i]),
                "bump_req_log10": float(bump_req[i]),
                "bump_pred_log10": float(bump_pred[i]),
            })

    df_gal = pd.DataFrame(gal_rows).sort_values("bumpiness_std_log10_req", ascending=False)
    df_bump = pd.DataFrame(bump_rows)

    gal_csv = os.path.join(DATA_DIR, "galaxy_summary_extra.csv")
    bump_csv = os.path.join(DATA_DIR, "galaxy_bumpiness_ranking.csv")
    df_gal.to_csv(gal_csv, index=False)
    df_gal[["Galaxy", "bumpiness_std_log10_req", "bump_corr_pearson"]].to_csv(bump_csv, index=False)

    print(f"[Saved] {gal_csv}")
    print(f"[Saved] {bump_csv}")

    # bump correlation hist
    fig_bhist = os.path.join(FIG_DIR, "bump_corr_hist.png")
    fig = plt.figure(figsize=(8, 5.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(df_gal["bump_corr_pearson"].to_numpy(), bins=20)
    ax.set_xlabel("Pearson corr(bump_req, bump_pred)")
    ax.set_ylabel("Count")
    ax.set_title(f"Bump correlation across galaxies (window={bump_window})")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(fig_bhist, dpi=300)
    plt.close(fig)
    print(f"[Saved] {fig_bhist}")

    # bump scatter (all bump points)
    fig_bsc = os.path.join(FIG_DIR, "bump_scatter.png")
    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df_bump["bump_req_log10"].to_numpy(), df_bump["bump_pred_log10"].to_numpy(), s=5, alpha=0.25)
    ax.set_xlabel("Bump req (log10-space, high-pass)")
    ax.set_ylabel("Bump pred (log10-space, high-pass)")
    ax.set_title("Bump-level comparison (all galaxies)")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(fig_bsc, dpi=300)
    plt.close(fig)
    print(f"[Saved] {fig_bsc}")

    # ----------------------------------------------------------------------
    # Per-galaxy overlay for top-K bumpiest galaxies
    # ----------------------------------------------------------------------
    topk = int(args.top_k_overlays)
    top_gals = df_gal["Galaxy"].head(max(0, topk)).tolist()

    for rank, gal in enumerate(top_gals, start=1):
        sub = df_pts[df_pts["Galaxy"] == gal].sort_values("r_kpc").copy()

        r = sub["r_kpc"].to_numpy()
        req = sub["extra_req_m_s2"].to_numpy()
        pred = sub["extra_isut_m_s2"].to_numpy()

        fig_path = os.path.join(FIG_DIR, f"overlay_top_bumpy_{rank:02d}_{gal}.png")

        fig = plt.figure(figsize=(9, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(r, req, linewidth=2.2, label=r"$g_{\rm DM,req}(r)$")
        ax.plot(r, pred, linestyle="--", linewidth=1.8, label=r"$g_{\rm ISUT,extra}(r)$")
        ax.set_yscale("log")
        ax.set_xlabel("Radius (kpc)")
        ax.set_ylabel("Extra acceleration (m/s^2)")
        ax.set_title(f"Local extra-acceleration overlay: {gal}")
        ax.grid(True, which="both", linestyle=":")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    # ----------------------------------------------------------------------
    # Metadata
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
        "model": {"g_star_m_s2": g_star, "p_fixed": p},
        "filters": {
            "r_min_kpc": args.r_min_kpc,
            "min_vobs_km_s": args.min_vobs,
            "min_points": args.min_points,
        },
        "bump": {"window": bump_window, "top_k_overlays": topk},
        "counts": {
            "galaxies_used": int(len(df_use)),
            "galaxies_with_points": int(df_pts["Galaxy"].nunique()),
            "points": int(len(df_pts)),
            "skipped_galaxies": int(len(df_skip)),
        },
    }
    with open(os.path.join(BASE_OUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Saved] {os.path.join(BASE_OUT_DIR, 'run_metadata.json')}")
    print("[Done] Dual reconstruction artifacts generated (DM-reverse vs ISUT-extra, plus bump tests).")


if __name__ == "__main__":
    main()
