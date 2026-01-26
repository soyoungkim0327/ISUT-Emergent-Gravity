# -*- coding: utf-8 -*-
"""
Clock-rate proxy robustness (All65): auto-find points CSV + holdout + sensitivity
================================================================================

Fixes
-----
- Automatically locates All65_clockrate_proxy_points.csv under the current project structure,
  instead of assuming it lives in the current directory.
- Chooses the most recently modified matching file if multiple candidates exist.

Purpose
-------
Tightens reviewer-facing claims for the proxy-based clock-rate test by adding:

(1) Galaxy-level K-fold holdout validation:
    - Fit global g_star on train galaxies (grid search)
    - Evaluate on held-out galaxies (no per-galaxy refit)

(2) Sensitivity analysis over transition sharpness p:
    - p in {0.5, 1.0, 2.0} by default

(3) Sensitivity analysis over inner-radius cut r_min_kpc:
    - r_min in {0.2, 0.5, 1.0} kpc by default

Input
-----
Point-level CSV produced by 1-3.clockrate_proxy_predict_all65.py:
  All65_clockrate_proxy_points.csv
Required columns (case-insensitive):
  Galaxy, r_kpc, gN_m_s2, nu_obs

Outputs
-------
./<SCRIPT_NAME>/
  figures/
    holdout_boxplot.png
    p_sensitivity.png
    rmin_sensitivity.png
  data/
    holdout_metrics.csv
    holdout_summary.csv
    p_sensitivity.csv
    rmin_sensitivity.csv
  run_metadata.json
"""

from __future__ import annotations

import os
import json
import time
import argparse
from typing import List, Tuple

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
FIG_DIR = os.path.join(BASE_OUT_DIR, "figures")
DATA_DIR = os.path.join(BASE_OUT_DIR, "data")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"[System] {SCRIPT_NAME} initialized")
print(f"[Info] Current Dir : {CURRENT_DIR}")
print(f"[Info] Output Base : {BASE_OUT_DIR}")


# ==============================================================================
# [1] Auto discovery for points CSV
# ==============================================================================
POINTS_FILENAME = "All65_clockrate_proxy_points.csv"


def _walk_find(root: str, filename: str) -> List[str]:
    hits = []
    for r, _, files in os.walk(root):
        if filename in files:
            hits.append(os.path.join(r, filename))
    return hits


def find_points_csv(explicit: str | None, search_root: str) -> Tuple[str, List[str]]:
    """
    Locate the points CSV.
    Priority:
      1) --points-csv (explicit)
      2) CURRENT_DIR/All65_clockrate_proxy_points.csv
      3) recursive search under search_root (default: CURRENT_DIR)
         - chooses most recently modified if multiple candidates found
    Returns: (chosen_path, candidates_list)
    """
    if explicit:
        if os.path.exists(explicit):
            return explicit, [explicit]
        raise FileNotFoundError(f"--points-csv was provided but not found: {explicit}")

    direct = os.path.join(CURRENT_DIR, POINTS_FILENAME)
    if os.path.exists(direct):
        return direct, [direct]

    # search recursively
    candidates = _walk_find(search_root, POINTS_FILENAME)
    if not candidates:
        raise FileNotFoundError(
            f"points CSV not found under search_root={search_root}. "
            f"Either run 1-3 script first or pass --points-csv explicitly."
        )

    # choose newest
    candidates_sorted = sorted(candidates, key=lambda p: os.path.getmtime(p), reverse=True)
    chosen = candidates_sorted[0]
    return chosen, candidates_sorted


# ==============================================================================
# [2] Column normalization (case-insensitive robustness)
# ==============================================================================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    def pick(name: str) -> str | None:
        return lower_map.get(name.lower(), None)

    required = ["galaxy", "r_kpc", "gn_m_s2", "nu_obs"]
    missing = []
    rename_map = {}

    for req in required:
        col = pick(req)
        if col is None:
            missing.append(req)
        else:
            rename_map[col] = req  # normalize to lowercase canonical names

    if missing:
        raise ValueError(f"Missing required columns (case-insensitive): {missing}. Columns={cols}")

    df2 = df.rename(columns=rename_map).copy()
    # numeric coercion
    df2["r_kpc"] = pd.to_numeric(df2["r_kpc"], errors="coerce")
    df2["gn_m_s2"] = pd.to_numeric(df2["gn_m_s2"], errors="coerce")
    df2["nu_obs"] = pd.to_numeric(df2["nu_obs"], errors="coerce")
    df2["galaxy"] = df2["galaxy"].astype(str)

    df2 = df2.dropna(subset=["r_kpc", "gn_m_s2", "nu_obs"]).copy()
    return df2


# ==============================================================================
# [3] Model + metrics
# ==============================================================================
def nu_pred_from_gN(gN_si: np.ndarray, g_star: float, p: float, g_floor: float = 1e-30) -> np.ndarray:
    g = np.maximum(gN_si, g_floor)
    return (1.0 + (g_star / g) ** p) ** (1.0 / (2.0 * p))


def median_abs_log10_residual(nu_obs: np.ndarray, nu_pred: np.ndarray, eps: float = 1e-30) -> float:
    x = np.maximum(nu_obs, eps)
    y = np.maximum(nu_pred, eps)
    res = np.log10(y / x)
    return float(np.median(np.abs(res)))


def percentile_abs_log10_residual(nu_obs: np.ndarray, nu_pred: np.ndarray, q: float, eps: float = 1e-30) -> float:
    x = np.maximum(nu_obs, eps)
    y = np.maximum(nu_pred, eps)
    res = np.abs(np.log10(y / x))
    return float(np.percentile(res, q))


def fit_gstar_grid(gN: np.ndarray, nu_obs: np.ndarray, p: float,
                   gstar_min: float, gstar_max: float, grid_n: int) -> Tuple[float, float]:
    g_grid = np.logspace(np.log10(gstar_min), np.log10(gstar_max), int(grid_n))
    losses = []
    for g_star in g_grid:
        nu_pred = nu_pred_from_gN(gN, g_star=g_star, p=p)
        losses.append(median_abs_log10_residual(nu_obs, nu_pred))
    losses = np.array(losses, dtype=float)
    best_idx = int(np.nanargmin(losses))
    return float(g_grid[best_idx]), float(losses[best_idx])


# ==============================================================================
# [4] K-fold split on galaxy IDs (deterministic)
# ==============================================================================
def kfold_split_galaxies(galaxies: List[str], k: int, seed: int) -> List[List[str]]:
    rng = np.random.default_rng(seed)
    gal = np.array(sorted(set(galaxies)))
    rng.shuffle(gal)
    folds = np.array_split(gal, k)
    return [f.tolist() for f in folds]


# ==============================================================================
# [5] Main
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Holdout + sensitivity for clock-rate proxy test (auto-find CSV).")

    parser.add_argument("--points-csv", type=str, default=None,
                        help="Point-level CSV from 1-3 script. If omitted, script searches automatically.")
    parser.add_argument("--search-root", type=str, default=CURRENT_DIR,
                        help="Root directory to recursively search for All65_clockrate_proxy_points.csv.")

    parser.add_argument("--seed", type=int, default=7, help="Random seed for fold split.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (galaxy-level).")

    parser.add_argument("--p-list", type=str, default="0.5,1.0,2.0", help="Comma-separated p values.")
    parser.add_argument("--rmin-list", type=str, default="0.2,0.5,1.0", help="Comma-separated r_min_kpc values.")

    parser.add_argument("--gstar-min", type=float, default=1e-12, help="Min g_star (m/s^2).")
    parser.add_argument("--gstar-max", type=float, default=1e-8, help="Max g_star (m/s^2).")
    parser.add_argument("--grid-n", type=int, default=240, help="Grid points for g_star search.")

    args = parser.parse_args()

    # --------------------------------------------------------------------------
    # Locate points CSV automatically
    # --------------------------------------------------------------------------
    points_csv, candidates = find_points_csv(args.points_csv, args.search_root)
    print(f"[Pick] points_csv = {points_csv}")
    if len(candidates) > 1:
        print("[Scan] candidates (newest first):")
        for p in candidates[:12]:
            print(f"  - {p}")

    # --------------------------------------------------------------------------
    # Load + normalize columns
    # --------------------------------------------------------------------------
    df_raw = pd.read_csv(points_csv)
    df = normalize_columns(df_raw)

    print(f"[Load] rows={len(df)}, galaxies={df['galaxy'].nunique()} from {points_csv}")

    # parse lists
    p_list = [float(x.strip()) for x in args.p_list.split(",") if x.strip()]
    rmin_list = [float(x.strip()) for x in args.rmin_list.split(",") if x.strip()]

    # --------------------------------------------------------------------------
    # A) Holdout validation (galaxy-level K-fold)
    # --------------------------------------------------------------------------
    holdout_rows = []
    galaxies = df["galaxy"].astype(str).tolist()
    folds = kfold_split_galaxies(galaxies, k=int(args.k), seed=int(args.seed))

    # Use baseline p=1.0 for holdout if present, else first p
    p_hold = 1.0 if 1.0 in p_list else p_list[0]

    for fold_idx, test_gals in enumerate(folds):
        train_mask = ~df["galaxy"].isin(test_gals)
        test_mask = df["galaxy"].isin(test_gals)

        train = df.loc[train_mask].copy()
        test = df.loc[test_mask].copy()

        # Fit g_star on train only
        g_star_best, loss_train = fit_gstar_grid(
            train["gn_m_s2"].to_numpy(),
            train["nu_obs"].to_numpy(),
            p=p_hold,
            gstar_min=args.gstar_min, gstar_max=args.gstar_max, grid_n=args.grid_n
        )

        # Evaluate on test
        nu_pred_test = nu_pred_from_gN(test["gn_m_s2"].to_numpy(), g_star=g_star_best, p=p_hold)
        loss_test = median_abs_log10_residual(test["nu_obs"].to_numpy(), nu_pred_test)

        holdout_rows.append({
            "fold": fold_idx,
            "p_fixed": p_hold,
            "g_star_best_m_s2": g_star_best,
            "train_median_abs_log10": loss_train,
            "test_median_abs_log10": loss_test,
            "n_train_points": int(len(train)),
            "n_test_points": int(len(test)),
            "n_test_galaxies": int(len(set(test_gals))),
        })

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_csv = os.path.join(DATA_DIR, "holdout_metrics.csv")
    holdout_df.to_csv(holdout_csv, index=False)

    holdout_summary = pd.DataFrame([{
        "k": int(args.k),
        "seed": int(args.seed),
        "p_fixed": p_hold,
        "test_median_abs_log10_median": float(np.median(holdout_df["test_median_abs_log10"].values)),
        "test_median_abs_log10_mean": float(np.mean(holdout_df["test_median_abs_log10"].values)),
    }])
    holdout_summary_csv = os.path.join(DATA_DIR, "holdout_summary.csv")
    holdout_summary.to_csv(holdout_summary_csv, index=False)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(holdout_df["test_median_abs_log10"].values, vert=True)
    ax.set_title("Holdout validation (galaxy-level K-fold): test median |log10(pred/obs)|")
    ax.set_ylabel("Median |log10(nu_pred/nu_obs)| (test)")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "holdout_boxplot.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # B) p sensitivity (fit on full set)
    # --------------------------------------------------------------------------
    p_rows = []
    for p in p_list:
        g_star_best, loss = fit_gstar_grid(
            df["gn_m_s2"].to_numpy(),
            df["nu_obs"].to_numpy(),
            p=p,
            gstar_min=args.gstar_min, gstar_max=args.gstar_max, grid_n=args.grid_n
        )
        nu_pred = nu_pred_from_gN(df["gn_m_s2"].to_numpy(), g_star=g_star_best, p=p)

        p_rows.append({
            "p": p,
            "g_star_best_m_s2": g_star_best,
            "median_abs_log10": loss,
            "p90_abs_log10": percentile_abs_log10_residual(df["nu_obs"].to_numpy(), nu_pred, 90.0),
            "p95_abs_log10": percentile_abs_log10_residual(df["nu_obs"].to_numpy(), nu_pred, 95.0),
            "p99_abs_log10": percentile_abs_log10_residual(df["nu_obs"].to_numpy(), nu_pred, 99.0),
        })

    p_df = pd.DataFrame(p_rows).sort_values("p")
    p_csv = os.path.join(DATA_DIR, "p_sensitivity.csv")
    p_df.to_csv(p_csv, index=False)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(p_df["p"].values, p_df["median_abs_log10"].values, marker="o")
    ax.set_title("Sensitivity to p: median |log10(pred/obs)|")
    ax.set_xlabel("p (transition sharpness)")
    ax.set_ylabel("Median |log10(nu_pred/nu_obs)|")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "p_sensitivity.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # C) r_min sensitivity (fit on filtered set, baseline p_hold)
    # --------------------------------------------------------------------------
    r_rows = []
    for rmin in rmin_list:
        sub = df[df["r_kpc"] >= rmin].copy()
        if len(sub) < 200:
            continue

        g_star_best, loss = fit_gstar_grid(
            sub["gn_m_s2"].to_numpy(),
            sub["nu_obs"].to_numpy(),
            p=p_hold,
            gstar_min=args.gstar_min, gstar_max=args.gstar_max, grid_n=args.grid_n
        )
        nu_pred = nu_pred_from_gN(sub["gn_m_s2"].to_numpy(), g_star=g_star_best, p=p_hold)

        r_rows.append({
            "r_min_kpc": rmin,
            "p_fixed": p_hold,
            "g_star_best_m_s2": g_star_best,
            "median_abs_log10": loss,
            "p90_abs_log10": percentile_abs_log10_residual(sub["nu_obs"].to_numpy(), nu_pred, 90.0),
            "N_points": int(len(sub)),
        })

    r_df = pd.DataFrame(r_rows).sort_values("r_min_kpc")
    r_csv = os.path.join(DATA_DIR, "rmin_sensitivity.csv")
    r_df.to_csv(r_csv, index=False)

    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(r_df["r_min_kpc"].values, r_df["median_abs_log10"].values, marker="o")
    ax.set_title("Sensitivity to inner-radius cut: median |log10(pred/obs)|")
    ax.set_xlabel("r_min (kpc)")
    ax.set_ylabel("Median |log10(nu_pred/nu_obs)|")
    ax.grid(True, linestyle=":")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "rmin_sensitivity.png"), dpi=300)
    plt.close(fig)

    # --------------------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------------------
    meta = {
        "script": SCRIPT_NAME,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "points_csv": points_csv,
        "search_root": args.search_root,
        "candidates_found": candidates[:30],
        "kfold": {"k": int(args.k), "seed": int(args.seed), "p_fixed": p_hold},
        "p_list": p_list,
        "rmin_list": rmin_list,
        "grid": {"gstar_min": args.gstar_min, "gstar_max": args.gstar_max, "grid_n": int(args.grid_n)},
        "counts": {"galaxies": int(df["galaxy"].nunique()), "points": int(len(df))}
    }
    with open(os.path.join(BASE_OUT_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[Saved] {holdout_csv}")
    print(f"[Saved] {holdout_summary_csv}")
    print(f"[Saved] {p_csv}")
    print(f"[Saved] {r_csv}")
    print(f"[Saved] figures under {FIG_DIR}")
    print("[Done] Robustness artifacts generated.")


if __name__ == "__main__":
    main()
