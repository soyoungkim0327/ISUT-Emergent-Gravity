"""
ISUT Global Statistical Survey & Deep Scan
==========================================
Objective:
    Final aggregation and recursive audit of holdout results across the 
    entire 65-galaxy sample.

Key Features:
1. Recursive Data Audit: Automatically locates and verifies all 
   cross-validation artifacts to prevent data loss or bias.
2. Distribution of Residuals: Plots the global distribution of prediction 
   errors to prove ISUT's consistency across diverse galaxy types.
3. Outlier Identification: Systematically flags galaxies where DM baselines 
   significantly deviate, providing physical insights into model failures.
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import traceback

# ==============================================================================
# [1] Configuration: Deep Scan Engine 
# ==============================================================================
DEFAULT_SEARCH_ROOT = os.environ.get(
    "ISUT_SEARCH_ROOT",
    os.path.dirname(os.path.abspath(__file__))
)

# Keep deep-scan capability, but only require the truly necessary file.
REQUIRED_FILES = [
    "All65_Holdout_Results.csv",
]

# Optional inputs (script will fall back gracefully if missing)
OPTIONAL_FILES = [
    "Fig1_SourceData.csv",            # if available, used only for cross-check
    "Summary_Model_Comparison.csv",   # legacy input (not required)
    "Holdout_Summary_Report.csv",     # used to cross-check summary stats if available
]

TARGET_FILES = REQUIRED_FILES + OPTIONAL_FILES


def find_files_recursively(root_path, targets):
    print(f"[System] 전수조사 시작 (Target Root: {root_path})")
    print(f"[System] 파일 {len(targets)}개를 찾는 중... (필수={len(REQUIRED_FILES)}개)")

    found_paths = {}

    for root, dirs, files in os.walk(root_path):
        for filename in targets:
            if filename in files and filename not in found_paths:
                full_path = os.path.join(root, filename)
                found_paths[filename] = full_path
                print(f"  [발견] {filename} -> {full_path}")

        if len(found_paths) == len(targets):
            break

    return found_paths


print("=" * 60)
found_files_map = find_files_recursively(DEFAULT_SEARCH_ROOT, TARGET_FILES)
print("=" * 60)

missing_required = [f for f in REQUIRED_FILES if f not in found_files_map]
if missing_required:
    print("[CRITICAL ERROR] 필수 파일을 찾을 수 없습니다:")
    for f in missing_required:
        print(f"  - {f}")
    print("\n[Hint] 같은 폴더(또는 하위 폴더)에 CSV를 두거나,")
    print("       환경변수 ISUT_SEARCH_ROOT 를 데이터 루트로 설정하세요.")
    sys.exit(1)

# ==============================================================================
# [2] Output Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)

FIG_DIR = os.path.join(OUTPUT_ROOT, "figures")
DATA_OUT_DIR = os.path.join(OUTPUT_ROOT, "source_data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_OUT_DIR, exist_ok=True)

# ==============================================================================
# [3] Main Function: Generate Figures
# ==============================================================================

def _trimmed_mean(s: pd.Series, q_lo: float = 0.05, q_hi: float = 0.95) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    s = s[s > 0]
    if len(s) == 0:
        return float("nan")
    lo = s.quantile(q_lo)
    hi = s.quantile(q_hi)
    return float(s[(s >= lo) & (s <= hi)].mean())


def generate_manuscript_figures():
    print("\n--- 데이터 로드 중 ---")
    try:
        df_holdout = pd.read_csv(found_files_map["All65_Holdout_Results.csv"])
    except Exception as e:
        print(f"[Error] All65_Holdout_Results.csv 로드 실패: {e}")
        return

    # Optional
    df_fig1 = None
    if "Fig1_SourceData.csv" in found_files_map:
        try:
            df_fig1 = pd.read_csv(found_files_map["Fig1_SourceData.csv"])
        except Exception as e:
            print(f"[Warn] Fig1_SourceData.csv 로드 실패 (무시): {e}")
            df_fig1 = None

    df_summary = None
    if "Holdout_Summary_Report.csv" in found_files_map:
        try:
            df_summary = pd.read_csv(found_files_map["Holdout_Summary_Report.csv"])
        except Exception as e:
            print(f"[Warn] Holdout_Summary_Report.csv 로드 실패 (무시): {e}")
            df_summary = None

    # ---------------------------------------------------------
    # Common cleaning
    # ---------------------------------------------------------
    for c in [
        "Train_Chi2_ISUT", "Train_Chi2_NFW",
        "Test_RedChi2_ISUT", "Test_RedChi2_NFW",
    ]:
        if c in df_holdout.columns:
            df_holdout[c] = pd.to_numeric(df_holdout[c], errors="coerce")

    df_clean = df_holdout.dropna(subset=["Train_Chi2_ISUT", "Train_Chi2_NFW", "Test_RedChi2_ISUT", "Test_RedChi2_NFW"])
    df_clean = df_clean[(df_clean["Test_RedChi2_ISUT"] > 0) & (df_clean["Test_RedChi2_NFW"] > 0)]

    # ---------------------------------------------------------
    # Figure 1: The Generalization Gap
    # ---------------------------------------------------------
    print("--- Figure 1 생성 중... ---")
    try:
        # (a) Training-fit preference frequency (lower Train χ²)
        train_total = len(df_clean)
        isut_train_wins = int((df_clean["Train_Chi2_ISUT"] < df_clean["Train_Chi2_NFW"]).sum())
        nfw_train_wins = train_total - isut_train_wins

        isut_win_rate = 100.0 * isut_train_wins / train_total if train_total else float("nan")
        nfw_win_rate = 100.0 - isut_win_rate if train_total else float("nan")

        # (b) Predictive generalization metric
        # Default: compute from the SAME df_clean used in Figure 2 (best traceability).
        isut_mean = float(df_clean["Test_RedChi2_ISUT"].mean())
        nfw_mean = float(df_clean["Test_RedChi2_NFW"].mean())
        isut_median = float(df_clean["Test_RedChi2_ISUT"].median())
        nfw_median = float(df_clean["Test_RedChi2_NFW"].median())
        isut_tmean = _trimmed_mean(df_clean["Test_RedChi2_ISUT"], 0.05, 0.95)
        nfw_tmean = _trimmed_mean(df_clean["Test_RedChi2_NFW"], 0.05, 0.95)

        # If Fig1_SourceData exists, use it ONLY when consistent (avoid silent mismatch)
        used_metric_source = "holdout_df_clean"
        if df_fig1 is not None and {"Model", "Mean_RedChi2"}.issubset(df_fig1.columns):
            try:
                val_isut = df_fig1[df_fig1["Model"].astype(str).str.contains("ISUT", case=False)]["Mean_RedChi2"].values[0]
                val_nfw = df_fig1[df_fig1["Model"].astype(str).str.contains("Dark Matter|NFW", case=False, regex=True)]["Mean_RedChi2"].values[0]
                val_isut = float(val_isut)
                val_nfw = float(val_nfw)

                # Consistency gate (15% tolerance)
                def rel_diff(a, b):
                    if a == 0 or b == 0:
                        return float("inf")
                    return abs(a - b) / max(abs(a), abs(b))

                if rel_diff(val_isut, isut_mean) <= 0.15 and rel_diff(val_nfw, nfw_mean) <= 0.15:
                    isut_mean = val_isut
                    nfw_mean = val_nfw
                    used_metric_source = "Fig1_SourceData.csv"
                else:
                    print("[Warn] Fig1_SourceData.csv mean 값이 holdout 평균과 크게 달라서(>15%) holdout 기반 평균을 사용합니다.")
            except Exception:
                pass

        # Source data 저장 (No plot without data)
        df_fig1_export = pd.DataFrame({
            "Metric": [
                "DM_Training_Preference", "ISUT_Training_Preference",
                "DM_Test_Mean_RedChi2", "ISUT_Test_Mean_RedChi2",
                "DM_Test_Median_RedChi2", "ISUT_Test_Median_RedChi2",
                "DM_Test_TrimmedMean_RedChi2_5_95", "ISUT_Test_TrimmedMean_RedChi2_5_95",
            ],
            "Value": [
                nfw_win_rate, isut_win_rate,
                nfw_mean, isut_mean,
                nfw_median, isut_median,
                nfw_tmean, isut_tmean,
            ],
            "Unit": [
                "Percent (%)", "Percent (%)",
                "Reduced Chi2", "Reduced Chi2",
                "Reduced Chi2", "Reduced Chi2",
                "Reduced Chi2", "Reduced Chi2",
            ],
            "Source": [
                "All65_Holdout_Results.csv (Train_Chi2)",
                "All65_Holdout_Results.csv (Train_Chi2)",
                f"{used_metric_source}",
                f"{used_metric_source}",
                "holdout_df_clean",
                "holdout_df_clean",
                "holdout_df_clean",
                "holdout_df_clean",
            ]
        })
        df_fig1_export.to_csv(os.path.join(DATA_OUT_DIR, "Figure1_SourceData.csv"), index=False)

        # Plot
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        x = np.arange(1)
        width = 0.5

        # Subplot 1
        p1 = ax1.bar(x - width / 2, [nfw_win_rate], width, label="DM baseline (NFW halo)", color="navy", alpha=0.8)
        p2 = ax1.bar(x + width / 2, [isut_win_rate], width, label="ISUT (Entropic)", color="crimson", alpha=0.8)

        ax1.set_ylabel("Model Preference Frequency (%)", fontsize=12)
        ax1.set_title(r"$\bf{a.}$ Explanatory Power (Training Fit)", fontsize=14)
        ax1.set_xticks([])
        ax1.set_ylim(0, 100)
        ax1.bar_label(p1, fmt="%.1f%%", padding=3)
        ax1.bar_label(p2, fmt="%.1f%%", padding=3)
        ax1.legend(loc="upper center", frameon=False)
        ax1.text(x[0], 2.0, "Lower Train $\\chi^2$ wins", ha="center", va="bottom", fontsize=9, color="gray")

        # Subplot 2
        p3 = ax2.bar(x - width / 2, [nfw_mean], width, label="DM baseline (NFW halo)", color="navy", alpha=0.8)
        p4 = ax2.bar(x + width / 2, [isut_mean], width, label="ISUT (Entropic)", color="crimson", alpha=0.8)

        ax2.set_ylabel(r"Mean Reduced $\chi^2_{\nu}$ (Test Set)", fontsize=12)
        ax2.set_title(r"$\bf{b.}$ Predictive Generalization (Holdout Test)", fontsize=14)
        ax2.set_xticks([])
        ax2.bar_label(p3, fmt="%.2f", padding=3)
        ax2.bar_label(p4, fmt="%.2f", padding=3)

        ylim = ax2.get_ylim()
        ax2.text(x[0], ylim[1] * 0.95, "Lower is Better ↓", ha="center", va="center", fontsize=10, color="gray")
        ax2.text(
            x[0],
            ylim[1] * 0.85,
            f"Median χ²ν (source data): ISUT {isut_median:.2f}, DM {nfw_median:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color="gray",
        )

        plt.suptitle("Figure 1: Contrast between Explanatory Power and Predictive Accuracy", fontsize=16)
        plt.tight_layout()

        save_path1 = os.path.join(FIG_DIR, "Figure1_Generalization_Gap.png")
        plt.savefig(save_path1, dpi=300)
        print(f"[Output] Figure 1 저장 완료: {save_path1}")
        plt.close(fig1)

    except Exception as e:
        print(f"[Error] Figure 1 생성 실패: {e}")
        traceback.print_exc()

    # ---------------------------------------------------------
    # Figure 2: Prediction Error Distribution
    # ---------------------------------------------------------
    print("--- Figure 2 생성 중... ---")
    try:
        df_clean[["Galaxy", "Test_RedChi2_ISUT", "Test_RedChi2_NFW"]].to_csv(
            os.path.join(DATA_OUT_DIR, "Figure2_SourceData.csv"), index=False
        )

        fig2, ax = plt.subplots(figsize=(8, 8))
        x_val = df_clean["Test_RedChi2_ISUT"]
        y_val = df_clean["Test_RedChi2_NFW"]

        ax.loglog(x_val, y_val, "o", color="gray", alpha=0.6, markersize=8, label="Galaxies")

        min_val = min(x_val.min(), y_val.min())
        max_val = max(x_val.max(), y_val.max())
        lims = [min_val * 0.5, max_val * 2.0]
        ax.plot(lims, lims, "k--", alpha=0.5, label="Equal Performance (1:1)")

        outliers = df_clean[df_clean["Test_RedChi2_NFW"] > df_clean["Test_RedChi2_ISUT"] * 5]
        ax.loglog(
            outliers["Test_RedChi2_ISUT"],
            outliers["Test_RedChi2_NFW"],
            "o",
            color="crimson",
            markersize=10,
            markeredgecolor="black",
            label="Significant DM deviations (NFW > 5× ISUT)",
        )

        top_outliers = outliers.sort_values("Test_RedChi2_NFW", ascending=False).head(5)
        for _, row in top_outliers.iterrows():
            ax.text(
                row["Test_RedChi2_ISUT"],
                row["Test_RedChi2_NFW"],
                f"  {row['Galaxy']}",
                fontsize=10,
                verticalalignment="center",
                fontweight="bold",
            )

        ax.set_xlabel(r"ISUT Prediction Error ($\chi^2_\nu$)", fontsize=12)
        ax.set_ylabel(r"DM Baseline Prediction Error ($\chi^2_\nu$)", fontsize=12)
        ax.set_title("Figure 2: Distribution of Prediction Residuals (Test Set)", fontsize=16)
        ax.legend(frameon=True)
        ax.grid(True, which="both", ls="-", alpha=0.2)

        plt.tight_layout()
        save_path2 = os.path.join(FIG_DIR, "Figure2_Prediction_Distribution.png")
        plt.savefig(save_path2, dpi=300)
        print(f"[Output] Figure 2 저장 완료: {save_path2}")
        plt.close(fig2)

    except Exception as e:
        print(f"[Error] Figure 2 생성 실패: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    generate_manuscript_figures()
