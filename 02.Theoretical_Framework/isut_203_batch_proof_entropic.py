# -*- coding: utf-8 -*-
"""
ISUT Entropic Gravity Ultimate Batch Suite (Final Ver. + Dual Summary)
======================================================================

[Goal]
Batch validation of Entropic Gravity with FULL statistical reporting.
Generates TWO types of summary visualizations:
1. Standard Summary Plot (2-Panel)
2. Comprehensive Dashboard (4-Panel: Beta, Chi2, Scatter, Top 10)

[Output Structure]
  ./3-1.isut_batch_proof_entropic/
    ├─ Golden12/
    │   ├─ figures/ (*.png, Summary_Plot_*.png, Summary_Dashboard_*.png)
    │   └─ data/    (*.csv, Batch_Summary_*.csv)
    └─ All65/
        ├─ figures/ (*.png, Summary_Plot_*.png, Summary_Dashboard_*.png)
        └─ data/    (*.csv, Batch_Summary_*.csv)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Seaborn is optional; provide a matplotlib fallback for reproducibility
try:
    import seaborn as sns
except ImportError:
    sns = None
    print("[Warning] seaborn not found. Falling back to matplotlib for summary plots.")
from scipy.optimize import minimize
import os
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# [1] Configuration
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)

# Smart Data Search
CANDIDATE_DIRS = [
    os.path.join(CURRENT_DIR, "galaxies", "65_galaxies"),        
    os.path.join(CURRENT_DIR, "..", "data", "galaxies", "65_galaxies"), 
    os.path.join(CURRENT_DIR, "..", "galaxies", "65_galaxies")   
]
DATA_DIR = CANDIDATE_DIRS[0]
for path in CANDIDATE_DIRS:
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        DATA_DIR = path
        break

print(f"[System] ISUT Ultimate Batch Suite Initialized")
print(f"[Info] Input: {DATA_DIR}") 
print(f"[Info] Output: {BASE_OUT_DIR}")

# Galaxy Lists
GOLDEN_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", 
    "NGC5055", "NGC7331", "NGC6946", "NGC7793", "NGC1560", "UGC02885"
]
FULL_GALAXIES = sorted(list(set([
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", "NGC5055", "NGC7331", 
    "NGC6946", "NGC7793", "NGC1560", "UGC02885", "NGC0801", "NGC2998", "NGC5033", "NGC5533", 
    "NGC5907", "NGC6674", "UGC06614", "UGC06786", "F568-3", "F571-8", "NGC0055", "NGC0247", 
    "NGC0300", "NGC1003", "NGC1365", "NGC2541", "NGC2683", "NGC2915", "NGC3109", "NGC3621", 
    "NGC3726", "NGC3741", "NGC3769", "NGC3877", "NGC3893", "NGC3917", "NGC3949", "NGC3953", 
    "NGC3972", "NGC3992", "NGC4013", "NGC4051", "NGC4085", "NGC4088", "NGC4100", "NGC4138", 
    "NGC4157", "NGC4183", "NGC4217", "NGC4559", "NGC5585", "NGC5985", "NGC6015", "NGC6195", 
    "UGC06399", "UGC06446", "UGC06667", "UGC06818", "UGC06917", "UGC06923", "UGC06930", 
    "UGC06983", "UGC07089"
])))

ACCEL_CONV = 3.24078e-14 
A0_SI = 1.2e-10 
A0_CODE = A0_SI / ACCEL_CONV  

# ==============================================================================
# [2] Physics Engine
# ==============================================================================
def nu_isut(y):
    return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / np.maximum(y, 1e-12))

def model_entropic(R, Vg, Vd, Vb, a0, beta, ups_d, ups_b):
    V_bary2 = (np.abs(Vg)*Vg) + (ups_d * np.abs(Vd)*Vd) + (ups_b * np.abs(Vb)*Vb)
    gN = np.abs(V_bary2) / np.maximum(R, 0.01)
    y = gN / (a0 * beta)
    g_final = gN * nu_isut(y)
    return np.sqrt(np.maximum(g_final * R, 0.0))

def loss_function(params, R, Vo, Ve, Vg, Vd, Vb):
    a0, beta, ud, ub = params
    V_pred = model_entropic(R, Vg, Vd, Vb, a0, beta, ud, ub)
    chi2 = np.sum(((Vo - V_pred) / np.maximum(Ve, 1.0))**2)
    pen = 0
    if ud < 0.1 or ud > 5.0: pen += 1000
    if ub < 0.0 or ub > 5.0: pen += 1000
    if beta < 0.5 or beta > 2.0: pen += 1000
    # Soft bounds for a0 (Nelder-Mead may ignore bounds argument)
    if a0 < A0_CODE*0.1 or a0 > A0_CODE*10.0:
        pen += 1000
    return chi2 + pen

def get_data(gal_name, return_reason=False):
    """Load a galaxy rotmod file.
    If return_reason=True, returns (data_or_None, reason_str).
    """
    path = os.path.join(DATA_DIR, f"{gal_name}_rotmod.dat")
    if not os.path.exists(path):
        return (None, "file_not_found") if return_reason else None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None).apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 5:
            return (None, "too_few_rows") if return_reason else None
        data = (
            df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values,
            df.iloc[:,3].values,
            (df.iloc[:,4].values if df.shape[1] > 4 else np.zeros(len(df))),
            (df.iloc[:,5].values if df.shape[1] > 5 else np.zeros(len(df))),
        )
        return (data, "ok") if return_reason else data
    except Exception as e:
        reason = f"parse_error:{type(e).__name__}"
        return (None, reason) if return_reason else None


# ==============================================================================
# [3] Visualization Tools (Dual Mode)
# ==============================================================================
def generate_standard_plot(df_sum, set_name, fig_dir, label=None):
    """Generates the standard 2-panel summary plot.
    Uses seaborn when available; otherwise falls back to matplotlib.
    """
    label_use = label if label is not None else set_name
    plt.figure(figsize=(12, 10))

    # Subplot 1: Beta Parameter Distribution
    plt.subplot(2, 1, 1)
    betas = df_sum["Best_Beta"].dropna() if "Best_Beta" in df_sum.columns else []
    if sns is not None:
        sns.histplot(betas, bins=15, kde=True, color="skyblue", edgecolor="black")
    else:
        plt.hist(betas, bins=15, color="skyblue", edgecolor="black", alpha=0.8)

    plt.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Theoretical (Beta=1.0)")
    mean_beta = float(np.mean(betas)) if len(betas) else float("nan")
    if np.isfinite(mean_beta):
        plt.axvline(mean_beta, color="blue", linestyle=":", linewidth=2, label=f"Mean ({mean_beta:.2f})")
    plt.title(f"Beta Parameter Distribution ({label_use})", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: MAPE vs N_Points
    plt.subplot(2, 1, 2)
    if sns is not None:
        sns.scatterplot(
            data=df_sum,
            x="N_Points",
            y="MAPE_Percent",
            hue="Chi2_Red",
            palette="viridis",
            size="BIC",
            sizes=(50, 200),
            alpha=0.7,
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        x = df_sum["N_Points"].values if "N_Points" in df_sum.columns else np.array([])
        y = df_sum["MAPE_Percent"].values if "MAPE_Percent" in df_sum.columns else np.array([])
        c = df_sum["Chi2_Red"].values if "Chi2_Red" in df_sum.columns else None
        b = df_sum["BIC"].values if "BIC" in df_sum.columns else None

        if b is not None and len(b):
            bmin, bmax = float(np.min(b)), float(np.max(b))
            denom = (bmax - bmin) if (bmax - bmin) != 0 else 1.0
            sizes = 50 + 150 * (b - bmin) / denom
        else:
            sizes = 80

        sc = plt.scatter(x, y, c=c, s=sizes, alpha=0.7)
        if c is not None and len(x):
            plt.colorbar(sc, label="Chi2_Red")

    plt.axhline(10.0, color="orange", linestyle="--", label="10% Error Threshold")
    plt.title(f"Accuracy (MAPE) vs Data Quality ({label_use})", fontsize=14)
    plt.xlabel("Number of Data Points")
    plt.ylabel("MAPE (%)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(fig_dir, f"Summary_Plot_{set_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   -> Standard Plot Saved: {path}")

def generate_dashboard_plot(df_sum, set_name, fig_dir, label=None):
    """(New) Generates the comprehensive 4-panel dashboard."""
    label_use = label if label is not None else set_name
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(f"ISUT Entropic Gravity: Statistical Summary ({label_use})", fontsize=16, fontweight='bold')

    # Panel 1: Beta Distribution
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.hist(df_sum['Best_Beta'], bins=12, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Theory (1.0)')
    ax1.axvline(df_sum['Best_Beta'].mean(), color='blue', linestyle=':', linewidth=2, label='Mean')
    ax1.set_title("Beta Parameter Distribution")
    ax1.set_xlabel("Beta Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Reduced Chi2 Distribution
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(df_sum['Chi2_Red'], bins=12, color='lightgreen', edgecolor='black', alpha=0.7)
    ax2.set_title("Reduced Chi-Square Distribution")
    ax2.set_xlabel("Reduced Chi2")
    ax2.grid(True, alpha=0.3)

    # Panel 3: Accuracy Scatter
    ax3 = fig.add_subplot(2, 2, 3)
    sc = ax3.scatter(df_sum['N_Points'], df_sum['MAPE_Percent'], 
                     c=df_sum['Chi2_Red'], cmap='viridis', s=80, alpha=0.8, edgecolors='k')
    plt.colorbar(sc, ax=ax3, label='Reduced Chi2')
    ax3.set_title("Model Accuracy vs Data Quality")
    ax3.set_xlabel("Data Points")
    ax3.set_ylabel("MAPE (%)")
    ax3.axhline(10, color='orange', linestyle='--', label='10% Line')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Panel 4: Top 10 Ranking
    ax4 = fig.add_subplot(2, 2, 4)
    top10 = df_sum.sort_values('Chi2_Red').head(10).sort_values('Chi2_Red', ascending=False)
    ax4.barh(top10['Galaxy'], top10['Chi2_Red'], color='salmon', edgecolor='black', alpha=0.8)
    ax4.set_title("Top 10 Best Fits (Lowest Chi2)")
    ax4.set_xlabel("Reduced Chi2")
    ax4.grid(True, axis='x', alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(fig_dir, f"Summary_Dashboard_{set_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"   -> Dashboard Plot Saved: {path}")

# ==============================================================================
# [4] Batch Process
# ==============================================================================
def process_batch(set_name, target_list):
    print(f"\n Processing {set_name} (N={len(target_list)})...")
    
    set_dir = os.path.join(BASE_OUT_DIR, set_name)
    fig_dir, dat_dir = os.path.join(set_dir, "figures"), os.path.join(set_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(dat_dir, exist_ok=True)
    
    summary_list = []
    skipped = []

    for gal in target_list:
        print(f"   Analyzing: {gal}...", end="\r")
        data, reason = get_data(gal, return_reason=True)
        if data is None:
            skipped.append({"Galaxy": gal, "Reason": reason})
            continue
        R, Vo, Ve, Vg, Vd, Vb = data
        
        init = [A0_CODE, 1.0, 0.5, 0.7]
        bnds = [(A0_CODE*0.1, A0_CODE*10), (0.5, 2.0), (0.1, 5.0), (0.0, 5.0)]
        
        res = minimize(loss_function, init, args=(R, Vo, Ve, Vg, Vd, Vb), bounds=bnds, method='Nelder-Mead')
        p = res.x
        
        V_model = model_entropic(R, Vg, Vd, Vb, p[0], p[1], p[2], p[3])
        residuals = Vo - V_model
        
        chi2_tot = np.sum((residuals / np.maximum(Ve, 1.0))**2)
        dof = max(1, len(R) - 4)
        chi2_red = chi2_tot / dof
        
        mask = Vo > 1.0
        mape = np.mean(np.abs(residuals[mask] / Vo[mask])) * 100 if np.any(mask) else 0.0
        bic = chi2_tot + 4 * np.log(len(R))

        summary_list.append({
            'Galaxy': gal, 'Best_Beta': round(p[1], 4), 'Best_a0': round(p[0], 2),
            'Ups_Disk': round(p[2], 3), 'Ups_Bulge': round(p[3], 3),
            'Chi2_Total': round(chi2_tot, 2), 'Chi2_Red': round(chi2_red, 3),
            'MAPE_Percent': round(mape, 2), 'BIC': round(bic, 2), 'N_Points': len(R)
        })
        
        V_bary = np.sqrt(np.abs((np.abs(Vg)*Vg) + (p[2]*np.abs(Vd)*Vd) + (p[3]*np.abs(Vb)*Vb)))
        pd.DataFrame({'R':R, 'V_Obs':Vo, 'V_Err':Ve, 'V_Model':V_model, 'V_Bary':V_bary})\
          .to_csv(os.path.join(dat_dir, f"FitData_{gal}.csv"), index=False)
          
        plt.figure(figsize=(10,6))
        plt.errorbar(R, Vo, yerr=Ve, fmt='ko', label='Observed')
        plt.plot(R, V_model, 'r-', lw=2, label=f'ISUT (Beta={p[1]:.2f})')
        plt.plot(R, V_bary, 'b--', label='Baryonic')
        plt.title(f"{gal}: MAPE={mape:.1f}%, Beta={p[1]:.2f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(fig_dir, f"FitPlot_{gal}.png"))
        plt.close()

    # Save Summary CSV
    df_sum = pd.DataFrame(summary_list)
    sum_csv_path = os.path.join(set_dir, f"Batch_Summary_{set_name}.csv")
    df_sum.to_csv(sum_csv_path, index=False)
    print(f"\n   -> Summary CSV Saved: {sum_csv_path}")


    # [NEW] Save skipped list + counts for auditability (no impact on scores)
    skip_path = os.path.join(set_dir, f"Skipped_{set_name}.csv")
    if skipped:
        pd.DataFrame(skipped).to_csv(skip_path, index=False)
    else:
        pd.DataFrame(columns=["Galaxy", "Reason"]).to_csv(skip_path, index=False)
    print(f"   -> Skipped CSV Saved: {skip_path}")

    n_target = len(target_list)
    n_valid = len(summary_list)
    n_skip = len(skipped)
    display_label = f"{set_name} (Valid={n_valid}/{n_target}, Skipped={n_skip})"
    print(f"   ℹ️ Target={n_target} | Valid={n_valid} | Skipped={n_skip}")

    # [NEW] Beta boundary cases report (hits/near-hits of bounds)
    beta_lo, beta_hi, tol = 0.5, 2.0, 0.02
    if not df_sum.empty and 'Best_Beta' in df_sum.columns:
        b = df_sum['Best_Beta'].astype(float)
        mask_lo = b <= (beta_lo + tol)
        mask_hi = b >= (beta_hi - tol)
        df_bound = df_sum.loc[mask_lo | mask_hi].copy()
        if not df_bound.empty:
            df_bound['Boundary'] = np.where(df_bound['Best_Beta'].astype(float) <= (beta_lo + tol), 'lower', 'upper')
        boundary_path = os.path.join(set_dir, f"Beta_Boundary_Cases_{set_name}.csv")
        df_bound.to_csv(boundary_path, index=False)
        print(f"   -> Beta boundary cases saved: {boundary_path} (N={len(df_bound)})")

    # Generate BOTH Plots (use display label, keep filenames stable)
    generate_standard_plot(df_sum, set_name, fig_dir, label=display_label)   # 2-Panel (Old)
    generate_dashboard_plot(df_sum, set_name, fig_dir, label=display_label)  # 4-Panel (New)

def main():
    process_batch("Golden12", GOLDEN_GALAXIES)
    process_batch("All65", FULL_GALAXIES)
    print(f"\n Completed. Output: {BASE_OUT_DIR}")

if __name__ == "__main__":
    main()