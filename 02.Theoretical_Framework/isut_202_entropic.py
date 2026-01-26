# -*- coding: utf-8 -*-
"""
ISUT Entropic Gravity & M/L Ratio Analysis (Representative Example)
===================================================================

[Goal]
Demonstrate the entropic gravity framework using a single representative galaxy (NGC6503).
Verifies:
  1. Mass-to-Light (M/L) Ratio Optimization
  2. Beta Parameter Landscape (Uniqueness of Solution)

[Output Structure]
  ./3.isut_entropic_new/
    ├─ figures/  (*.png Analysis Plots)
    └─ data/     (*.csv Numerical Data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import minimize
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] Configuration: Dynamic Path Setup
# ==============================================================================
# Determine current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# Input Data Directory (Smart Search)
CANDIDATE_DIRS = [
    os.path.join(CURRENT_DIR, "galaxies", "65_galaxies"),        
    os.path.join(CURRENT_DIR, "..", "data", "galaxies", "65_galaxies"), 
    os.path.join(CURRENT_DIR, "..", "galaxies", "65_galaxies")   
]

DATA_DIR = CANDIDATE_DIRS[0] # Default
for path in CANDIDATE_DIRS:
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        DATA_DIR = path
        break

# Output Directory Setup
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(BASE_OUT_DIR, "figures")
DAT_DIR = os.path.join(BASE_OUT_DIR, "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DAT_DIR, exist_ok=True)

TARGET_GALAXY = "NGC6503"

# Plot Style Settings
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif',
    'axes.labelsize': 14, 'axes.titlesize': 16,
    'legend.fontsize': 11, 'figure.dpi': 150
})

print(f"[System] ISUT Entropic Analysis Initialized (Target: {TARGET_GALAXY})")
print(f"[Info] Input Directory : {DATA_DIR}")
print(f"[Info] Output Base     : {BASE_OUT_DIR}")
print(f"       |- Figures      : {FIG_DIR}")
print(f"       |- Data         : {DAT_DIR}")

# ==============================================================================
# [2] Physics Engine (ISUT Entropic)
# ==============================================================================
# Physical Constants (Normalized for Code)
ACCEL_CONV = 3.24078e-14 
A0_SI = 1.2e-10 
A0_CODE = A0_SI / ACCEL_CONV  

def nu_isut(y):
    # Interpolating Function: nu(y) = 0.5 + 0.5*sqrt(1 + 4/y)
    # y = g_N / a0
    y_safe = np.maximum(y, 1e-12)
    return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y_safe)

def predict_velocity_ML(R, V_gas, V_disk, V_bul, a0, beta, ups_disk, ups_bulge):
    """
    Predicts velocity based on Mass-to-Light ratios.
    beta: Theoretical parameter (should be close to 1.0)
    """
    # 1. Baryonic Newtonian Acceleration
    # V_bary^2 = V_gas^2 + Y_disk*V_disk^2 + Y_bulge*V_bulge^2
    V_bary2 = (np.abs(V_gas)*V_gas) + (ups_disk * np.abs(V_disk)*V_disk) + (ups_bulge * np.abs(V_bul)*V_bul)
    R_safe = np.maximum(R, 0.01)
    gN = np.abs(V_bary2) / R_safe
    
    # 2. ISUT Modification
    y = gN / (a0 * beta) # Beta scales the acceleration scale if needed
    g_final = gN * nu_isut(y)
    
    return np.sqrt(np.maximum(g_final * R, 0.0))

def loss_function(params, R, V_obs, V_err, V_gas, V_disk, V_bul, fixed_beta=None):
    # Params: [a0, ups_disk, ups_bulge, (beta)]
    if fixed_beta is None:
        a0_val, ups_d, ups_b, beta_val = params
    else:
        a0_val, ups_d, ups_b = params
        beta_val = fixed_beta
        
    V_pred = predict_velocity_ML(R, V_gas, V_disk, V_bul, a0_val, beta_val, ups_d, ups_b)
    chi2 = np.sum(((V_obs - V_pred) / np.maximum(V_err, 1.0))**2)
    
    # Regularization (Priors)
    # Penalize unrealistic M/L ratios
    penalty = 0
    if ups_d < 0.1 or ups_d > 5.0: penalty += 1000
    if ups_b < 0.1 or ups_b > 5.0: penalty += 1000
    
    return chi2 + penalty

# ==============================================================================
# [3] Data Loading
# ==============================================================================
def load_data(gal_name):
    path = os.path.join(DATA_DIR, f"{gal_name}_rotmod.dat")
    if not os.path.exists(path):
        print(f"[Error] Data file not found: {path}")
        return None
        
    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None)
        # Columns: R, Vobs, Verr, Vgas, Vdisk, Vbul
        data = df.apply(pd.to_numeric, errors='coerce').dropna()
        return (data.iloc[:,0].values, data.iloc[:,1].values, data.iloc[:,2].values,
                data.iloc[:,3].values, data.iloc[:,4].values, data.iloc[:,5].values)
    except Exception as e:
        print(f"[Error] Failed to load data: {e}")
        return None

# ==============================================================================
# [4] Main Analysis Logic
# ==============================================================================
def main():
    # Load Data
    data = load_data(TARGET_GALAXY)
    if data is None: return
    R, V_obs, V_err, V_gas, V_disk, V_bul = data
    
    print(f"\n[Analysis] Processing {TARGET_GALAXY}...")
    
    # 1. Optimize Parameters (a0, M/L_disk, M/L_bulge) fixing Beta=1.0
    # Initial guess: a0=A0_CODE, M/L=0.5
    init_guess = [A0_CODE, 0.5, 0.7]
    bounds = [(A0_CODE*0.1, A0_CODE*10), (0.1, 5.0), (0, 5.0)]
    
    res = minimize(loss_function, init_guess, 
                   args=(R, V_obs, V_err, V_gas, V_disk, V_bul, 1.0), # Fixed Beta=1
                   bounds=bounds, method='Nelder-Mead')
    
    best_a0, best_ups_d, best_ups_b = res.x
    best_beta = 1.0
    
    print(f"   -> Optimized M/L Disk : {best_ups_d:.2f}")
    print(f"   -> Optimized M/L Bulge: {best_ups_b:.2f}")
    print(f"   -> Chi-Square         : {res.fun:.2f}")

    # ==========================================================================
    # [Vis 1] Rotation Curve Decomposition & Data Export
    # ==========================================================================
    print("\n[Step 1] Generating Rotation Curve Decomposition...")
    
    V_final = predict_velocity_ML(R, V_gas, V_disk, V_bul, best_a0, best_beta, best_ups_d, best_ups_b)
    
    # Calculate components for visualization
    V_bary_total = np.sqrt(np.abs((np.abs(V_gas)*V_gas) + (best_ups_d * np.abs(V_disk)*V_disk) + (best_ups_b * np.abs(V_bul)*V_bul)))
    
    # [1] Save Data (CSV)
    df_fit = pd.DataFrame({
        'Radius_kpc': R,
        'V_Observed': V_obs,
        'V_Error': V_err,
        'V_Total_Model': V_final,
        'V_Baryonic': V_bary_total,
        'V_Gas': V_gas,
        'V_Disk_Scaled': V_disk * np.sqrt(best_ups_d),
        'V_Bulge_Scaled': V_bul * np.sqrt(best_ups_b)
    })
    csv_path1 = os.path.join(DAT_DIR, "Fig1_Data_Fitting.csv")
    df_fit.to_csv(csv_path1, index=False)
    print(f"   -> Data Saved: {csv_path1}")

    # [2] Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(R, V_obs, yerr=V_err, fmt='ko', mfc='none', label='Observed')
    plt.plot(R, V_final, 'r-', lw=2.5, label='ISUT Model (Total)')
    plt.plot(R, V_bary_total, 'b--', label='Baryonic Contribution')
    plt.plot(R, V_gas, 'c:', label='Gas')
    plt.plot(R, V_disk * np.sqrt(best_ups_d), 'g:', label=f'Disk (M/L={best_ups_d:.2f})')
    
    plt.xlabel('Radius (kpc)')
    plt.ylabel('Velocity (km/s)')
    plt.title(f'Entropic Gravity Decomposition: {TARGET_GALAXY}')
    plt.legend()
    plt.grid(alpha=0.3)
    
    fig_path1 = os.path.join(FIG_DIR, "Fig1_Rotation_Curve_Decomposition.png")
    plt.savefig(fig_path1)
    plt.close()
    print(f"   -> Figure Saved: {fig_path1}")

    # ==========================================================================
    # [Vis 2] Beta Landscape & Data Export
    # ==========================================================================
    print("\n[Step 2] Generating Parameter Landscape...")
    beta_scan = np.linspace(0.5, 2.0, 50)
    chi2_scan = []
    
    for b in beta_scan:
        # Scan Beta while keeping a0 and M/L fixed
        v_tmp = predict_velocity_ML(R, V_gas, V_disk, V_bul, best_a0, b, best_ups_d, best_ups_b)
        chi2_scan.append(np.sum(((V_obs - v_tmp) / np.maximum(V_err, 1.0))**2))
    
    # [NEW] Best beta from the scan (argmin)
    best_beta_scan = float(beta_scan[int(np.argmin(chi2_scan))])

    # [1] Save Data (CSV)
    df_landscape = pd.DataFrame({
        'Beta_Parameter': beta_scan,
        'Chi_Square_Error': chi2_scan
    })
    csv_path2 = os.path.join(DAT_DIR, "Fig2_Data_Landscape.csv")
    df_landscape.to_csv(csv_path2, index=False)
    print(f"   -> Data Saved: {csv_path2}")

    # [2] Plot
    plt.figure(figsize=(8, 6))
    plt.plot(beta_scan, chi2_scan, 'k-', lw=2)
    plt.axvline(best_beta_scan, color='r', ls='--', label=f'Best (scan) = {best_beta_scan:.3f}')
    plt.axvline(1.0, color='g', ls=':', label='Reference (1.0)')
    plt.xlabel(r'Beta $\beta$')
    plt.ylabel(r'$\chi^2$ Error')
    plt.title(r'Conditional $\chi^2$ scan of $\beta$ (fixed $a_0$, fixed M/L)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    fig_path2 = os.path.join(FIG_DIR, "Fig2_Beta_Landscape.png")
    plt.savefig(fig_path2)
    plt.close()
    print(f"   -> Figure Saved: {fig_path2}")

    print("\n" + "="*60)
    print(f"Validation Suite Completed.")
    print(f"Check outputs in: {BASE_OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()