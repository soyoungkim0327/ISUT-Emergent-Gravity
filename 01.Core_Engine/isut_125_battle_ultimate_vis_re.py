# -*- coding: utf-8 -*-
"""
ISUT Model Visualization & Data Archiving Suite (Paper Readiness)
================================================================
Objective:
    Generates high-fidelity visualizations and standardized data exports 
    to support peer-review claims and ensure results transparency.

Key Scientific Standards:
1. Systematic Data Archiving:
   - Separates results into 'Golden12' (high-signal) and 'All65' (statistical) 
     directories with dedicated subfolders for figures and raw CSV data.
2. Visualizing Correction Impacts:
   - Explicitly renders the vector shift between original and corrected 
     observational data ($z_D$, $z_i$) to provide visual proof of stability.
3. Comparative Analytics:
   - Simultaneous plotting of ISUT and NFW (Dark Matter) models with 
     associated BIC-style scores for direct performance comparison.
4. Export for Meta-Analysis:
   - Automatically generates per-galaxy CSV files (`VisData_*.csv`) 
     to allow third-party verification of the fitting curves.
============================================================

[Output Structure]
  ./5-6.isut_battle_ultimate_vis/
    ├─ Golden12/
    │   ├─ figures/ (*.png)
    │   └─ data/    (*.csv)
    └─ All65/
        ├─ figures/ (*.png)
        └─ data/    (*.csv)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import warnings
import sys

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] Configuration: Dynamic Path Setup
# ==============================================================================
# 1. Determine current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)

# 2. Input Data Directory (Smart Search)
# Attempts to locate the 'galaxies' data folder in common relative paths
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

print(f"[System] ISUT Visualization Suite Initialized")
print(f"[Info] Input Directory : {DATA_DIR}") 
print(f"[Info] Output Base     : {BASE_OUT_DIR}")

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

# Physical Constants
ACCEL_CONV = 3.24078e-14 
A0_SI = 1.2e-10 
A0_CODE = A0_SI / ACCEL_CONV  

# ==============================================================================
# [2] Physics & Logic
# ==============================================================================
def nu_isut(y):
    return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / np.maximum(y, 1e-12))

def model_isut(R, ud, ub, Vg, Vd, Vb):
    Vb2 = (np.abs(Vg)*Vg) + (ud*np.abs(Vd)*Vd) + (ub*np.abs(Vb)*Vb)
    gN = np.abs(Vb2) / np.maximum(R, 0.01)
    g_final = gN * nu_isut(gN / A0_CODE)
    return np.sqrt(np.maximum(g_final * R, 0.0))

def velocity_nfw(R, V200, c):
    R200 = V200 / 10.0 
    x = R / np.maximum(R200, 0.01)
    numerator = np.log(1.0 + c*x) - (c*x) / (1.0 + c*x)
    denominator = np.log(1.0 + c) - c / (1.0 + c)
    return np.sqrt(np.maximum(V200**2 * (numerator / denominator) / x, 0.0))

def model_dm(R, ud, ub, V200, c, Vg, Vd, Vb):
    Vb2 = (np.abs(Vg)*Vg) + (ud*np.abs(Vd)*Vd) + (ub*np.abs(Vb)*Vb)
    return np.sqrt(np.maximum(Vb2 + velocity_nfw(R, V200, c)**2, 0.0))

def apply_corrections(R, Vo, Ve, Vg, Vd, Vb, zD, zi, i0=60.0):
    d_factor = 1.0 + 0.1 * zD 
    R_n = R * d_factor
    v_scale = np.sqrt(d_factor)
    i_new = np.clip(i0 + 5.0 * zi, 10, 89)
    sin_ratio = np.sin(np.radians(i0)) / np.sin(np.radians(i_new))
    return R_n, Vo * sin_ratio, Ve * sin_ratio, Vg*v_scale, Vd*v_scale, Vb*v_scale

def loss_isut(p, R, Vo, Ve, Vg, Vd, Vb, mode):
    ud, ub = p[0], p[1]
    if mode == 'none': zD, zi = 0, 0
    else: zD, zi = p[2], p[3]
    Rc, Voc, Vec, Vgc, Vdc, Vbc = apply_corrections(R, Vo, Ve, Vg, Vd, Vb, zD, zi)
    Vp = model_isut(Rc, ud, ub, Vgc, Vdc, Vbc)
    return np.sum(((Voc - Vp) / np.maximum(Vec, 1.0))**2) + (0 if mode=='none' else (zD**2 + zi**2) * (2 if mode=='smart' else 0.5))

def loss_dm(p, R, Vo, Ve, Vg, Vd, Vb):
    Vp = model_dm(R, p[0], p[1], p[2], p[3], Vg, Vd, Vb)
    chi2 = np.sum(((Vo - Vp) / np.maximum(Ve, 1.0))**2)
    return chi2 + (1000 if (p[3] < 1 or p[3] > 100 or p[2] < 10 or p[2] > 500) else 0)

def get_data(gal_name, return_reason=False):
    path = os.path.join(DATA_DIR, f"{gal_name}_rotmod.dat")
    if not os.path.exists(path):
        return (None, 'file_not_found') if return_reason else None
    try:
        df = pd.read_csv(path, delim_whitespace=True, comment='#', header=None)
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        if df.shape[1] < 6:
            return (None, 'too_few_columns') if return_reason else None
        df = df.iloc[:, :6]
        if len(df) < 5:
            return (None, 'too_few_rows') if return_reason else None
        data = df.values.T
        return (data, 'ok') if return_reason else data
    except Exception as e:
        reason = f"parse_error:{type(e).__name__}"
        return (None, reason) if return_reason else None

# ==============================================================================
# [3] Visualization Loop (Structured)
# ==============================================================================
def run_vis(target_list, mode, set_name):
    # Determine label for filename safety
    label = f"{set_name}_{mode}"
    print(f"\n[Processing] Set: {set_name} | Mode: {mode}")
    
    # Structure: BASE_OUT_DIR / SetName / {figures, data}
    set_dir = os.path.join(BASE_OUT_DIR, set_name)
    fig_dir = os.path.join(set_dir, "figures")
    dat_dir = os.path.join(set_dir, "data")
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(dat_dir, exist_ok=True)
    skipped = []
    fit_rows = []
    
    for i, gal in enumerate(target_list):
        print(f"   Generating: {gal}...", end="\r")
        d, reason = get_data(gal, return_reason=True)
        if d is None:
            skipped.append({"Galaxy": gal, "Reason": reason})
            continue
        R, Vo, Ve, Vg, Vd, Vb = d
        
        # 1. Fit ISUT
        if mode == 'none': 
            res_i = minimize(loss_isut, [0.5, 0.7], args=(R, Vo, Ve, Vg, Vd, Vb, 'none'), bounds=[(0.01,5),(0,5)])
            ud_i, ub_i, zd, zi = res_i.x[0], res_i.x[1], 0, 0
            k_i = 2
        else:
            res_i = minimize(loss_isut, [0.5, 0.7, 0, 0], args=(R, Vo, Ve, Vg, Vd, Vb, mode), bounds=[(0.01,5),(0,5),(-3,3),(-3,3)])
            ud_i, ub_i, zd, zi = res_i.x
            k_i = 4
        
        # 2. Fit DM
        v_max = np.max(Vo)
        res_d = minimize(loss_dm, [0.5, 0.7, v_max, 10], args=(R, Vo, Ve, Vg, Vd, Vb), bounds=[(0.01,5),(0,5),(10,500),(1,100)])
        
        # 3. Calculate BIC
        bic_i = res_i.fun + k_i * np.log(len(R))
        bic_d = res_d.fun + 4 * np.log(len(R))
        diff = bic_d - bic_i
        winner = "ISUT" if diff > 0 else "DM"
        fit_rows.append({
            'Galaxy': gal,
            'Set': set_name,
            'Mode': mode,
            'n': int(len(R)),
            'ud_isut': float(ud_i),
            'ub_isut': float(ub_i),
            'zD': float(zd),
            'zi': float(zi),
            'V200_dm': float(res_d.x[2]),
            'c_dm': float(res_d.x[3]),
            'score_isut': float(res_i.fun),
            'score_dm': float(res_d.fun),
            'k_isut': int(k_i),
            'k_dm': int(4),
            'bic_style_isut': float(bic_i),
            'bic_style_dm': float(bic_d),
            'delta_bic_style': float(diff),
            'winner': winner,
            'success_isut': bool(getattr(res_i, 'success', True)),
            'success_dm': bool(getattr(res_d, 'success', True)),
            'message_isut': str(getattr(res_i, 'message', '')),
            'message_dm': str(getattr(res_d, 'message', '')),
            'nfev_isut': int(getattr(res_i, 'nfev', -1)),
            'nfev_dm': int(getattr(res_d, 'nfev', -1)),
        })
        
        # 4. Prepare Data for Export
        Rc, Voc, Vec, Vgc, Vdc, Vbc = apply_corrections(R, Vo, Ve, Vg, Vd, Vb, zd, zi)
        V_dm_pred = model_dm(R, res_d.x[0], res_d.x[1], res_d.x[2], res_d.x[3], Vg, Vd, Vb)
        V_isut_pred = model_isut(Rc, ud_i, ub_i, Vgc, Vdc, Vbc)
        
        # Save CSV (Data Export)
        df_out = pd.DataFrame({
            'Radius_kpc': R, 'V_Obs': Vo, 'V_Err': Ve,
            'V_ISUT_Model': V_isut_pred, 'V_DM_Model': V_dm_pred,
            'R_Corrected': Rc, 'V_Obs_Corrected': Voc
        })
        # Filename includes mode to avoid overwrite
        df_out.to_csv(os.path.join(dat_dir, f"VisData_{gal}_{mode}.csv"), index=False)

        # 5. Plot
        plt.figure(figsize=(10, 6))
        
        # DM Curve
        plt.plot(R, V_dm_pred, 'b--', linewidth=2, label=f'Dark Matter (NFW)  BIC-style: {bic_d:.1f}')
        plt.errorbar(R, Vo, yerr=Ve, fmt='bx', alpha=0.3, label='Original Data')
        
        # ISUT Curve
        plt.plot(Rc, V_isut_pred, 'r-', linewidth=2.5, label=f'ISUT (Mode: {mode})  BIC-style: {bic_i:.1f}')
        
        if mode != 'none':
            plt.errorbar(Rc, Voc, yerr=Vec, fmt='ro', alpha=0.6, label=f'Corrected Data\n($z_D$={zd:.2f}, $z_i$={zi:.2f})')
            # Connect original to corrected
            for j in range(len(R)): plt.plot([R[j], Rc[j]], [Vo[j], Voc[j]], 'k-', alpha=0.1)
        else:
            plt.errorbar(R, Vo, yerr=Ve, fmt='ro', alpha=0.6, label='Data (No Correction)')

        plt.title(f'Galaxy: {gal} ({mode}) | Winner: {winner} (ΔBIC-style={diff:.1f})', fontsize=14)
        plt.xlabel("Radius [kpc]"); plt.ylabel("Velocity [km/s]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(fig_dir, f"VisPlot_{gal}_{mode}.png"), dpi=150)
        plt.close()


    # --- Save skipped list for auditability (no impact on scores) ---
    skip_path = os.path.join(set_dir, f'Skipped_{label}.csv')
    if skipped:
        pd.DataFrame(skipped).to_csv(skip_path, index=False)
    else:
        pd.DataFrame(columns=['Galaxy', 'Reason']).to_csv(skip_path, index=False)

    # --- Save per-galaxy fit summary (no impact on scores) ---
    fit_path = os.path.join(set_dir, f'FitSummary_{label}.csv')
    if fit_rows:
        pd.DataFrame(fit_rows).to_csv(fit_path, index=False)
    else:
        pd.DataFrame(columns=[
            'Galaxy','Set','Mode','n','ud_isut','ub_isut','zD','zi','V200_dm','c_dm',
            'score_isut','score_dm','k_isut','k_dm','bic_style_isut','bic_style_dm','delta_bic_style','winner',
            'success_isut','success_dm','message_isut','message_dm','nfev_isut','nfev_dm'
        ]).to_csv(fit_path, index=False)

    n_target = len(target_list)
    n_skip = len(skipped)
    n_valid = n_target - n_skip
    print(f'   ℹ️ {set_name}/{mode}: Target={n_target} | Valid={n_valid} | Skipped={n_skip}  ->  {os.path.basename(skip_path)}')
    print(f'   ℹ️ Fit summary saved -> {os.path.basename(fit_path)}')
def main():
    # Run scenarios with Set Names mapping to folder structure
    run_vis(GOLDEN_GALAXIES, 'none', "Golden12")
    run_vis(GOLDEN_GALAXIES, 'smart', "Golden12")
    
    # Note: 'none' and 'smart' for All65 will go into the same 'All65' folder
    # but distinguished by filenames.
    run_vis(FULL_GALAXIES, 'none', "All65") 
    run_vis(FULL_GALAXIES, 'smart', "All65") 
    
    print("\n" + "="*60)
    print(f"✅ All tasks finished. Check output: {BASE_OUT_DIR}")

if __name__ == "__main__":
    main()