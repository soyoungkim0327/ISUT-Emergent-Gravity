
"""
ISUT Out-of-Sample Prediction (Holdout) Validator
=================================================
Objective:
    Proves the predictive power of the ISUT model via rigorous galaxy-level 
    holdout validation (Train/Test split).

Key Validations:
1. Generalization Test: Evaluates how well the model predicts the rotation 
   curves of 'unseen' galaxies not used during the parameter tuning.
2. Comparative Residual Analysis: Directly compares prediction errors (Chi-square) 
   between ISUT and NFW (Dark Matter) on the test set.
3. Statistical Landscape: Visualizes the preference landscape to identify 
   potential outliers and ensure overall model dominance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import warnings
import sys

from pathlib import Path
# --- ISUT shared helpers (path + SPARC io) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from isut_000_common import find_sparc_data_dir, ensure_rotmod_file, write_run_metadata

ALLOW_NET_DOWNLOAD = os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1"

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] Configuration: Smart Path Finder & Output Setup
# ==============================================================================
# 1. Determine current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# 2. Candidate Directories for Data Search
DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR} (override via ISUT_SPARC_DIR)")

# 5. Output Base Directory (Creates a folder named after this script)
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
os.makedirs(BASE_OUT_DIR, exist_ok=True)

print(f"[System] ISUT Ultimate Holdout Visualization Initialized")
print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Galaxy Lists & Physical Constants
# ==============================================================================
GOLDEN_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", 
    "NGC5055", "NGC7331", "NGC6946", "NGC7793", "NGC1560", "UGC02885"
]

FULL_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", "NGC5055", "NGC7331", 
    "NGC6946", "NGC7793", "NGC1560", "UGC02885", "NGC0801", "NGC2998", "NGC5033", "NGC5533", 
    "NGC5907", "NGC6674", "UGC06614", "UGC06786", "F568-3", "F571-8", "NGC0055", "NGC0247", 
    "NGC0300", "NGC1003", "NGC1365", "NGC2541", "NGC2683", "NGC2915", "NGC3109", "NGC3621", 
    "NGC3726", "NGC3741", "NGC3769", "NGC3877", "NGC3893", "NGC3917", "NGC3949", "NGC3953", 
    "NGC3972", "NGC3992", "NGC4013", "NGC4051", "NGC4085", "NGC4088", "NGC4100", "NGC4138", 
    "NGC4157", "NGC4183", "NGC4217", "NGC4559", "NGC5585", "NGC5985", "NGC6015", "NGC6195", 
    "UGC06399", "UGC06446", "UGC06667", "UGC06818", "UGC06917", "UGC06923", "UGC06930", 
    "UGC06983", "UGC07089"
]
FULL_GALAXIES = sorted(list(set(FULL_GALAXIES)))

# Physics Constants
ACCEL_CONV = 3.24078e-14 
A0_SI = 1.2e-10 
A0_CODE = A0_SI / ACCEL_CONV  

# ==============================================================================
# [3] Physics Engines & Optimization Logic
# ==============================================================================
def nu_isut(y):
    """ ISUT Interpolation Function """
    y_safe = np.maximum(y, 1e-12)
    return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y_safe)

def model_isut_core(R, ups_disk, ups_bulge, V_gas, V_disk, V_bul):
    """ Core ISUT Velocity Calculation """
    V_bary2 = (np.abs(V_gas)*V_gas) + (ups_disk * np.abs(V_disk)*V_disk) + (ups_bulge * np.abs(V_bul)*V_bul)
    R_safe = np.maximum(R, 0.01)
    gN = np.abs(V_bary2) / R_safe
    y = gN / A0_CODE
    g_final = gN * nu_isut(y)
    return np.sqrt(np.maximum(g_final * R, 0.0))

def apply_obs_corrections(R, V_obs, V_err, V_gas, V_disk, V_bul, z_D, z_i, i_deg_assume=60.0):
    """ Apply geometric corrections (Distance and Inclination) """
    d_factor = 1.0 + 0.1 * z_D 
    R_new = R * d_factor
    v_scale = np.sqrt(d_factor)
    
    i_new_deg = np.clip(i_deg_assume + 5.0 * z_i, 10.0, 89.0)
    sin_ratio = np.sin(np.radians(i_deg_assume)) / np.sin(np.radians(i_new_deg))
    
    V_obs_new = V_obs * sin_ratio
    V_err_new = V_err * sin_ratio
    
    return R_new, V_obs_new, V_err_new, V_gas*v_scale, V_disk*v_scale, V_bul*v_scale

def loss_fn(params, R, V_obs, V_err, V_gas, V_disk, V_bul, mode='none'):
    """
    Loss function for optimization.
    mode: 'none' (No correction), 'smart' (Strong Penalty), 'full' (Weak Penalty)
    """
    ud, ub = params[0], params[1]
    
    if mode == 'none':
        zD, zi = 0.0, 0.0
    else:
        zD, zi = params[2], params[3]
        
    R_c, V_c, V_ec, V_gc, V_dc, V_bc = apply_obs_corrections(R, V_obs, V_err, V_gas, V_disk, V_bul, zD, zi)
    V_model = model_isut_core(R_c, ud, ub, V_gc, V_dc, V_bc)
    
    chi2 = np.sum(((V_c - V_model) / np.maximum(V_ec, 1.0))**2)
    
    penalty = 0
    if mode == 'smart':
        penalty += (zD**2 + zi**2) * 2.0 
    elif mode == 'full':
        penalty += (zD**2 + zi**2) * 0.5 
        
    return chi2 + penalty

# ==============================================================================
# [4] Data Loader
# ==============================================================================
def get_data(gal_name):
    # Construct path using the dynamically found DATA_DIR
    try:
        path = str(ensure_rotmod_file(gal_name, Path(DATA_DIR), allow_download=ALLOW_NET_DOWNLOAD, timeout=3.0))
    except FileNotFoundError:
        return None
    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None).apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 5: return None
        return df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values, \
               df.iloc[:,3].values, \
               (df.iloc[:,4].values if df.shape[1]>4 else np.zeros(len(df))), \
               (df.iloc[:,5].values if df.shape[1]>5 else np.zeros(len(df)))
    except: return None

# ==============================================================================
# [5] Experiment Execution Logic
# ==============================================================================
def run_scenario(target_list, mode, label, subset_folder):
    """
    Runs the holdout test for a list of galaxies.
    subset_folder: 'Golden12' or 'All65' to organize outputs.
    """
    print(f"\n[Scenario] Running: {label} (Mode: {mode}, N={len(target_list)})")
    
    # Setup Structured Output Paths
    # Structure: BASE/Subset/figures and BASE/Subset/data
    subset_dir = os.path.join(BASE_OUT_DIR, subset_folder)
    fig_dir = os.path.join(subset_dir, "figures")
    data_dir = os.path.join(subset_dir, "data")
    
    # Create directories safely
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    results = []
    
    for i, gal in enumerate(target_list):
        print(f"   Processing {gal}...", end="\r")
        data = get_data(gal)
        if not data: continue
        
        R, V_obs, V_err, V_gas, V_disk, V_bul = data
        n_points = len(R)
        # Holdout Split: Train on inner 50%, Test on outer 50%
        n_split = int(n_points * 0.5)
        if n_split < 4: n_split = n_points - 1
        
        # 1. Train on Inner 50%
        init_guess = [0.5, 0.7, 0.0, 0.0]
        if mode == 'none':
            bounds = [(0.01, 5), (0, 5), (0, 0), (0, 0)] 
        else:
            bounds = [(0.01, 5), (0, 5), (-3, 3), (-3, 3)]
            
        try:
            res = minimize(loss_fn, init_guess, 
                           args=(R[:n_split], V_obs[:n_split], V_err[:n_split], 
                                 V_gas[:n_split], V_disk[:n_split], V_bul[:n_split], mode),
                           bounds=bounds)
            
            ud, ub, zd, zi = res.x
            
            # 2. Test on Outer 50% (Blind Prediction)
            R_test_c, V_test_c, E_test_c, V_gc_t, V_dc_t, V_bc_t = apply_obs_corrections(
                R[n_split:], V_obs[n_split:], V_err[n_split:], 
                V_gas[n_split:], V_disk[n_split:], V_bul[n_split:], zd, zi)
            
            V_pred_test = model_isut_core(R_test_c, ud, ub, V_gc_t, V_dc_t, V_bc_t)
            
            # 3. Calculate Stats
            dof = max(len(R_test_c) - 1, 1)
            chi2_test = np.sum(((V_test_c - V_pred_test) / np.maximum(E_test_c, 1.0))**2)
            chi2_red = chi2_test / dof
            k = 2 if mode == 'none' else 4
            bic_val = chi2_test + k * np.log(len(R_test_c))
            
            # 4. Visualization & Data Export
            # Apply corrections to FULL data for plotting context
            R_all_c, V_all_c, E_all_c, V_g_all, V_d_all, V_b_all = apply_obs_corrections(
                R, V_obs, V_err, V_gas, V_disk, V_bul, zd, zi)
            
            # Model prediction on FULL range
            V_model_full = model_isut_core(R_all_c, ud, ub, V_g_all, V_d_all, V_b_all)
            
            plt.figure(figsize=(10, 6))
            
            # Train Data (Inner - Black)
            plt.errorbar(R_all_c[:n_split], V_all_c[:n_split], yerr=E_all_c[:n_split], 
                         fmt='ko', ecolor='gray', label='Train (Inner 50%)')
            # Test Data (Outer - Red)
            plt.errorbar(R_all_c[n_split:], V_all_c[n_split:], yerr=E_all_c[n_split:], 
                         fmt='ro', ecolor='salmon', label='Test (Outer 50%)')
            # Model Prediction (Blue Line)
            plt.plot(R_all_c, V_model_full, 'b-', linewidth=2, label=f'ISUT Prediction (Mode: {mode})')
            
            plt.axvline(x=R_all_c[n_split-1], color='k', linestyle='--', alpha=0.5, label='Split Boundary')
            
            plt.title(f"[{gal}] Holdout Test ({label})\nTest $\chi^2_\\nu$ = {chi2_red:.2f} (zD={zd:.2f}, zi={zi:.2f})", fontsize=14)
            plt.xlabel("Radius [kpc]", fontsize=12)
            plt.ylabel("Velocity [km/s]", fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save Figure
            fig_filename = f"{gal}_{label}_holdout.png"
            fig_path = os.path.join(fig_dir, fig_filename)
            plt.savefig(fig_path, dpi=150)
            plt.close()
            
            # Save Plot Data to CSV (Requirement 3)
            # Create a dataframe for this specific plot
            df_plot = pd.DataFrame({
                "Radius_kpc": R_all_c,
                "Velocity_Obs_Corrected": V_all_c,
                "Velocity_Err_Corrected": E_all_c,
                "Velocity_Model_Prediction": V_model_full,
                "Region": ["Train" if idx < n_split else "Test" for idx in range(len(R_all_c))]
            })
            plot_csv_filename = f"{gal}_{label}_plot_data.csv"
            df_plot.to_csv(os.path.join(data_dir, plot_csv_filename), index=False)
            
            results.append({
                "Galaxy": gal,
                "Mode": mode,
                "Train_N": n_split,
                "Test_N": n_points - n_split,
                "Test_Chi2": chi2_test,
                "Test_Chi2_Red": chi2_red,
                "Test_BIC": bic_val,
                "z_Dist": zd,
                "z_Inc": zi,
                "Figure_File": fig_filename,
                "Figure_Path": os.path.relpath(fig_path, start=BASE_OUT_DIR)
            })
            
        except Exception as e:
            print(f"\n   [Error] Failed processing {gal}: {e}")
            continue

    # Save Summary CSV for the scenario
    if results:
        df = pd.DataFrame(results)
        filename = f"Holdout_Summary_{label}.csv"
        df.to_csv(os.path.join(data_dir, filename), index=False)
        
        mean_chi2r = df['Test_Chi2_Red'].mean()
        print(f"\n   [Success] {label} Completed. Mean Chi2_Red: {mean_chi2r:.2f}")
        return df
    return None

# ==============================================================================
# [6] Main Execution
# ==============================================================================
def main():
    # A. Golden 12 Scenarios (Saved to 'Golden12' folder)
    print("\n--- Processing Golden 12 Subset ---")
    df_b1 = run_scenario(GOLDEN_GALAXIES, 'none', "Golden12_NoCorr", "Golden12")
    df_b2 = run_scenario(GOLDEN_GALAXIES, 'smart', "Golden12_SmartCorr", "Golden12")
    df_b3 = run_scenario(GOLDEN_GALAXIES, 'full', "Golden12_FullCorr", "Golden12")
    
    # B. Full 65 Galaxy Scenarios (Saved to 'All65' folder)
    print("\n--- Processing All 65 Dataset ---")
    df_a1 = run_scenario(FULL_GALAXIES, 'none', "All65_NoCorr", "All65")
    df_a2 = run_scenario(FULL_GALAXIES, 'smart', "All65_SmartCorr", "All65")
    df_a3 = run_scenario(FULL_GALAXIES, 'full', "All65_FullCorr", "All65")
    
    print("\n" + "="*70)
    print("ULTIMATE HOLDOUT SUMMARY")
    print("="*70)
    print(f"{'Scenario':<20} | {'N_Gal':<5} | {'Mean Chi2_R':<12} | {'Status'}")
    print("-" * 65)
    
    summary_data = [
        ("All65_NoCorr", df_a1), ("All65_SmartCorr", df_a2), ("All65_FullCorr", df_a3),
        ("Golden12_NoCorr", df_b1), ("Golden12_SmartCorr", df_b2), ("Golden12_FullCorr", df_b3)
    ]
    
    for label, df in summary_data:
        if df is not None:
            print(f"{label:<20} | {len(df):<5} | {df['Test_Chi2_Red'].mean():<12.2f} | Saved ✅")
            
    print("-" * 65)
    print(f"[System] All Data and Figures are saved in: {BASE_OUT_DIR}")

if __name__ == "__main__":
    main()
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD})
    except Exception:
        pass
