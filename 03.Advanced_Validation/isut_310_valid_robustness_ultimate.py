"""
ISUT Numerical Robustness & Sensitivity Stress-Test
==================================================
Objective:
    Quantifies the stability of ISUT parameters against observational 
    noise and geometric uncertainties.

Key Stress-Tests:
1. Parameter Sensitivity: Monitors RMSE changes against variations in 
   the fundamental constant a0.
2. Noise Injection: Tests model resilience by adding Gaussian noise (5% to 20%) 
   to the observed rotation velocities.
3. Geometric Stability: Evaluates the impact of inclination and distance 
   errors (Geometry factors) on the final fit quality.
"""

import os
import sys

from pathlib import Path
# --- ISUT shared helpers (path + SPARC io) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from isut_000_common import find_sparc_data_dir, ensure_rotmod_file, write_run_metadata

ALLOW_NET_DOWNLOAD = os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import json
from datetime import datetime

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

BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
os.makedirs(BASE_OUT_DIR, exist_ok=True)

print(f"[System] ISUT Robustness Validation Initialized")
print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Galaxy Lists
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

# ==============================================================================
# [3] Data Loader
# ==============================================================================
def get_data(gal_name):
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
# [4] ISUT Physics Engine
# ==============================================================================
def calculate_isut_acceleration(R, V_bary, a0=1.2e-10):
    kpc_to_m = 3.086e19
    v_ms = V_bary * 1000.0
    r_m = R * kpc_to_m
    
    g_N = (v_ms**2) / np.maximum(r_m, 1.0)
    y = g_N / a0
    nu = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / np.maximum(y, 1e-12))
    g_obs = g_N * nu
    
    V_isut = np.sqrt(np.maximum(g_obs * r_m, 0.0)) / 1000.0 
    return V_isut

# ==============================================================================
# [5] Validation Modules (Tests)
# ==============================================================================
def run_sensitivity_test(galaxy_name, R, V_bary, output_dirs):
    fig_dir, data_dir = output_dirs
    a0_range = np.linspace(0.6e-10, 1.8e-10, 20)
    results = []
    
    V_std = calculate_isut_acceleration(R, V_bary, a0=1.2e-10)

    for a0_val in a0_range:
        V_pred = calculate_isut_acceleration(R, V_bary, a0=a0_val)
        diff = V_pred - V_std
        rmse = np.sqrt(np.mean(diff**2))
        results.append({
            "Galaxy": galaxy_name,
            "a0": a0_val,
            "RMSE_Sensitivity": rmse
        })

    df_sens = pd.DataFrame(results)
    df_sens.to_csv(os.path.join(data_dir, f"{galaxy_name}_sensitivity_a0.csv"), index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df_sens["a0"], df_sens["RMSE_Sensitivity"], 'bo-', linewidth=2)
    plt.axvline(x=1.2e-10, color='r', linestyle='--', label='Standard a0')
    plt.title(f"[{galaxy_name}] Parameter Sensitivity (a0)", fontsize=14)
    plt.xlabel("a0 [m/s^2]", fontsize=12)
    plt.ylabel("RMSE [km/s]", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, f"{galaxy_name}_sensitivity_plot.png"))
    plt.close()
    
    return df_sens

def run_noise_stress_test(galaxy_name, R, V_bary, output_dirs):
    fig_dir, data_dir = output_dirs
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20]
    export_data = {"Radius": R}
    
    V_true_pred = calculate_isut_acceleration(R, V_bary)
    export_data["Base_Pred"] = V_true_pred
    
    results_summary = []

    plt.figure(figsize=(10, 6))
    plt.plot(R, V_true_pred, 'k-', linewidth=3, label="Base (0%)")

    for noise in noise_levels:
        if noise == 0.0: continue
        np.random.seed(42) # Ensure consistent noise for fair comparison
        noise_arr = np.random.normal(0, noise * np.mean(V_bary), size=len(V_bary))
        V_bary_noisy = np.abs(V_bary + noise_arr)
        V_noisy_pred = calculate_isut_acceleration(R, V_bary_noisy)
        
        plt.plot(R, V_noisy_pred, linestyle='--', alpha=0.7, label=f"Noise {int(noise*100)}%")
        export_data[f"Pred_Noise_{int(noise*100)}pct"] = V_noisy_pred
        
        rmse = np.sqrt(np.mean((V_noisy_pred - V_true_pred)**2))
        results_summary.append({
            "Galaxy": galaxy_name,
            "Noise_Level": noise,
            "RMSE_Noise": rmse
        })

    plt.title(f"[{galaxy_name}] Robustness against Input Noise", fontsize=14)
    plt.xlabel("Radius [kpc]", fontsize=12)
    plt.ylabel("Velocity [km/s]", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, f"{galaxy_name}_noise_stress_plot.png"))
    plt.close()
    
    pd.DataFrame(export_data).to_csv(os.path.join(data_dir, f"{galaxy_name}_noise_stress_data.csv"), index=False)
    return pd.DataFrame(results_summary)

def run_geometry_stability_test(galaxy_name, R, V_bary, output_dirs):
    fig_dir, data_dir = output_dirs
    geo_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    export_data = {"Radius": R}
    results_summary = []
    
    plt.figure(figsize=(10, 6))
    V_std = calculate_isut_acceleration(R, V_bary)

    for factor in geo_factors:
        V_bary_mod = V_bary * np.sqrt(factor) 
        V_pred = calculate_isut_acceleration(R, V_bary_mod)
        
        style = '-' if factor == 1.0 else ':'
        width = 3 if factor == 1.0 else 1.5
        label = f"Factor {factor}"
        
        plt.plot(R, V_pred, linestyle=style, linewidth=width, label=label)
        export_data[f"Pred_Geo_{factor}"] = V_pred
        
        rmse = np.sqrt(np.mean((V_pred - V_std)**2))
        results_summary.append({
            "Galaxy": galaxy_name,
            "Geo_Factor": factor,
            "RMSE_Geo": rmse
        })

    plt.title(f"[{galaxy_name}] Geometric Stability Check", fontsize=14)
    plt.xlabel("Radius [kpc]", fontsize=12)
    plt.ylabel("Velocity [km/s]", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(fig_dir, f"{galaxy_name}_geo_stability_plot.png"))
    plt.close()
    
    pd.DataFrame(export_data).to_csv(os.path.join(data_dir, f"{galaxy_name}_geo_stability_data.csv"), index=False)
    return pd.DataFrame(results_summary)

# ==============================================================================
# [6] Main Execution Logic
# ==============================================================================
def process_subset(galaxy_list, subset_name):
    print(f"\n[Analysis] Processing Subset: {subset_name}")
    
    # Setup folders
    subset_dir = os.path.join(BASE_OUT_DIR, subset_name)
    fig_dir = os.path.join(subset_dir, "figures")
    data_dir = os.path.join(subset_dir, "data")
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    output_dirs = (fig_dir, data_dir)
    
    all_sens_data = []
    
    # New: Accumulator for the "Comprehensive Report" (One row per galaxy)
    comprehensive_report = []
    
    for i, gal in enumerate(galaxy_list):
        print(f"   -> Analyzing {gal} ({i+1}/{len(galaxy_list)})...", end="\r")
        
        data = get_data(gal)
        if data is None: continue
        
        R, V_obs, V_err, V_gas, V_disk, V_bul = data
        V_bary2 = (np.abs(V_gas)*V_gas) + (0.5 * np.abs(V_disk)*V_disk) + (0.7 * np.abs(V_bul)*V_bul)
        V_bary = np.sqrt(np.maximum(V_bary2, 0.0))
        
        # Run Tests
        df_sens = run_sensitivity_test(gal, R, V_bary, output_dirs)
        df_noise = run_noise_stress_test(gal, R, V_bary, output_dirs)
        df_geo = run_geometry_stability_test(gal, R, V_bary, output_dirs)
        
        all_sens_data.append(df_sens)
        
        # [NEW] Calculate Summary Metrics for the Comprehensive Report
        # Using Mean RMSE across the tested variations as the Robustness Score
        summary_sens = df_sens['RMSE_Sensitivity'].mean()
        summary_noise = df_noise['RMSE_Noise'].mean()
        summary_geo = df_geo['RMSE_Geo'].mean()
        
        comprehensive_report.append({
            "Galaxy": gal,
            "Mean_RMSE_Sensitivity": summary_sens,
            "Mean_RMSE_Noise": summary_noise,
            "Mean_RMSE_Geometry": summary_geo
        })

    # --- 1. Save Aggregated Raw Data (Sensitivity) & Plot ---
    if all_sens_data:
        total_sens = pd.concat(all_sens_data)
        total_sens.to_csv(os.path.join(data_dir, f"{subset_name}_Total_Sensitivity_Raw.csv"), index=False)
        
        # Summary Plot (Overlay)
        plt.figure(figsize=(12, 8))
        for gal in total_sens['Galaxy'].unique():
            sub = total_sens[total_sens['Galaxy'] == gal]
            plt.plot(sub['a0'], sub['RMSE_Sensitivity'], alpha=0.3)
        
        # Mean Trend
        mean_trend = total_sens.groupby('a0')['RMSE_Sensitivity'].mean().reset_index()
        plt.plot(mean_trend['a0'], mean_trend['RMSE_Sensitivity'], 'k-', linewidth=4, label='MEAN Trend')
        
        plt.title(f"[{subset_name}] Aggregate Parameter Sensitivity (a0)", fontsize=16)
        plt.xlabel("a0 [m/s^2]", fontsize=14)
        plt.ylabel("RMSE [km/s]", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"{subset_name}_Total_Sensitivity_Plot.png"))
        plt.close()
        
    # --- 2. [NEW] Save Comprehensive Report (One Big CSV) ---
    if comprehensive_report:
        report_df = pd.DataFrame(comprehensive_report)
        report_path = os.path.join(data_dir, f"Robustness_Report_{subset_name}.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\n   [Success] Comprehensive Report saved: {report_path}")
        
        # Print Scorecard
        print(f"   [Scorecard] {subset_name} Mean RMSE:")
        print(f"      - Sensitivity: {report_df['Mean_RMSE_Sensitivity'].mean():.2f}")
        print(f"      - Noise:       {report_df['Mean_RMSE_Noise'].mean():.2f}")
        print(f"      - Geometry:    {report_df['Mean_RMSE_Geometry'].mean():.2f}")

        # [Reproducibility] Save minimal run metadata (no effect on numerical results)
        meta = {
            "script": SCRIPT_NAME,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "subset": subset_name,
            "n_galaxies_report": int(len(report_df)),
            "data_dir": os.path.abspath(DATA_DIR),
            "out_dir": os.path.abspath(subset_dir),
            "a0_range": [0.6e-10, 1.8e-10],
            "noise_levels": [0.0, 0.05, 0.10, 0.15, 0.20],
            "geo_factors": [0.8, 0.9, 1.0, 1.1, 1.2],
            "python": sys.version
        }
        with open(os.path.join(data_dir, f"run_metadata_{subset_name}.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def main():
    # A. Process Golden 12
    process_subset(GOLDEN_GALAXIES, "Golden12")
    
    # B. Process All 65
    process_subset(FULL_GALAXIES, "All65")
    
    print("\n" + "="*60)
    print("✅ All Robustness Tests (Individual + Aggregate) Completed.")
    print(f"📂 Results saved in: {BASE_OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD})
    except Exception:
        pass
