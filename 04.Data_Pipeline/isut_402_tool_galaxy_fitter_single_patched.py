"""
ISUT Interactive Galaxy Fitter (Unit Test & Diagnostic Tool)
==========================================================
Objective:
    Provides a real-time, single-galaxy fitting interface to inspect 
    model performance and parameter sensitivity (Beta) on-the-fly.

Key Features:
1. Instant Visual Feedback: Generates rotation curve plots and Chi-square 
   metrics for any galaxy in the SPARC dataset.
2. Beta-Parameter Optimization: Uses bounded scalar minimization to find 
   the optimal field-strength scaling for specific galactic profiles.
3. Diagnostic Export: Saves high-resolution PNGs and raw fitting data 
   for individual case studies requested by reviewers.
"""

import os

from pathlib import Path
# --- ISUT shared helpers (path + SPARC io) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
import sys as _sys
if str(REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(REPO_ROOT))
from isut_000_common import find_sparc_data_dir, ensure_rotmod_file, write_run_metadata

ALLOW_NET_DOWNLOAD = os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
import warnings

# [Setting] Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
plt.style.use('default')

# ==============================================================================
# [1] Configuration: Smart Path Finder & Output Setup
# ==============================================================================
# 1. Determine current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# 2. Candidate Directories for Data Search
DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR}")

# 4. Define Output Directory Structure
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)

# Definition of Galaxy Lists for Folder Sorting
GOLDEN_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", 
    "NGC5055", "NGC7331", "NGC6946", "NGC7793", "NGC1560", "UGC02885"
]

print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Physics Constants & Model (Robust Version)
# ==============================================================================
A0_SI = 1.2e-10 
ACCEL_CONV = 3.24078e-14 
A0_CODE = A0_SI / ACCEL_CONV  


# Default mass-to-light scaling factors used to combine SPARC baryonic components
# (kept as previous hard-coded values for backward-compatibility)
UPS_D_DEFAULT = 0.5
UPS_B_DEFAULT = 0.7
def mu_beta(x, beta):
    """Interpolating function mu(x) for the Beta-family model."""
    return x / (1.0 + x**beta)**(1.0/beta)

def solve_nu_from_y(y, beta):
    """Inverse solver to find nu from y = g_N / a0 using Brent's method."""
    def func(x): return x * mu_beta(x, beta) - y
    try:
        x_min = np.sqrt(y) * 0.1
        x_max = y * 10.0 + 10.0
        x_sol = brentq(func, x_min, x_max)
        return x_sol / y
    except: return 1.0

class BetaModel:
    """Class to handle Beta model calculations with lookup tables."""
    def __init__(self, beta):
        self.beta = beta
        self._build_lookup()
        
    def _build_lookup(self):
        y_vals = np.logspace(-5, 5, 1000)
        nu_vals = [solve_nu_from_y(y, self.beta) for y in y_vals]
        self.interp = interp1d(np.log10(y_vals), nu_vals, kind='linear', fill_value="extrapolate")
        
    def get_nu(self, y):
        return self.interp(np.log10(np.maximum(y, 1e-10)))

# ==============================================================================
# [3] Data Loader
# ==============================================================================
def get_data(gal_name):
    """Loads galaxy rotation curve data.

    Note:
      - Uses shared helper `ensure_rotmod_file()` so the download behavior is consistent
        across the repo.
      - Set environment variable ISUT_NO_DOWNLOAD=1 to disable network access.
    """
    try:
        path = str(ensure_rotmod_file(gal_name, Path(DATA_DIR), allow_download=ALLOW_NET_DOWNLOAD, timeout=3.0))
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None).apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 5:
            print("[Error] Data too short.")
            return None

        # Returns: R, V_obs, V_err, V_gas, V_disk, V_bul
        return (
            df.iloc[:, 0].values,
            df.iloc[:, 1].values,
            df.iloc[:, 2].values,
            df.iloc[:, 3].values,
            df.iloc[:, 4].values if df.shape[1] > 4 else np.zeros(len(df)),
            df.iloc[:, 5].values if df.shape[1] > 5 else np.zeros(len(df)),
        )
    except Exception as e:
        print(f"[Error] Failed to read data: {e}")
        return None

# ==============================================================================
# [4] Fitting & Visualization Logic
# ==============================================================================
def save_results(gal_name, R, V_obs, V_err, V_bary, V_pred, best_beta, chi2):
    """
    Saves the figure and the source data (CSV) into the correct folder.
    """
    # 1. Determine Category (Golden12 vs All65)
    if gal_name in GOLDEN_GALAXIES:
        category = "Golden12"
    else:
        category = "All65"
    
    # 2. Create Directory Structure
    # Structure: SCRIPT_NAME / Category / [figures | data]
    fig_dir = os.path.join(BASE_OUT_DIR, category, "figures")
    data_dir = os.path.join(BASE_OUT_DIR, category, "data")
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # 3. Save Plot Data (CSV) - Requirement: Image source file
    df_out = pd.DataFrame({
        "Radius_kpc": R,
        "V_Observed": V_obs,
        "V_Error": V_err,
        "V_Baryonic": V_bary,
        "V_Predicted_ISUT": V_pred,
        "Best_Beta": [best_beta] * len(R),
        "Chi2": [chi2] * len(R)
    })
    csv_path = os.path.join(data_dir, f"{gal_name}_fit_data.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"[File] Saved Data: {csv_path}")

    # 3b. Save run metadata (helps reviewer trace assumptions; does not affect results)
    meta = {
        'Galaxy': gal_name,
        'Category': category,
        'Best_Beta': best_beta,
        'Chi2': chi2,
        'UPS_D_DEFAULT': UPS_D_DEFAULT,
        'UPS_B_DEFAULT': UPS_B_DEFAULT,
        'A0_SI': A0_SI,
        'ACCEL_CONV': ACCEL_CONV,
        'A0_CODE': A0_CODE,
        'DATA_DIR': DATA_DIR,
        'DATA_URL_TEMPLATE': 'https://raw.githubusercontent.com/jobovy/sparc-rotation-curves/master/data/{gal}_rotmod.dat'
    }
    meta_path = os.path.join(data_dir, f"{gal_name}_run_metadata.csv")
    pd.DataFrame([meta]).to_csv(meta_path, index=False)
    print(f"[File] Saved metadata: {meta_path}")

    # 4. Save Figure
    plt.figure(figsize=(10, 6))
    
    # Plotting
    plt.errorbar(R, V_obs, yerr=V_err, fmt='ko', ecolor='gray', alpha=0.6, label='Observed Data', capsize=3)
    plt.plot(R, V_bary, 'b--', linewidth=2, label='Newtonian (Baryons Only)')
    plt.plot(R, V_pred, 'r-', linewidth=3, label=f'ISUT Fit (Beta={best_beta:.2f})')
    
    # Styling
    plt.title(f"Rotation Curve: {gal_name} (Beta={best_beta:.2f}, Chi2={chi2:.2f})", fontsize=14)
    plt.xlabel("Radius [kpc]", fontsize=12)
    plt.ylabel("Velocity [km/s]", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    fig_path = os.path.join(fig_dir, f"{gal_name}_fit_result.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[File] Saved Figure: {fig_path}")


def analyze_galaxy(gal_name):
    print(f"\n[Process] Analyzing {gal_name}...")
    
    # 1. Load Data
    data = get_data(gal_name)
    if data is None: return

    R, V_obs, V_err, V_gas, V_disk, V_bul = data
    
    # 2. Calculate Baryonic Component
    V_bary_sq = (np.abs(V_gas) * V_gas) \
              + (UPS_D_DEFAULT * np.abs(V_disk) * V_disk) \
              + (UPS_B_DEFAULT * np.abs(V_bul) * V_bul)
    V_bary = np.sqrt(np.maximum(V_bary_sq, 0))
    g_N = np.maximum(V_bary_sq, 0) / np.maximum(R, 0.01)
    y = g_N / A0_CODE
    
    # 3. Fit Beta Model
    def loss(beta):
        model = BetaModel(beta)
        nu = model.get_nu(y)
        g_pred = g_N * nu
        V_pred = np.sqrt(np.maximum(g_pred * R, 0))
        # Robust Error Handling
        err = np.maximum(V_err, 1.0)
        chi2 = np.sum(((V_obs - V_pred) / err)**2)
        return chi2

    # Optimization
    res = minimize_scalar(loss, bounds=(0.1, 5.0), method='bounded')
    best_beta = res.x
    min_chi2 = res.fun
    
    # 4. Generate Prediction for Plotting
    model = BetaModel(best_beta)
    nu = model.get_nu(y)
    g_pred = g_N * nu
    V_pred = np.sqrt(np.maximum(g_pred * R, 0))
    
    # 5. Output Result
    print(f"   -> Best Beta: {best_beta:.4f}")
    print(f"   -> Min Chi2:  {min_chi2:.2f}")
    
    # 6. Save Files (CSV & PNG)
    save_results(gal_name, R, V_obs, V_err, V_bary, V_pred, best_beta, min_chi2)


# ==============================================================================
# [5] Main Execution Loop
# ==============================================================================
def main():
    print("="*60)
    print("   ISUT Galaxy Fitter Tool (Single Galaxy Analysis)")
    print("="*60)
    
    while True:
        try:
            user_input = input("\nInput Galaxy Name (e.g., NGC6503) or 'q' to quit: ").strip()
            
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("[System] Exiting tool.")
                break
                
            if not user_input:
                continue
                
            analyze_galaxy(user_input)
            
        except KeyboardInterrupt:
            print("\n[System] Interrupted.")
            break
        except Exception as e:
            print(f"[Error] Unexpected error: {e}")

if __name__ == "__main__":
    main()
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD})
    except Exception:
        pass
