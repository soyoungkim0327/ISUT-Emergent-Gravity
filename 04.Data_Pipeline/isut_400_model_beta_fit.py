"""
ISUT Beta-Parameter Optimization & Global Distribution Suite
============================================================
Objective:
    Systematic optimization of the field-strength scaling parameter (Beta) 
    across the galactic sample to ensure physical consistency and model parsimony.

Key Methodologies:
1. Automated Scalar Minimization: 
   - Uses bounded optimization (minimize_scalar) to find the unique Beta 
     that minimizes Chi-square for each individual galaxy.
2. Statistical Distribution Profile: 
   - Analyzes the probability density of best-fit Beta values to determine 
     if the parameter converges to a universal physical range.
3. Comparative subset auditing: 
   - Benchmarks the Beta distribution of the 'Golden 12' sample against 
     the full 65-galaxy survey to detect sampling bias.
4. Convergence Verification: 
   - Implements Brent's method (brentq) to solve the interpolating 
     function nu(y) precisely without numerical divergence.

Output Structure:
    ./1.isut_model_beta_fit_Final_patched_v2/
      ├─ figures/ (Beta distribution & Comparison plots)
      └─ data/    (Per-galaxy best-fit Beta parameters)
"""


import os
import sys

from pathlib import Path
# --- ISUT shared helpers (path + SPARC io) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from isut_000_common import find_sparc_data_dir, ensure_rotmod_file, write_run_metadata

# Network download toggle (set ISUT_NO_DOWNLOAD=1 to disable)
ALLOW_NET_DOWNLOAD = os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None  # optional

from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")
plt.style.use('default') 
# ==============================================================================
# [1] Configuration: Smart Path Discovery (Automatic Directory Mapping)
# ==============================================================================

# Select the first valid path among candidates
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR}")

# Folder Structure Generation
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
DIRS = {
    "all_data": os.path.join(BASE_OUT_DIR, "All65", "data"),
    "all_figs": os.path.join(BASE_OUT_DIR, "All65", "figures"), # 개별 그래프 저장될 곳
    "all_fitdata": os.path.join(BASE_OUT_DIR, "All65", "fitdata"),
    "gold_data": os.path.join(BASE_OUT_DIR, "Golden12", "data"),
    "gold_figs": os.path.join(BASE_OUT_DIR, "Golden12", "figures"),
    "gold_fitdata": os.path.join(BASE_OUT_DIR, "Golden12", "fitdata"),
    "comp_data": os.path.join(BASE_OUT_DIR, "Comparison", "data"),
    "comp_figs": os.path.join(BASE_OUT_DIR, "Comparison", "figures"),
}

for d in DIRS.values():
    os.makedirs(d, exist_ok=True)

print(f"[System] Output directory initialized: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Galaxy List and Physical Constants
# ==============================================================================
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

GOLDEN_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", 
    "NGC5055", "NGC7331", "NGC6946", "NGC7793", "NGC1560", "UGC02885"
]

A0_SI = 1.2e-10 
ACCEL_CONV = 3.24078e-14 
A0_CODE = A0_SI / ACCEL_CONV  

# ==============================================================================
# [3] Physics Engine: Beta-Family Model (Strict Numerical Solver)
# ==============================================================================
def mu_beta(x, beta):
    """Interpolating function mu(x)."""
    return x / (1.0 + x**beta)**(1.0/beta)

def solve_nu_from_y(y, beta):
    """Inverse solver using Brent's method (High Precision).

    Minimal-risk robustness patch:
    - Keep the original bracket attempt.
    - If it fails, expand the bracket once before falling back to nu=1.0.
    """
    def func(x):
        return x * mu_beta(x, beta) - y
    try:
        x_min = np.sqrt(y) * 0.1
        x_max = y * 10.0 + 10.0
        x_sol = brentq(func, x_min, x_max)
        return x_sol / y
    except Exception:
        try:
            x_min = 1e-12
            x_max = max(1e3, float(y) * 1e3)
            x_sol = brentq(func, x_min, x_max)
            return x_sol / max(float(y), 1e-30)
        except Exception:
            return 1.0


class BetaModel:
    """Class to handle Beta model calculations efficiently."""
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
# [4] Data Load and Fitting/Visualization Functions
# ==============================================================================
def get_data(gal_name):
    path = os.path.join(DATA_DIR, f"{gal_name}_rotmod.dat")
    try:
        if not os.path.exists(path):
            try:
                path = str(ensure_rotmod_file(gal_name, Path(DATA_DIR), allow_download=ALLOW_NET_DOWNLOAD, timeout=3.0))
            except FileNotFoundError:
                return None
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None).apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 5: return None
        
        return df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values, \
               df.iloc[:,3].values, \
               df.iloc[:,4].values if df.shape[1]>4 else np.zeros(len(df)), \
               df.iloc[:,5].values if df.shape[1]>5 else np.zeros(len(df))
    except Exception:
        return None

def plot_individual_galaxy(gal_name, R, V_obs, V_err, V_bary, best_beta, save_dir, save_data_dir=None):
    """Generates and saves the rotation curve fit for a single galaxy.

    Reproducibility patch (No plot without data):
    - Alongside PNG, save per-point source data CSV (R, Vobs, Verr, Vbary, Vpred, gN, nu, etc.).
    """
    # Recalculate best fit curve
    V_bary_sq = V_bary**2
    g_N = np.maximum(V_bary_sq, 0) / np.maximum(R, 0.01)
    y = g_N / A0_CODE

    model = BetaModel(best_beta)
    nu = model.get_nu(y)
    g_pred = g_N * nu
    V_pred = np.sqrt(np.maximum(g_pred * R, 0))

    plt.figure(figsize=(10, 6))
    plt.errorbar(R, V_obs, yerr=V_err, fmt='ko', ecolor='gray', alpha=0.6, label='Observed')
    plt.plot(R, V_bary, 'b--', linewidth=2, label='Newtonian (Baryons)')
    plt.plot(R, V_pred, 'r-', linewidth=3, label=f'ISUT Fit (Beta={best_beta:.2f})')

    plt.title(f"Rotation Curve: {gal_name} (Beta={best_beta:.2f})", fontsize=14)
    plt.xlabel("Radius [kpc]")
    plt.ylabel("Velocity [km/s]")
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(save_dir, f"{gal_name}_fit.png")
    plt.savefig(save_path)
    plt.close()

    # Save source data CSV (optional)
    if save_data_dir is not None:
        os.makedirs(save_data_dir, exist_ok=True)
        df_src = pd.DataFrame({
            'R_kpc': R,
            'Vobs_kms': V_obs,
            'Verr_kms': V_err,
            'Vbary_kms': V_bary,
            'Vpred_kms': V_pred,
            'gN_code': g_N,
            'y': y,
            'nu': nu,
            'beta': np.full_like(R, float(best_beta), dtype=float),
            'a0_code': np.full_like(R, float(A0_CODE), dtype=float),
        })
        df_src.to_csv(os.path.join(save_data_dir, f"{gal_name}_fit_source_data.csv"), index=False)


def fit_beta_for_galaxy(gal):
    """Fits Beta model and plots the result."""
    data = get_data(gal)
    if data is None: return None
    
    R, V_obs, V_err, V_gas, V_disk, V_bul = data
    V_bary_sq = (V_gas**2) + 0.5*(V_disk**2) + 0.7*(V_bul**2)
    V_bary = np.sqrt(np.maximum(V_bary_sq, 0))
    g_N = np.maximum(V_bary_sq, 0) / np.maximum(R, 0.01)
    y = g_N / A0_CODE
    
    # Loss Function
    def loss(beta):
        model = BetaModel(beta)
        nu = model.get_nu(y)
        g_pred = g_N * nu
        V_pred = np.sqrt(np.maximum(g_pred * R, 0))
        chi2 = np.sum(((V_obs - V_pred) / np.maximum(V_err, 1.0))**2)
        return chi2

    # Minimize
    res = minimize_scalar(loss, bounds=(0.5, 5.0), method='bounded')
    best_beta = res.x
    chi2_free = res.fun
    
    is_golden = gal in GOLDEN_GALAXIES
    
    # Save Individual Plot (Feature from New code ported to Old code)
    target_dir = DIRS['all_figs'] # Default to All folder
    if is_golden:
        # If golden, also save a copy or just save in golden folder?
        # Requirement was separate folders. We save to both or just specific.
        # Let's save to Golden folder if golden, All folder if not, or both.
        # To keep it clean: Save Golden ones to Golden folder ONLY? 
        # Or save ALL to All folder, and Golden to Golden folder also?
        # Let's save Golden to Golden folder.
        plot_individual_galaxy(gal, R, V_obs, V_err, V_bary, best_beta, DIRS['gold_figs'], save_data_dir=DIRS['gold_fitdata'])
    
    # Also save everything to All_figs for completeness
    plot_individual_galaxy(gal, R, V_obs, V_err, V_bary, best_beta, DIRS['all_figs'], save_data_dir=DIRS['all_fitdata'])
    
    return {
        "Galaxy": gal,
        "Best_Beta": best_beta,
        "Chi2": chi2_free,
        "Is_Golden": is_golden
    }

# ==============================================================================
# [5] Comparative Analysis (Core)
# ==============================================================================
def run_comparative_analysis(df):
    print("\n" + "="*60)
    print("[ANALYSIS] COMPARATIVE ANALYSIS: GOLDEN 12 vs ALL 65")
    print("="*60)
    
    df_golden = df[df['Is_Golden']]
    stats_all = df['Best_Beta'].describe()
    stats_gold = df_golden['Best_Beta'].describe()
    
    print(f"All 65 Median Beta    : {stats_all['50%']:.4f}")
    print(f"Golden 12 Median Beta : {stats_gold['50%']:.4f}")

    # Basic completeness check (helps reviewer transparency)
    if len(df) != len(FULL_GALAXIES):
        print(f"[Warning] Missing galaxies in All65 results: {len(FULL_GALAXIES) - len(df)}")
    if len(df_golden) != len(GOLDEN_GALAXIES):
        print(f"[Warning] Missing galaxies in Golden12 results: {len(GOLDEN_GALAXIES) - len(df_golden)}")
    
    # Save Data
    df.to_csv(os.path.join(DIRS['all_data'], "Beta_Fitting_All65.csv"), index=False)
    df_golden.to_csv(os.path.join(DIRS['gold_data'], "Beta_Fitting_Golden12.csv"), index=False)

    # Visualization (Overlay Plot)
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(data=df, x='Best_Beta', fill=True, color='gray', alpha=0.3, label='All 65 Galaxies')
    sns.kdeplot(data=df_golden, x='Best_Beta', fill=True, color='gold', alpha=0.5, label='Golden 12 Subset')
    
    plt.axvline(x=1.0, color='r', linestyle='--', linewidth=2, label='Simple Model (Beta=1)')
    plt.axvline(x=stats_gold['50%'], color='orange', linestyle=':', linewidth=2, label=f'Golden Median ({stats_gold["50%"]:.2f})')
    
    plt.title("Optimal Beta Distribution: All 65 vs Golden 12", fontsize=14)
    plt.xlabel("Best Fit Beta Parameter")
    plt.ylabel("Density")
    plt.xlim(0.5, 5.0)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    path_fig = os.path.join(DIRS['comp_figs'], "Beta_Comparison_Analysis.png")
    plt.savefig(path_fig, dpi=300)
    print(f"[File] Saved Comparison Plot: {path_fig}")
    
    # Save Source Data for Plot
    df_export = df[['Galaxy', 'Best_Beta', 'Is_Golden']].copy()
    df_export.to_csv(os.path.join(DIRS['comp_data'], "Beta_Comparison_Source_Data.csv"), index=False)
    
    plt.close()

# ==============================================================================
# [6] Main Execution Entry Point
# ==============================================================================
def main():
    print(f"[Process] Starting Beta Model Fitting for {len(FULL_GALAXIES)} Galaxies...")
    results = []
    failed = []
    
    for i, gal in enumerate(FULL_GALAXIES):
        sys.stdout.write(f"\r[Process] Fitting {gal} [{i+1}/{len(FULL_GALAXIES)}]")
        sys.stdout.flush()
        
        res = fit_beta_for_galaxy(gal)
        if res:
            results.append(res)
        else:
            failed.append(gal)
            
    print("\n[Process] Fitting Complete.")

    if failed:
        miss_path = os.path.join(BASE_OUT_DIR, "Beta_Fitting_Missing.csv")
        pd.DataFrame({"Missing_Galaxy": failed}).to_csv(miss_path, index=False)
        print(f"[Warning] {len(failed)} galaxies failed to load/fit. Saved list: {miss_path}")
    
    if results:
        df = pd.DataFrame(results)
        run_comparative_analysis(df)
    else:
        print("[Error] No results generated.")


    # --- Reviewer-facing metadata (inputs, environment, counts) ---
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD}, notes={"n_results": len(results), "n_failed": len(failed)})
    except Exception:
        pass

if __name__ == "__main__":
    main()