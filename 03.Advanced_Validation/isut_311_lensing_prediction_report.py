"""
ISUT Cross-Domain Prediction: Gravitational Lensing Proxy
=========================================================
Objective:
    Validates the ISUT framework by predicting gravitational lensing 
    signatures using parameters derived solely from rotation curves.

Key Validations:
1. Kinetic-to-Lensing Bridge: Tests if the mass-acceleration relation 
   holds true for light bending (lensing) as well as orbital motion.
2. Ratio Analysis: Computes the ISUT-to-Newtonian lensing ratio to 
   quantify the expected 'extra' bending without dark matter.
3. Radius-Dependent Scaling: Analyzes how the lensing signal evolves at 
   outer radii, providing a clear testable prediction for future surveys.
"""


import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# [0] Compatibility Patch (NumPy Trapz)
# ==============================================================================
try:
    trapz = np.trapezoid # NumPy 2.0+
except AttributeError:
    trapz = np.trapz # NumPy < 2.0

# ==============================================================================
# [1] Configuration: Smart Path Finder & Output Setup
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR} (override via ISUT_SPARC_DIR)")

BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
os.makedirs(BASE_OUT_DIR, exist_ok=True)

print(f"[System] ISUT Lensing Proxy Prediction (Report Ver) Initialized")
print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Physics Constants & Engines
# ==============================================================================
G_const = 4.301e-6  # kpc (km/s)^2 / M_sun
c_light = 300000.0  # km/s
A0_SI = 1.2e-10     # m/s^2
ACCEL_CONV = 3.24078e-14 
A0_CODE = A0_SI / ACCEL_CONV 

def nu_isut(y):
    """ ISUT Interpolation Function (Entropic Force) """
    y_safe = np.maximum(y, 1e-12)
    return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y_safe)

# Lensing geometry factor (D_ls/D_s). Default 1.0 keeps legacy numeric outputs.
DLS_OVER_DS = 1.0

def calculate_deflection_angle_proxy(V_circ_sq, dls_over_ds=DLS_OVER_DS, gamma_ppn: float = 1.0):
    """ 
    Lensing **proxy** (NOT a full lens equation / ray-tracing):
      alpha_proxy = (D_ls/D_s) * 2(1+gamma) V^2 / c^2  [radians]

    Notes:
      * gamma=1 reproduces the legacy 4 V^2/c^2 factor.
      * This is only a proxy scaling; it is *not* a projected-mass integral.
    Returned in arcsec purely as a convenient angular scaling.
    """
    alpha = float(dls_over_ds) * 2.0 * (1.0 + float(gamma_ppn)) * V_circ_sq / (c_light**2)
    return alpha * 206265.0

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
# [4] Analysis Logic
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

def process_subset(galaxy_list, subset_name, *, gamma_isut: float = 1.0, dls_over_ds: float = DLS_OVER_DS):
    print(f"\n[Analysis] Processing Lensing Prediction for: {subset_name}")
    
    subset_dir = os.path.join(BASE_OUT_DIR, subset_name)
    fig_dir = os.path.join(subset_dir, "figures")
    data_dir = os.path.join(subset_dir, "data")
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    detailed_report = []
    
    for i, gal in enumerate(galaxy_list):
        print(f"   -> Analyzing {gal} ({i+1}/{len(galaxy_list)})...", end="\r")
        
        data = get_data(gal)
        if data is None: continue
        
        R, V_obs, V_err, V_gas, V_disk, V_bul = data
        
        # 1. Calculate Accelerations
        V_bary2 = (np.abs(V_gas)*V_gas) + (0.5 * np.abs(V_disk)*V_disk) + (0.7 * np.abs(V_bul)*V_bul)
        V_bary2 = np.maximum(V_bary2, 0.0)
        
        g_N = V_bary2 / np.maximum(R, 0.01)
        y = g_N / A0_CODE
        nu = nu_isut(y)
        g_ISUT = g_N * nu
        V_ISUT2 = g_ISUT * R
        
        # 2. Calculate Deflection Angles
        # Newtonian baseline is assumed to follow GR with PPN gamma=1.
        alpha_N = calculate_deflection_angle_proxy(V_bary2, dls_over_ds=dls_over_ds, gamma_ppn=1.0)
        # ISUT metric completion may in general have gamma != 1; we expose it as a parameter.
        alpha_ISUT = calculate_deflection_angle_proxy(V_ISUT2, dls_over_ds=dls_over_ds, gamma_ppn=float(gamma_isut))
        
        # 3. Ratio
        ratio = np.divide(alpha_ISUT, alpha_N, out=np.ones_like(alpha_ISUT), where=alpha_N > 1e-6)
        
        # 4. Visualization
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(R, alpha_ISUT, 'r-', linewidth=2, label='ISUT Lensing (Proxy)')
        plt.plot(R, alpha_N, 'b--', linewidth=2, label='Newtonian (Baryon Only)')
        plt.title(f"[{gal}] Deflection Angle Proxy Profile")
        plt.xlabel("Impact Parameter b [kpc]")
        plt.ylabel("Deflection Proxy [arcsec]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(R, ratio, 'k-', linewidth=2)
        plt.axhline(y=1.0, color='gray', linestyle='--')
        plt.title("Lensing Proxy Enhancement Ratio")
        plt.xlabel("Impact Parameter b [kpc]")
        plt.ylabel("Ratio (Enhancement Factor)")
        plt.grid(True, alpha=0.3)
        
        fig_path = os.path.join(fig_dir, f"{gal}_Lensing.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close()
        
        # 5. Save Individual Data
        df_gal = pd.DataFrame({
            "Radius_kpc": R,
            "Alpha_Newton_proxy_arcsec": alpha_N,
            "Alpha_ISUT_proxy_arcsec": alpha_ISUT,
            "Enhancement_Ratio": ratio,
            "gamma_newton": 1.0,
            "gamma_isut": float(gamma_isut),
            "dls_over_ds": float(dls_over_ds),
        })
        df_gal.to_csv(os.path.join(data_dir, f"{gal}_Lensing_Data.csv"), index=False)
        
        # 6. Collect Detailed Report Data (Snapshot at Max Radius)
        # Taking the last point (max radius) as representative
        # Representative summaries (avoid single-point cherry-pick)
        n = len(R)
        k0 = int(np.floor(0.7 * n))  # outer 30% region
        outer_mean_ratio = float(np.mean(ratio[k0:])) if n > 0 else float('nan')

        detailed_report.append({
            "Galaxy": gal,
            "Impact_Radius_kpc": float(R[-1]),
            "Alpha_Newton_proxy_arcsec": float(alpha_N[-1]),
            "Alpha_ISUT_proxy_arcsec": float(alpha_ISUT[-1]),
            "Enhancement_Ratio_last": float(ratio[-1]),
            "Mean_Ratio": float(np.mean(ratio)),
            "OuterMean_Ratio": outer_mean_ratio,
            "Max_Ratio": float(np.max(ratio)),
            "gamma_isut": float(gamma_isut),
            "dls_over_ds": float(dls_over_ds),
            "Figure_File": os.path.basename(fig_path),
            "Figure_Path": os.path.relpath(fig_path, start=BASE_OUT_DIR)
        })
        
    # Save Report
    if detailed_report:
        # Save "Old Style" Detailed Report (Specific Values)
        df_report = pd.DataFrame(detailed_report)
        report_path = os.path.join(data_dir, f"Lensing_Prediction_Report_{subset_name}.csv")
        df_report.to_csv(report_path, index=False)
        print(f"\n   [Success] Detailed Report Saved: {report_path}")

        # Save a tiny metadata file clarifying proxy assumptions
        meta_path = os.path.join(data_dir, f"Lensing_Proxy_Meta_{subset_name}.csv")
        pd.DataFrame([{
            'subset': subset_name,
            'dls_over_ds': float(dls_over_ds),
            'gamma_newton': 1.0,
            'gamma_isut': float(gamma_isut),
            'definition': 'alpha_proxy = (D_ls/D_s) * 2(1+gamma) V^2 / c^2 (arcsec scaling)',
            'note': 'Proxy only; not a full lensing calculation (no projected-mass integral).',
        }]).to_csv(meta_path, index=False)


def main():
    ap = argparse.ArgumentParser(description="ISUT lensing proxy report (with optional PPN gamma)")
    ap.add_argument("--gamma", type=float, default=1.0, help="PPN gamma for the ISUT metric completion (default 1)")
    ap.add_argument("--dls_over_ds", type=float, default=DLS_OVER_DS, help="Geometry factor D_ls/D_s (default 1)")
    ap.add_argument("--subset", choices=["Golden12", "All65", "both"], default="both")
    args = ap.parse_args()

    if args.subset in ("Golden12", "both"):
        process_subset(GOLDEN_GALAXIES, "Golden12", gamma_isut=args.gamma, dls_over_ds=args.dls_over_ds)
    if args.subset in ("All65", "both"):
        process_subset(FULL_GALAXIES, "All65", gamma_isut=args.gamma, dls_over_ds=args.dls_over_ds)

    print("\n" + "="*60)
    print("✅ All Lensing Predictions & Reports Completed.")
    print(f"📂 Results saved in: {BASE_OUT_DIR}")
    print("="*60)

if __name__ == "__main__":
    main()
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD}, notes={"n_golden": len(GOLDEN_GALAXIES), "n_all65": len(FULL_GALAXIES)})
    except Exception:
        pass
