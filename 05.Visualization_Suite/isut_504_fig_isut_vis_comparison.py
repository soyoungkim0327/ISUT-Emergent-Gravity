import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import seaborn as sns
except Exception:
    sns = None  # optional

import warnings
import sys

from pathlib import Path
# --- ISUT shared helpers (path + SPARC io) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from isut_000_common import find_sparc_data_dir, ensure_rotmod_file, write_run_metadata

ALLOW_NET_DOWNLOAD = os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1"

# [Configuration] Suppress warnings and set academic plot style
warnings.filterwarnings("ignore")
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif', 'axes.grid': True,
    'grid.alpha': 0.3, 'lines.linewidth': 2.0, 'figure.dpi': 150, 'savefig.dpi': 300
})

# ==============================================================================
# [1] System Configuration: Smart Path Finder & Output Setup
# ==============================================================================

# 1.1 Identify Current Directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# 1.2 Smart Path Finder (Data Directory)
DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR}")

BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(BASE_OUT_DIR, "Comparison", "figures")
DATA_DIR_OUT = os.path.join(BASE_OUT_DIR, "Comparison", "data")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR_OUT, exist_ok=True)

print(f"[System] Output directory initialized: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Physics & Galaxy Configuration
# ==============================================================================
# Physical Constants
ACCEL_CONV = 3.24078e-14 
A0_SI = 1.2e-10 
A0_CODE = A0_SI / ACCEL_CONV

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

# ==============================================================================
# [3] Data Loader
# ==============================================================================
def get_data(gal_name):
    """Loads galaxy data, downloading if necessary."""
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
    except:
        return None

# ==============================================================================
# [4] Model Definitions
# ==============================================================================
def model_heuristic(g_bary, g_obs, a0=A0_CODE):
    """Old Heuristic Model: Simple interpolation"""
    return g_bary / (1 - np.exp(-np.sqrt(g_bary/a0)))

def model_lagrangian(g_bary, g_obs, a0=A0_CODE):
    """New Lagrangian (Beta) Model: Physically derived"""
    y = g_bary / a0
    # Beta=1.0 assumption for Golden 12
    nu = 0.5 + 0.5 * np.sqrt(1 + 4/y)
    return g_bary * nu

# ==============================================================================
# [5] Visualization & Analysis Suite
# ==============================================================================
def main():
    print(f"[Process] Starting Ultimate Comparison Suite Generation...")
    
    # 5.1 Data Collection
    stats_data = []
    skipped = []
    
    for gal in FULL_GALAXIES:
        data = get_data(gal)
        if data is None:
            skipped.append({"Galaxy": gal, "Reason": "data_load_failed_or_too_short"})
            continue
        
        R, V_obs, V_err, V_gas, V_disk, V_bul = data
        V_bary_sq = np.abs(V_gas)*V_gas + 0.5*np.abs(V_disk)*V_disk + 0.7*np.abs(V_bul)*V_bul
        V_bary = np.sqrt(np.maximum(V_bary_sq, 0))
        
        g_obs = (V_obs**2) / np.maximum(R, 0.01)
        g_bar = np.maximum(V_bary_sq, 0) / np.maximum(R, 0.01)
        
        # Predictions
        g_pred_new = model_lagrangian(g_bar, g_obs)
        g_pred_old = model_heuristic(g_bar, g_obs)
        
        # Residuals (log-space); add epsilon to avoid log10(0)
        eps = 1e-30
        res_new = np.log10(np.maximum(g_obs, eps)) - np.log10(np.maximum(g_pred_new, eps))
        res_old = np.log10(np.maximum(g_obs, eps)) - np.log10(np.maximum(g_pred_old, eps))
        
        # Summary Stats
        # NOTE: The SPARC "rotmod" file provides velocity contributions. Without distances and M/L priors,
        # we keep a transparent *proxy* for baryonic mass: M_proxy ~ V_bary^2 * R (up to constants).
        m_proxy_v2r = float(np.maximum(V_bary_sq[-1], 0) * np.maximum(R[-1], 0))

        # NOTE: These are *not* chi-square statistics (no uncertainty model used here).
        sse_new = float(np.sum((g_obs - g_pred_new) ** 2))
        sse_old = float(np.sum((g_obs - g_pred_old) ** 2))

        stats_data.append({
            "Galaxy": gal,
            "Is_Golden": gal in GOLDEN_GALAXIES,
            # Backward compatible column name (do NOT interpret as physical mass)
            "M_bary": m_proxy_v2r,
            "M_proxy_V2R": m_proxy_v2r,
            "V_flat": np.mean(V_obs[-3:]) if len(V_obs) >=3 else V_obs[-1],
            "Res_New_Mean": np.mean(res_new),
            "Res_Old_Mean": np.mean(res_old),
            # Backward compatible column names (actually SSE)
            "Chi2_New": sse_new,
            "Chi2_Old": sse_old,
            "SSE_New": sse_new,
            "SSE_Old": sse_old,
        })
    
    df = pd.DataFrame(stats_data)
    df_gold = df[df['Is_Golden']]
    
    # Save Source Data
    csv_path = os.path.join(DATA_DIR_OUT, "Ultimate_Comparison_Source_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"[File] Saved Source Data: {csv_path}")

    # Save skipped list (reproducibility + reviewer defense)
    skipped_path = os.path.join(DATA_DIR_OUT, "Ultimate_Comparison_Skipped_Galaxies.csv")
    pd.DataFrame(skipped).to_csv(skipped_path, index=False)
    if len(skipped) > 0:
        print(f"[Warning] Skipped {len(skipped)} galaxies. Saved list: {skipped_path}")

    # 5.2 Visualization (8-Panel Figure)
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)
    
    # Panel 1: BTFR
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.loglog(df['M_proxy_V2R'], df['V_flat'], 'ko', alpha=0.3, label='All')
    ax1.loglog(df_gold['M_proxy_V2R'], df_gold['V_flat'], 'ro', label='Golden 12')
    ax1.set_xlabel('Baryonic mass proxy ~ V_bary^2 R (arb. units)')
    ax1.set_ylabel('Flat Velocity')
    ax1.set_title('1. BTFR-like proxy (consistency check)')
    ax1.legend()
    
    # Panel 2: Residuals Histogram (Accuracy)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(df['Res_New_Mean'], bins=20, alpha=0.5, color='blue', label='New Lagrangian')
    ax2.hist(df['Res_Old_Mean'], bins=20, alpha=0.5, color='gray', label='Old Heuristic')
    ax2.set_title('2. Model Accuracy (Residuals)')
    ax2.set_xlabel('Log(g_obs) - Log(g_pred)')
    ax2.legend()
    
    # Panel 3: Golden 12 Comparison (Zoom-in)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(['New', 'Old'], [df_gold['SSE_New'].mean(), df_gold['SSE_Old'].mean()], color=['blue', 'gray'])
    ax3.set_title('3. Golden 12 mean SSE (not chi-square)')
    ax3.set_ylabel('Sum of squared error (g-space)')
    
    # Panel 4: RAR (Theory vs Obs)
    ax4 = fig.add_subplot(gs[0, 3])
    x = np.logspace(-13, -8, 100)
    ax4.loglog(x, x, 'k--', label='1:1')
    ax4.loglog(x, model_lagrangian(x, x), 'b-', label='New Theory')
    ax4.set_title('4. RAR Prediction Curve')
    ax4.legend()
    
    # Panel 5: Theoretical basis (Fixed: clean text formatting)
    ax5 = fig.add_subplot(gs[1, 0])
    ax5.axis('off')

    # FIX: Separated raw strings from newlines to prevent "\n" literals in output
    # FIX: Changed \tfrac to \frac for better compatibility
    basis_text = (
        "Proxy mapping (disk kinematics):\n" +
        r"$g_{\rm obs}=g_{\rm bar}\,\nu(g_{\rm bar}/a_0)$" + "\n" +
        r"$\nu_{\rm simple}(y)=\frac{1}{2}\,(1+\sqrt{1+4/y})$" + "\n\n" +
        "Clock-rate interpretation:\n" +
        r"$\nu=\alpha^2$ (update/clock factor)" + "\n\n" +
        "Conservative completion (field solve):\n" +
        r"QUMOND (two Poisson solves), $a=-\nabla\Phi$"
    )
    # Adjusted position slightly
    ax5.text(0.02, 1.0, basis_text, va='top', ha='left', fontsize=11)
    ax5.set_title("5. Theoretical Basis (summary)")

    # Panel 6: Error Distribution (KDE) — seaborn optional
    ax6 = fig.add_subplot(gs[1, 1])
    if sns is not None:
        sns.kdeplot(data=df, x='Res_New_Mean', fill=True, alpha=0.3, ax=ax6, label='New')
        sns.kdeplot(data=df, x='Res_Old_Mean', fill=True, alpha=0.3, ax=ax6, label='Old')
        ax6.set_title('6. Error Distribution (KDE)')
    else:
        # Fallback: density-normalized histograms (no seaborn dependency)
        ax6.hist(df['Res_New_Mean'], bins=30, density=True, alpha=0.3, label='New')
        ax6.hist(df['Res_Old_Mean'], bins=30, density=True, alpha=0.3, label='Old')
        ax6.set_title('6. Error Distribution (density hist)')
    ax6.legend()

    # Panel 7: Residual vs V_flat (diagnostic)
    ax7 = fig.add_subplot(gs[1, 2])
    ax7.scatter(df['V_flat'], df['Res_New_Mean'], c='b', alpha=0.5)
    ax7.axhline(0, color='k', linestyle='--')
    ax7.set_xlabel('Velocity')
    ax7.set_ylabel('Residual (New Model)')
    ax7.set_title('7. Residual vs V_flat (diagnostic)')

    # Panel 8: Conclusion (Fixed: Layout and Alignment)
    ax8 = fig.add_subplot(gs[1, 3])
    ax8.axis('off')

    n_all = int(len(df))
    n_gold = int(len(df_gold))
    # Compare on this script's internal metric only (SSE). Do not generalize beyond this.
    mean_new_gold = float(df_gold['SSE_New'].mean()) if n_gold > 0 else float('nan')
    mean_old_gold = float(df_gold['SSE_Old'].mean()) if n_gold > 0 else float('nan')
    better = "New" if (mean_new_gold < mean_old_gold) else "Old"

    text_content = (
        "CONCLUSION:\n\n"
        f"1) Dataset coverage: {n_all} / {len(FULL_GALAXIES)} galaxies\n"
        "   (see skipped list if any).\n\n"
        "2) Accuracy metric (this figure):\n"
        f"   Golden12 mean SSE:\n"
        f"   New={mean_new_gold:.3e}\n"
        f"   Old={mean_old_gold:.3e}\n"
        f"   Better on this metric: {better}\n\n"
        "3) Interpretation:\n"
        "   This is a visualization/diagnostic summary.\n"
        "   Paper-level claims must rely on the\n"
        "   main holdout/robustness analyses."
    )
    # FIX: Changed va='top' and position to 0.9 to prevent cutoff/overlap
    ax8.text(0.05, 0.95, text_content, fontsize=11, va='top', ha='left')
    ax8.set_title("8. Final Verdict")

    plt.tight_layout()
    
    # Save Figure
    fig_path = os.path.join(FIG_DIR, "Ultimate_Comparison_Suite.png")
    plt.savefig(fig_path, dpi=300)
    print(f"[File] Saved Comparison Figure: {fig_path}")
    plt.close()
    
    print(f"\n[System] All tasks completed successfully.")

if __name__ == "__main__":
    main()
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD})
    except Exception:
        pass