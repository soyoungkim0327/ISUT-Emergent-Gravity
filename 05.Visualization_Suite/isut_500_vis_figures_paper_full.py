import os
import sys

from pathlib import Path
# --- ISUT shared helpers (path + SPARC io) ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from isut_000_common import find_sparc_data_dir, ensure_rotmod_file, write_run_metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except Exception:
    sns = None  # optional

from scipy.optimize import brentq, minimize_scalar
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
import warnings
import requests

# [설정] 경고 무시 및 스타일 설정
warnings.filterwarnings("ignore")
plt.style.use('default')

# ============================================================================== 
# [0] Reproducibility knobs (minimal, result-preserving)
# - Keep defaults identical to current behavior, but make failures visible.
# ============================================================================== 
ALLOW_NET_DOWNLOAD = os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1"
NET_TIMEOUT_SEC = 3

_DOWNLOAD_LOG = []        # (galaxy, url, status)
_SKIPPED_GALAXIES = []    # galaxies with missing or unreadable data
_BRENTQ_FAIL_COUNT = 0

# ==============================================================================
# [1] 시스템 설정: 스마트 경로 탐색 & 폴더 구조화
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# 데이터 경로 자동 탐색
DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR}")

# 출력 폴더 생성
BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
DIRS = {
    "all_data": os.path.join(BASE_OUT_DIR, "All65", "data"),
    "all_figs": os.path.join(BASE_OUT_DIR, "All65", "figures"),
    "gold_data": os.path.join(BASE_OUT_DIR, "Golden12", "data"),
    "gold_figs": os.path.join(BASE_OUT_DIR, "Golden12", "figures"),
    "comp_data": os.path.join(BASE_OUT_DIR, "Comparison", "data"),
    "comp_figs": os.path.join(BASE_OUT_DIR, "Comparison", "figures"),
}
for d in DIRS.values(): os.makedirs(d, exist_ok=True)

print(f"[System] Output directory initialized: {BASE_OUT_DIR}")

# ==============================================================================
# [2] 은하 리스트 & 물리 상수
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
# [3] 물리 엔진: Beta Model (정밀 수치해석)
# ==============================================================================
def mu_beta(x, beta): return x / (1.0 + x**beta)**(1.0/beta)

def solve_nu_from_y(y, beta):
    def func(x): return x * mu_beta(x, beta) - y
    try:
        x_min = np.sqrt(y) * 0.1; x_max = y * 10.0 + 10.0
        return brentq(func, x_min, x_max) / y
    except:
        # Keep legacy fallback (result-preserving) but record failure count
        global _BRENTQ_FAIL_COUNT
        _BRENTQ_FAIL_COUNT += 1
        return 1.0

class BetaModel:
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
# [4] 데이터 로드 및 저장 유틸리티
# ==============================================================================
def get_data(gal_name):
    path = os.path.join(DATA_DIR, f"{gal_name}_rotmod.dat")
    url = f"https://raw.githubusercontent.com/jobovy/sparc-rotation-curves/master/data/{gal_name}_rotmod.dat"
    try:
        # Attempt to fetch missing files (legacy behavior) but record outcomes.
        if not os.path.exists(path):
            if ALLOW_NET_DOWNLOAD:
                try:
                    r = requests.get(url, timeout=NET_TIMEOUT_SEC)
                    _DOWNLOAD_LOG.append((gal_name, url, int(getattr(r, 'status_code', -1))))
                    if getattr(r, "status_code", None) == 200:
                        os.makedirs(DATA_DIR, exist_ok=True)
                        with open(path, "w") as f:
                            f.write(r.text)
                except Exception as e:
                    _DOWNLOAD_LOG.append((gal_name, url, f"EXC:{type(e).__name__}"))
            else:
                _DOWNLOAD_LOG.append((gal_name, url, "SKIP"))

        if not os.path.exists(path):
            _SKIPPED_GALAXIES.append((gal_name, "missing_dat"))
            return None

        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None).apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 5:
            _SKIPPED_GALAXIES.append((gal_name, "too_few_rows"))
            return None

        return (
            df.iloc[:,0].values,
            df.iloc[:,1].values,
            df.iloc[:,2].values,
            df.iloc[:,3].values,
            df.iloc[:,4].values if df.shape[1]>4 else np.zeros(len(df)),
            df.iloc[:,5].values if df.shape[1]>5 else np.zeros(len(df)),
        )
    except Exception as e:
        _SKIPPED_GALAXIES.append((gal_name, f"read_error:{type(e).__name__}"))
        return None

def save_plot_and_data(fig, data_df, filename_base, target_dirs):
    for fig_dir, data_dir in target_dirs:
        fig.savefig(os.path.join(fig_dir, f"{filename_base}.png"), dpi=300, bbox_inches='tight')
        if data_df is not None:
            data_df.to_csv(os.path.join(data_dir, f"{filename_base}_source_data.csv"), index=False)

# ==============================================================================
# [5] 분석 코어: 개별 피팅 + RAR + BTFR 데이터 수집
# ==============================================================================
def process_galaxy(gal):
    data = get_data(gal)
    if data is None: return None
    R, V_obs, V_err, V_gas, V_disk, V_bul = data
    
    # Physics
    # NOTE: Keep legacy component combination (result-preserving). If any component contains
    # negative values, the sign-preserving |V|*V may yield negative contributions; we later clip.
    if (np.any(V_gas < 0) or np.any(V_disk < 0) or np.any(V_bul < 0)):
        _SKIPPED_GALAXIES.append((gal, "has_negative_component_values"))
    V_bary_sq = np.abs(V_gas)*V_gas + 0.5*np.abs(V_disk)*V_disk + 0.7*np.abs(V_bul)*V_bul
    g_N = np.maximum(V_bary_sq, 0) / np.maximum(R, 0.01)
    y = g_N / A0_CODE
    
    # Fit Beta
    def loss(beta):
        model = BetaModel(beta)
        nu = model.get_nu(y)
        g_pred = g_N * nu
        V_pred = np.sqrt(np.maximum(g_pred * R, 0))
        return np.sum(((V_obs - V_pred) / np.maximum(V_err, 1.0))**2)

    res = minimize_scalar(loss, bounds=(0.5, 5.0), method='bounded')
    best_beta = res.x
    
    # Model Curve for Plot
    model = BetaModel(best_beta)
    nu = model.get_nu(y)
    g_pred = g_N * nu
    V_pred = np.sqrt(np.maximum(g_pred * R, 0))
    
    # --- [Plot 1] Individual Rotation Curve ---
    fig = plt.figure(figsize=(8, 6))
    plt.errorbar(R, V_obs, yerr=V_err, fmt='ko', ecolor='gray', alpha=0.6, label='Observed')
    plt.plot(R, V_bary_sq**0.5, 'b--', label='Baryonic')
    plt.plot(R, V_pred, 'r-', linewidth=2, label=f'ISUT (Beta={best_beta:.2f})')
    plt.title(f"{gal} Rotation Curve")
    plt.xlabel("Radius [kpc]"); plt.ylabel("Velocity [km/s]"); plt.legend()
    
    targets = [(DIRS['all_figs'], DIRS['all_data'])]
    if gal in GOLDEN_GALAXIES: targets.append((DIRS['gold_figs'], DIRS['gold_data']))
    
    save_plot_and_data(fig, pd.DataFrame({"R":R, "V_obs":V_obs, "V_pred":V_pred}), f"{gal}_curve", targets)
    plt.close(fig)
    
    # Data for Aggregate Analysis (RAR, BTFR)
    return {
        "Galaxy": gal, "Best_Beta": best_beta, "Is_Golden": gal in GOLDEN_GALAXIES,
        "V_flat": np.mean(V_obs[-3:]), 
        "M_bary": float(V_bary_sq[-1] * R[-1]), # Proxy Mass (code-units)
        # For RAR (Arrays)
        "g_obs_list": (V_obs**2 / R).tolist(),
        "g_bar_list": g_N.tolist()
    }

# ==============================================================================
# [6] 통합 분석: RAR, BTFR, Correlation (구형 코드 기능 복원)
# ==============================================================================
def run_full_analysis(results):
    df = pd.DataFrame(results)
    df_gold = df[df['Is_Golden']]
    
    # --- [Analysis 1] Beta Distribution (Comparison) ---
    fig = plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Best_Beta', fill=True, color='gray', label='All 65')
    sns.kdeplot(data=df_gold, x='Best_Beta', fill=True, color='gold', label='Golden 12')
    plt.axvline(1.0, color='r', linestyle='--'); plt.legend()
    plt.title("Beta Parameter Distribution")
    save_plot_and_data(fig, df[['Galaxy', 'Best_Beta', 'Is_Golden']], "Beta_Distribution", [(DIRS['comp_figs'], DIRS['comp_data'])])
    plt.close(fig)
    
    # --- [Analysis 2] Velocity vs Beta (Correlation) ---
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(df['V_flat'], df['Best_Beta'], c='gray', alpha=0.5, label='All 65')
    plt.scatter(df_gold['V_flat'], df_gold['Best_Beta'], c='gold', edgecolors='k', s=80, label='Golden 12')
    plt.axhline(1.0, color='r', linestyle='--'); plt.xlabel("V_flat"); plt.ylabel("Beta"); plt.legend()
    plt.title("Correlation: V_flat vs Beta")
    save_plot_and_data(fig, df, "Corr_Vflat_Beta", [(DIRS['comp_figs'], DIRS['comp_data'])])
    plt.close(fig)

    # --- [Analysis 3] RAR (Radial Acceleration Relation) ---
    # Unpack all points
    all_g_obs, all_g_bar, all_is_gold = [], [], []
    for res in results:
        all_g_obs.extend(res['g_obs_list'])
        all_g_bar.extend(res['g_bar_list'])
        all_is_gold.extend([res['Is_Golden']] * len(res['g_obs_list']))
    
    rar_df = pd.DataFrame({"g_obs": all_g_obs, "g_bar": all_g_bar, "Is_Golden": all_is_gold})
    rar_gold = rar_df[rar_df['Is_Golden']]
    
    fig = plt.figure(figsize=(8, 8))
    plt.loglog(rar_df['g_bar'], rar_df['g_obs'], 'k.', alpha=0.1, label='All Points')
    plt.loglog(rar_gold['g_bar'], rar_gold['g_obs'], 'r.', alpha=0.3, label='Golden Points')
    plt.plot([1e-13, 1e-8], [1e-13, 1e-8], 'b--', label='1:1 Line')
    plt.xlabel("g_bary (Newtonian)"); plt.ylabel("g_obs (Observed)"); plt.legend()
    plt.title("Radial Acceleration Relation (RAR)")
    save_plot_and_data(fig, rar_df, "RAR_Plot", [(DIRS['comp_figs'], DIRS['comp_data'])])
    plt.close(fig)

    # --- [Analysis 4] BTFR (Baryonic Tully-Fisher) ---
    fig = plt.figure(figsize=(8, 6))
    plt.loglog(df['M_bary'], df['V_flat'], 'ko', alpha=0.3, label='All 65')
    plt.loglog(df_gold['M_bary'], df_gold['V_flat'], 'ro', label='Golden 12')
    plt.xlabel("Baryonic Mass (Proxy)"); plt.ylabel("Flat Rotation Velocity"); plt.legend()
    plt.title("Baryonic Tully-Fisher Relation (BTFR)")
    save_plot_and_data(fig, df[['Galaxy', 'M_bary', 'V_flat', 'Is_Golden']], "BTFR_Plot", [(DIRS['comp_figs'], DIRS['comp_data'])])
    plt.close(fig)

# ==============================================================================
# [7] 메인 실행
# ==============================================================================
def main():
    print(f"[Process] Starting FULL Analysis (RAR, BTFR included) for {len(FULL_GALAXIES)} Galaxies...")
    results = []
    for i, gal in enumerate(FULL_GALAXIES):
        sys.stdout.write(f"\r[Process] {gal} [{i+1}/{len(FULL_GALAXIES)}]")
        sys.stdout.flush()
        res = process_galaxy(gal)
        if res: results.append(res)

    # Persist transparency logs (does not change any numerical results)
    try:
        if _DOWNLOAD_LOG:
            pd.DataFrame(_DOWNLOAD_LOG, columns=["Galaxy", "URL", "Status"]).to_csv(
                os.path.join(DIRS['all_data'], "Download_Log.csv"),
                index=False,
            )
        if _SKIPPED_GALAXIES:
            pd.DataFrame(_SKIPPED_GALAXIES, columns=["Galaxy", "Reason"]).to_csv(
                os.path.join(DIRS['all_data'], "Skipped_Galaxies.csv"),
                index=False,
            )
        with open(os.path.join(DIRS['all_data'], "Brentq_Fail_Count.txt"), "w") as f:
            f.write(str(_BRENTQ_FAIL_COUNT))
    except Exception:
        pass
            
    print("\n[Process] Generating Global Figures (RAR, BTFR, Beta Dist)...")
    if results:
        run_full_analysis(results)
        pd.DataFrame(results).drop(columns=['g_obs_list', 'g_bar_list']).to_csv(os.path.join(DIRS['all_data'], "Final_Results_Summary.csv"), index=False)
        print(f"[System] Done. Check {BASE_OUT_DIR}")

if __name__ == "__main__":
    main()
    try:
        write_run_metadata(Path(BASE_OUT_DIR), args={"data_dir": DATA_DIR, "allow_download": ALLOW_NET_DOWNLOAD}, notes={"n_downloaded": len(_DOWNLOAD_LOG), "n_skipped": len(_SKIPPED_GALAXIES), "brentq_fail": int(_BRENTQ_FAIL_COUNT)})
        # also persist download log for reviewers
        if len(_DOWNLOAD_LOG) > 0:
            pd.DataFrame(_DOWNLOAD_LOG).to_csv(os.path.join(DIRS["all_data"], "Download_Log.csv"), index=False)
    except Exception:
        pass
