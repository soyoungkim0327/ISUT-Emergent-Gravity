"""
ISUT vs. Cold Dark Matter (CDM) Model Comparison Suite
======================================================
Objective: 
    Rigorous statistical model selection using the Bayesian Information Criterion (BIC).
    This module evaluates whether the ISUT framework provides a more parsimonious 
    explanation of galaxy rotation curves compared to the NFW (Dark Matter) profile.

Key Methodologies:
1. Model Selection (BIC): 
   - Penalizes over-fitting by accounting for the number of free parameters.
   - Formula: BIC = chi^2 + k * ln(n)
2. Correction Bias Control: 
   - Implements three levels of correction (None, Smart, Full) to test the 
     robustness of the ISUT model against observational uncertainties (distance, inclination).
3. Sample Integrity: 
   - Comparative analysis across both 'Golden Sample' (high-quality data) 
     and 'Full Sample' (65 galaxies) to ensure statistical significance.
4. Auditability: 
   - Automated logging of skipped data and fit failures for full transparency.

Output:
    - Comparative CSV results for each scenario.
    - Summary visualization of ISUT preference rates.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
import requests
import warnings
import sys

# 경고 메시지 무시
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] 설정: 똑똑한 경로 찾기 (수정됨)
# ==============================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# [핵심 수정] 데이터를 찾을 후보 경로들 (현재 폴더 vs 상위 폴더의 data)
CANDIDATE_DIRS = [
    os.path.join(CURRENT_DIR, "galaxies", "65_galaxies"),        # 1. 현재 폴더 안
    os.path.join(CURRENT_DIR, "..", "data", "galaxies", "65_galaxies"), # 2. 상위 폴더의 data 안
    os.path.join(CURRENT_DIR, "..", "galaxies", "65_galaxies")   # 3. 상위 폴더 바로 안
]

# 실제로 존재하는 경로 선택
DATA_DIR = CANDIDATE_DIRS[0] # 기본값
for path in CANDIDATE_DIRS:
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        DATA_DIR = path
        break

# 결과 저장 폴더
OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[System] ISUT Model Comparison Suite Initialized")
print(f"[Info] Input Directory : {DATA_DIR}") 
print(f"[Info] Output Directory: {OUT_DIR}")

# 은하 리스트
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

# 물리 상수
ACCEL_CONV = 3.24078e-14 
A0_SI = 1.2e-10 
A0_CODE = A0_SI / ACCEL_CONV  

# ==============================================================================
# [2] 물리 엔진
# ==============================================================================
def nu_isut(y):
    y_safe = np.maximum(y, 1e-12)
    return 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y_safe)

def model_isut(R, ups_disk, ups_bulge, V_gas, V_disk, V_bul):
    V_bary2 = (np.abs(V_gas)*V_gas) + (ups_disk * np.abs(V_disk)*V_disk) + (ups_bulge * np.abs(V_bul)*V_bul)
    R_safe = np.maximum(R, 0.01)
    gN = np.abs(V_bary2) / R_safe
    y = gN / A0_CODE
    g_final = gN * nu_isut(y)
    return np.sqrt(np.maximum(g_final * R, 0.0))

def velocity_nfw(R, V200, c):
    R200 = V200 / 10.0 
    x = R / np.maximum(R200, 0.01)
    numerator = np.log(1.0 + c*x) - (c*x) / (1.0 + c*x)
    denominator = np.log(1.0 + c) - c / (1.0 + c)
    V2 = V200**2 * (numerator / denominator) / x
    return np.sqrt(np.maximum(V2, 0.0))

def model_dm(R, ups_disk, ups_bulge, V200, c, V_gas, V_disk, V_bul):
    V_bary2 = (np.abs(V_gas)*V_gas) + (ups_disk * np.abs(V_disk)*V_disk) + (ups_bulge * np.abs(V_bul)*V_bul)
    V_dm = velocity_nfw(R, V200, c)
    return np.sqrt(np.maximum(V_bary2 + V_dm**2, 0.0))

# ==============================================================================
# [3] 로스 함수
# ==============================================================================
def apply_corrections(R, V_obs, V_err, V_gas, V_disk, V_bul, z_D, z_i, i_deg_assume=60.0):
    d_factor = 1.0 + 0.1 * z_D 
    R_new = R * d_factor
    v_scale = np.sqrt(d_factor)
    i_new_deg = np.clip(i_deg_assume + 5.0 * z_i, 10.0, 89.0)
    sin_ratio = np.sin(np.radians(i_deg_assume)) / np.sin(np.radians(i_new_deg))
    V_obs_new = V_obs * sin_ratio
    V_err_new = V_err * sin_ratio
    return R_new, V_obs_new, V_err_new, V_gas*v_scale, V_disk*v_scale, V_bul*v_scale

def loss_isut(params, R, V_obs, V_err, V_gas, V_disk, V_bul, mode):
    ud, ub = params[0], params[1]
    if mode == 'none': zD, zi = 0.0, 0.0
    else: zD, zi = params[2], params[3]
    R_c, V_c, V_ec, V_gc, V_dc, V_bc = apply_corrections(R, V_obs, V_err, V_gas, V_disk, V_bul, zD, zi)
    V_pred = model_isut(R_c, ud, ub, V_gc, V_dc, V_bc)
    chi2 = np.sum(((V_c - V_pred) / np.maximum(V_ec, 1.0))**2)
    penalty = 0
    if mode == 'smart': penalty += (zD**2 + zi**2) * 2.0
    elif mode == 'full': penalty += (zD**2 + zi**2) * 0.5
    return chi2 + penalty

def loss_dm(params, R, V_obs, V_err, V_gas, V_disk, V_bul):
    ud, ub, v200, c = params
    V_pred = model_dm(R, ud, ub, v200, c, V_gas, V_disk, V_bul)
    chi2 = np.sum(((V_obs - V_pred) / np.maximum(V_err, 1.0))**2)
    if c < 1 or c > 100: chi2 += 1000.0
    if v200 < 10 or v200 > 500: chi2 += 1000.0
    return chi2


# ==============================================================================
# [4] 실행 로직
# ==============================================================================
def get_data(gal_name, return_reason=False):
    """
    Load a galaxy rotation-curve file.

    NOTE: This helper optionally returns a failure reason for auditability.
    - return_reason=False (default): returns data tuple or None (backward-compatible)
    - return_reason=True: returns (data_tuple_or_None, reason_str)
    """
    path = os.path.join(DATA_DIR, f"{gal_name}_rotmod.dat")
    if not os.path.exists(path):
        return (None, 'file_not_found') if return_reason else None

    try:
        df = pd.read_csv(path, sep=r'\s+', comment='#', header=None).apply(pd.to_numeric, errors='coerce').dropna()
        if len(df) < 5:
            return (None, 'too_few_rows') if return_reason else None

        data = (
            df.iloc[:,0].values, df.iloc[:,1].values, df.iloc[:,2].values,
            df.iloc[:,3].values,
            (df.iloc[:,4].values if df.shape[1] > 4 else np.zeros(len(df))),
            (df.iloc[:,5].values if df.shape[1] > 5 else np.zeros(len(df))),
        )
        return (data, 'ok') if return_reason else data

    except Exception as e:
        reason = f"parse_error:{type(e).__name__}"
        return (None, reason) if return_reason else None





def run_analysis_scenario(target_list, mode, label):
    print(f"\n[Analysis] Scenario: {label} (Mode: {mode}, N={len(target_list)})")
    results = []
    skipped = []

    for i, gal in enumerate(target_list):



        # 진행상황 표시
        if i % 5 == 0: print(f"   Processing {gal}...", end="\r")

        # data = get_data(gal)
        # if not data: continue
        data, reason = get_data(gal, return_reason=True)
        if data is None:
            skipped.append({"Galaxy": gal, "Reason": reason})
            continue

        
        R, V_obs, V_err, V_gas, V_disk, V_bul = data
        n = len(R)
        
        # 1. ISUT Fit
        if mode == 'none':
            init_i, k_i, args_i = [0.5, 0.7], 2, (R, V_obs, V_err, V_gas, V_disk, V_bul, 'none')
            bounds_i = [(0.01, 5), (0, 5)]
        else:
            init_i, k_i, args_i = [0.5, 0.7, 0.0, 0.0], 4, (R, V_obs, V_err, V_gas, V_disk, V_bul, mode)
            bounds_i = [(0.01, 5), (0, 5), (-3, 3), (-3, 3)]
            
        res_i = minimize(loss_isut, init_i, args=args_i, bounds=bounds_i)
        bic_i = res_i.fun + k_i * np.log(n)
        
        # 2. DM Fit
        v_max = np.max(V_obs)
        res_d = minimize(loss_dm, [0.5, 0.7, v_max, 10.0], 
                         args=(R, V_obs, V_err, V_gas, V_disk, V_bul),
                         bounds=[(0.01, 5), (0, 5), (10, 500), (1, 100)])
        bic_d = res_d.fun + 4 * np.log(n)
        
        diff = bic_d - bic_i
        results.append({
            "Galaxy": gal, "Delta_BIC": diff, "Selected_Model": "ISUT" if diff > 0 else "DarkMatter"
        })


    # --- Save skipped list for auditability (no impact on scores) ---
    skip_path = os.path.join(OUT_DIR, f"Skipped_{label}.csv")
    if skipped:
        pd.DataFrame(skipped).to_csv(skip_path, index=False)
    else:
        pd.DataFrame(columns=["Galaxy", "Reason"]).to_csv(skip_path, index=False)

    n_valid = len(results)
    n_skip = len(skipped)
    print(f"   ℹ️ Valid={n_valid} / Target={len(target_list)}  |  Skipped={n_skip} -> {os.path.basename(skip_path)}")


    if results:
        df = pd.DataFrame(results)
        csv_path = os.path.join(OUT_DIR, f"Result_{label}.csv")
        df.to_csv(csv_path, index=False)
        wins = len(df[df['Delta_BIC'] > 0])
        print(f"    {label} Completed. ISUT Preference: {wins}/{len(df)} ({(wins/len(df)*100) if len(df)>0 else 0:.1f}%)")
        return df, wins, len(df)
    
    # [수정] 데이터가 없으면 0 리턴
    print(f"    No valid data found for {label}. Check data directory.")
    return None, 0, 0

def main():
    # A. Run All Scenarios
    scenarios = [
        (GOLDEN_GALAXIES, 'none', "Golden12_NoCorr"),
        (GOLDEN_GALAXIES, 'smart', "Golden12_SmartCorr"),
        (GOLDEN_GALAXIES, 'full', "Golden12_FullCorr"),
        (FULL_GALAXIES, 'none', "All65_NoCorr"),
        (FULL_GALAXIES, 'smart', "All65_SmartCorr"),
        (FULL_GALAXIES, 'full', "All65_FullCorr")
    ]
    
    summary_list = []
    
    for targets, mode, label in scenarios:
        df, w, n = run_analysis_scenario(targets, mode, label)
        # [핵심 수정] 0으로 나누기 방지 (n이 0이면 승률도 0)
        rate = (w / n) if n > 0 else 0.0
        summary_list.append({"Scenario": label, "ISUT_Wins": w, "Total": n, "Win_Rate": rate})

    # C. Summary Visualization
    print("\n" + "="*60)
    print("Generating Comparative Summary Visualization...")
    print("="*60)
    
    df_summary = pd.DataFrame(summary_list)
    summary_csv_path = os.path.join(OUT_DIR, "Summary_Model_Comparison.csv")
    df_summary.to_csv(summary_csv_path, index=False)
    
    if df_summary['Total'].sum() == 0:
        print(" Error: No data processed at all. Please check [Info] Input Directory path.")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_summary["Scenario"], df_summary["Win_Rate"]*100, color='skyblue', edgecolor='navy')
    plt.axhline(50, color='red', linestyle='--', alpha=0.5, label="Parity (50%)")
    plt.ylabel("ISUT Preference Rate (%)")
    plt.title("Model Comparison: ISUT vs Dark Matter (BIC)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    summary_fig_path = os.path.join(OUT_DIR, "Summary_Model_Comparison.png")
    plt.savefig(summary_fig_path)
    print(f"   L Figure Saved: {summary_fig_path}")
    print(f"   L Data Saved  : {summary_csv_path}")
    print("-" * 55)

if __name__ == "__main__":
    main()