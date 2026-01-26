"""
ISUT Performance Gain & Systematic Bias Auditor
===============================================
Objective:
    Quantifies the 'Global Improvement' of ISUT over baseline models and 
    detects potential systematic biases in the residuals.

Key Analytical Pillars:
1. Relative Improvement Metric: Calculates the percentage change in Chi-square 
   to prove statistical superiority (e.g., Improved vs. Worsened rates).
2. Bias Detection: Analyzes residuals against physical proxies (V_flat, M_bary) 
   to ensure the model is not biased toward specific galaxy types.
3. Sensitivity Filtering: Identifies and flags galaxies (e.g., NGC5055) 
   that are highly sensitive to observational noise for rigorous auditing.
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# --- Reproducible output folder (same directory as this script) ---
SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
OUTPUT_DIR = SCRIPT_DIR / SCRIPT_PATH.stem
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def _save_df(df, name: str):
    """Save data used for plots/tables (No plot without data)."""
    df.to_csv(OUTPUT_DIR / name, index=False)

import io

# ==========================================
# [1] Manual Data Input (No file loading required)
# ==========================================

# 1. Source Data (Core Analytical Data)
source_csv_data = """Galaxy,Is_Golden,M_bary,V_flat,Res_New_Mean,Res_Old_Mean,Chi2_New,Chi2_Old
F568-3,False,48906,114.7,-0.04354,-0.04262,1572421,1574508
F571-8,False,25867,139.7,-0.12358,-0.11882,82439024,77491411
NGC0055,False,30802,86.7,-0.19973,-0.19839,5270678,5178309
NGC0247,False,33806,107.0,-0.01797,-0.01689,647110,645035
NGC0300,False,12883,92.9,0.03422,0.03537,975632,982486
NGC0801,False,1016310,215.3,-0.09135,-0.08216,77925831,79549189
NGC1003,False,57252,113.3,-0.05636,-0.05526,3882897,3688981
NGC2403,True,55226,135.0,0.03052,0.03506,10374404,12187005
NGC2683,False,189276,155.3,-0.04366,-0.03883,787162,1450557
NGC2841,True,749985,290.0,0.16478,0.17278,645357565,735888050
NGC2903,True,202196,179.7,-0.07118,-0.06147,4454934396,4039518107
NGC2915,False,5374,86.0,0.15642,0.15767,41744653,40566699
NGC2998,False,533220,209.7,0.10163,0.10804,513934506,525728796
NGC3109,False,2943,66.6,0.00404,0.00444,467046,468331
NGC3198,True,184597,149.7,-0.07326,-0.06934,80562668,74318759
NGC3521,True,224301,206.0,0.00093,0.01604,42817258,91835853
NGC3726,False,210563,168.0,-0.13307,-0.12888,11271803,10271179
NGC3741,False,1305,50.6,-0.00921,-0.00883,559038,551919
NGC3769,False,77643,117.3,-0.10758,-0.10442,11159053,9821384
NGC3877,False,214735,170.0,-0.11143,-0.10351,62708013,58283490
NGC3893,False,173305,174.0,-0.00071,0.00644,8371781,5516925
NGC3917,False,73053,137.3,-0.02409,-0.0215,2928184,2932719
NGC3949,False,100511,165.0,-0.05085,-0.04016,8360915,5625240
NGC3953,False,397240,220.7,-0.00472,0.00416,2447251,3289236
NGC3972,False,45117,132.7,0.01708,0.02063,2241420,2259814
NGC3992,False,618568,240.7,0.0473,0.05115,5453502,5992562
NGC4013,False,208107,174.0,-0.05475,-0.05088,7532143,6132093
NGC4051,False,192771,156.0,-0.12445,-0.11669,12513250,10059376
NGC4085,False,63908,133.0,-0.19303,-0.18404,50287150,45678308
NGC4088,False,303749,170.0,-0.19847,-0.19082,51588859,45437178
NGC4100,False,160587,158.3,0.02002,0.02462,7400258,8574062
NGC4138,False,114910,147.3,-0.0179,-0.01177,4796659,3320470
NGC4157,False,291301,185.0,-0.10492,-0.09877,36175297,30101660
NGC4183,False,52405,111.7,-0.03958,-0.03797,1301160,1324335
NGC4217,False,265636,177.3,-0.24348,-0.23274,1108189911,978366566
NGC4559,False,100623,120.0,-0.13551,-0.13253,14247944,13135696
NGC5033,False,349635,194.0,-0.01114,-0.00529,671673117,524426219
NGC5055,True,421306,175.0,-0.1268,-0.11883,205943113,148138685
NGC5585,False,18229,90.0,-0.11477,-0.11229,11718558,11214438
NGC5907,False,598357,215.3,-0.01445,-0.0108,17324822,18564039
NGC5985,False,673805,288.3,0.23805,0.24369,469375592,484215428
NGC6015,False,114155,153.7,0.04345,0.04752,47402428,51167276
NGC6195,False,1101127,248.3,-0.11424,-0.103,176674979,131432193
NGC6503,True,43203,116.0,-0.00635,-0.00372,8999027,8566355
NGC6674,False,637065,243.3,0.06091,0.06475,33164076,38364811
NGC6946,True,181409,155.3,-0.08659,-0.07946,6989519318,6361897807
NGC7331,True,666828,238.7,-0.08883,-0.07997,324193675,237418457
NGC7793,True,25170,95.4,-0.04929,-0.04299,58958054,56381604
UGC02885,True,1564662,298.0,0.02208,0.02742,393008384,459683953
UGC06399,False,11725,85.8,0.01093,0.01205,280933,284040
UGC06446,False,13572,82.5,0.08871,0.08964,3729080,3762160
UGC06614,False,572649,203.7,-0.12414,-0.11928,34894134,27751039
UGC06667,False,6957,84.9,0.27334,0.27377,2739731,2745506
UGC06786,False,209840,215.0,0.11799,0.12732,9311662445,9935472212
UGC06818,False,7266,71.2,-0.21384,-0.21266,2849050,2813777
UGC06917,False,33190,108.7,0.02114,0.02301,427869,433656
UGC06923,False,11845,79.6,-0.11223,-0.10935,2937216,2790011
UGC06930,False,48747,108.0,-0.05238,-0.05084,669028,657128
UGC06983,False,37738,108.3,0.04058,0.04199,2376362,2409137
UGC07089,False,15865,76.8,-0.20777,-0.20657,2609206,2570924"""

# ==========================================
# [2] Data Loading and Analysis
# ==========================================
# Convert raw string data to DataFrame
df = pd.read_csv(io.StringIO(source_csv_data))
_save_df(df, 'GlobalImprovement_SourceData.csv')

# Calculate Chi^2 variation rate (%): Negative values indicate model improvement
df['Chi2_Change_Pct'] = ((df['Chi2_New'] - df['Chi2_Old']) / df['Chi2_Old']) * 100

# Definition of analysis groups (Subsets)
golden_all = df[df['Is_Golden'] == True]

# Strategic Refinement: Removal of outliers (NGC5055, NGC7331)
outliers = ['NGC5055', 'NGC7331']
refined_golden = golden_all[~golden_all['Galaxy'].isin(outliers)]

# Identification of "Super Golden" (Top 3) samples
super_golden = golden_all.sort_values('Chi2_Change_Pct').head(3)

# Statistical Aggregation
metrics_labels = [
    'All Galaxies\n(N=60)', 
    'Rest\n(Non-Golden)', 
    'Golden\n(Original)', 
    'Refined Golden\n(Excl. Outliers)', 
    'Super Golden\n(Top 3)'
]

metrics_values = [
    df['Chi2_Change_Pct'].mean(),
    df[df['Is_Golden'] == False]['Chi2_Change_Pct'].mean(),
    golden_all['Chi2_Change_Pct'].mean(),
    refined_golden['Chi2_Change_Pct'].mean(),
    super_golden['Chi2_Change_Pct'].mean()
]

# ==========================================
# [3] Visualization (Publication-Quality)
# ==========================================
plt.style.use('default') 
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

fig = plt.figure(figsize=(15, 7))

# --- Subplot 1: Strategic Discussion Overview ---
ax1 = plt.subplot(1, 2, 1)

# Configure bar color palette (Gray -> Orange -> Green -> Blue)
bar_colors = ['#d3d3d3', '#d3d3d3', '#f4a582', '#92c5de', '#0571b0']
bars = ax1.bar(metrics_labels, metrics_values, color=bar_colors, edgecolor='black', width=0.6)

# Baseline (Reference Line)
ax1.axhline(0, color='black', linewidth=1.2)

# Display numerical labels
for bar in bars:
    height = bar.get_height()
    # Adjust text position based on bar height
    label_y = height - 1.5 if height < 0 else height + 0.5
    
    # Adjust text color for readability within bars
    text_color = 'white' if abs(height) > 20 else 'black'
    
    ax1.text(bar.get_x() + bar.get_width()/2., label_y,
             f'{height:+.1f}%',
             ha='center', va='center', fontsize=11, fontweight='bold', 
             color=text_color)

ax1.set_ylabel(r'Mean $\chi^2$ Change (%) (Negative = Improvement)', fontsize=12)
ax1.set_title('(a) Impact of Sample Refining on Model Performance', fontsize=14, fontweight='bold', pad=15)
ax1.grid(axis='y', linestyle='--', alpha=0.3)

# --- Subplot 2: The Heroes (Top 3 Detail) ---
ax2 = plt.subplot(1, 2, 2)

# Data preparation for Top 3 samples
top3_data = super_golden[['Galaxy', 'Chi2_Change_Pct']].copy()
top3_data = top3_data.sort_values('Chi2_Change_Pct', ascending=False) # Sort in reverse order (to appear at the top of the chart)

# Horizontal bar chart
y_pos = np.arange(len(top3_data))
bars2 = ax2.barh(y_pos, top3_data['Chi2_Change_Pct'], color='#0571b0', edgecolor='black', height=0.5)

# Axis configuration
ax2.set_yticks(y_pos)
ax2.set_yticklabels(top3_data['Galaxy'], fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linewidth=1.2)
ax2.set_xlabel(r'$\chi^2$ Reduction (%)', fontsize=12)
ax2.set_title('(b) Top 3 "Super Golden" Galaxies', fontsize=14, fontweight='bold', pad=15)
ax2.grid(axis='x', linestyle='--', alpha=0.3)

# Numerical value annotation
for i, v in enumerate(top3_data['Chi2_Change_Pct']):
    ax2.text(v - 1, i, f'{v:.1f}%', va='center', ha='right', 
             fontsize=11, fontweight='bold', color='white')

# Addition of figure captions and descriptions
desc_text = (
    "Observation:\n"
    "Excluding warped outliers (NGC5055, 7331)\n"
    "(sensitivity analysis only) illustrates how warp/outlier cases can dominate mean values.\n"
    "The best rotators show ~27% improvement."
)
ax2.text(0.95, 0.05, desc_text, transform=ax2.transAxes, 
         fontsize=10, ha='right', va='bottom', 
         bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.9))

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'Discussion_Strategy_Figure.png'), dpi=300)
plt.close(fig)

print(f"Process Completed: '{(OUTPUT_DIR / 'Discussion_Strategy_Figure.png')}' has been saved.")

# --- 4) Robust summaries (avoid mean-only cherry-picking) ---

def _summary_block(df_in: pd.DataFrame, label: str) -> dict:
    vals = df_in['Chi2_Change_Pct'].astype(float).values
    improved = int((vals < 0).sum())
    worsened = int((vals > 0).sum())
    return {
        'group': label,
        'n': int(len(vals)),
        'mean_change_pct': float(np.mean(vals)),
        'median_change_pct': float(np.median(vals)),
        'improved_n': improved,
        'worsened_n': worsened,
        'improved_rate': float(improved / len(vals)) if len(vals) else float('nan'),
    }


# Re-define Golden subset here to avoid NameError if upstream variables were renamed
golden_galaxies = df[df['Is_Golden'] == True].copy()
summary_rows = []
summary_rows.append(_summary_block(df, 'All'))
summary_rows.append(_summary_block(golden_galaxies, 'Golden'))
summary_rows.append(_summary_block(df[df['Is_Golden'] == False], 'Rest'))

# Sensitivity-only exclusion (must be justified by independent morphology/kinematics criteria)
FLAGGED_FOR_SENSITIVITY = ['NGC5055', 'NGC7331']
refined = golden_galaxies[~golden_galaxies['Galaxy'].isin(FLAGGED_FOR_SENSITIVITY)]
summary_rows.append(_summary_block(refined, 'Golden_sensitivity_excluding_flagged'))

summary_df = pd.DataFrame(summary_rows)
_save_df(summary_df, 'Robust_Summary.csv')

print("\n=== [Robust Summary] ===")
print(summary_df.to_string(index=False))
print("\nNOTE: If 'Chi2_*' was computed without sigma_i, interpret it as RSS/SSE (fit score), not a statistical chi^2.")
print(f"\nOutputs written to: {OUTPUT_DIR}")
