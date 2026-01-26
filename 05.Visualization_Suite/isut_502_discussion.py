import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import os
import sys

# ==============================================================================
# [Configuration] Path Setup
# ==============================================================================
# 1. Apply Publication Style (Seaborn)
sns.set_theme(style="whitegrid", context="talk") # Enable Background Grid + Increase Font Size
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

# ==============================================================================
# NOTE FOR PAPER / REVIEWERS
# ------------------------------------------------------------------------------
# This script is meant to create *discussion* diagnostics (post-hoc sensitivity
# summaries) from the comparison table below.
#
# IMPORTANT: The columns named "Chi2_*" here are treated as a *generic fit-score*
# (chi-square-like). Calling them a true "chi-square" is only correct if the
# underlying per-point uncertainties were used in the usual definition.
# If not, label them in the manuscript as "fit score" or "SSE-like misfit".
# ==============================================================================
METRIC_LABEL = "Fit-score"  # safer than claiming a formal chi^2 unless defined

# 2. 경로 설정
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)

DIR_ALL65 = os.path.join(OUTPUT_ROOT, "All65")
DIR_GOLDEN12 = os.path.join(OUTPUT_ROOT, "Golden12")

for path in [
    os.path.join(OUTPUT_ROOT, 'figures'), os.path.join(OUTPUT_ROOT, 'data'),
    os.path.join(DIR_ALL65, 'figures'), os.path.join(DIR_ALL65, 'data'),
    os.path.join(DIR_GOLDEN12, 'figures'), os.path.join(DIR_GOLDEN12, 'data')
]:
    os.makedirs(path, exist_ok=True)

print(f"[System] Output Directories Ready: {OUTPUT_ROOT}")

# ==============================================================================
# [Data Load] Ultimate Comparison Source Data (Preserve Original Data)
# ==============================================================================
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

df = pd.read_csv(io.StringIO(source_csv_data))

# Data Preprocessing
df['Chi2_Change_Pct'] = ((df['Chi2_New'] - df['Chi2_Old']) / df['Chi2_Old']) * 100
df['Res_Bias_Change'] = df['Res_New_Mean'].abs() - df['Res_Old_Mean'].abs()

# Group Definitions
golden_all = df[df['Is_Golden'] == True]
non_golden = df[df['Is_Golden'] == False]

# NOTE: outlier removal must NOT be used as the *main* conclusion in a paper.
# Treat this as a *sensitivity analysis* only, unless you define the exclusion
# rule *a priori* (e.g., strong warp/non-circular motion flags from external
# catalogs). Keep the full-sample results in the main text.
OUTLIERS = ['NGC5055', 'NGC7331']
OUTLIER_REASON = {
    'NGC5055': 'Known strong warp / kinematic complexity (see Discussion in manuscript).',
    'NGC7331': 'Kinematic asymmetry / non-axisymmetry concerns (see Discussion in manuscript).'
}

refined_golden = golden_all[~golden_all['Galaxy'].isin(OUTLIERS)]

# "Top 3" is *post-hoc* (best improvements). Use only as illustrative case studies.
super_golden = golden_all.sort_values('Chi2_Change_Pct', ascending=True).head(3)

def group_stats(name: str, dfi: pd.DataFrame) -> dict:
    """Compute robust summary stats used in the Discussion section."""
    if len(dfi) == 0:
        return {"Group": name, "N": 0, "MeanPct": np.nan, "MedianPct": np.nan, "WinRate": np.nan}
    vals = dfi['Chi2_Change_Pct']
    return {
        "Group": name,
        "N": int(len(dfi)),
        "MeanPct": float(vals.mean()),
        "MedianPct": float(vals.median()),
        "WinRate": float((vals < 0).mean())  # fraction improved
    }

# Save discussion-level summary (for reproducibility / paper table)
summary_rows = [
    group_stats(f"All(N={len(df)})", df),
    group_stats(f"Rest(N={len(non_golden)})", non_golden),
    group_stats(f"Golden(N={len(golden_all)})", golden_all),
    group_stats(f"RefinedGolden(N={len(refined_golden)})", refined_golden),
    group_stats(f"Top3(N={len(super_golden)})", super_golden),
]
df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUTPUT_ROOT, "data", "Discussion_GroupSummary.csv"), index=False)
df_summary.to_csv(os.path.join(DIR_ALL65, "data", "Discussion_GroupSummary.csv"), index=False)
df_summary.to_csv(os.path.join(DIR_GOLDEN12, "data", "Discussion_GroupSummary.csv"), index=False)

# Save explicit outlier list used for the sensitivity run
pd.DataFrame(
    [{"Galaxy": g, "Reason": OUTLIER_REASON.get(g, "")}
     for g in OUTLIERS]
).to_csv(os.path.join(OUTPUT_ROOT, "data", "Discussion_Outliers.csv"), index=False)

# ==============================================================================
# [Figure 1] Strategic Overview (All65) - Clean Bar Chart
# ==============================================================================
def plot_strategic_overview():
    metrics_labels = [
        f"All (N={len(df)})",
        f"Rest (N={len(non_golden)})",
        f"Golden (N={len(golden_all)})",
        f"Refined-Golden (N={len(refined_golden)})",
        f"Top-3 (N={len(super_golden)})",
    ]
    metrics_values = [
        df['Chi2_Change_Pct'].mean(),
        non_golden['Chi2_Change_Pct'].mean(),
        golden_all['Chi2_Change_Pct'].mean(),
        refined_golden['Chi2_Change_Pct'].mean(),
        super_golden['Chi2_Change_Pct'].mean()
    ]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    # Color Palette: Gray (Base) -> Orange (Warning) -> Light Blue (Good) -> Deep Blue (Best)
    colors = ['#bdbdbd', '#bdbdbd', '#fc8d59', '#91bfdb', '#4575b4']
    
    bars = ax.bar(metrics_labels, metrics_values, color=colors, edgecolor='k', linewidth=1.2, width=0.6)
    
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_ylabel('Mean fit-score change (%)', fontsize=14, fontweight='bold')
    ax.set_title('(a) Sensitivity to sample selection (post-hoc)', fontsize=16, fontweight='bold', pad=20)
    
    # Display Values
    for bar in bars:
        height = bar.get_height()
        label_y = height - 1.5 if height < 0 else height + 0.5
        font_col = 'white' if abs(height) > 10 else 'black'
        ax.text(bar.get_x() + bar.get_width()/2., label_y, f'{height:+.1f}%', 
                ha='center', va='center', fontsize=12, fontweight='bold', color=font_col)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ALL65, 'figures', 'Fig1_Strategic_Overview.png'))
    plt.close()
    
    # Save CSV
    pd.DataFrame({'Group': metrics_labels, 'Value': metrics_values}).to_csv(
        os.path.join(DIR_ALL65, 'data', 'Fig1_Data.csv'), index=False)
    print(" [OK] Fig1 Created (Clean Style)")

# ==============================================================================
# [Figure 2] Systematic Bias Scatter (All65) - Transparency + Regression Line
# ==============================================================================
def plot_bias_check():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter with edge color for definition
    sns.scatterplot(x=df['Res_Old_Mean'], y=df['Res_New_Mean'], 
                    s=100, color='#4575b4', edgecolor='w', alpha=0.7, ax=ax)
    
    # Identity line (y=x)
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'k--', alpha=0.6, linewidth=1.5, label='No Change')
    
    ax.set_xlabel('Old Mean Residual', fontsize=13, fontweight='bold')
    ax.set_ylabel('New Mean Residual', fontsize=13, fontweight='bold')
    ax.set_title('(b) Systematic Bias Check', fontsize=15, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ALL65, 'figures', 'Fig2_Systematic_Bias.png'))
    plt.close()
    
    # Save CSV
    df[['Galaxy', 'Res_Old_Mean', 'Res_New_Mean']].to_csv(
        os.path.join(DIR_ALL65, 'data', 'Fig2_Data.csv'), index=False)
    print(" [OK] Fig2 Created (Scatter Style)")

# ==============================================================================
# [Figure 3] All Galaxies Detail (All65) - Tall Bar Chart (Red/Blue)
# ==============================================================================
def plot_all_galaxies_detail():
    sorted_df = df.sort_values('Chi2_Change_Pct', ascending=False)
    
    fig, ax = plt.subplots(figsize=(9, 15)) # Tall figure
    
    # Red for worsening, Blue for improvement
    colors = ['#4575b4' if x < 0 else '#d73027' for x in sorted_df['Chi2_Change_Pct']]
    
    ax.barh(sorted_df['Galaxy'], sorted_df['Chi2_Change_Pct'], color=colors, alpha=0.85, edgecolor='none')
    
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel(f"{METRIC_LABEL} change (%)", fontsize=14, fontweight='bold')
    ax.set_title('(c) Full Sample Performance (N=60)', fontsize=16, fontweight='bold')
    ax.grid(axis='y', alpha=0.2) # Horizontal grid strictly
    
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_ALL65, 'figures', 'Fig3_All_Galaxies_Detailed.png'))
    plt.close()
    
    sorted_df.to_csv(os.path.join(DIR_ALL65, 'data', 'Fig3_Data.csv'), index=False)
    print(" [OK] Fig3 Created (Tall Bar Style)")

# ==============================================================================
# [Figure 4] Golden Individual Performance (Golden12)
# ==============================================================================
def plot_golden_individual():
    golden_sorted = golden_all.sort_values('Chi2_Change_Pct', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    # Coolwarm colors manually
    colors = ['#4575b4' if x < 0 else '#d73027' for x in golden_sorted['Chi2_Change_Pct']]
    
    bars = ax.barh(golden_sorted['Galaxy'], golden_sorted['Chi2_Change_Pct'], 
                   color=colors, edgecolor='k', linewidth=1, height=0.6)
    
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_xlabel(f"{METRIC_LABEL} change (%) (Negative = Improvement)", fontsize=13, fontweight='bold')
    ax.set_title('(d) Golden Group Performance', fontsize=15, fontweight='bold')
    
    for bar in bars:
        width = bar.get_width()
        label_x = width - 2 if width < 0 else width + 0.5
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', 
                va='center', fontsize=11, fontweight='bold', color='black')

    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GOLDEN12, 'figures', 'Fig4_Golden_Performance.png'))
    plt.close()
    
    golden_sorted.to_csv(os.path.join(DIR_GOLDEN12, 'data', 'Fig4_Data.csv'), index=False)
    print(" [OK] Fig4 Created (Golden Detail)")

# ==============================================================================
# [Figure 5] Top-3 illustrative cases (Golden) - scorecard style (post-hoc)
# ==============================================================================
def plot_top3_scorecards():
    top3 = super_golden.copy()
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    
    for i, (idx, row) in enumerate(top3.iterrows()):
        ax = axes[i]
        galaxy_name = row['Galaxy']
        improvement = abs(row['Chi2_Change_Pct'])
        
        # Comparison Bar (Old vs New)
        # Normalize Old to 100%
        # New is (100 - improvement)
        vals = [100, 100 - improvement]
        labels = ['Old Model', 'New Model']
        bar_colors = ['#bdbdbd', '#4575b4'] # Grey vs Blue
        
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor='k', linewidth=1.5, width=0.6)
        
        ax.set_title(f"{galaxy_name}", fontsize=18, fontweight='bold', color='#252525')
        ax.set_ylim(0, 115)
        ax.set_ylabel(f"Relative misfit ({METRIC_LABEL})", fontsize=11)
        ax.set_yticks([]) # Hide y ticks for cleaner look
        
        # Add improvement badge
        ax.text(0.5, 50, f"-{improvement:.1f}%", ha='center', va='center', 
                fontsize=22, fontweight='bold', color='white', 
                bbox=dict(boxstyle="round,pad=0.3", fc="#d73027", ec="none", alpha=0.8))
        
        # Value labels
        ax.text(0, 102, "100%", ha='center', fontsize=12, fontweight='bold', color='#525252')
        ax.text(1, 100 - improvement + 2, f"{100-improvement:.1f}%", ha='center', fontsize=12, fontweight='bold', color='#4575b4')
        
    plt.suptitle('Figure 5. Top-3 illustrative cases (post-hoc sensitivity)', fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(DIR_GOLDEN12, 'figures', 'Fig5_Top3_Scorecards.png'))
    plt.close()
    
    top3.to_csv(os.path.join(DIR_GOLDEN12, 'data', 'Fig5_Data.csv'), index=False)
    print(" [OK] Fig5 Created (Scorecard)")

# ==============================================================================
# [Main Execution]
# ==============================================================================
if __name__ == "__main__":
    print("--- Generating Professional Visualizations---")
    plot_strategic_overview()     # Fig 1
    plot_bias_check()             # Fig 2
    plot_all_galaxies_detail()    # Fig 3
    plot_golden_individual()      # Fig 4
    plot_top3_scorecards()        # Fig 5
    print("-----------------------------------------------------------")
    print(f"All outputs saved to: {OUTPUT_ROOT}")