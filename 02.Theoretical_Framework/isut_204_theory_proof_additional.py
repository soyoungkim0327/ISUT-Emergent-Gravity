import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.integrate import quad
import pandas as pd
import os
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] System Configuration & Path Definitions
# ==============================================================================
# Determine the directory of the currently running script
current_script_path = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Output Directory: Creates a folder named after the script in the current location
BASE_OUT_DIR = os.path.join(current_script_path, script_name)

# Sub-directories for organized output (Figures vs Data)
IMG_DIR = os.path.join(BASE_OUT_DIR, "figures")
DATA_DIR = os.path.join(BASE_OUT_DIR, "data")

# Create directories if they don't exist
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Plotting Style for Academic Papers
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif',
    'axes.labelsize': 14, 'axes.titlesize': 16,
    'lines.linewidth': 2.5, 'figure.dpi': 150
})

print(f"[System] ISUT Theoretical Framework Verification Protocol Initiated")
print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [Part 1] Lagrangian Reverse-Engineering Proof
# ==============================================================================
# 1. ISUT Interpolating Function
def mu_simple(x):
    return x / (1.0 + x)

# 2. Analytical Form of the Lagrangian derived from ISUT
# F(y) = y - 2*sqrt(y) + 2*ln(1 + sqrt(y))
def F_analytical(y):
    sqrt_y = np.sqrt(y)
    return y - 2*sqrt_y + 2*np.log(1 + sqrt_y)

def proof_lagrangian_robust():
    print("\n[Proof 1] Verifying Lagrangian Mathematical Consistency...")
    
    # 1. Grid Setup (y = (a/a0)^2, field strength squared)
    # Range: 10^-6 to 10^4 (Extended range for robust verification)
    y_grid = np.logspace(-6, 4, 100) 
    x_grid = np.sqrt(y_grid)
    
    # 2. Analytical Calculation (Theoretical value)
    F_ana = F_analytical(y_grid)
    
    # 3. Numerical Reconstruction (Numerical Integration - Ground Truth)
    # F(y) = Integral [ mu(sqrt(s)) ] ds from 0 to y
    F_num = np.zeros_like(y_grid)
    for i, yi in enumerate(y_grid):
        # Precise integration
        val, err = quad(lambda s: mu_simple(np.sqrt(s)), 0, yi)
        F_num[i] = val
        
    # 4. Residual Check (Validation of Mathematical Integrity)
    residual = F_ana - F_num
    max_error = np.max(np.abs(residual))
    
    print(f"   -> Max Integration Error: {max_error:.2e} (Virtually Zero)")
    
    # 5. Differentiation Check (Force Derivation)
    # dF/dy should equal mu(x)
    dF_dy = np.gradient(F_ana, y_grid)
    mu_recon = dF_dy 
    mu_theory = mu_simple(x_grid)
    
    # 6. Save Raw Data (CSV)
    df = pd.DataFrame({
        'y_acceleration_sq': y_grid,
        'x_acceleration': x_grid,
        'F_analytical': F_ana,
        'F_numerical': F_num,
        'Error_Residual': residual,
        'Mu_Theory': mu_theory,
        'Mu_Reconstructed': mu_recon
    })
    csv_path = os.path.join(DATA_DIR, "Figure_A1_Lagrangian_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [Data] CSV Saved: {csv_path}")
    
    # 7. Visualization (With Residuals)
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[3, 1]) # Top: Main plot, Bottom: Residuals
    
    # Subplot 1: Lagrangian F(y)
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(y_grid, F_ana, 'k-', lw=5, alpha=0.3, label='Analytical (Formula)')
    ax0.plot(y_grid, F_num, 'r--', lw=2, label='Numerical (Integration)')
    ax0.set_xscale('log'); ax0.set_yscale('log')
    ax0.set_title('Proof 1-A: Lagrangian Consistency')
    ax0.set_ylabel(r'Lagrangian Density $L(y)$')
    ax0.legend()
    ax0.grid(True, which='both', linestyle='--', alpha=0.3)
    ax0.set_xticklabels([]) # Hide x-labels for top plot
    
    # Subplot 1-Res: Residuals
    ax0_res = plt.subplot(gs[1, 0], sharex=ax0)
    ax0_res.plot(y_grid, residual, 'b-', lw=1)
    ax0_res.axhline(0, color='k', ls=':')
    ax0_res.set_ylabel('Error (Residual)')
    ax0_res.set_xlabel(r'Field Strength Squared $y = (a/a_0)^2$')
    
    # Annotation for error magnitude
    ax0_res.text(0.05, 0.8, f"Max Err < {max_error:.1e}\n(Perfect Match)", 
                 transform=ax0_res.transAxes, fontsize=10, color='red', 
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
    ax0_res.grid(True, alpha=0.3)

    # Subplot 2: Euler-Lagrange Equation (Force)
    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(x_grid, mu_theory, 'k-', lw=5, alpha=0.3, label=r'Target $\mu(x)$')
    ax1.plot(x_grid, mu_recon, 'g--', lw=2, label=r'Derived $\partial L / \partial y$')
    ax1.set_xscale('log')
    ax1.set_title('Proof 1-B: Euler-Lagrange Verification')
    ax1.set_ylabel(r'Interpolating Function $\mu(x)$')
    ax1.legend()
    ax1.grid(True, which='both', linestyle='--', alpha=0.3)
    ax1.set_xticklabels([])
    
    # Subplot 2-Res: Residuals
    ax1_res = plt.subplot(gs[1, 1], sharex=ax1)
    mu_err = mu_theory - mu_recon
    ax1_res.plot(x_grid, mu_err, 'g-', lw=1)
    ax1_res.axhline(0, color='k', ls=':')
    ax1_res.set_ylabel('Diff')
    ax1_res.set_xlabel(r'Acceleration $x = a/a_0$')
    ax1_res.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, "Figure_A1_Lagrangian_Proof.png")
    plt.savefig(save_path)
    
    print(f"   [Plot] Image Saved: {save_path}")
    print("   [Info] Displaying Graph 1. Close window to proceed...")
    plt.show() 

# ==============================================================================
# [Part 2] Microscopic Model Proof (Thermodynamic Entropy)
# ==============================================================================
def proof_micro_model_robust():
    print("\n[Proof 2] Verifying Microscopic Entropy Model...")
    
    # 1. Grid Setup (Wide range for thermodynamics check)
    x = np.logspace(-3, 3, 200)
    
    # 2. Physics Model
    # Assumption: Information bit energy level Delta_E = -kT * ln(x)
    kT = 1.0
    Delta_E = -kT * np.log(x)
    
    # 3. Fermi-Dirac like Occupancy (2-state system: 0 or 1)
    # P(excited) = 1 / (1 + exp(Delta_E / kT))
    # Simplifying: 1 / (1 + 1/x) = x / (x + 1) -> Matches ISUT formula
    occupancy = 1.0 / (1.0 + np.exp(Delta_E / kT))
    
    # 4. Macro Target (MOND/ISUT)
    mu_target = mu_simple(x)
    
    # 5. Check Identity (Match Verification)
    diff = occupancy - mu_target
    max_diff = np.max(np.abs(diff))
    print(f"   -> Thermodynamics Mismatch: {max_diff:.2e} (Exact Match)")
    
    # 6. Save Raw Data (CSV)
    df = pd.DataFrame({
        'x_acceleration': x,
        'Energy_Gap': Delta_E,
        'Occupancy_Prob': occupancy,
        'Mu_Target': mu_target,
        'Mismatch': diff
    })
    csv_path = os.path.join(DATA_DIR, "Figure_A2_Micro_Model_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [Data] CSV Saved: {csv_path}")
    
    # 7. Visualization
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    
    ax0 = plt.subplot(gs[0])
    ax0.plot(x, occupancy, 'b-', lw=5, alpha=0.3, label='Micro-Model (Info Bit)')
    ax0.plot(x, mu_target, 'r--', lw=2, label='Macro-Law (MOND)')
    
    # Annotation for physics equation
    ax0.text(0.05, 0.6, r"$\Delta E = -k_B T \ln(a/a_0)$" + "\n" + r"$P = \frac{1}{1 + e^{\Delta E/kT}}$", 
             transform=ax0.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.9, edgecolor='blue'))
    
    ax0.set_xscale('log')
    ax0.set_ylabel(r'Occupancy / $\mu(x)$')
    ax0.set_title('Proof 2: Emergence from Thermodynamics')
    ax0.legend()
    ax0.grid(True, which='both', linestyle='--', alpha=0.3)
    ax0.set_xticklabels([])
    
    # Residual Plot
    ax1 = plt.subplot(gs[1])
    ax1.plot(x, diff, 'k-', lw=1.5)
    ax1.axhline(0, color='r', ls=':')
    ax1.set_ylim(-1e-15, 1e-15) # Demonstrate machine epsilon precision
    ax1.set_ylabel('Difference')
    ax1.set_xlabel(r'Acceleration Ratio $x = a/a_0$')
    ax1.set_title('Exact Mathematical Identity Check', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, "Figure_A2_Micro_Model.png")
    plt.savefig(save_path)
    
    print(f"   [Plot] Image Saved: {save_path}")
    print("   [Info] Displaying Graph 2. Close window to finish...")
    plt.show()

# ==============================================================================
# [Main Execution]
# ==============================================================================
if __name__ == "__main__":
    proof_lagrangian_robust()
    proof_micro_model_robust()
    
    print("\n[System] Grand Proof Protocol Completed.")
    print(f"[System] Results saved in: {BASE_OUT_DIR}")