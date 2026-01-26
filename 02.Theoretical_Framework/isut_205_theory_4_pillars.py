import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] System Configuration & Path Definitions
# ==============================================================================
# Determine the directory of the currently running script
current_script_path = os.path.dirname(os.path.abspath(__file__))
script_name = os.path.splitext(os.path.basename(__file__))[0]

# Output Base Directory
BASE_OUT_DIR = os.path.join(current_script_path, script_name)

# Sub-directories for organized output
IMG_DIR = os.path.join(BASE_OUT_DIR, "figures")
DATA_DIR = os.path.join(BASE_OUT_DIR, "data")

# Create directories
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Plotting Style for Academic Papers
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif',
    'axes.labelsize': 14, 'axes.titlesize': 16,
    'lines.linewidth': 2.5, 'figure.dpi': 150
})

print(f"[System] ISUT Theory Visualizer: 4 Pillars (Conceptual Schematics) Initiated")
print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [Pillar 1] Zipfian Redundancy (Origin of Flat Rotation)
# ==============================================================================
def proof_1_zipfian_redundancy():
    print("\n[Pillar 1] Processing Zipfian Redundancy (The Origin of Flat Rotation)...")
    
    # 1. Physics Model Setup
    r = np.linspace(0.1, 20, 200) # Distance (kpc)
    
    # (A) Newtonian Dynamics: Information dilution proportional to surface area (~ r^2)
    # Force ~ 1/r^2
    force_newton = 1.0 / (r**2)
    
    # (B) ISUT/Zipf: Information Redundancy (Network Theory)
    # Effective Information ~ r (Linear Scaling due to Holography)
    # Force ~ 1/r
    force_isut = 1.0 / r
    
    # 2. Data Export (Reviewer Defense: Numerical Basis)
    df = pd.DataFrame({
        'Radius_kpc': r,
        'Force_Newton_InvSq': force_newton,
        'Force_ISUT_Inv': force_isut,
        'Ratio_ISUT_Newton': force_isut / force_newton # Proportional to r
    })
    csv_path = os.path.join(DATA_DIR, "Pillar1_Zipfian_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [Data] CSV Saved: {csv_path}")
    
    # 3. Visualization
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(r, force_newton, 'b--', label=r'Newtonian ($1/r^2$) - Dilution')
    plt.plot(r, force_isut, 'r-', lw=3, label=r'ISUT / Zipf ($1/r$) - Conservation')
    
    plt.yscale('log')
    plt.xlabel('Distance $r$')
    plt.ylabel('Gravity Strength (Log Scale)')
    plt.title('Pillar 1: Why Gravity Decays Slowly?\n(Zipfian Information Redundancy)')
    
    # Explanatory Text
    plt.text(5, 0.01, "Reviewer Note:\nZipf's Law preserves information\nover distance, leading to\nflat rotation curves.",
             bbox=dict(facecolor='white', edgecolor='red', alpha=0.9))
    
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, "Fig1_Zipfian_Redundancy.png")
    plt.savefig(save_path)
    
    print("   [Plot] Graph 1 Generated. Proceeding...")
    # plt.show() # Optional: Comment in to view interactively
    plt.close(fig)

# ==============================================================================
# [Pillar 2] Landauer's Principle (Information-Energy Equivalence)
# ==============================================================================
def proof_2_landauer_limit():
    print("\n[Pillar 2] Processing Landauer's Principle (Information is Energy)...")
    
    # 1. Physics Model
    # Energy cost calculation based on bit count
    bits = np.linspace(0, 100, 100)
    kT = 4.11e-21 # Reference value at 300K
    
    # E = N * kT * ln(2)
    energy = bits * kT * np.log(2)
    
    # 2. Data Export
    df = pd.DataFrame({'Bits': bits, 'Energy_Joules': energy})
    csv_path = os.path.join(DATA_DIR, "Pillar2_Landauer_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [Data] CSV Saved: {csv_path}")
    
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(bits, energy, 'g-', lw=3)
    ax.fill_between(bits, 0, energy, color='green', alpha=0.1)
    
    ax.set_xlabel('Information Content (Bits)')
    ax.set_ylabel('Minimal Energy Cost (Joules)')
    ax.set_title("Pillar 2: Landauer's Limit\n(Gravity is the Cost of Information Erasure)")
    
    # Key Formula Display
    ax.text(20, energy.max()*0.8, r"$E = k_B T \ln(2) \times N_{bits}$", fontsize=16,
            bbox=dict(facecolor='white', edgecolor='green'))
    
    ax.text(40, energy.max()*0.4, "Gravity emerges as an\nentropic force to pay\nthis energy debt.", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, "Fig2_Landauer_Principle.png")
    plt.savefig(save_path)
    
    print("   [Plot] Graph 2 Generated. Proceeding...")
    # plt.show()
    plt.close(fig)

# ==============================================================================
# [Pillar 3] Euler's Template (Topological Necessity)
# ==============================================================================
def proof_3_euler_template():
    print("\n[Pillar 3] Processing Euler's Template (Topological Necessity)...")
    
    # 1. Unit Circle on Complex Plane
    theta = np.linspace(0, np.pi, 100)
    z_real = np.cos(theta)
    z_imag = np.sin(theta)
    
    # 2. Data Export (Trajectory)
    df = pd.DataFrame({'Theta': theta, 'Real': z_real, 'Imag': z_imag})
    csv_path = os.path.join(DATA_DIR, "Pillar3_Euler_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [Data] CSV Saved: {csv_path}")
    
    # 3. Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit Circle Trajectory
    ax.plot(z_real, z_imag, 'k--', label='Phase Evolution', alpha=0.5)
    
    # Key Points (0, 1, -1)
    ax.scatter([1, -1], [0, 0], c=['blue', 'red'], s=150, zorder=5)
    ax.text(1.1, 0, "Initial State (+1)\n(Matter)", color='blue', fontweight='bold')
    ax.text(-1.4, 0, "Final State (-1)\n(Dark Matter?)", color='red', fontweight='bold')
    
    # Vectors
    ax.arrow(0, 0, 1, 0, head_width=0.05, fc='b', ec='b', alpha=0.3)
    ax.arrow(0, 0, -1, 0, head_width=0.05, fc='r', ec='r', alpha=0.3)
    
    # Formula
    ax.text(-0.5, 0.5, r"$e^{i\pi} + 1 = 0$", fontsize=20, 
            bbox=dict(facecolor='white', edgecolor='purple'))
    
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.set_title("Pillar 3: Euler's Identity\n(Balancing the Entropic Equation)")
    ax.set_xlabel('Real Axis (Physical)')
    ax.set_ylabel('Imaginary Axis (Information/Entropy)')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, "Fig3_Euler_Template.png")
    plt.savefig(save_path)
    
    print("   [Plot] Graph 3 Generated. Proceeding...")
    # plt.show()
    plt.close(fig)

# ==============================================================================
# [Pillar 4] Holographic Screen (Area Law)
# ==============================================================================
def proof_4_holographic_screen():
    print("\n[Pillar 4] Processing Holographic Screen (Area Law)...")
    
    # 1. Physics Model: Sphere Information Capacity
    R = np.linspace(0.1, 10, 100)
    Area = 4 * np.pi * R**2
    Bits = Area / 4 # (Assuming Planck Area units, Bekenstein-Hawking)
    
    # 2. Data Export
    df = pd.DataFrame({'Radius': R, 'Area': Area, 'Max_Entropy_Bits': Bits})
    csv_path = os.path.join(DATA_DIR, "Pillar4_Holography_Data.csv")
    df.to_csv(csv_path, index=False)
    print(f"   [Data] CSV Saved: {csv_path}")
    
    # 3. Visualization
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(R, Bits, 'm-', lw=3, label=r'Info Capacity $\propto Area$')
    
    plt.xlabel('Horizon Radius $R$')
    plt.ylabel('Information Bits (N)')
    plt.title('Pillar 4: Holographic Principle\n(Gravity is encoded on the boundary)')
    
    # Fill Area
    plt.fill_between(R, 0, Bits, color='purple', alpha=0.1)
    
    plt.text(2, Bits.max()*0.6, r"$N = \frac{A}{4 L_P^2}$ (Planck units)", fontsize=20, color='purple')
    plt.text(5, Bits.max()*0.3, "The universe is a pixelated screen.\nGravity is the logic processing\nof these pixels.", fontsize=12)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(IMG_DIR, "Fig4_Holographic_Screen.png")
    plt.savefig(save_path)
    
    print("   [Plot] Graph 4 Generated. Protocol Completed.")
    # plt.show()
    plt.close(fig)

# ==============================================================================
# [Main Execution]
# ==============================================================================
if __name__ == "__main__":
    print("==================================================")
    print("   ISUT Theoretical Framework: 4 Pillars (Conceptual Schematics)")
    print("==================================================")
    
    proof_1_zipfian_redundancy()
    proof_2_landauer_limit()
    proof_3_euler_template()
    proof_4_holographic_screen()
    
    print("\n[System] All 4 Pillars have been Visualized (conceptual schematics).")
    
    # --- Save minimal run metadata (reproducibility) ---
    import json, platform
    meta = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": plt.matplotlib.__version__,
        "script": os.path.basename(__file__),
    }
    with open(os.path.join(DATA_DIR, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[System] Check the output folder: {BASE_OUT_DIR}")