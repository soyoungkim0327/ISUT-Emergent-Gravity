# -*- coding: utf-8 -*-
"""
QUMOND Phantom Density 3D Visualization (Interactive + Paper)
=============================================================
[Modifications]
 1. Removed 'Agg' mode -> 3D window pops up immediately upon execution.
 2. Auto-open HTML -> Browser opens automatically after execution.
 3. High-quality paper figure saving functionality is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser  # [Added] For auto-opening the browser
import numpy as np

# --- Visualization Setup ---
# [Important] Disable 'Agg' mode for paper saving. Now the window appears!
# matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Check for Plotly
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("[Info] Plotly not found. Web 3D window might not open. (pip install plotly)")

# --- Import Physics Engine ---
# (isut_100_qumond_pm.py must be in the same folder)
try:
    from isut_100_qumond_pm import (
        PMGrid3D,
        QUMONDSolverFFT,
        exponential_disk_density,
        nu_simple,
    )
except ImportError:
    print("[Error] 'isut_100_qumond_pm.py' not found in the same folder!")
    sys.exit(1)

# =============================================================================
# Path Setup
# =============================================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUT_ROOT = os.path.join(CURRENT_DIR, SCRIPT_NAME)
FIG_DIR = os.path.join(OUT_ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Paper Style (White Background)
plt.rcParams.update({
    'font.size': 12, 'font.family': 'sans-serif',
    'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'grid.alpha': 0.3,
})

# =============================================================================
# Physics Calculation Functions (Kept as is)
# =============================================================================

def _grad_periodic(phi, grid):
    phi_k = np.fft.fftn(phi)
    return (np.fft.ifftn(1j * grid.kx * phi_k).real,
            np.fft.ifftn(1j * grid.ky * phi_k).real,
            np.fft.ifftn(1j * grid.kz * phi_k).real)

def _div_periodic(Fx, Fy, Fz, grid):
    div_k = 1j * (grid.kx * np.fft.fftn(Fx) + 
                  grid.ky * np.fft.fftn(Fy) + 
                  grid.kz * np.fft.fftn(Fz))
    return np.fft.ifftn(div_k).real

# =============================================================================
# Main Execution
# =============================================================================

def main():
    # 1. Settings
    N = 64         # Resolution (Lower to 48 if slow)
    L = 200.0      # Box Size
    M = 1000.0     # Galaxy Mass
    quantile = 0.98 # Visualize top 2% only (to prevent lag)
    max_points = 15000 

    print(f"[System] Starting 3D Simulation (Resolution N={N})...")

    # 2. Physics Calculation
    grid = PMGrid3D(N=N, boxsize=L, G=1.0)
    rho_bar = exponential_disk_density(grid, M_total=M, R_d=3.0, z_d=0.4, renormalize=True)
    
    solver = QUMONDSolverFFT(grid)
    res = solver.solve(rho=rho_bar, a0=0.12, nu_func=nu_simple, pad_factor=1)

    # Calculate Phantom Density
    dx, dy, dz = _grad_periodic(res.phi_N, grid)
    gN_mag = np.sqrt(dx**2 + dy**2 + dz**2)
    nu_val = nu_simple(gN_mag / 0.12)
    
    div_F = _div_periodic(-nu_val*dx, -nu_val*dy, -nu_val*dz, grid)
    rho_ph = (-(1.0 / (4.0 * np.pi)) * div_F) - (rho_bar - np.mean(rho_bar))

    # 3. Data Extraction (For Visualization)
    abs_rho = np.abs(rho_ph)
    threshold = np.quantile(abs_rho, quantile)
    
    # Coordinate Grid
    x_1d = np.linspace(-L/2, L/2, N)
    X, Y, Z = np.meshgrid(x_1d, x_1d, x_1d, indexing='ij')

    # Select Points (Red=Positive, Blue=Negative)
    pos_mask = rho_ph > threshold
    neg_mask = rho_ph < -threshold

    # Downsample Points (Prevent Lag)
    def subsample(mask, limit):
        idx = np.argwhere(mask)
        if len(idx) > limit:
            rng = np.random.default_rng(42)
            idx = idx[rng.choice(len(idx), limit, replace=False)]
        return idx

    limit = max_points // 2
    idx_p = subsample(pos_mask, limit)
    idx_n = subsample(neg_mask, limit)

    xp, yp, zp = X[idx_p[:,0], idx_p[:,1], idx_p[:,2]], Y[idx_p[:,0], idx_p[:,1], idx_p[:,2]], Z[idx_p[:,0], idx_p[:,1], idx_p[:,2]]
    xn, yn, zn = X[idx_n[:,0], idx_n[:,1], idx_n[:,2]], Y[idx_n[:,0], idx_n[:,1], idx_n[:,2]], Z[idx_n[:,0], idx_n[:,1], idx_n[:,2]]

    # =========================================================================
    # [1] Open Matplotlib Window (Python Default)
    # =========================================================================
    print("[Display 1] Generating Python 3D Window... (Popup imminent)")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(xp, yp, zp, c='crimson', s=2, alpha=0.4, label='Disk (+) Mass')
    ax.scatter(xn, yn, zn, c='royalblue', s=2, alpha=0.4, label='Holes (-) Mass')

    ax.set_xlabel('X [kpc]'); ax.set_ylabel('Y [kpc]'); ax.set_zlabel('Z [kpc]')
    ax.set_title(f'Phantom Density Structure (Thresh > {threshold:.1e})')
    ax.legend()
    
    # Save for Paper
    png_path = os.path.join(FIG_DIR, "fig_phantom_3d_paper.png")
    plt.savefig(png_path)
    print(f"   -> Image Saved: {png_path}")

    # =========================================================================
    # [2] Open Plotly Web Window (Smoother)
    # =========================================================================
    if HAS_PLOTLY:
        print("[Display 2] Generating Web Browser 3D Window...")
        fig_html = go.Figure()

        fig_html.add_trace(go.Scatter3d(
            x=xp, y=yp, z=zp, mode='markers',
            marker=dict(size=3, color='red', opacity=0.3, symbol='square'),
            name='Positive Mass (Disk)'
        ))

        fig_html.add_trace(go.Scatter3d(
            x=xn, y=yn, z=zn, mode='markers',
            marker=dict(size=3, color='blue', opacity=0.3, symbol='square'),
            name='Negative Mass (Holes)'
        ))

        fig_html.update_layout(
            title="Interactive QUMOND Phantom Density",
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z', aspectmode='cube'),
            template="plotly_white"
        )

        html_path = os.path.join(FIG_DIR, "fig_phantom_3d_interactive.html")
        fig_html.write_html(html_path)
        
        # [Core] Auto-open browser!
        webbrowser.open('file://' + os.path.abspath(html_path))
        print("   -> Browser opened! Rotate with mouse.")

    # Show Python window last (This pauses the script and keeps window open)
    print("Rotate the graph with your mouse. Close window to exit.")
    plt.show()

if __name__ == "__main__":
    main()