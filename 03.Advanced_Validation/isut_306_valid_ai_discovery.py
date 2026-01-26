
"""
ISUT Symbolic Regression & Law Discovery Suite
==============================================
Objective:
    Leverages Symbolic Regression (PySR) to discover the underlying functional 
    form of the modified gravity law directly from observational data.

Key Methodologies:
1. Data-Driven Discovery: Uses PySR to search for the most parsimonious 
   mathematical expression that describes the MOND-like behavior.
2. Symbolic Parsimony: Penalizes complex expressions to find 'physical' 
   laws rather than just over-fitted numerical approximations.
3. Domain Flexibility: Supports both linear and log-space fitting to ensure 
   numerical stability across different acceleration scales.
"""


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
import argparse

# Check for PySR availability (Symbolic Regression Library)
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("[System] PySR not found. Proceeding with pre-calculated symbolic model (Simulation Mode).")

# Suppress warnings
warnings.filterwarnings("ignore")

# ==============================================================================
# [1] Configuration: Dynamic Path Setup (Smart Path Finder)
# ==============================================================================
# 1. Determine current script location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]

# 2. Input Data Directory (Smart Search)
DATA_DIR = str(find_sparc_data_dir(Path(CURRENT_DIR)))
print(f"[System] SPARC data directory: {DATA_DIR} (override via ISUT_SPARC_DIR)")

BASE_OUT_DIR = os.path.join(CURRENT_DIR, SCRIPT_NAME)
os.makedirs(BASE_OUT_DIR, exist_ok=True)

print(f"[System] ISUT AI Symbolic Discovery Initialized")
print(f"[System] Output Directory: {BASE_OUT_DIR}")

# ==============================================================================
# [2] Galaxy Definitions
# ==============================================================================
# Full Survey (65 Galaxies)
FULL_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521", "NGC5055", "NGC7331",
    "NGC6946", "NGC7793", "NGC1560", "UGC02885", "NGC0801", "NGC2998", "NGC5033", "NGC5533",
    "NGC5907", "NGC6674", "UGC06614", "UGC06786", "F568-3", "F571-8", "NGC0055", "NGC0247",
    "NGC0300", "NGC1003", "NGC1365", "NGC2541", "NGC2683", "NGC2915", "NGC3109", "NGC3621",
    "NGC3726", "NGC3741", "NGC3769", "NGC3877", "NGC3893", "NGC3917", "NGC3949", "NGC3953",
    "NGC3972", "NGC3992", "NGC4013", "NGC4051", "NGC4085", "NGC4088", "NGC4100", "NGC4138",
    "NGC4157", "NGC4183", "NGC4217", "NGC4559", "NGC5585", "NGC5985", "NGC6015", "NGC6195",
    "UGC06399", "UGC06446", "UGC06667", "UGC06818", "UGC06917", "UGC06923", "UGC06930",
    "UGC06983", "UGC07089",
]
FULL_GALAXIES = sorted(list(set(FULL_GALAXIES)))

# Golden Set (12 High-Quality Galaxies)
GOLDEN_GALAXIES = [
    "NGC6503", "NGC3198", "NGC2403", "NGC2841", "NGC2903", "NGC3521",
    "NGC5055", "NGC7331", "NGC6946", "NGC7793", "NGC1560", "UGC02885",
]

# Physical Constants
ACCEL_CONV = 3.24078e-14  # km^2/s^2/kpc -> m/s^2
A0_TARGET = 1.2e-10       # m/s^2

# ==============================================================================
# [2.5] Baselines + Metrics (Reviewer-grade)
# ==============================================================================

def rar_mcgaugh(g_bar, a0):
    """McGaugh et al. RAR: g_obs = g_bar / (1 - exp(-sqrt(g_bar/a0)))"""
    g_safe = np.maximum(g_bar, 1e-30)
    x = np.sqrt(g_safe / a0)
    denom = 1.0 - np.exp(-x)
    denom = np.maximum(denom, 1e-12)
    return g_safe / denom


def fit_a0_mcgaugh(g_bar, g_obs, a0_min=2e-11, a0_max=5e-10, n=240):
    """Deterministic grid search to avoid optimizer randomness."""
    grid = np.logspace(np.log10(a0_min), np.log10(a0_max), n)
    logy = np.log10(np.maximum(g_obs, 1e-30))
    best_a0 = grid[0]
    best_rmse = np.inf
    for a0 in grid:
        pred = rar_mcgaugh(g_bar, a0)
        rmse = float(np.sqrt(np.mean((np.log10(np.maximum(pred, 1e-30)) - logy) ** 2)))
        if rmse < best_rmse:
            best_a0 = a0
            best_rmse = rmse
    return float(best_a0), float(best_rmse)


def metrics_log10(y_true, y_pred):
    yt = np.log10(np.maximum(y_true, 1e-30))
    yp = np.log10(np.maximum(y_pred, 1e-30))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    bias = float(np.mean(yp - yt))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"RMSE_log10": rmse, "Bias_log10": bias, "R2_log10": r2}


# ==============================================================================
# [3] Data Loading Logic
# ==============================================================================

def get_data(gal_name):
    # Try multiple common naming conventions
    candidates = [f"{gal_name}_rotmod.dat", f"{gal_name}.dat"]
    path = None
    for fname in candidates:
        temp_path = os.path.join(DATA_DIR, fname)
        if os.path.exists(temp_path):
            path = temp_path
            break

    # Download if missing
    if path is None or not os.path.exists(path):
        try:
            path = str(ensure_rotmod_file(gal_name, Path(DATA_DIR), allow_download=ALLOW_NET_DOWNLOAD, timeout=10.0))
        except FileNotFoundError:
            return None
    try:
        df = pd.read_csv(path, sep=r"\s+", comment="#", header=None).apply(pd.to_numeric, errors="coerce").dropna()
        if len(df) < 5:
            return None
        # R, Vobs, Verr, Vgas, Vdisk, Vbul
        return (
            df.iloc[:, 0].values,
            df.iloc[:, 1].values,
            df.iloc[:, 2].values,
            df.iloc[:, 3].values,
            (df.iloc[:, 4].values if df.shape[1] > 4 else np.zeros(len(df))),
            (df.iloc[:, 5].values if df.shape[1] > 5 else np.zeros(len(df))),
        )
    except Exception:
        return None


# ==============================================================================
# [4] Analysis Pipeline
# ==============================================================================

def process_galaxies_for_rar(galaxy_list, return_labels=False):
    """Aggregates RAR data for a list of galaxies. Returns arrays in m/s^2."""
    g_bar_list = []
    g_obs_list = []
    labels = []

    print(f"   [Process] Loading data for {len(galaxy_list)} galaxies...")

    for gal in galaxy_list:
        data = get_data(gal)
        if data is None:
            continue

        R, V_obs, V_err, V_gas, V_disk, V_bul = data
        R_safe = np.maximum(R, 0.01)

        # Baryonic Mass (Assuming M/L_disk=0.5, M/L_bulge=0.7 as baseline)
        V_bary2 = (np.abs(V_gas) * V_gas) + (0.5 * np.abs(V_disk) * V_disk) + (0.7 * np.abs(V_bul) * V_bul)
        g_bar = np.abs(V_bary2) / R_safe * ACCEL_CONV  # Convert to m/s^2
        g_obs = (V_obs ** 2) / R_safe * ACCEL_CONV     # Convert to m/s^2

        g_bar_list.extend(g_bar)
        g_obs_list.extend(g_obs)
        if return_labels:
            labels.extend([gal] * len(g_bar))

    if return_labels:
        return np.array(g_bar_list), np.array(g_obs_list), np.array(labels, dtype=object)
    return np.array(g_bar_list), np.array(g_obs_list)


def run_analysis(galaxy_list, subset_name, pysr_domain="linear", do_holdout=False, holdout_frac=0.2, seed=42):
    """Main logic: Loads data, runs AI/Sim, generates plots and CSVs for a specific subset."""
    print(f"\n[Analysis] Starting analysis for subset: {subset_name}")

    # Setup Output Folders
    target_dir = os.path.join(BASE_OUT_DIR, subset_name)
    fig_dir = os.path.join(target_dir, "figures")
    data_dir = os.path.join(target_dir, "data")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # Galaxy-level holdout split (OFF by default)
    rng = np.random.default_rng(seed)
    if do_holdout and len(galaxy_list) >= 5:
        gals = list(galaxy_list)
        rng.shuffle(gals)
        k = max(1, int(round(len(gals) * holdout_frac)))
        test_gals = sorted(gals[:k])
        train_gals = sorted(gals[k:])
        print(f"   [Holdout] Train galaxies: {len(train_gals)}, Test galaxies: {len(test_gals)}")

        g_bar_tr, g_obs_tr = process_galaxies_for_rar(train_gals)
        g_bar_te, g_obs_te = process_galaxies_for_rar(test_gals)
        g_bar, g_obs, labels = process_galaxies_for_rar(galaxy_list, return_labels=True)
    else:
        g_bar, g_obs, labels = process_galaxies_for_rar(galaxy_list, return_labels=True)
        g_bar_tr, g_obs_tr = g_bar, g_obs
        g_bar_te, g_obs_te = np.array([]), np.array([])

    if len(g_bar) == 0:
        print(f"   [Error] No valid data found for {subset_name}.")
        return

    # 2. AI Discovery / Simulation
    y_pred_ai_full = None
    y_pred_ai_te = None

    model = None
    if PYSR_AVAILABLE:
        # Prepare data for AI (PySR)
        if pysr_domain == "log":
            X = np.log10(np.maximum(g_bar_tr, 1e-30)).reshape(-1, 1)
            y = np.log10(np.maximum(g_obs_tr, 1e-30))
        else:
            X = g_bar_tr.reshape(-1, 1)
            y = g_obs_tr

        print(f"   [AI] Running PySR Symbolic Regression ({pysr_domain}-domain) on {len(X)} points...")
        try:
            model = PySRRegressor(
                niterations=20,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["sqrt", "log", "exp"],
                loss="loss(prediction, target) = (prediction - target)^2",
                verbosity=0,
            )
            model.fit(X, y)

            # Predict on FULL set
            if pysr_domain == "log":
                X_full = np.log10(np.maximum(g_bar, 1e-30)).reshape(-1, 1)
                y_pred_log = model.predict(X_full)
                y_pred_ai_full = 10 ** y_pred_log
            else:
                X_full = g_bar.reshape(-1, 1)
                y_pred_ai_full = model.predict(X_full)

            # Predict on TEST set (if enabled)
            if do_holdout and len(g_bar_te) > 0:
                if pysr_domain == "log":
                    X_te = np.log10(np.maximum(g_bar_te, 1e-30)).reshape(-1, 1)
                    y_pred_ai_te = 10 ** model.predict(X_te)
                else:
                    X_te = g_bar_te.reshape(-1, 1)
                    y_pred_ai_te = model.predict(X_te)

            print("   [AI] Discovery Complete. Best equation found.")
            # Save Equation
            model.equations_.to_csv(os.path.join(data_dir, f"AI_Equations_{subset_name}.csv"), index=False)
        except Exception as e:
            print(f"   [Warning] PySR failed: {e}. Falling back to simulation.")
            model = None

    # Fallback / Simulation Model (ISUT Formula)
    if y_pred_ai_full is None:
        # ISUT: g_obs = g_bar * nu(g_bar/a0)
        y_norm = g_bar / A0_TARGET
        nu = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y_norm)
        y_pred_ai_full = g_bar * nu

        if do_holdout and len(g_bar_te) > 0:
            y_norm_te = g_bar_te / A0_TARGET
            nu_te = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 / y_norm_te)
            y_pred_ai_te = g_bar_te * nu_te

    # 2.5 McGaugh baseline (fit a0 on FULL set)
    a0_fit_full, rmse_fit_full = fit_a0_mcgaugh(g_bar, g_obs)
    y_pred_mcg_full = rar_mcgaugh(g_bar, a0_fit_full)

    # Test-set McGaugh (optional)
    if do_holdout and len(g_bar_te) > 0:
        a0_fit_te, rmse_fit_te = fit_a0_mcgaugh(g_bar_te, g_obs_te)
        y_pred_mcg_te = rar_mcgaugh(g_bar_te, a0_fit_te)
    else:
        a0_fit_te, rmse_fit_te, y_pred_mcg_te = None, None, None

    # 3. Visualization: Radial Acceleration Relation (RAR)
    plt.figure(figsize=(8, 8))

    # Observed points
    plt.loglog(g_bar, g_obs, "ko", alpha=0.15, markersize=3, label="Observed Data")

    # Newtonian line y=x
    xy_line = np.logspace(np.log10(min(g_bar)), np.log10(max(g_bar)), 200)
    plt.loglog(xy_line, xy_line, "b--", linewidth=2, label="Newtonian (1:1)")

    # McGaugh RAR curve
    plt.loglog(
        xy_line,
        rar_mcgaugh(xy_line, a0_fit_full),
        "g-",
        linewidth=2,
        label=fr"McGaugh RAR (a0={a0_fit_full:.2e})",
    )

    # AI / ISUT prediction line (sorted)
    sort_idx = np.argsort(g_bar)
    plt.loglog(g_bar[sort_idx], y_pred_ai_full[sort_idx], "r-", linewidth=2.5, label="AI / ISUT Law")

    plt.xlabel(r"Baryonic Acceleration $g_{bar}$ $[m/s^2]$")
    plt.ylabel(r"Observed Acceleration $g_{obs}$ $[m/s^2]$")
    plt.title(f"Radial Acceleration Relation ({subset_name})")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)

    fig_path = os.path.join(fig_dir, f"Fig_RAR_{subset_name}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # 4. Save Source Data (CSV) — keep legacy cols + add baseline + label (backward-compatible)
    df_export = pd.DataFrame({
        "g_baryonic": g_bar,
        "g_observed": g_obs,
        "g_predicted_AI": y_pred_ai_full,
        "g_predicted_Newton": g_bar,
        "g_predicted_McGaugh": y_pred_mcg_full,
        "Galaxy": labels,
    })
    csv_path = os.path.join(data_dir, f"Fig_RAR_SourceData_{subset_name}.csv")
    df_export.to_csv(csv_path, index=False)

    # 5. Save Metrics (Reviewer-grade)
    rows = [
        {"Subset": subset_name, "Split": "FULL", "Model": "Newton", **metrics_log10(g_obs, g_bar)},
        {"Subset": subset_name, "Split": "FULL", "Model": "AI_or_ISUT", **metrics_log10(g_obs, y_pred_ai_full)},
        {"Subset": subset_name, "Split": "FULL", "Model": "McGaugh_RAR", **metrics_log10(g_obs, y_pred_mcg_full)},
    ]

    if do_holdout and len(g_bar_te) > 0 and y_pred_ai_te is not None and y_pred_mcg_te is not None:
        rows += [
            {"Subset": subset_name, "Split": "TEST", "Model": "Newton", **metrics_log10(g_obs_te, g_bar_te)},
            {"Subset": subset_name, "Split": "TEST", "Model": "AI_or_ISUT", **metrics_log10(g_obs_te, y_pred_ai_te)},
            {"Subset": subset_name, "Split": "TEST", "Model": "McGaugh_RAR", **metrics_log10(g_obs_te, y_pred_mcg_te)},
        ]

    metrics_path = os.path.join(data_dir, f"RAR_Metrics_{subset_name}.csv")
    pd.DataFrame(rows).to_csv(metrics_path, index=False)

    meta = {
        "subset": subset_name,
        "pysr_domain": pysr_domain,
        "do_holdout": bool(do_holdout),
        "holdout_frac": float(holdout_frac),
        "seed": int(seed),
        "a0_fit_full": float(a0_fit_full),
        "rmse_fit_full": float(rmse_fit_full),
    }
    if a0_fit_te is not None:
        meta.update({"a0_fit_test": float(a0_fit_te), "rmse_fit_test": float(rmse_fit_te)})

    meta_path = os.path.join(data_dir, f"RAR_Meta_{subset_name}.csv")
    pd.DataFrame([meta]).to_csv(meta_path, index=False)

    print(f"   [Success] Analysis saved to: {target_dir}")


# ==============================================================================
# [5] Main Execution
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISUT RAR + Symbolic Regression audit")
    parser.add_argument(
        "--pysr-domain",
        choices=["linear", "log"],
        default="linear",
        help="Fit domain for PySR. 'linear' keeps original behavior; 'log' fits log10 space.",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Enable galaxy-level holdout (train/test split by galaxy). Default OFF.",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Fraction of galaxies held out for testing (only if --holdout). Default 0.2",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for holdout split. Default 42")
    parser.add_argument(
        "--subset",
        choices=["both", "Golden12", "All65"],
        default="both",
        help="Which subset to run. Default both.",
    )
    args = parser.parse_args()

    if args.subset in ("both", "Golden12"):
        run_analysis(
            GOLDEN_GALAXIES,
            "Golden12",
            pysr_domain=args.pysr_domain,
            do_holdout=args.holdout,
            holdout_frac=args.holdout_frac,
            seed=args.seed,
        )

    if args.subset in ("both", "All65"):
        run_analysis(
            FULL_GALAXIES,
            "All65",
            pysr_domain=args.pysr_domain,
            do_holdout=args.holdout,
            holdout_frac=args.holdout_frac,
            seed=args.seed,
        )

    print(f"\n[System] All protocols completed. Output at: {BASE_OUT_DIR}")