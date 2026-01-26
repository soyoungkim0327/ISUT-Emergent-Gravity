""" 
External-Field Effect (EFE) Sensitivity Scan (Reviewer Defense)
==============================================================

Why this exists
---------------
One common reviewer attack on MOND-like frameworks is:
  "What about the External Field Effect (EFE)?"

In full AQUAL/QUMOND, the EFE arises from the nonlinearity of the field
equation and boundary conditions. Implementing a fully self-consistent EFE
solve requires an external-field boundary condition (or a multi-scale
environment model), which is outside the scope of the lightweight SPARC
rotation-curve pipeline.

This script therefore provides a **controlled, clearly-labeled sensitivity
scan** that answers the practical reviewer question:

  "If I include a plausible constant external field g_ext, how much do the
   predicted rotation curves change?"

We use a scalar proxy prescription (colinear worst/best case):

    g_int(r) = g_N(r) * nu( |g_N(r) + s*g_ext| / a0 )

where s = +1 (parallel, maximal EFE impact) or s = -1 (anti-parallel,
minimal EFE impact). This ensures g_int -> 0 when g_N -> 0.

Outputs (relative, reviewer-friendly)
------------------------------------
Creates a folder next to this script:

  03.Advanced_Validation/11.efe_sensitivity_scan__/
      data/<gal>_efe_scan.csv
      figures/<gal>_efe_scan.png
      logs/run_metadata.json

Run
---
    python 03.Advanced_Validation/11.efe_sensitivity_scan__.py --gal NGC3198 \
        --gext_a0 0 0.3 1.0 3.0 --nu simple --mode parallel
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required for this script") from e

import matplotlib.pyplot as plt

# --- ISUT shared helpers ---
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isut_000_common import ScriptPaths, find_sparc_data_dir, ensure_rotmod_file, write_run_metadata


# -----------------------------
# Units / constants
# -----------------------------

# 1 (km/s)^2 / kpc  ->  m/s^2
ACCEL_CONV = 3.24078e-14

# Reference MOND acceleration scale (SI). Keep consistent with the rest of repo.
A0_SI_DEFAULT = 1.2e-10


def nu_simple(y: np.ndarray) -> np.ndarray:
    """'Simple' nu-function: nu = 0.5*(1 + sqrt(1 + 4/y))."""
    y = np.maximum(y, 1e-30)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y))


def nu_standard(y: np.ndarray) -> np.ndarray:
    """nu corresponding to mu(x)=x/sqrt(1+x^2)."""
    y = np.maximum(y, 1e-30)
    return np.sqrt(0.5 * (1.0 + np.sqrt(1.0 + 4.0 / (y**2))))


def load_rotmod(path: Path) -> tuple[np.ndarray, ...]:
    """Load SPARC *_rotmod.dat (whitespace-separated).

    Expected columns (SPARC convention):
      0:R[kpc], 1:Vobs, 2:Verr, 3:Vgas, 4:Vdisk, 5:Vbulge
    """
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None)
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    if df.shape[1] < 6:
        raise ValueError(f"Unexpected rotmod format: {path} (ncol={df.shape[1]})")
    R = df.iloc[:, 0].to_numpy(float)
    Vobs = df.iloc[:, 1].to_numpy(float)
    Verr = df.iloc[:, 2].to_numpy(float)
    Vgas = df.iloc[:, 3].to_numpy(float)
    Vdisk = df.iloc[:, 4].to_numpy(float)
    Vbul = df.iloc[:, 5].to_numpy(float)
    return R, Vobs, Verr, Vgas, Vdisk, Vbul


def compute_v_bary2(Vgas: np.ndarray, Vdisk: np.ndarray, Vbul: np.ndarray, ups_d: float, ups_b: float) -> np.ndarray:
    Vb2 = (np.abs(Vgas) * Vgas) + (ups_d * np.abs(Vdisk) * Vdisk) + (ups_b * np.abs(Vbul) * Vbul)
    return np.maximum(Vb2, 0.0)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="EFE sensitivity scan (scalar proxy)")
    ap.add_argument("--gal", "--galaxy", dest="gal", required=True, help="Galaxy name, e.g., NGC3198")
    ap.add_argument("--nu", choices=["simple", "standard"], default="simple")
    ap.add_argument("--mode", choices=["parallel", "antiparallel"], default="parallel",
                    help="Colinear worst/best-case external field orientation.")
    ap.add_argument("--gext_a0", nargs="+", type=float, default=[0.0, 0.3, 1.0, 3.0],
                    help="External field magnitudes in units of a0 (e.g., 0 0.3 1 3)")
    ap.add_argument("--a0_si", type=float, default=A0_SI_DEFAULT, help="a0 in SI (m/s^2)")
    ap.add_argument("--ups_disk", type=float, default=0.5)
    ap.add_argument("--ups_bulge", type=float, default=0.7)
    ap.add_argument("--no_download", action="store_true", help="Disable auto-download of missing SPARC files")

    args = ap.parse_args(argv)

    sp = ScriptPaths.for_script(__file__)

    allow_download = (not args.no_download) and (os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1")

    data_dir = find_sparc_data_dir(sp.script_dir)
    rotmod_path = ensure_rotmod_file(args.gal, data_dir, allow_download=allow_download)

    R, Vobs, Verr, Vgas, Vdisk, Vbul = load_rotmod(rotmod_path)

    a0_code = args.a0_si / ACCEL_CONV

    Vb2 = compute_v_bary2(Vgas, Vdisk, Vbul, args.ups_disk, args.ups_bulge)
    gN = Vb2 / np.maximum(R, 0.01)

    nu_fn = nu_simple if args.nu == "simple" else nu_standard

    # Baseline (no EFE)
    y0 = gN / a0_code
    g0 = gN * nu_fn(y0)
    V0 = np.sqrt(np.maximum(g0 * R, 0.0))

    s = +1.0 if args.mode == "parallel" else -1.0

    out = {
        "R_kpc": R,
        "Vobs_kms": Vobs,
        "Verr_kms": Verr,
        "Vbar_kms": np.sqrt(np.maximum(Vb2, 0.0)),
        "Vpred_no_EFE_kms": V0,
    }

    # Scan g_ext
    for gext_a0 in args.gext_a0:
        gext = float(gext_a0) * a0_code
        y_eff = np.abs(gN + s * gext) / a0_code
        g_int = gN * nu_fn(y_eff)
        V_pred = np.sqrt(np.maximum(g_int * R, 0.0))
        out[f"Vpred_EFE_{args.mode}_gext{gext_a0:g}a0_kms"] = V_pred

    df = pd.DataFrame(out)
    csv_path = sp.data_dir / f"{args.gal}_efe_scan.csv"
    df.to_csv(csv_path, index=False)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(R, Vobs, yerr=Verr, fmt="k.", alpha=0.6, label="Observed")
    plt.plot(R, V0, lw=2, label=f"ISUT/MOND (no EFE), nu={args.nu}")
    for gext_a0 in args.gext_a0:
        key = f"Vpred_EFE_{args.mode}_gext{gext_a0:g}a0_kms"
        if gext_a0 == 0.0:
            continue
        plt.plot(R, df[key], lw=1.8, label=f"EFE proxy: g_ext={gext_a0:g} a0 ({args.mode})")
    plt.xlabel("R [kpc]")
    plt.ylabel("V [km/s]")
    plt.title(f"EFE Sensitivity Scan (proxy) — {args.gal}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    fig_path = sp.fig_dir / f"{args.gal}_efe_scan.png"
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160)
    plt.close()

    # Metadata
    write_run_metadata(
        sp.log_dir,
        args={
            "gal": args.gal,
            "nu": args.nu,
            "mode": args.mode,
            "gext_a0": args.gext_a0,
            "a0_si": args.a0_si,
            "ups_disk": args.ups_disk,
            "ups_bulge": args.ups_bulge,
            "allow_download": allow_download,
            "sparc_dir": str(data_dir),
        },
        notes={
            "efe_proxy": "g_int = gN * nu(|gN + s*gext|/a0) (scalar sensitivity; not a full boundary-condition solve)",
            "csv": str(csv_path.relative_to(sp.out_root)),
            "figure": str(fig_path.relative_to(sp.out_root)),
        },
    )

    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
