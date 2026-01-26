""" 
Laplace-Approximate Bayesian Evidence (Reviewer Defense)
=======================================================

Motivation
----------
The repository already reports a BIC* score as an internal
complexity-penalized benchmark. A harsh reviewer may still ask:

  "Can you provide a Bayesian evidence / Bayes factor comparison?"

Full nested sampling (MultiNest/dynesty) is heavier than this repo's
dependency-light goal. This script therefore provides a *transparent* and
*reproducible* middle ground: a Laplace approximation to the marginal
likelihood around each model's best-fit point.

What it does
------------
For a small galaxy subset (default: Golden12), fit two proxy models:
  - ISUT/MOND baryon-coupled proxy: (ups_disk, ups_bulge)
  - NFW halo proxy: (ups_disk, ups_bulge, V200, c)

Then estimate a "tilde" log-evidence (constants common to both models drop
out in the Bayes factor) using:

  log Z~ = -0.5 * chi2(θ̂) + (k/2) log(2π) + 0.5 log det(Σ) - log(V_prior)

where Σ is the inverse Hessian of the negative log-likelihood at θ̂ and
V_prior is the uniform prior volume.

Important caveats
-----------------
* This is an approximation.
* The result depends on prior ranges.
* We report it as *supplementary* reviewer-facing context, not as a single
  definitive verdict.

Outputs
-------
Creates:
  03.Advanced_Validation/13.bayes_evidence_laplace__/data/evidence_summary.csv

Run
---
  python 03.Advanced_Validation/13.bayes_evidence_laplace__.py --nmax 12
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise RuntimeError("pandas is required for this script") from e

from scipy.optimize import minimize

# --- ISUT shared helpers ---
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import isut_000_common as _common
from isut_000_common import ScriptPaths, find_sparc_data_dir, ensure_rotmod_file, write_run_metadata


# -----------------------------
# Units / constants
# -----------------------------

ACCEL_CONV = 3.24078e-14  # (km/s)^2/kpc -> m/s^2
A0_SI_DEFAULT = 1.2e-10


def nu_simple(y: np.ndarray) -> np.ndarray:
    y = np.maximum(y, 1e-30)
    return 0.5 * (1.0 + np.sqrt(1.0 + 4.0 / y))


def load_rotmod(path: Path) -> Tuple[np.ndarray, ...]:
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


def velocity_nfw(R: np.ndarray, V200: float, c: float, R200: float = 200.0) -> np.ndarray:
    """Very standard NFW circular velocity proxy (kpc, km/s)."""
    R = np.maximum(R, 1e-6)
    x = c * R / R200
    f = np.log(1.0 + x) - x / (1.0 + x)
    fc = np.log(1.0 + c) - c / (1.0 + c)
    return V200 * np.sqrt(np.maximum(f / (x * fc), 0.0))


def predict_isut(R, Vgas, Vdisk, Vbul, ups_d, ups_b, a0_code) -> np.ndarray:
    Vb2 = compute_v_bary2(Vgas, Vdisk, Vbul, ups_d, ups_b)
    gN = Vb2 / np.maximum(R, 0.01)
    y = gN / a0_code
    g = gN * nu_simple(y)
    return np.sqrt(np.maximum(g * R, 0.0))


def predict_dm(R, Vgas, Vdisk, Vbul, ups_d, ups_b, V200, c) -> np.ndarray:
    Vb2 = compute_v_bary2(Vgas, Vdisk, Vbul, ups_d, ups_b)
    Vdm = velocity_nfw(R, V200, c)
    return np.sqrt(np.maximum(Vb2 + Vdm**2, 0.0))


def chi2(Vobs: np.ndarray, Verr: np.ndarray, Vpred: np.ndarray) -> float:
    Verr = np.maximum(Verr, 1e-3)
    return float(np.sum(((Vobs - Vpred) / Verr) ** 2))


def finite_hessian(f, x0: np.ndarray, step: np.ndarray) -> np.ndarray:
    """Central finite-difference Hessian for small k."""
    x0 = x0.astype(float)
    k = len(x0)
    H = np.zeros((k, k), float)
    f0 = f(x0)
    for i in range(k):
        ei = np.zeros(k); ei[i] = 1.0
        for j in range(i, k):
            ej = np.zeros(k); ej[j] = 1.0
            if i == j:
                fp = f(x0 + step[i] * ei)
                fm = f(x0 - step[i] * ei)
                H[i, i] = (fp - 2 * f0 + fm) / (step[i] ** 2)
            else:
                fpp = f(x0 + step[i]*ei + step[j]*ej)
                fpm = f(x0 + step[i]*ei - step[j]*ej)
                fmp = f(x0 - step[i]*ei + step[j]*ej)
                fmm = f(x0 - step[i]*ei - step[j]*ej)
                H[i, j] = (fpp - fpm - fmp + fmm) / (4 * step[i] * step[j])
                H[j, i] = H[i, j]
    return H


def laplace_logZ(chi2_hat: float, H: np.ndarray, prior_vol: float) -> Tuple[float, Dict[str, float]]:
    """Return logZ~ and diagnostics."""
    k = H.shape[0]
    # Ensure SPD-ish
    w, V = np.linalg.eigh(H)
    w_floor = np.maximum(w, 1e-9)
    Hs = (V * w_floor) @ V.T
    Sigma = np.linalg.inv(Hs)
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        logdet = float('nan')
    logZ = -0.5 * chi2_hat + 0.5 * k * np.log(2.0 * np.pi) + 0.5 * logdet - np.log(prior_vol)
    diag = {
        "k": float(k),
        "logdet_Sigma": float(logdet),
        "min_eig_H": float(np.min(w)),
        "prior_vol": float(prior_vol),
    }
    return float(logZ), diag


def ensure_rotmod_file_resilient(
    gal_name: str,
    data_dir: Path,
    *,
    allow_download: bool,
    timeout: float = 5.0,
) -> Path:
    '''
    Wrapper around ``ensure_rotmod_file`` that is resilient to SPARC GitHub mirror
    branch-name changes (e.g. ``master`` -> ``main``).

    The core repo helper uses a single URL template. This wrapper tries a small
    list of equivalent templates and returns the first successful download.

    If the file exists locally, no network is used.
    '''
    # Start from whatever the common module is configured with.
    base = getattr(_common, "SPARC_URL_TEMPLATE", "")
    templates = []
    if isinstance(base, str) and base:
        templates.append(base)
        if "/master/" in base:
            templates.append(base.replace("/master/", "/main/"))
        if "/main/" in base:
            templates.append(base.replace("/main/", "/master/"))

    # Hard-coded fallbacks (kept minimal on purpose).
    templates += [
        "https://raw.githubusercontent.com/jobovy/sparc-rotation-curves/main/data/{gal}_rotmod.dat",
        "https://raw.githubusercontent.com/jobovy/sparc-rotation-curves/master/data/{gal}_rotmod.dat",
    ]

    # De-duplicate while preserving order.
    seen = set()
    templates = [t for t in templates if t and not (t in seen or seen.add(t))]

    last_err: Exception | None = None
    for tmpl in templates:
        try:
            _common.SPARC_URL_TEMPLATE = tmpl
            return ensure_rotmod_file(
                gal_name, data_dir, allow_download=allow_download, timeout=timeout
            )
        except FileNotFoundError as e:
            last_err = e
            if not allow_download:
                # If downloads are disabled, we can fail fast.
                raise
            continue

    # If we get here, all attempts failed.
    if last_err is not None:
        raise last_err
    raise FileNotFoundError(f"Missing SPARC file for {gal_name} (no URL templates configured).")

GOLDEN12 = [
    "NGC6503", "NGC3198", "NGC2403", "NGC6946", "NGC2998", "NGC3953",
    "NGC5055", "NGC3521", "NGC7331", "DDO154", "IC2574", "UGC128"
]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Laplace-approx evidence (ISUT vs NFW)")
    ap.add_argument("--nmax", type=int, default=12, help="How many galaxies from Golden12")
    ap.add_argument("--a0_si", type=float, default=A0_SI_DEFAULT)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args(argv)

    sp = ScriptPaths.for_script(__file__)
    allow_download = (not args.no_download) and (os.environ.get("ISUT_NO_DOWNLOAD", "0") != "1")

    sparc_dir = find_sparc_data_dir(sp.script_dir)
    a0_code = args.a0_si / ACCEL_CONV

    rows = []
    for gal in GOLDEN12[: max(1, min(args.nmax, len(GOLDEN12)))]:
        try:
            fp = ensure_rotmod_file_resilient(gal, sparc_dir, allow_download=allow_download)
        except FileNotFoundError as e:
            print(f"[WARN] Skipping {gal} (missing SPARC rotmod): {e}")
            continue
        R, Vobs, Verr, Vgas, Vdisk, Vbul = load_rotmod(fp)
        n = len(R)

        # ---- ISUT fit (ups_d, ups_b)
        def obj_isut(p):
            ups_d, ups_b = p
            Vpred = predict_isut(R, Vgas, Vdisk, Vbul, ups_d, ups_b, a0_code)
            return 0.5 * chi2(Vobs, Verr, Vpred)

        res_i = minimize(obj_isut, x0=np.array([0.5, 0.7]), bounds=[(0.05, 2.0), (0.05, 2.5)], method="L-BFGS-B")
        p_i = res_i.x
        chi2_i = 2.0 * float(res_i.fun)
        # Hessian step sizes
        step_i = np.array([0.02, 0.02])
        H_i = finite_hessian(obj_isut, p_i, step_i)
        prior_i = (2.0 - 0.05) * (2.5 - 0.05)
        logZ_i, diag_i = laplace_logZ(chi2_i, H_i, prior_i)

        # ---- DM fit (ups_d, ups_b, V200, c)
        def obj_dm(p):
            ups_d, ups_b, V200, c = p
            Vpred = predict_dm(R, Vgas, Vdisk, Vbul, ups_d, ups_b, V200, c)
            return 0.5 * chi2(Vobs, Verr, Vpred)

        res_d = minimize(
            obj_dm,
            x0=np.array([0.5, 0.7, 150.0, 10.0]),
            bounds=[(0.05, 2.0), (0.05, 2.5), (40.0, 300.0), (4.0, 25.0)],
            method="L-BFGS-B",
        )
        p_d = res_d.x
        chi2_d = 2.0 * float(res_d.fun)
        step_d = np.array([0.02, 0.02, 2.0, 0.3])
        H_d = finite_hessian(obj_dm, p_d, step_d)
        prior_d = (2.0 - 0.05) * (2.5 - 0.05) * (300.0 - 40.0) * (25.0 - 4.0)
        logZ_d, diag_d = laplace_logZ(chi2_d, H_d, prior_d)

        dlogZ = logZ_i - logZ_d

        rows.append({
            "Galaxy": gal,
            "n_points": n,
            "chi2_ISUT": chi2_i,
            "chi2_DM": chi2_d,
            "logZtilde_ISUT": logZ_i,
            "logZtilde_DM": logZ_d,
            "Delta_logZtilde_ISUT_minus_DM": dlogZ,
            "ISUT_ups_disk": p_i[0],
            "ISUT_ups_bulge": p_i[1],
            "DM_ups_disk": p_d[0],
            "DM_ups_bulge": p_d[1],
            "DM_V200": p_d[2],
            "DM_c": p_d[3],
            "diag_ISUT_min_eig_H": diag_i["min_eig_H"],
            "diag_DM_min_eig_H": diag_d["min_eig_H"],
        })

    df = pd.DataFrame(rows)
    out_csv = sp.data_dir / "evidence_summary.csv"
    df.to_csv(out_csv, index=False)

    write_run_metadata(
        sp.log_dir,
        args={"nmax": args.nmax, "a0_si": args.a0_si, "allow_download": allow_download, "sparc_dir": str(sparc_dir), "sparc_url_template": getattr(_common, "SPARC_URL_TEMPLATE", "")},
        notes={
            "model": "Laplace approximation around MAP (uniform priors)",
            "priors": {
                "ISUT": {"ups_disk": "U(0.05,2.0)", "ups_bulge": "U(0.05,2.5)"},
                "DM": {"ups_disk": "U(0.05,2.0)", "ups_bulge": "U(0.05,2.5)", "V200": "U(40,300)", "c": "U(4,25)"},
            },
            "file": str(out_csv.relative_to(sp.out_root)),
        },
    )

    print(f"[OK] Wrote: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())