

# ISUT — Information Entropy-based Emergent Gravity

> **"What if the missing mass is actually missing information?"**

We present **ISUT**, a project that reimagines galactic gravity not through invisible particles, but through the lens of **information entropy**.

It starts with a simple question: **Does spacetime have an information capacity?**

Surprisingly, when we model this limit (*Zipfian redundancy*) in code, the need for Dark Matter is reduced at the scale of galaxy rotation curves, and the observed rotation-curve phenomenology emerges naturally.

We believe that extraordinary claims require transparent code.  
So, this repository is released as an **Open Evidence Pack**, exposing everything—from core logic to verification scripts and output conventions.

### 🎥 Simulation Demo
[![ISUT Simulation Video](https://img.youtube.com/vi/Qj1V799ct60/0.jpg)](https://www.youtube.com/watch?v=Qj1V799ct60)


1. **The Code**  
   A Python-based simulation/analysis engine that runs on your laptop. It includes an optional **Relational Time** mode designed for long-integration numerical stability (bounded energy drift) and a built-in **Audit Suite** that regenerates reproducible artifacts.

2. **The Data & Pipeline**  
   A consistent pipeline over SPARC-based samples (“All65”, “Golden12”), running key diagnostics and scaling relations such as RAR / TFR.

3. **The AI Check**  
   (Optional) A **Symbolic Regression** track that inspects the data structure “without physics priors,” to test whether discovered patterns are intrinsic to the signal rather than hand-crafted.

4. **The Verification (reviewer-ready scripts)**  
   Reviewer-ready scripts that regenerate concrete artifacts (CSV/PNG + run metadata).  
   They cover internal-consistency checks (identity-by-construction), holdout/sensitivity sweeps, and rebuttal-oriented diagnostics such as “DM-reverse (extra-acceleration)” tests.

This repository is an invitation.  
Everything is open—from the core engine to the plotting scripts—so you don’t have to take our word for it.  
Run it, challenge it, and inspect it end-to-end.

---

## ISUT — QUMOND-based Open Evidence Pack

This repository provides two complementary layers:

1. **Rotation-curve / RAR / TFR pipeline (observational-facing, fast)**  
   Fits SPARC profiles using an algebraic response form such as `g = ν(y) g_N` (and related diagnostics),
   producing summary tables and running holdout / sensitivity / audit checks.

2. **Conservative QUMOND completion (engine-facing, potential-based)**  
   Implements a **QUMOND (two-Poisson) conservative completion** that actually solves for the potential Φ
   and computes forces via `a = -∇Φ`.  
   This layer aligns the implementation with the “potential-based / conservative” claims.

---

## Repository map

```text
.
├── 01.Core_Engine/              # QUMOND conservative solver + engine-level audits
├── 02.Theoretical_Framework/    # derivations / theory checks exported as artifacts
├── 03.Advanced_Validation/      # population tests, holdouts, sensitivity scans, AI check
├── 04.Data_Pipeline/            # deterministic loaders + fitting tools
├── 05.Visualization_Suite/      # figure generation scripts
├── data/
│   └── galaxies/
│       ├── 65_galaxies/         # “All65” (packaged: 60 galaxies; legacy label kept)
│       └── 12_galaxies/         # “Golden12” (packaged: 11 galaxies; legacy label kept)
├── QUMOND_QUICKSTART.md
├── INSTALL.md
├── requirements.txt
└── isut_000_common.py           # shared utilities (paths, JSON/CSV helpers, hashing, etc.)
```

---

## Quickstart (QUMOND conservative completion)

For the full walkthrough, see **`QUMOND_QUICKSTART.md`**.

Minimal “does it run?” commands from the repo root:

### 1) Core demo (field solve → rotation curve → orbit energy audit)

```bash
python 01.Core_Engine/isut_121_engine_core_qumond.py --headless
```

### 2) Audit suite (smoke / fiducial / sensitivity)

```bash
python 01.Core_Engine/isut_123_engine_audit_qumond.py --mode all --headless
```

### 3) One-command reviewer pack runner

```bash
python 01.Core_Engine/isut_111_qumond_reviewer_pack_runner.py --fast
```

---

## Installation

For environment setup (including VS Code kernel notes) and dependencies, see **`INSTALL.md`**.

```bash
pip install -r requirements.txt
```

Notes:

* Python **3.9+** recommended
* `seaborn` is optional. Some figure scripts use it; most fall back to matplotlib.
* Symbolic-regression scripts run without PySR (they fall back to a “mock mode”), but installing PySR enables the full AI blind-recovery experiment.

---

## Output convention (reviewer/reproducibility-critical)

Scripts write outputs to a **relative sibling folder** next to the script:

* `.../isut_###_script.py`
* outputs → `.../isut_###_script/{data,figures,logs}`

Some runner scripts additionally create timestamped run folders:

* `.../isut_###_script/runs/<run_id>/{data,figures,logs}`

This convention makes runs **self-contained and auditable**.

---

## Reviewer-defense / verification scripts

Key scripts that can be executed as a “verification pack” even without the manuscript.  
(Paths are relative to the repo root.)

### Conservative QUMOND engine checks (field-level; potential-based)

* **QUMOND reviewer pack runner**: `01.Core_Engine/isut_111_qumond_reviewer_pack_runner.py`
* **Poisson residual checks**: `01.Core_Engine/isut_104_qumond_poisson_residuals.py`
* **Curl-free field check**: `01.Core_Engine/isut_106_qumond_curl_free_check.py`
* **Convergence sweep (mesh/padding)**: `01.Core_Engine/isut_105_qumond_convergence_suite.py`
* **Energy-drift audit**: `01.Core_Engine/isut_103_qumond_energy_audit.py`

### External Field Effect (EFE)

* **EFE sensitivity scan (scalar proxy)**: `03.Advanced_Validation/isut_314_efe_sensitivity_scan.py`
* **EFE field-level proxy demo (in QUMOND source term)**: `01.Core_Engine/isut_109_qumond_efe_field_bc_demo.py`

### “Reviewer objections” pack

* **Synthetic-data discriminability**: `03.Advanced_Validation/isut_315_synthetic_dm_rejection_test.py`
* **Laplace evidence (prior-dependent)**: `03.Advanced_Validation/isut_316_bayes_evidence_laplace.py`
* **Residual whiteness test**: `03.Advanced_Validation/isut_317_residual_whiteness_test.py`
* **Solar-system sanity checks (high-g Newtonian limit + EFE quadrupole)**:
  `01.Core_Engine/isut_108_qumond_solar_system_sanity.py`  
  `01.Core_Engine/isut_110_qumond_solar_system_efe_quadrupole.py`  
  `03.Advanced_Validation/isut_313_solar_system_high_g_limit.py`

### Clock-rate interpretation & “dark matter reverse” diagnostics

* **All65 proxy prediction (population-level ν prediction)**: `03.Advanced_Validation/isut_302_clockrate_proxy_predict_all65.py`
* **Holdout + sensitivity sweeps (global-scale robustness)**: `03.Advanced_Validation/isut_303_clockrate_proxy_holdout_sensitivity.py`
* **Clock-rate identity (equivalence) check**: `01.Core_Engine/isut_126_verification_clock_gravity.py`
* **Toy rotation-curve demo**: `01.Core_Engine/isut_127_verification_rotationcurve_demo.py`
* **DM-reverse local extra-acceleration test**: `03.Advanced_Validation/isut_304_dm_reverse_local_test.py`
* **DM-reverse holdout/weighted variant**: `03.Advanced_Validation/isut_305_dm_reverse_local_test_holdout_weighted.py`

### Evidence-pack integrity

* **Artifact manifest (hashes + sizes + mtimes)**: `03.Advanced_Validation/isut_312_reviewer_artifact_manifest.py`

---

## License

See `LICENSE.txt`.

* **Software code**: MIT License
* **Research documents & data**: CC BY-NC 4.0 (attribution required; non-commercial)

---

## Contact / repository

Repository: `https://github.com/soyoungkim0327/ISUT-Emergent-Gravity`





