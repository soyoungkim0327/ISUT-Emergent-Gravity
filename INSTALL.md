# ISUT Reproducibility: installation & running (Windows / macOS / Linux)

This repository is designed to be runnable on a standard laptop and to export **auditable artifacts**
(**CSV/PNG/JSON + run_metadata**) via the numbered `isut_###_*.py` scripts.

## 0) Recommended setup (Conda)

If you already have Python 3.9+ working, you can skip to **1)**.

```bash
conda create -n isut python=3.9 -y
conda activate isut
```

### VS Code / Jupyter kernel (optional but recommended)
```bash
pip install ipykernel
python -m ipykernel install --user --name isut --display-name "Python (isut)"
```

## 1) Install required dependencies

From the repository root:

```bash
pip install -r requirements.txt
```

Notes:
- `seaborn` is used for a few “paper style” plots. If you prefer a minimal install, you can still run most scripts without it (they fall back to matplotlib), but figures may look different.
- `PySR` (symbolic regression) is **optional**. If it is not installed, the AI scripts will run in a “mock mode” and still export the non-AI artifacts.

## 2) Quick sanity checks (fast)

### QUMOND conservative field-solve demo (two Poisson solves)
```bash
python 01.Core_Engine/isut_100_qumond_fieldsolve_demo.py --N 96 --L 200 --pad 2
```

### Engine stability audit (energy drift + curl-free + residual checks)
```bash
python 01.Core_Engine/isut_120_engine_audit_qumond.py --N 96 --L 200 --pad 2 --steps 2000 --dt 0.03
```

Outputs are written **relative to the script path**, into a folder with the same stem as the script:
`<script_dir>/<script_name>/...` (CSV/PNG/JSON).

## 3) Population validation (All65 / Golden12)

### a0 constancy validator
```bash
python 03.Advanced_Validation/isut_300_valid_a0_constancy.py --subset All65
```

### All65 proxy RAR extraction & plots
```bash
python 03.Advanced_Validation/isut_302_clockrate_proxy_predict_all65.py
```

If you see a *validator not found* error, confirm that the file exists:
`03.Advanced_Validation/isut_300_valid_a0_constancy.py`.

## 4) Optional: GPU acceleration

The current conservative QUMOND solver uses NumPy FFTs.
If you want GPU acceleration, consider CuPy (drop-in NumPy-like arrays) or a dedicated Poisson solver.
This is optional and **not required** for reproducing the paper’s shipped artifacts.

## 5) Troubleshooting

- **Missing `requests`**:
  ```bash
  pip install requests
  ```
- **Missing `seaborn`**:
  ```bash
  pip install seaborn
  ```
- **PySR not installed**: scripts will print a warning and continue in mock mode.

