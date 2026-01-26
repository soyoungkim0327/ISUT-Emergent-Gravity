# QUMOND quickstart (conservative completion)

This repository includes a conservative (potential-based) QUMOND kernel where
accelerations are computed as `a = -∇Φ` from a solved scalar potential.

## 1) Core demo (rotation curve + orbit energy audit)

From repo root:

```bash
python 01.Core_Engine/isuit_121_engine_core_qumond.py --headless
```

Outputs:

- `01.Core_Engine/isuit_121_engine_core_qumond/runs/core_<timestamp>/data/*.csv`
- `01.Core_Engine/isuit_121_engine_core_qumond/runs/core_<timestamp>/figures/*.png`

## 2) Audit suite (smoke / fiducial / sensitivity)

```bash
python 01.Core_Engine/isuit_123_engine_audit_qumond.py --mode all --headless
```

Outputs:

- `01.Core_Engine/isuit_123_engine_audit_qumond/runs/audit_qumond_<timestamp>/{data,figures,logs}`

## 3) Solar-system sanity check (high-g Newtonian limit)

This script is a reviewer-facing regime check: in the Solar System, where
accelerations are far above a0, the adopted response nu(y) should reduce to
nu -> 1 (i.e., Newtonian limit).

```bash
python 01.Core_Engine/isuit_108_qumond_solar_system_sanity.py --nu nu_standard
```

Outputs:

- `01.Core_Engine/isuit_108_qumond_solar_system_sanity/data/*.csv`
- `01.Core_Engine/isuit_108_qumond_solar_system_sanity/figures/*.png`

Tip: the one-command runner also includes this check:

```bash
python 01.Core_Engine/isuit_111_qumond_reviewer_pack_runner.py --fast
```


## 4) Solar-system EFE quadrupole / perihelion scale (order-of-magnitude)

```bash
python 01.Core_Engine/isuit_110_qumond_solar_system_efe_quadrupole.py --nu nu_beta --beta 12 --eta 1.6
```

Outputs:

- `01.Core_Engine/isuit_110_qumond_solar_system_efe_quadrupole/data/*.csv`
- `01.Core_Engine/isuit_110_qumond_solar_system_efe_quadrupole/figures/*.png`

## Notes

- The solver uses FFTs (periodic boundaries). Use `--pad 2` or `--pad 3` to reduce
  periodic-image artifacts by zero-padding.
- This demo uses an analytic exponential disk density on a 3D mesh.
## Additional reviewer-defense scripts (Advanced Validation)

These scripts live under `03.Advanced_Validation/` and follow the same convention:
they create a sibling output folder next to the `.py` script, with `data/`, `figures/`,
and `logs/` subfolders.

### External-Field Effect (EFE) sensitivity (scalar proxy)

```bash
python 03.Advanced_Validation/isuit_314_efe_sensitivity_scan.py --galaxy NGC3198 --nu simple --gext_a0 0 0.3 1.0 3.0
```

Outputs:

- `03.Advanced_Validation/isuit_314_efe_sensitivity_scan/data/*.csv`
- `03.Advanced_Validation/isuit_314_efe_sensitivity_scan/figures/*.png`

### EFE field-level proxy (constant external field in QUMOND source term)

```bash
python 01.Core_Engine/isuit_109_qumond_efe_field_bc_demo.py --gext 0.02*a0@x
```

Outputs:

- `01.Core_Engine/isuit_109_qumond_efe_field_bc_demo/data/*.csv`
- `01.Core_Engine/isuit_109_qumond_efe_field_bc_demo/figures/*.png`


### Synthetic-data discriminability (DM-generated vs ISUT-generated)

```bash
python 03.Advanced_Validation/isuit_315_synthetic_dm_rejection_test.py --subset Golden12 --n_rep 25 --seed 123
```

Outputs:

- `03.Advanced_Validation/isuit_315_synthetic_dm_rejection_test/data/*.csv`
- `03.Advanced_Validation/isuit_315_synthetic_dm_rejection_test/figures/*.png`

### Laplace-approximate evidence / Bayes factors (prior-dependent)

```bash
python 03.Advanced_Validation/isuit_316_bayes_evidence_laplace.py --subset Golden12
```

Outputs:

- `03.Advanced_Validation/isuit_316_bayes_evidence_laplace/data/evidence_summary.csv`
- `03.Advanced_Validation/isuit_316_bayes_evidence_laplace/logs/run_metadata.json`

### Residual whiteness / structure test (All65)

```bash
python 03.Advanced_Validation/isuit_317_residual_whiteness_test.py --subset All65
```

Outputs:

- `03.Advanced_Validation/isuit_317_residual_whiteness_test/data/*.csv`
- `03.Advanced_Validation/isuit_317_residual_whiteness_test/figures/*.png`
