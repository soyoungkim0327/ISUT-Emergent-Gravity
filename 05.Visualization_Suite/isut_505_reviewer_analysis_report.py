# -*- coding: utf-8 -*-
"""isut_505_reviewer_analysis_report.py

Reviewer-style diagnostic report that summarizes per-galaxy fit deltas and
residual-bias shifts between two proxy pipelines.

This script is intentionally lightweight and *data-driven*: it reads the
CSV exported by `isut_504_fig_isut_vis_comparison.py` and produces:

- A per-galaxy Δχ²% bar plot (new vs old)
- A residual-mean shift scatter (new vs old)
- A short CSV summary (counts, quantiles, golden subset breakdown)

Outputs are written to a script-local folder:

    05.Visualization_Suite/isut_505_reviewer_analysis_report/
        data/
        figures/
        logs/

This file is part of the Open Evidence Pack.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    raise SystemExit("pandas is required for this script. Install pandas and retry.") from e

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isut_000_common import ScriptPaths, find_repo_root, write_run_metadata, sha256_file


def _load_inputs(repo_root: Path) -> Tuple[pd.DataFrame, Path]:
    """Locate and load the comparison CSV produced by isut_504."""
    csv_path = repo_root / "05.Visualization_Suite" / "isut_504_fig_isut_vis_comparison" / "Comparison" / "data" / "Ultimate_Comparison_Source_Data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            "Missing comparison CSV. Run `05.Visualization_Suite/isut_504_fig_isut_vis_comparison.py` first to generate:\n"
            f"  {csv_path}"
        )
    df = pd.read_csv(csv_path)
    return df, csv_path


def summarize(df: pd.DataFrame) -> Dict[str, float]:
    """Return key scalar metrics."""
    # Defensive casting
    for col in ["Chi2_Old", "Chi2_New", "Res_Old_Mean", "Res_New_Mean"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Chi2_Old", "Chi2_New"]).copy()
    df["Chi2_Change_Pct"] = (df["Chi2_New"] - df["Chi2_Old"]) / df["Chi2_Old"] * 100.0

    # Improvement = χ² decreased
    improved = (df["Chi2_Change_Pct"] < 0).sum()
    worsened = (df["Chi2_Change_Pct"] > 0).sum()

    metrics: Dict[str, float] = {
        "n": float(len(df)),
        "improved_n": float(improved),
        "worsened_n": float(worsened),
        "improved_frac": float(improved) / float(len(df)) if len(df) else float("nan"),
        "median_chi2_change_pct": float(np.nanmedian(df["Chi2_Change_Pct"].values)) if len(df) else float("nan"),
        "mean_chi2_change_pct": float(np.nanmean(df["Chi2_Change_Pct"].values)) if len(df) else float("nan"),
        "p10_chi2_change_pct": float(np.nanpercentile(df["Chi2_Change_Pct"].values, 10)) if len(df) else float("nan"),
        "p90_chi2_change_pct": float(np.nanpercentile(df["Chi2_Change_Pct"].values, 90)) if len(df) else float("nan"),
    }

    if "Is_Golden" in df.columns:
        gold = df[df["Is_Golden"] == True].copy()  # noqa: E712
        if len(gold):
            metrics.update(
                {
                    "gold_n": float(len(gold)),
                    "gold_improved_frac": float((gold["Chi2_Change_Pct"] < 0).sum()) / float(len(gold)),
                    "gold_mean_chi2_change_pct": float(np.nanmean(gold["Chi2_Change_Pct"].values)),
                    "gold_median_chi2_change_pct": float(np.nanmedian(gold["Chi2_Change_Pct"].values)),
                }
            )
    return metrics


def plot_report(df: pd.DataFrame, out_png: Path, *, title: str = "Reviewer Diagnostic Report") -> None:
    df = df.copy()

    # Basic guards
    if "Galaxy" not in df.columns:
        df["Galaxy"] = [f"gal_{i}" for i in range(len(df))]

    df["Chi2_Old"] = pd.to_numeric(df.get("Chi2_Old"), errors="coerce")
    df["Chi2_New"] = pd.to_numeric(df.get("Chi2_New"), errors="coerce")
    df["Res_Old_Mean"] = pd.to_numeric(df.get("Res_Old_Mean"), errors="coerce")
    df["Res_New_Mean"] = pd.to_numeric(df.get("Res_New_Mean"), errors="coerce")

    df = df.dropna(subset=["Chi2_Old", "Chi2_New"]).copy()
    df["Chi2_Change_Pct"] = (df["Chi2_New"] - df["Chi2_Old"]) / df["Chi2_Old"] * 100.0

    # Sort for readability
    df = df.sort_values("Chi2_Change_Pct", ascending=True).reset_index(drop=True)

    # Create figure
    fig = plt.figure(figsize=(12, 8), dpi=150)

    # Panel A: Δχ²% bar
    ax1 = fig.add_subplot(2, 2, 1)
    y = np.arange(len(df))
    ax1.barh(y, df["Chi2_Change_Pct"].values)
    ax1.axvline(0.0)
    ax1.set_yticks(y)
    ax1.set_yticklabels(df["Galaxy"].values, fontsize=6)
    ax1.set_xlabel("Δχ²% = (χ²_new − χ²_old) / χ²_old × 100")
    ax1.set_title("Per-galaxy χ² change")

    # Panel B: residual mean shift
    ax2 = fig.add_subplot(2, 2, 2)
    if df["Res_Old_Mean"].notna().any() and df["Res_New_Mean"].notna().any():
        ax2.scatter(df["Res_Old_Mean"].values, df["Res_New_Mean"].values, s=12)
        lim = np.nanmax(np.abs(np.concatenate([df["Res_Old_Mean"].values, df["Res_New_Mean"].values])))
        lim = float(lim) if np.isfinite(lim) else 1.0
        ax2.plot([-lim, lim], [-lim, lim], linestyle="--")
        ax2.set_xlim(-lim, lim)
        ax2.set_ylim(-lim, lim)
        ax2.set_xlabel("mean residual (old)")
        ax2.set_ylabel("mean residual (new)")
    else:
        ax2.text(0.5, 0.5, "Residual mean columns missing", ha="center", va="center")
    ax2.set_title("Residual-bias shift")

    # Panel C: histogram of Δχ²%
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(df["Chi2_Change_Pct"].values, bins=20)
    ax3.axvline(0.0)
    ax3.set_xlabel("Δχ²%")
    ax3.set_ylabel("count")
    ax3.set_title("Distribution of χ² deltas")

    # Panel D: small text summary
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    m = summarize(df)
    lines = [
        f"N = {int(m.get('n', 0))}",
        f"Improved (Δχ²%<0): {int(m.get('improved_n', 0))} ({m.get('improved_frac', float('nan')):.2%})",
        f"Worsened (Δχ²%>0): {int(m.get('worsened_n', 0))}",
        f"Mean Δχ²%: {m.get('mean_chi2_change_pct', float('nan')):.2f}",
        f"Median Δχ²%: {m.get('median_chi2_change_pct', float('nan')):.2f}",
        f"P10/P90 Δχ²%: {m.get('p10_chi2_change_pct', float('nan')):.2f} / {m.get('p90_chi2_change_pct', float('nan')):.2f}",
    ]
    if "gold_n" in m:
        lines += [
            "",
            f"Golden subset N = {int(m.get('gold_n', 0))}",
            f"Golden improved frac: {m.get('gold_improved_frac', float('nan')):.2%}",
            f"Golden mean Δχ²%: {m.get('gold_mean_chi2_change_pct', float('nan')):.2f}",
        ]
    ax4.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=10)

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a reviewer-style diagnostic report from Ultimate_Comparison_Source_Data.csv")
    ap.add_argument("--title", default="Reviewer Diagnostic Report (proxy comparison)", help="Figure title")
    args = ap.parse_args()

    paths = ScriptPaths.for_script(__file__)
    repo_root = find_repo_root(Path(__file__).resolve())

    df, csv_path = _load_inputs(repo_root)

    # Save a cleaned copy + summary
    df_out = df.copy()
    df_out["Chi2_Change_Pct"] = (df_out["Chi2_New"] - df_out["Chi2_Old"]) / df_out["Chi2_Old"] * 100.0
    out_csv = paths.data_dir / "reviewer_proxy_comparison_table.csv"
    df_out.to_csv(out_csv, index=False)

    metrics = summarize(df)
    out_metrics = paths.data_dir / "reviewer_proxy_comparison_summary.json"
    out_metrics.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    # Plot
    out_png = paths.fig_dir / "reviewer_analysis_report.png"
    plot_report(df, out_png, title=args.title)

    # Metadata
    write_run_metadata(
        paths.out_root,
        args={"title": args.title},
        notes={
            "input_csv": str(csv_path),
            "input_csv_sha256": sha256_file(csv_path),
            "outputs": {
                "table_csv": str(out_csv),
                "summary_json": str(out_metrics),
                "figure_png": str(out_png),
            },
        },
    )

    print(f"[OK] Wrote:\n- {out_csv}\n- {out_metrics}\n- {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
