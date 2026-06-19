#!/usr/bin/env python3
"""
Step 4 - Generate benchmark report figures and summary table.

Reads results/<ACCESSION>/ produced by 03_run_pipeline.py and produces:
    results/<ACCESSION>/
        fig_spectra_overlay.png      stacked spectral overlay (coloured by group)
        fig_pca.png                  PCA scores with 95% confidence ellipses
        fig_vip.png                  VIP score spectrum (ppm vs VIP)
        benchmark_summary.txt        plain-text metrics table for the manuscript

Usage:
    python 04_report.py [ACCESSION]
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

ACCESSION   = sys.argv[1] if len(sys.argv) > 1 else "MTBLS1"
RESULTS_DIR = Path(__file__).parent / "results" / ACCESSION

PALETTE = ["#1f4e79", "#c00000", "#375623", "#7030a0", "#833c00"]


def load_or_exit(path, label):
    if not path.exists():
        print(f"[error] {path} not found. Run 03_run_pipeline.py first.")
        sys.exit(1)
    return path


# ---------------------------------------------------------------------------
# load artefacts
# ---------------------------------------------------------------------------

load_or_exit(RESULTS_DIR / "spectral_matrix.csv", "spectral_matrix")
load_or_exit(RESULTS_DIR / "pca_scores.csv", "pca_scores")

print(f"\n=== Generating report for {ACCESSION} ===")

X       = pd.read_csv(RESULTS_DIR / "spectral_matrix.csv", index_col=0)
ppm     = X.columns.astype(float).to_numpy()
scores  = pd.read_csv(RESULTS_DIR / "pca_scores.csv", index_col=0)
groups  = scores["group"].unique()
cmap    = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(groups))}

with open(RESULTS_DIR / "opls_metrics.json") as fh:
    metrics = json.load(fh)

vip_path = RESULTS_DIR / "vip_scores.csv"
has_vip  = vip_path.exists()

# ---------------------------------------------------------------------------
# fig 1: spectral overlay
# ---------------------------------------------------------------------------

print("  Generating spectral overlay...")
fig, ax = plt.subplots(figsize=(9, 3.5))

for group in sorted(groups):
    idx   = scores[scores["group"] == group].index
    sub   = X.loc[X.index.isin(idx)]
    color = cmap[group]
    for i, row in sub.iterrows():
        ax.plot(ppm, row.values, color=color, alpha=0.25, linewidth=0.4)
    ax.plot(ppm, sub.mean().values, color=color, linewidth=1.2, label=group)

ax.set_xlabel("Chemical shift (ppm)", fontsize=10)
ax.set_ylabel("Intensity (a.u.)", fontsize=10)
ax.set_xlim(ppm.max(), ppm.min())   # NMR convention: high ppm on left
ax.set_title(f"{ACCESSION} - ¹H NMR spectral overlay", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig_spectra_overlay.png", dpi=200)
plt.close(fig)
print(f"  Saved: fig_spectra_overlay.png")


# ---------------------------------------------------------------------------
# fig 2: PCA scores
# ---------------------------------------------------------------------------

print("  Generating PCA scores plot...")
pc1_col = next((c for c in scores.columns if "PC1" in c or c == "0"), scores.columns[0])
pc2_col = next((c for c in scores.columns if "PC2" in c or c == "1"), scores.columns[1])

# try to extract explained variance from column names (e.g. "PC1 (34.5%)")
def extract_var(col):
    import re
    m = re.search(r"(\d+\.?\d*)\s*%", col)
    return float(m.group(1)) if m else None

var1 = extract_var(pc1_col)
var2 = extract_var(pc2_col)

fig, ax = plt.subplots(figsize=(5, 4.5))

for group in sorted(groups):
    sub = scores[scores["group"] == group]
    ax.scatter(sub[pc1_col], sub[pc2_col],
               color=cmap[group], s=40, alpha=0.8,
               label=group, edgecolors="white", linewidths=0.3)
    # 95% confidence ellipse
    try:
        from matplotlib.patches import Ellipse
        import matplotlib.transforms as transforms
        x, y = sub[pc1_col].values, sub[pc2_col].values
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell = Ellipse((0, 0),
                      width=np.sqrt(1 + pearson) * 2,
                      height=np.sqrt(1 - pearson) * 2,
                      facecolor=cmap[group], alpha=0.12,
                      edgecolor=cmap[group], linewidth=1.0)
        tf = (transforms.Affine2D()
              .rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * 2, np.sqrt(cov[1, 1]) * 2)
              .translate(np.mean(x), np.mean(y)))
        ell.set_transform(tf + ax.transData)
        ax.add_patch(ell)
    except Exception:
        pass

xlabel = f"PC1 ({var1:.1f}%)" if var1 else "PC1"
ylabel = f"PC2 ({var2:.1f}%)" if var2 else "PC2"
ax.set_xlabel(xlabel, fontsize=10)
ax.set_ylabel(ylabel, fontsize=10)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
ax.set_title(f"{ACCESSION} - PCA", fontsize=11)
ax.legend(frameon=False, fontsize=9)
ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "fig_pca.png", dpi=200)
plt.close(fig)
print(f"  Saved: fig_pca.png")


# ---------------------------------------------------------------------------
# fig 3: VIP spectrum (if available)
# ---------------------------------------------------------------------------

if has_vip:
    print("  Generating VIP spectrum...")
    vip_df = pd.read_csv(vip_path)
    # column may be named 'Features' (ppm axis) or 'ppm'
    ppm_col = "Features" if "Features" in vip_df.columns else "ppm"
    vip_df = vip_df.rename(columns={ppm_col: "ppm"})
    vip_df["ppm"] = pd.to_numeric(vip_df["ppm"], errors="coerce")
    vip_df = vip_df.dropna(subset=["ppm"]).sort_values("ppm")

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.fill_between(vip_df["ppm"], vip_df["VIP"],
                    where=(vip_df["VIP"] >= 1.0),
                    color="#c00000", alpha=0.7, label="VIP >= 1.0")
    ax.fill_between(vip_df["ppm"], vip_df["VIP"],
                    where=(vip_df["VIP"] < 1.0),
                    color="#a6a6a6", alpha=0.4, label="VIP < 1.0")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Chemical shift (ppm)", fontsize=10)
    ax.set_ylabel("VIP score", fontsize=10)
    ax.set_xlim(vip_df["ppm"].max(), vip_df["ppm"].min())
    ax.set_title(f"{ACCESSION} - VIP scores", fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig_vip.png", dpi=200)
    plt.close(fig)
    print(f"  Saved: fig_vip.png")


# ---------------------------------------------------------------------------
# plain-text summary for manuscript Table
# ---------------------------------------------------------------------------

print("  Writing benchmark_summary.txt...")

lines = [
    f"Benchmark summary - {ACCESSION}",
    "=" * 50,
    f"Samples processed      : {metrics.get('n_samples_total', 'n/a')}",
    f"PPM data points        : {metrics.get('ppm_points', 'n/a')}",
    f"Groups                 : {metrics.get('groups', 'n/a')}",
    "",
    "Preprocessing QC",
    f"  TSP shift deviation  : {metrics['calibration_mean_dev_ppm']*1000:.2f} +/- "
        f"{metrics.get('calibration_std_dev_ppm',0)*1000:.2f} m-ppm"
    if metrics.get("calibration_mean_dev_ppm") else
    "  TSP shift deviation  : n/a",
    f"  Mean spectral SNR    : {metrics['mean_snr']:.1f}"
    if metrics.get("mean_snr") else
    "  Mean spectral SNR    : n/a",
    "",
    "Multivariate performance (OPLS-DA)",
    f"  R2Y                  : {metrics.get('R2Y', 'n/a')}",
    f"  Q2                   : {metrics.get('Q2', 'n/a')}",
    f"  p-permutation (1000) : {metrics.get('p_permutation', 'n/a')}",
    f"  AUROC (7-fold CV)    : {metrics.get('AUROC_cv', 'n/a')}",
    "",
    "Runtime",
    f"  Preprocessing        : {metrics.get('preprocessing_time_s', 'n/a')} s",
    f"  PCA                  : {metrics.get('pca_time_s', 'n/a')} s",
    f"  OPLS-DA              : {metrics.get('opls_time_s', 'n/a')} s",
]

summary_path = RESULTS_DIR / "benchmark_summary.txt"
summary_path.write_text("\n".join(lines))
print(f"  Saved: benchmark_summary.txt")

print(f"\n{'='*50}")
print(f"All outputs in: {RESULTS_DIR}")
print("  fig_spectra_overlay.png")
print("  fig_pca.png")
if has_vip:
    print("  fig_vip.png")
print("  opls_metrics.json")
print("  benchmark_summary.txt")
