#!/usr/bin/env python3
"""
Step 3 - Run the full metbit pipeline on downloaded MetaboLights data.

Reads:
    data/<ACCESSION>/FILES/          Bruker FID directories
    data/<ACCESSION>/samples_parsed.csv   sample metadata with group labels

Outputs:
    results/<ACCESSION>/
        spectral_matrix.csv          samples x ppm (float32)
        ppm.npy                      ppm axis array
        pca_scores.csv               PC1/PC2 scores + group label
        opls_metrics.json            R2Y, Q2, p_perm, AUROC
        vip_scores.csv               ppm -> VIP score
        preprocessing_qc.csv         per-sample shift deviation + SNR

Usage:
    python 03_run_pipeline.py [ACCESSION] [--factor "Factor Value[Disease]"]

Options:
    --factor   Column name in samples_parsed.csv to use as group label.
               Default: first column whose name contains "Factor Value".
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# parse args
# ---------------------------------------------------------------------------

ACCESSION    = "MTBLS1"
FACTOR_COL   = None

i = 1
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg == "--factor" and i + 1 < len(sys.argv):
        FACTOR_COL = sys.argv[i + 1]
        i += 2
    elif not arg.startswith("--"):
        ACCESSION = arg
        i += 1
    else:
        i += 1

DATA_DIR    = Path(__file__).parent / "data" / ACCESSION
FID_DIR     = DATA_DIR / "FILES"
SAMPLE_CSV  = DATA_DIR / "samples_parsed.csv"
RESULTS_DIR = Path(__file__).parent / "results" / ACCESSION
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== metbit pipeline benchmark: {ACCESSION} ===")

# ---------------------------------------------------------------------------
# 1. load group labels from ISA-Tab
# ---------------------------------------------------------------------------

if not SAMPLE_CSV.exists():
    print(f"[error] {SAMPLE_CSV} not found. Run 01_fetch_study.py first.")
    sys.exit(1)

meta_df = pd.read_csv(SAMPLE_CSV)
print(f"  Metadata: {meta_df.shape[0]} rows, {meta_df.shape[1]} columns")

_SKIP_TERMS = {"gender", "sex", "age", "race", "ethnicity"}

if FACTOR_COL is None:
    factor_cols = [c for c in meta_df.columns if "Factor Value" in c]
    # prefer disease/condition columns; deprioritise demographic confounders
    preferred = [c for c in factor_cols
                 if not any(t in c.lower() for t in _SKIP_TERMS)]
    factor_cols = preferred if preferred else factor_cols
    if not factor_cols:
        # fallback to any column with few unique values
        factor_cols = [c for c in meta_df.columns
                       if meta_df[c].nunique() <= 10 and meta_df[c].dtype == object
                       and "Source" not in c and "Sample" not in c]
    if not factor_cols:
        print("[error] No Factor Value columns found in sample metadata.")
        print("        Columns:", list(meta_df.columns))
        sys.exit(1)
    FACTOR_COL = factor_cols[0]

print(f"  Group factor: {FACTOR_COL}")
groups = meta_df[FACTOR_COL].value_counts()
print(f"  Group counts:\n{groups.to_string()}")

# build sample -> label mapping
# ISA-Tab uses "Sample Name" as the sample identifier
sample_id_col = next((c for c in meta_df.columns
                      if "Sample Name" in c or c == "Sample Name"), None)
if sample_id_col is None:
    sample_id_col = meta_df.columns[0]

label_map = dict(zip(meta_df[sample_id_col].astype(str), meta_df[FACTOR_COL].astype(str)))


# ---------------------------------------------------------------------------
# 2. run nmr_preprocessing on the FID directory
# ---------------------------------------------------------------------------

print(f"\n  Running nmr_preprocessing on {FID_DIR} ...")

if not FID_DIR.exists() or not any(FID_DIR.iterdir()):
    print(f"[error] No FID directories found in {FID_DIR}.")
    print("        Run 02_download_fids.py first.")
    sys.exit(1)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from metbit import nmr_preprocessing, Normalise, pca, opls_da

import time
t0 = time.time()

nmr = nmr_preprocessing(
    str(FID_DIR),
    bin_size=0.0004,
    auto_phasing=True,
    fn_="acme",
    baseline_correction=True,
    baseline_type="corrector",
    calibration=True,
    calib_type="tsp",
    align=True,
    align_reference="median",
    align_max_shift_ppm=0.02,
    align_top_n=20,
)

preproc_time = time.time() - t0
print(f"  Preprocessing done in {preproc_time:.1f}s")

X = nmr.get_data()
ppm_axis = X.columns.to_numpy(dtype=float)

print(f"  Spectral matrix: {X.shape[0]} samples x {X.shape[1]} ppm points")
print(f"  PPM range: {ppm_axis.min():.3f} - {ppm_axis.max():.3f}")

X.astype("float32").to_csv(RESULTS_DIR / "spectral_matrix.csv")
np.save(RESULTS_DIR / "ppm.npy", ppm_axis)
print(f"  Saved: spectral_matrix.csv, ppm.npy")


# ---------------------------------------------------------------------------
# 3. preprocessing QC - calibration accuracy
# ---------------------------------------------------------------------------

print("\n  Computing preprocessing QC metrics...")

# TSP reference peak should be at 0.000 ppm
# Find the closest index to 0.000 ppm and measure actual peak position
tsp_region = (ppm_axis >= -0.05) & (ppm_axis <= 0.05)
if tsp_region.sum() > 0:
    tsp_intensities = X.iloc[:, tsp_region]
    tsp_peak_ppm_per_sample = [
        ppm_axis[tsp_region][np.argmax(tsp_intensities.iloc[i].values)]
        for i in range(len(X))
    ]
    shift_dev = np.array(tsp_peak_ppm_per_sample)
    mean_dev  = float(np.mean(np.abs(shift_dev)))
    std_dev   = float(np.std(np.abs(shift_dev)))
    print(f"  TSP calibration deviation: {mean_dev*1000:.2f} +/- {std_dev*1000:.2f} m-ppm (mean +/- SD)")
else:
    mean_dev, std_dev = None, None
    print("  [warn] TSP region not found in ppm range.")

# SNR: signal region 0.5-9.0 ppm vs noise region 9.5-10.0 ppm
sig_region   = (ppm_axis >= 0.5)  & (ppm_axis <= 9.0)
noise_region = (ppm_axis >= 9.5)  & (ppm_axis <= 10.0)
if sig_region.sum() > 0 and noise_region.sum() > 0:
    sig_rms   = X.iloc[:, sig_region].values.std(axis=1)
    noise_rms = X.iloc[:, noise_region].values.std(axis=1)
    snr_per_sample = sig_rms / (noise_rms + 1e-12)
    mean_snr = float(np.mean(snr_per_sample))
    print(f"  Mean spectral SNR: {mean_snr:.1f}")
else:
    snr_per_sample = np.full(len(X), np.nan)
    mean_snr = None

qc_df = pd.DataFrame({
    "sample": X.index,
    "tsp_shift_ppm": shift_dev if tsp_region.sum() > 0 else np.nan,
    "snr": snr_per_sample,
})
qc_df.to_csv(RESULTS_DIR / "preprocessing_qc.csv", index=False)
print(f"  Saved: preprocessing_qc.csv")


# ---------------------------------------------------------------------------
# 4. align sample labels to spectral matrix index
# ---------------------------------------------------------------------------

sample_names = X.index.astype(str).tolist()
y_labels = []
unmatched = 0
for s in sample_names:
    label = label_map.get(s)
    if label is None:
        # try partial match (some studies suffix the sample name)
        matches = [v for k, v in label_map.items() if s in k or k in s]
        label = matches[0] if matches else "unknown"
        if label == "unknown":
            unmatched += 1
    y_labels.append(label)

print(f"\n  Label assignment: {len(y_labels) - unmatched} matched, {unmatched} unmatched")

y_series = pd.Series(y_labels, index=X.index, name="group")
unique_groups = y_series.unique()
print(f"  Groups in spectral matrix: {list(unique_groups)}")


# ---------------------------------------------------------------------------
# 5. PQN normalisation
# ---------------------------------------------------------------------------

print("\n  Normalising (PQN)...")
norm = Normalise(X, compute_missing=True)
X_norm = norm.pqn_normalise(plot=False)
print(f"  Normalised matrix: {X_norm.shape}")


# ---------------------------------------------------------------------------
# 6. PCA
# ---------------------------------------------------------------------------

print("\n  Running PCA...")
t_pca = time.time()

pca_model = pca(X_norm, label=y_labels, n_components=5, scaling_method="pareto")
pca_model.fit()
scores = pca_model.get_scores()
explained_df = pca_model.get_explained_variance()

pca_time = time.time() - t_pca
print(f"  PCA done in {pca_time:.2f}s")

# get_explained_variance returns DataFrame with columns PC / Explained variance / Cumulative
# Row 0 is a header row with empty PC - skip it
ev_vals = explained_df[explained_df["PC"].str.startswith("PC", na=False)]["Explained variance"].astype(float).values
pc1_var = float(ev_vals[0]) * 100 if len(ev_vals) > 0 else 0.0
pc2_var = float(ev_vals[1]) * 100 if len(ev_vals) > 1 else 0.0
print(f"  Variance explained: PC1={pc1_var:.1f}%, PC2={pc2_var:.1f}%")

scores_out = scores.copy()
scores_out["group"] = y_labels
scores_out.to_csv(RESULTS_DIR / "pca_scores.csv")
print(f"  Saved: pca_scores.csv")


# ---------------------------------------------------------------------------
# 7. OPLS-DA (binary groups only)
# ---------------------------------------------------------------------------

valid_for_oplsda = [g for g in unique_groups if g != "unknown"]
if len(valid_for_oplsda) == 2:
    g0, g1 = valid_for_oplsda
    # Re-build y_series with the same index as X_norm to avoid alignment issues
    y_series = pd.Series(y_labels, index=X_norm.index, name="group")
    mask = y_series.isin([g0, g1]).values   # numpy bool array - no index alignment needed
    X_bin  = X_norm.iloc[mask]
    y_bin  = y_series.iloc[mask]

    print(f"\n  Running OPLS-DA ({g0} vs {g1}) ...")
    print(f"    n={mask.sum()}: {g0}={int((y_bin==g0).sum())}, {g1}={int((y_bin==g1).sum())}")

    t_opls = time.time()
    model = opls_da(
        X=X_bin,
        y=y_bin,
        n_components=2,
        scaling_method="pareto",
        kfold=7,
        estimator="opls",
        random_state=42,
    )
    model.fit()
    opls_time = time.time() - t_opls
    print(f"  OPLS-DA fit done in {opls_time:.2f}s")

    r2y = float(model.R2y)
    q2  = float(model.q2)

    # permutation test (500 permutations for speed; use 1000 for publication)
    print("  Running permutation test (500 permutations)...")
    t_perm = time.time()
    try:
        model.permutation_test(n_permutations=500, cv=7, n_jobs=-1, verbose=0)
        perm_scores = model.get_permutation_scores()
        # p-value: fraction of permuted Q2 >= observed Q2
        p_perm = float(np.mean(perm_scores >= q2))
        print(f"  Permutation done in {time.time()-t_perm:.1f}s, p={p_perm:.4f}")
    except Exception as exc:
        print(f"  [warn] Permutation test failed: {exc}")
        p_perm = None

    # AUROC via CV scores (take max since OPLS-DA sign direction is arbitrary)
    try:
        from sklearn.metrics import roc_auc_score
        opls_scores = model.get_oplsda_scores()
        t_pred = opls_scores["t_pred"].values
        y_bin_num = (y_bin.values == g1).astype(int)
        auroc_raw = float(roc_auc_score(y_bin_num, t_pred))
        auroc = max(auroc_raw, 1.0 - auroc_raw)   # direction is arbitrary in OPLS-DA
        print(f"  AUROC (t_pred vs group): {auroc:.3f}")
    except Exception as exc:
        print(f"  [warn] AUROC calculation failed: {exc}")
        auroc = None

    print(f"  R2Y={r2y:.3f}, Q2={q2:.3f}, p_perm={p_perm}, AUROC={auroc}")

    metrics = {
        "accession": ACCESSION,
        "groups": [g0, g1],
        "n_samples": int(mask.sum()),
        "R2Y": r2y,
        "Q2": q2,
        "p_permutation": p_perm,
        "AUROC_cv": auroc,
        "preprocessing_time_s": round(preproc_time, 2),
        "pca_time_s": round(pca_time, 3),
        "opls_time_s": round(opls_time, 2),
        "ppm_points": int(X.shape[1]),
        "n_samples_total": int(X.shape[0]),
        "calibration_mean_dev_ppm": mean_dev,
        "calibration_std_dev_ppm": std_dev,
        "mean_snr": mean_snr,
    }

    with open(RESULTS_DIR / "opls_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(f"  Saved: opls_metrics.json")

    # VIP scores
    try:
        model.vip_scores()
        vip_df = model.get_vip_scores(filter_=False)
        vip_df.to_csv(RESULTS_DIR / "vip_scores.csv", index=False)
        print(f"  Saved: vip_scores.csv ({len(vip_df)} entries)")
    except Exception as exc:
        print(f"  [warn] VIP extraction failed: {exc}")

else:
    print(f"\n  [info] {len(valid_for_oplsda)} unique groups found - skipping OPLS-DA (requires exactly 2).")
    print(f"  Groups: {valid_for_oplsda}")
    metrics = {
        "accession": ACCESSION,
        "groups": valid_for_oplsda,
        "note": "OPLS-DA requires binary groups",
        "ppm_points": int(X.shape[1]),
        "n_samples_total": int(X.shape[0]),
        "calibration_mean_dev_ppm": mean_dev,
        "mean_snr": mean_snr,
    }
    with open(RESULTS_DIR / "opls_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)


# ---------------------------------------------------------------------------
# 8. summary report
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
print(f"Benchmark complete: {ACCESSION}")
print(f"  Samples processed : {X.shape[0]}")
print(f"  PPM points        : {X.shape[1]}")
if mean_dev is not None:
    print(f"  TSP deviation     : {mean_dev*1000:.2f} +/- {std_dev*1000:.2f} m-ppm")
if mean_snr is not None:
    print(f"  Mean SNR          : {mean_snr:.1f}")
print(f"  Preprocessing time: {preproc_time:.1f}s")
print(f"  Results saved in  : {RESULTS_DIR}")
print(f"{'='*60}")
print(f"\nNext step: python 04_report.py {ACCESSION}")
