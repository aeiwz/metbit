"""Validate metbit's raw-FID processing on MTBLS2052 (5/6-Nx rat CKD model).

ST002087 has no FID, so this dataset exercises the half ST002087 could not:
raw Bruker FID -> auto-phased, calibrated, aligned spectra. We process a clean
single-matrix / single-timepoint subset (kidney, week 6) and validate:

  1. Processing correctness -- spectra are chemically sane: TSP/DSS reference
     lands at 0 ppm after calibration, upright peaks, flat baseline.
  2. Biological separation -- PCA + OPLS-DA distinguish CKD (nephrectomy) from
     sham controls in the target organ, matching the original study's finding
     of a strong renal metabolic phenotype.

Usage: python Benchmark/MTBLS2052/validate_mtbls2052.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metbit import Normalise, opls_da, pca  # noqa: E402
from metbit import nmr_preprocessing  # noqa: E402

DATA = REPO / "Benchmark" / "data" / "MTBLS2052"
ARCHIVE = DATA / "5_6_Nx_rat_CKD_model_archive.zip"
OUT = Path(__file__).resolve().parent / "result"
OUT.mkdir(parents=True, exist_ok=True)
WEEK = "6"
MATRICES = ["kidney", "blood serum", "urine", "lung", "heart", "spleen", "liver"]
RANDOM_STATE = 42


def build_subset(matrix):
    work = DATA / ("proc_subset_" + matrix.replace(" ", "_") + "_wk6")
    m = pd.read_csv(DATA / "noesy_sample_map.csv", dtype=str)
    sub = m[(m["matrix"] == matrix) & (m["Factor Value[week]"] == WEEK)].copy()
    z = zipfile.ZipFile(ARCHIVE)
    work.mkdir(parents=True, exist_ok=True)
    labels = {}
    for _, r in sub.iterrows():
        sid = r["fid"]
        try:
            inner = zipfile.ZipFile(io.BytesIO(z.read(f"{sid}.zip")))
            if inner.testzip() is not None:
                continue
            if not (work / sid / "fid").exists():
                inner.extractall(work)
            labels[sid] = ("CKD" if r["Factor Value[nephrectomy]"] == "nephrectomy"
                           else "sham")
        except Exception:
            continue
    return work, pd.Series(labels, name="group")


def validate_matrix(matrix):
    work, labels = build_subset(matrix)
    if labels.nunique() < 2 or (labels == "CKD").sum() < 3 or (labels == "sham").sum() < 3:
        print(f"  [{matrix}] skipped (insufficient group sizes)")
        return None
    prep = nmr_preprocessing(str(work), calib_type="tsp", auto_phasing=True, align=True)
    X = prep.get_data(); ppm = prep.get_ppm()
    X.index = X.index.astype(str); labels.index = labels.index.astype(str)
    common = [s for s in X.index if s in labels.index]
    X, y = X.loc[common], labels.loc[common]

    win = (ppm > -0.2) & (ppm < 0.2)
    tsp_med = float(np.median([ppm[win][np.argmax(X.iloc[i].to_numpy(float)[win])]
                               for i in range(len(X))]))

    fig, ax = plt.subplots(figsize=(11, 3.2))
    for i in X.index:
        ax.plot(ppm, X.loc[i].to_numpy(float), lw=0.3)
    ax.set_xlim(10, -1); ax.axhline(0, color="k", lw=0.3)
    ax.set_title(f"MTBLS2052 {matrix} wk6 auto-phased spectra (n={len(X)}), TSP@{tsp_med:+.3f}")
    ax.set_xlabel("ppm")
    tag = matrix.replace(" ", "_")
    plt.tight_layout(); plt.savefig(OUT / f"spectra_qc_{tag}.png", dpi=105); plt.close(fig)

    Xn = Normalise(X).pqn_normalise(plot=False); Xn.columns = X.columns
    model = opls_da(X=Xn, y=y, n_components=2, scaling_method="pareto",
                    kfold=7, estimator="opls", random_state=RANDOM_STATE)
    model.fit(); model.vip_scores()
    vip = model.get_vip_scores().sort_values("VIP", ascending=False)
    return {
        "matrix": matrix, "n_samples": int(len(X)),
        "n_CKD": int((y == "CKD").sum()), "n_sham": int((y == "sham").sum()),
        "median_TSP_peak_ppm": round(tsp_med, 4),
        "TSP_within_0.02ppm": bool(abs(tsp_med) < 0.02),
        "R2Y": round(float(model.R2y), 4), "Q2": round(float(model.q2), 4),
        "top5_VIP_ppm": [round(float(p), 3) for p in vip["Features"].head(5)],
    }


def main():
    results = []
    for matrix in MATRICES:
        print(f"=== {matrix} wk{WEEK} ===")
        r = validate_matrix(matrix)
        if r:
            results.append(r)
            print(f"  n={r['n_samples']} ({r['n_CKD']}CKD/{r['n_sham']}sham) "
                  f"TSP={r['median_TSP_peak_ppm']:+.3f} R2Y={r['R2Y']} Q2={r['Q2']}")
    report = {"study": "MTBLS2052 5/6-Nx rat CKD, week 6, CKD vs sham",
              "contrast": "nephrectomy vs sham control", "per_matrix": results}
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
