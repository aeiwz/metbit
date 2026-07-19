"""Validate metbit against the original COMETA article (ST002087).

The deposited data are processed bucket tables (no FID), plus the article's
quantified metabolite matrix (uM). We therefore validate metbit's analysis
half, in two ways:

  1. Feature fidelity -- for well-known metabolites with a known 1H ppm peak,
     correlate metbit's (PQN-normalised) bucket intensity at that ppm against
     the article's reported concentration across all 368 samples. High
     correlation = metbit's spectral features track the article's quantified
     values.
  2. Discrimination -- run metbit PCA + OPLS-DA + VIP on the buckets for the
     acute-vs-post-COVID contrast, and check the top-VIP ppm regions fall on
     known discriminant metabolites.

Usage: python Benchmark/ST002087/validate_st002087.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metbit import Normalise, opls_da, pca  # noqa: E402

DATA = REPO / "Benchmark" / "data" / "ST002087"
OUT = Path(__file__).resolve().parent / "result"
OUT.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# Known 1H NMR peak positions (ppm, TSP-referenced) for reference metabolites.
MARKER_PPM = {
    "Lacticacid": 1.33, "3-Hydroxybutyricacid": 1.20, "Alanine": 1.48,
    "Aceticacid": 1.92, "Pyruvicacid": 2.37, "Citricacid": 2.54,
    "Creatinine": 3.04, "Glucose": 5.23,
}


def load_buckets(name):
    df = pd.read_excel(DATA / name)
    df = df.set_index("sample_code")
    df.index = df.index.astype(str)
    centers = np.array([np.mean([float(x) for x in c.split("_")]) for c in df.columns])
    df.columns = centers  # ppm bin centres
    return df.sort_index(axis=1)  # ascending ppm


def load_reference():
    ref = pd.read_csv(DATA / "reference_metabolite_matrix.csv", index_col=0)
    ref.columns = [str(c) for c in ref.columns]
    return ref  # metabolites x samples


def nearest_bucket(buckets, ppm):
    j = int(np.argmin(np.abs(buckets.columns.to_numpy(float) - ppm)))
    return buckets.columns[j]


def feature_fidelity(buckets, ref):
    from scipy.stats import pearsonr, spearmanr
    samples = [s for s in buckets.index if s in ref.columns]
    B = buckets.loc[samples]
    rows = []
    for met, ppm in MARKER_PPM.items():
        if met not in ref.index:
            continue
        col = nearest_bucket(B, ppm)
        x = B[col].to_numpy(float)
        y = pd.to_numeric(ref.loc[met, samples], errors="coerce").to_numpy(float)
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 20:
            continue
        pr, _ = pearsonr(x[m], y[m])
        sr, _ = spearmanr(x[m], y[m])
        rows.append({"metabolite": met, "target_ppm": ppm,
                     "bucket_ppm": round(float(col), 3), "n": int(m.sum()),
                     "pearson_r": round(float(pr), 3), "spearman_r": round(float(sr), 3)})
    return pd.DataFrame(rows).sort_values("spearman_r", ascending=False)


def discrimination(buckets, factors):
    grp = factors["group"]
    mask = grp.isin(["COVID-19<21days", "Post-COVID-19"])
    samples = [s for s in buckets.index if s in grp.index and mask.get(s, False)]
    X = buckets.loc[samples]
    y = grp.loc[samples].map({"COVID-19<21days": "acute", "Post-COVID-19": "post"})

    Xn = Normalise(X).pqn_normalise(plot=False)
    Xn.columns = X.columns  # keep ppm labels for VIP

    pca_model = pca(Xn, label=y, n_components=5, scaling_method="pareto")
    pca_model.fit()

    model = opls_da(X=Xn, y=y, n_components=2, scaling_method="pareto",
                    kfold=7, estimator="opls", random_state=RANDOM_STATE)
    model.fit()
    model.vip_scores()
    vip = model.get_vip_scores().sort_values("VIP", ascending=False)
    return {
        "contrast": "acute (COVID-19<21days, n=%d) vs post (Post-COVID-19, n=%d)"
                    % (int((y == "acute").sum()), int((y == "post").sum())),
        "R2Y": round(float(model.R2y), 4),
        "Q2": round(float(model.q2), 4),
        "top15_VIP_ppm": [round(float(p), 3) for p in vip["Features"].head(15)],
    }, Xn, X, y


def main():
    ref = load_reference()
    factors = pd.read_csv(DATA / "sample_factors.csv", index_col=0)
    factors.index = factors.index.astype(str)
    factors["group"] = factors["factors"].str.extract(r"Group:([^|]+)")[0].str.strip()

    results = {}
    for exp, fname in [("cpmg", "cpmg_bucket_table.xlsx"),
                       ("noesy", "noesy_bucket_table.xlsx")]:
        buckets = load_buckets(fname)
        fid = feature_fidelity(buckets, ref)
        fid.to_csv(OUT / f"feature_fidelity_{exp}.csv", index=False)
        disc, *_ = discrimination(buckets, factors)
        results[exp] = {
            "feature_fidelity": fid.to_dict(orient="records"),
            "median_spearman": round(float(fid["spearman_r"].median()), 3),
            "discrimination": disc,
        }
        print(f"\n=== {exp.upper()} ===")
        print(fid.to_string(index=False))
        print("median |Spearman| across markers:", results[exp]["median_spearman"])
        print("OPLS-DA:", disc)

    (OUT / "validation_report.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT/'validation_report.json'}")


if __name__ == "__main__":
    main()
