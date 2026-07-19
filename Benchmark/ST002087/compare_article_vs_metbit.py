"""Directly compare the COMETA article's result with metbit's result (ST002087).

Article result: the deposited quantified concentration matrix IS the article's
result. For each metabolite we compute the acute-vs-post-COVID difference
(log2 fold-change + Mann-Whitney p, BH-corrected) from those concentrations.

metbit result: run metbit's own pipeline (PQN -> OPLS-DA) on the spectra
(bucket tables) and, for each metabolite's known 1H ppm, take metbit's
class-correlation (does the peak go up or down in acute) and its VIP.

Comparison: do metbit and the article agree on (a) the DIRECTION of change and
(b) WHICH metabolites are the strongest discriminators?

Usage: python Benchmark/ST002087/compare_article_vs_metbit.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, pointbiserialr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from metbit import Normalise, opls_da  # noqa: E402

DATA = REPO / "Benchmark" / "data" / "ST002087"
OUT = Path(__file__).resolve().parent / "result"
RANDOM_STATE = 42

# Confident 1H ppm for well-resolved plasma metabolites in the article matrix.
MET_PPM = {
    "3-Hydroxybutyricacid": 1.20, "Aceticacid": 1.92, "Acetone": 2.23,
    "Acetoaceticacid": 2.28, "Alanine": 1.48, "Citricacid": 2.54,
    "Creatine": 3.03, "Creatinine": 3.04, "Formicacid": 8.46, "Glucose": 5.23,
    "Glutamine": 2.45, "Glycine": 3.56, "Histidine": 7.05, "Isoleucine": 1.01,
    "Lacticacid": 1.33, "Leucine": 0.96, "Methionine": 2.14, "Phenylalanine": 7.42,
    "Pyruvicacid": 2.37, "Succinicacid": 2.41, "Tyrosine": 6.90, "Valine": 1.04,
    "Glycoproteins": 2.04,
}
BH_ALPHA = 0.05


def bh(pvals):
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        r = n - rank
        prev = min(prev, p[i] * n / r)
        adj[i] = prev
    return adj


def main():
    ref = pd.read_csv(DATA / "reference_metabolite_matrix.csv", index_col=0)
    ref.columns = [str(c) for c in ref.columns]
    fac = pd.read_csv(DATA / "sample_factors.csv", index_col=0)
    fac.index = fac.index.astype(str)
    grp = fac["factors"].str.extract(r"Group:([^|]+)")[0].str.strip()
    acute = grp[grp == "COVID-19<21days"].index
    post = grp[grp == "Post-COVID-19"].index

    # ---- ARTICLE result: per-metabolite acute vs post from concentrations ----
    rows = []
    for met in MET_PPM:
        if met not in ref.index:
            continue
        a = pd.to_numeric(ref.loc[met, ref.columns.intersection(acute)], errors="coerce").dropna()
        p_ = pd.to_numeric(ref.loc[met, ref.columns.intersection(post)], errors="coerce").dropna()
        if len(a) < 10 or len(p_) < 10:
            continue
        try:
            _, pv = mannwhitneyu(a, p_, alternative="two-sided")
        except ValueError:
            pv = 1.0
        log2fc = float(np.log2((a.median() + 1e-9) / (p_.median() + 1e-9)))
        rows.append({"metabolite": met, "ppm": MET_PPM[met],
                     "article_log2FC_acute_vs_post": round(log2fc, 3),
                     "article_p": pv, "article_dir": "up" if log2fc > 0 else "down"})
    df = pd.DataFrame(rows)
    df["article_p_BH"] = bh(df["article_p"].values)
    df["article_significant"] = df["article_p_BH"] < BH_ALPHA

    # ---- metbit result: OPLS-DA on spectra, per-metabolite ppm ----
    buck = pd.read_excel(DATA / "cpmg_bucket_table.xlsx").set_index("sample_code")
    buck.index = buck.index.astype(str)
    centers = np.array([np.mean([float(x) for x in c.split("_")]) for c in buck.columns])
    buck.columns = centers
    samples = [s for s in buck.index if s in acute or s in post]
    X = buck.loc[samples]
    y = pd.Series(["acute" if s in acute else "post" for s in samples], index=samples)
    Xn = Normalise(X).pqn_normalise(plot=False); Xn.columns = X.columns
    model = opls_da(X=Xn, y=y, n_components=2, scaling_method="pareto",
                    kfold=7, estimator="opls", random_state=RANDOM_STATE)
    model.fit(); model.vip_scores()
    vip = model.get_vip_scores()
    vip_by_ppm = dict(zip(vip["Features"].astype(float).round(3), vip["VIP"].astype(float)))
    vip_ppm_arr = np.array(sorted(vip_by_ppm))
    ybin = (y == "acute").astype(int).to_numpy()

    def metbit_row(ppm):
        j = int(np.argmin(np.abs(Xn.columns.to_numpy(float) - ppm)))
        col = Xn.columns[j]
        r, _ = pointbiserialr(ybin, Xn[col].to_numpy(float))
        vppm = vip_ppm_arr[np.argmin(np.abs(vip_ppm_arr - float(col)))]
        return round(float(r), 3), round(float(vip_by_ppm[vppm]), 3)

    mb = df["ppm"].apply(metbit_row)
    df["metbit_class_corr_acute"] = [m[0] for m in mb]
    df["metbit_VIP"] = [m[1] for m in mb]
    df["metbit_dir"] = np.where(df["metbit_class_corr_acute"] > 0, "up", "down")
    df["direction_agree"] = df["article_dir"] == df["metbit_dir"]

    df = df.sort_values("metbit_VIP", ascending=False)
    df.to_csv(OUT / "article_vs_metbit_comparison.csv", index=False)

    # ---- concordance summary ----
    sig = df[df["article_significant"]]
    dir_agree_all = float(df["direction_agree"].mean())
    dir_agree_sig = float(sig["direction_agree"].mean()) if len(sig) else float("nan")
    # rank concordance: article -log10(p) vs metbit VIP
    from scipy.stats import spearmanr
    rho, _ = spearmanr(-np.log10(df["article_p"].clip(lower=1e-300)), df["metbit_VIP"])
    summary = {
        "n_metabolites_compared": int(len(df)),
        "n_article_significant_BH": int(sig.shape[0]),
        "direction_agreement_all": round(dir_agree_all, 3),
        "direction_agreement_article_significant": round(dir_agree_sig, 3),
        "rank_concordance_spearman(article_-log10p vs metbit_VIP)": round(float(rho), 3),
    }
    (OUT / "article_vs_metbit_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- figure: article log2FC vs metbit class-correlation ----
    fig, ax = plt.subplots(figsize=(8, 6))
    col = np.where(df["direction_agree"], "#2980b9", "#c0392b")
    ax.scatter(df["article_log2FC_acute_vs_post"], df["metbit_class_corr_acute"],
               c=col, s=np.clip(df["metbit_VIP"] * 40, 15, 300), alpha=0.75, edgecolor="w")
    for _, r in df.iterrows():
        ax.annotate(r["metabolite"], (r["article_log2FC_acute_vs_post"], r["metbit_class_corr_acute"]),
                    fontsize=7, alpha=0.8)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel("ARTICLE: log2 fold-change (acute / post), from reported concentrations")
    ax.set_ylabel("metbit: peak-vs-class correlation (acute)")
    ax.set_title("Article vs metbit, acute vs post-COVID (blue=direction agrees; size=metbit VIP)")
    plt.tight_layout(); plt.savefig(OUT / "article_vs_metbit.png", dpi=115); plt.close(fig)

    print(df[["metabolite", "ppm", "article_log2FC_acute_vs_post", "article_p_BH",
              "article_dir", "metbit_class_corr_acute", "metbit_dir",
              "metbit_VIP", "direction_agree"]].to_string(index=False))
    print("\nSUMMARY:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
