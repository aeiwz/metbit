"""Validate metbit on MTBLS12785 (infant urinary 1H NMR, fully public).

Both halves are open: raw Bruker FID (196 subjects x timepoints) and the MAF
reference listing 39 identified metabolites with their assigned 1H ppm. We:

  1. Process raw FID with metbit (one timepoint, 1-month urine).
  2. Peak-assignment concordance -- for each metabolite the article assigns,
     verify metbit's processed spectra show a resolved peak at that ppm. This
     checks metbit reproduces the article's assignments against a public
     reference (no paywall, no fabricated numbers).

Usage: python Benchmark/MTBLS12785/validate_mtbls12785.py
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
from metbit import nmr_preprocessing  # noqa: E402

DATA = REPO / "Benchmark" / "data" / "MTBLS12785"
NMR = DATA / "nmr_file"
OUT = Path(__file__).resolve().parent / "result"
OUT.mkdir(parents=True, exist_ok=True)
TIMEPOINT = "1M_urine"   # one-month infant urine
MAX_N = 60               # cap for a fast, representative run


def parse_ppm(cs):
    """Parse a MAF chemical_shift cell ('8.870-8.930' or '1.323') to a center ppm."""
    if not isinstance(cs, str):
        return None
    first = cs.split(",")[0].strip()
    try:
        if "-" in first and not first.startswith("-"):
            a, b = first.split("-")
            return (float(a) + float(b)) / 2
        return float(first)
    except ValueError:
        return None


def load_reference():
    maf = pd.read_csv(DATA / "m_MTBLS12785_NMR___metabolite_profiling_v2_maf.tsv", sep="\t")
    ref = maf[["metabolite_identification", "chemical_shift", "multiplicity"]].copy()
    ref["ppm"] = ref["chemical_shift"].apply(parse_ppm)
    ref = ref.dropna(subset=["ppm"])
    ref = ref[(ref["ppm"] > 0.2) & (ref["ppm"] < 9.5)]  # exclude water/ref edges
    return ref.reset_index(drop=True)


def extract_subset():
    work = DATA / ("proc_" + TIMEPOINT)
    work.mkdir(parents=True, exist_ok=True)
    zips = sorted(NMR.glob(f"{TIMEPOINT}-*.zip"))[:MAX_N]
    n = 0
    for zp in zips:
        try:
            inner = zipfile.ZipFile(zp)
            if inner.testzip() is not None:
                continue
            sid = zp.stem  # e.g. 1M_urine-100
            if not (work / sid / "fid").exists():
                inner.extractall(work / sid)
            n += 1
        except Exception:
            continue
    return work, n


def main():
    ref = load_reference()
    work, n = extract_subset()
    print(f"{TIMEPOINT}: extracted {n} FID samples; {len(ref)} assigned metabolite peaks in MAF")

    prep = nmr_preprocessing(str(work), calib_type="tsp", auto_phasing=True, align=True)
    X = prep.get_data(); ppm = prep.get_ppm()
    mean_spec = X.mean(0).to_numpy(float)
    ppm_arr = np.asarray(ppm, float)

    # peak-assignment concordance: is there a local maximum near each assigned
    # ppm, rising above the LOCAL baseline noise (robust to the residual-water
    # region that would inflate a global noise estimate)?
    rows = []
    for _, r in ref.iterrows():
        p = r["ppm"]
        win = (ppm_arr > p - 0.02) & (ppm_arr < p + 0.02)          # peak window
        loc = (ppm_arr > p - 0.15) & (ppm_arr < p + 0.15)          # local context
        if win.sum() == 0:
            continue
        peak = float(mean_spec[win].max())
        peak_ppm = float(ppm_arr[win][np.argmax(mean_spec[win])])
        local = mean_spec[loc]
        baseline = float(np.median(local))
        local_noise = float(np.median(np.abs(local - baseline)) * 1.4826)  # local robust sigma
        snr = (peak - baseline) / (local_noise + 1e-9)
        rows.append({"metabolite": r["metabolite_identification"], "assigned_ppm": round(p, 3),
                     "multiplicity": r["multiplicity"], "metbit_peak_ppm": round(peak_ppm, 3),
                     "ppm_offset": round(abs(peak_ppm - p), 4),
                     "metbit_peak_snr": round(snr, 1), "resolved": bool(snr > 3)})
    conc = pd.DataFrame(rows).sort_values("metbit_peak_snr", ascending=False)
    conc.to_csv(OUT / "peak_assignment_concordance.csv", index=False)
    n_res = int(conc["resolved"].sum())

    # TSP check
    win0 = (ppm_arr > -0.2) & (ppm_arr < 0.2)
    tsp = float(ppm_arr[win0][np.argmax(mean_spec[win0])])

    # plot mean spectrum with assigned ppm marked
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ppm_arr, mean_spec, lw=0.5, color="#333")
    for _, r in conc.iterrows():
        c = "#2980b9" if r["resolved"] else "#c0392b"
        ax.axvline(r["assigned_ppm"], color=c, lw=0.5, alpha=0.5)
    ax.set_xlim(9.5, 0); ax.set_xlabel("ppm")
    ax.set_title(f"MTBLS12785 {TIMEPOINT} mean spectrum (n={len(X)}); "
                 f"blue=article-assigned peak resolved by metbit, red=not")
    plt.tight_layout(); plt.savefig(OUT / "mean_spectrum_assignments.png", dpi=115); plt.close(fig)

    report = {
        "dataset": "MTBLS12785 infant urinary 1H NMR (open Scientific Data)",
        "timepoint": TIMEPOINT, "n_samples": int(len(X)),
        "median_TSP_peak_ppm": round(tsp, 4), "TSP_within_0.02ppm": bool(abs(tsp) < 0.02),
        "n_assigned_metabolite_peaks": int(len(conc)),
        "n_resolved_by_metbit_SNR_gt_3": n_res,
        "fraction_resolved": round(n_res / len(conc), 3) if len(conc) else None,
        "resolved_peak_ppm_accuracy_median": round(float(conc[conc["resolved"]]["ppm_offset"].median()), 4),
        "resolved_peak_ppm_accuracy_max": round(float(conc[conc["resolved"]]["ppm_offset"].max()), 4),
        "note": ("MAF assignments span ages 0-5y; low-abundance/dietary metabolites "
                 "(trigonelline, aromatics, sugars) are expectedly weak in 1-month "
                 "milk-fed infant urine, so unresolved != processing error."),
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(conc.to_string(index=False))
    print("\n", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
