"""Shared benchmark metrics for the metbit multi-dataset benchmark suite.

Grouped by the four benchmark tiers (see Benchmark/README.md):
  correctness  -- compare processed spectra / buckets against a reference
  qc           -- technical-replicate reproducibility and QC dispersion
  chemometrics -- classifier quality and VIP-ranking stability
  performance  -- timing, memory, throughput, cross-backend numerical agreement

Dependency-light on purpose (numpy / scipy / scikit-learn only) so every
per-dataset script can import it without pulling extra packages. Run this file
directly to execute the self-checks:  python Benchmark/metrics.py
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Correctness tier (e.g. NMRProcFlow Tomato: metbit output vs reference bucket)
# --------------------------------------------------------------------------- #

def rmse(x, x_ref):
    """Root-mean-square error between an array and its reference."""
    x, x_ref = np.asarray(x, float).ravel(), np.asarray(x_ref, float).ravel()
    return float(np.sqrt(np.mean((x - x_ref) ** 2)))


def reference_correlation(x, x_ref):
    """Pearson correlation between a processed spectrum and a reference."""
    x, x_ref = np.asarray(x, float).ravel(), np.asarray(x_ref, float).ravel()
    return float(np.corrcoef(x, x_ref)[0, 1])


def integrated_area_relative_error(areas, areas_ref, eps=1e-12):
    """Mean relative error of integrated bucket/region areas vs reference."""
    a, r = np.asarray(areas, float).ravel(), np.asarray(areas_ref, float).ravel()
    return float(np.mean(np.abs(a - r) / (np.abs(r) + eps)))


def negative_area_fraction(spectra):
    """Fraction of intensities below zero (baseline-correction quality proxy)."""
    X = np.asarray(spectra, float)
    return float((X < 0).sum() / X.size)


def alignment_error_ppm(peak_ppm, target_ppm):
    """Mean absolute chemical-shift alignment error, in ppm."""
    p, t = np.asarray(peak_ppm, float).ravel(), np.asarray(target_ppm, float).ravel()
    return float(np.mean(np.abs(p - t)))


# --------------------------------------------------------------------------- #
# QC / reproducibility tier (e.g. nPYc DEVSET / MTBLS694 replicates + QC)
# --------------------------------------------------------------------------- #

def median_feature_cv(X):
    """Median per-feature coefficient of variation (%) across samples.

    Report before vs after normalisation: a good normalisation lowers it.
    """
    X = np.asarray(X, float)
    mean = X.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cv = np.where(np.abs(mean) > 0, X.std(0, ddof=1) / np.abs(mean), np.nan)
    return float(np.nanmedian(cv) * 100.0)


def replicate_mean_cv(X, replicate_labels):
    """Mean within-replicate-group CV (%). Lower = better repeatability."""
    X = np.asarray(X, float)
    labels = np.asarray(replicate_labels)
    cvs = []
    for g in np.unique(labels):
        Xg = X[labels == g]
        if Xg.shape[0] < 2:
            continue
        m = Xg.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            cv = np.where(np.abs(m) > 0, Xg.std(0, ddof=1) / np.abs(m), np.nan)
        cvs.append(np.nanmedian(cv))
    return float(np.mean(cvs) * 100.0) if cvs else float("nan")


def pairwise_replicate_correlation(X, replicate_labels):
    """Mean pairwise Pearson correlation within replicate groups."""
    X = np.asarray(X, float)
    labels = np.asarray(replicate_labels)
    cors = []
    for g in np.unique(labels):
        Xg = X[labels == g]
        if Xg.shape[0] < 2:
            continue
        c = np.corrcoef(Xg)
        iu = np.triu_indices_from(c, k=1)
        cors.extend(c[iu].tolist())
    return float(np.mean(cors)) if cors else float("nan")


def qc_cluster_dispersion(scores, qc_mask):
    """Mean distance of QC-sample PCA scores to their centroid.

    Tight QC clustering (small value) indicates good analytical precision.
    """
    S = np.asarray(scores, float)
    qc = np.asarray(qc_mask, bool)
    if qc.sum() < 2:
        return float("nan")
    Sq = S[qc]
    return float(np.mean(np.linalg.norm(Sq - Sq.mean(0), axis=1)))


# --------------------------------------------------------------------------- #
# Chemometrics tier (e.g. MTBLS2052 CKD: OPLS-DA quality + VIP stability)
# --------------------------------------------------------------------------- #

def classification_metrics(y_true, y_pred, y_score=None, pos_label=None):
    """Balanced accuracy, sensitivity, specificity, and ROC-AUC (if scores)."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    pos = classes[-1] if pos_label is None else pos_label
    yt = (y_true == pos).astype(int)
    yp = (y_pred == pos).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    out = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }
    if y_score is not None:
        out["roc_auc"] = float(roc_auc_score(yt, np.asarray(y_score, float)))
    return out


def vip_rank_correlation(vip_a, vip_b):
    """Spearman rank correlation of two VIP vectors (ranking stability)."""
    from scipy.stats import spearmanr
    r, _ = spearmanr(np.asarray(vip_a, float), np.asarray(vip_b, float))
    return float(r)


def bootstrap_vip_stability(vip_matrix, top_k=20):
    """Mean Jaccard overlap of the top-k VIP feature sets across repeats.

    vip_matrix: shape (n_repeats, n_features). 1.0 = identical top-k every run.
    """
    V = np.asarray(vip_matrix, float)
    tops = [set(np.argsort(v)[::-1][:top_k]) for v in V]
    j = []
    for i in range(len(tops)):
        for k in range(i + 1, len(tops)):
            u = len(tops[i] | tops[k])
            j.append(len(tops[i] & tops[k]) / u if u else np.nan)
    return float(np.nanmean(j)) if j else float("nan")


# --------------------------------------------------------------------------- #
# Performance tier (e.g. ST002087 COMETA: scaling + cross-backend agreement)
# --------------------------------------------------------------------------- #

def throughput(n_items, seconds):
    """Items processed per second (samples/s or features/s)."""
    return float(n_items / seconds) if seconds > 0 else float("inf")


def speedup(baseline_seconds, candidate_seconds):
    """Speed-up of a candidate backend relative to a baseline (e.g. NumPy)."""
    return float(baseline_seconds / candidate_seconds) if candidate_seconds > 0 else float("inf")


def max_abs_error(a, b):
    """Maximum absolute element-wise difference between two backends' output."""
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


# --------------------------------------------------------------------------- #

def _selfcheck():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(6, 200))
    noisy = ref + rng.normal(scale=0.01, size=ref.shape)
    assert rmse(noisy, ref) < 0.02
    assert reference_correlation(noisy[0], ref[0]) > 0.99
    assert integrated_area_relative_error([10, 20], [10, 20]) == 0.0
    assert 0 <= negative_area_fraction(ref) <= 1
    assert abs(alignment_error_ppm([0.01, -0.01], [0, 0]) - 0.01) < 1e-9

    labels = np.array([0, 0, 1, 1, 2, 2])
    # normalisation should not increase within-replicate CV vs raw scaled rows
    raw = np.abs(ref) + 1.0
    assert replicate_mean_cv(raw, labels) >= 0
    assert -1 <= pairwise_replicate_correlation(raw, labels) <= 1
    assert median_feature_cv(raw) >= 0
    assert qc_cluster_dispersion(rng.normal(size=(6, 2)), [1, 1, 0, 0, 0, 0]) >= 0

    yt = np.array(["a", "a", "b", "b"])
    yp = np.array(["a", "b", "b", "b"])
    m = classification_metrics(yt, yp, y_score=[0.1, 0.6, 0.7, 0.9], pos_label="b")
    assert 0 <= m["balanced_accuracy"] <= 1 and "roc_auc" in m
    assert abs(vip_rank_correlation([3, 2, 1], [3, 2, 1]) - 1.0) < 1e-9
    V = np.tile(np.arange(50)[::-1], (3, 1))
    assert bootstrap_vip_stability(V, top_k=10) == 1.0

    assert throughput(100, 2.0) == 50.0
    assert speedup(10.0, 2.0) == 5.0
    assert max_abs_error([1, 2, 3], [1, 2, 3.5]) == 0.5
    print("metrics self-check passed")


if __name__ == "__main__":
    _selfcheck()
