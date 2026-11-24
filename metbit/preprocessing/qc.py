"""Sample-wise normalization with exact coefficients, QC drift correction and outlier detection.

Matches the Normalize, QC drift and Outliers steps of the metbit desktop app.

SNV: https://docs.labcognition.com/panorama/en/standard_normal_variate_correction/
MSC: https://search.r-project.org/CRAN/refmans/prospectr/html/msc.html
PQN: Dieterle et al. 2006, median quotients to the median reference spectrum.
"""
import numpy as np
import pandas as pd
from scipy.stats import f as f_dist
from sklearn.decomposition import PCA

SCALING_POWER = {"mean": 0.0, "pareto": 0.5, "uv": 1.0}


def _positive(values, what, *, allow_negative=False):
    bad = ~np.isfinite(values) | ((values == 0) if allow_negative else (values <= 0))
    if bad.any():
        rows = ", ".join(str(i + 1) for i in np.flatnonzero(bad)[:8])
        need = "nonzero" if allow_negative else "positive"
        raise ValueError(f"{what} must be finite and {need}; check sample rows {rows} for flat, inverted or invalid spectra.")
    return values


def _pqn(X):
    reference = np.median(X, axis=0)
    usable = reference != 0
    if not usable.any():
        raise ValueError("PQN needs a nonzero median reference spectrum.")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        ratios = X[:, usable] / reference[usable]
    coef = np.array([np.median(r[np.isfinite(r)]) if np.isfinite(r).any() else np.nan for r in ratios])
    return _positive(coef, "PQN coefficients")


def normalize_spectra(X, method: str = "pqn"):
    """Normalize each sample by PQN, SNV, SNV then PQN ('snv_pqn') or MSC.

    Returns (normalized, factors, offsets) with normalized = (X - offsets[:, None]) * factors[:, None],
    so the exact applied transform can be reported or reapplied. A DataFrame input gives a DataFrame.
    """
    frame = X if isinstance(X, pd.DataFrame) else None
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or not X.size or X.shape[1] < 2:
        raise ValueError("Normalization needs samples and at least two spectral features.")
    if not np.isfinite(X).all():
        raise ValueError("Normalization input contains missing or infinite intensities.")
    offsets = np.zeros(len(X))
    if method == "pqn":
        factors = 1 / _pqn(X)
    elif method in ("snv", "snv_pqn"):
        offsets = X.mean(axis=1)
        factors = 1 / _positive(X.std(axis=1, ddof=1), "Sample standard deviations")
        if method == "snv_pqn":
            factors = factors / _pqn((X - offsets[:, None]) * factors[:, None])
    elif method == "msc":
        reference = X.mean(axis=0)
        centered = reference - reference.mean()
        denom = centered @ centered
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError("MSC needs a varying mean reference spectrum.")
        slopes = (X - X.mean(axis=1, keepdims=True)) @ centered / denom
        factors = 1 / _positive(slopes, "MSC regression slopes", allow_negative=True)
        offsets = X.mean(axis=1) - slopes * reference.mean()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    with np.errstate(over="ignore", invalid="ignore"):
        out = (X - offsets[:, None]) * factors[:, None]
    if not np.isfinite(out).all() or not np.isfinite(factors).all():  # pragma: no cover - guard behind the checks above
        raise ValueError("Normalization produced non-finite values.")
    if frame is not None:
        out = pd.DataFrame(out, index=frame.index, columns=frame.columns)
    return out, factors, offsets


def scale_columns(X, how: str = "pareto") -> np.ndarray:
    """Mean-centre, then divide by SD ** power (mean: 0, pareto: 0.5, uv: 1). Constant columns keep SD 1."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    power = SCALING_POWER[how]
    if not power:
        return Xc
    sd = X.std(axis=0, ddof=1)
    sd[~(sd > 0)] = 1.0
    return Xc / sd**power


# QC drift (QC-RLSC) ---------------------------------------------------------

def qc_rsd(X) -> float:
    """Median relative standard deviation (%) of the features across QC samples."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    ok = mean > 0
    return float(np.median(X[:, ok].std(axis=0, ddof=1) / mean[ok]) * 100) if ok.any() else float("nan")


def _loess_matrix(x: np.ndarray, span: float) -> np.ndarray:
    """S with fitted = S @ y for a local linear LOESS (tricube weights) at sorted points x."""
    m = len(x)
    k = max(2, min(m, int(np.ceil(span * m))))
    S = np.zeros((m, m))
    for i in range(m):
        d = np.abs(x - x[i])
        h = np.sort(d)[k - 1] or 1.0
        w = np.clip(1 - (d / (h * (1 + 1e-10))) ** 3, 0, None) ** 3
        A = np.column_stack([np.ones(m), x - x[i]]) * np.sqrt(w)[:, None]
        S[i] = (np.linalg.pinv(A) * np.sqrt(w))[0]
    return S


def _interpolation_matrix(x: np.ndarray, at: np.ndarray) -> np.ndarray:
    """M with M @ f == np.interp(at, x, f) for every column f (x sorted)."""
    M = np.zeros((len(at), len(x)))
    j = np.clip(np.searchsorted(x, at) - 1, 0, len(x) - 2)
    t = np.clip((at - x[j]) / (x[j + 1] - x[j]), 0, 1)
    M[np.arange(len(at)), j] = 1 - t
    M[np.arange(len(at)), j + 1] = t
    return M


def qc_drift_correct(X, order, qc, span: float = 0.75):
    """QC-RLSC: a LOESS through the pooled QC samples by run order, read at every sample's run
    order, and every feature divided by that trend and scaled back to the QC median.

    order: run (injection) order per sample. qc: boolean mask of pooled QC samples (at least 4).
    Returns the corrected matrix (DataFrame for a DataFrame input).
    """
    frame = X if isinstance(X, pd.DataFrame) else None
    X = np.asarray(X, dtype=float)
    order = np.asarray(order, dtype=float)
    qc = np.asarray(qc, dtype=bool)
    if np.isnan(order).any():
        raise ValueError("Run order must be numeric with no missing values")
    if qc.sum() < 4:
        raise ValueError(f"Drift correction needs at least 4 QC samples; found {int(qc.sum())}")
    # ponytail: no robustness iterations; samples outside the QC run range get the end's trend
    x_qc = order[qc]
    rank = np.argsort(x_qc)
    trend = _interpolation_matrix(x_qc[rank], order) @ (_loess_matrix(x_qc[rank], span) @ X[qc][rank])
    level = np.median(X[qc], axis=0)
    safe = trend > 0
    out = np.where(safe, X / np.where(safe, trend, 1) * level, X)
    return pd.DataFrame(out, index=frame.index, columns=frame.columns) if frame is not None else out


# Outliers -------------------------------------------------------------------

def detect_outliers(X, n_components: int = 2, scaling: str = "pareto", confidence: float = 0.95) -> pd.DataFrame:
    """PCA Hotelling T2 (F-based limit) and SIMCA-normalized DModX for every sample.

    Returns a DataFrame with T2, DModX, their limits and an 'outlier' text ('T2', 'DModX', both or '').
    """
    index = X.index if isinstance(X, pd.DataFrame) else None
    X = np.asarray(X, dtype=float)
    n, k = X.shape
    a = n_components
    if a >= min(n - 1, k):
        raise ValueError(f"Too many components ({a}) for {n} samples")
    Xs = scale_columns(X, scaling)
    model = PCA(n_components=a).fit(Xs)
    T = model.transform(Xs)
    t2 = ((T / T.std(axis=0, ddof=1)) ** 2).sum(axis=1)
    t2_limit = a * (n - 1) / (n - a) * f_dist.ppf(confidence, a, n - a)
    E = Xs - T @ model.components_ - model.mean_
    s_i = np.sqrt((E**2).sum(axis=1) / (k - a))
    s0 = np.sqrt((E**2).sum() / ((n - a - 1) * (k - a)))
    dmodx = s_i / s0
    dmodx_limit = float(np.sqrt(f_dist.ppf(confidence, k - a, (n - a - 1) * (k - a))))
    flag = [" and ".join(name for name, hit in (("T2", t > t2_limit), ("DModX", d > dmodx_limit)) if hit)
            for t, d in zip(t2, dmodx)]
    return pd.DataFrame({"T2": t2, "T2_limit": float(t2_limit), "DModX": dmodx, "DModX_limit": dmodx_limit,
                         "outlier": flag}, index=index)


__all__ = ["normalize_spectra", "scale_columns", "qc_rsd", "qc_drift_correct", "detect_outliers"]
