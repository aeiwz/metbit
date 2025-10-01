"""
Baseline correction utilities for 1D NMR spectra.

Adds multiple algorithms with a consistent interface. Depends on `pybaselines`
for many methods, with a pure-Python fallback for a rubberband baseline.
"""

from __future__ import annotations

from typing import Optional, Tuple, Literal, Dict, Any
import numpy as np
import pandas as pd

# Try to import optional algorithms from pybaselines
try:
    from pybaselines.whittaker import asls as _asls
except Exception:  # pragma: no cover
    _asls = None

try:
    from pybaselines.whittaker import arpls as _arpls  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _arpls = None

try:
    from pybaselines.whittaker import airpls as _airpls  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _airpls = None

try:
    from pybaselines.polynomial import modpoly as _modpoly, imodpoly as _imodpoly
except Exception:  # pragma: no cover
    _modpoly = _imodpoly = None


def _rubberband_baseline(y: np.ndarray, x: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute a simple 'rubberband' baseline via the lower convex hull.

    Parameters
    ----------
    y : np.ndarray
        1D intensity array.
    x : np.ndarray, optional
        1D x-axis. If None, uses np.arange(len(y)).
    """
    if x is None:
        x = np.arange(len(y))
    # Monotone chain to compute the lower convex hull
    pts = np.column_stack((x.astype(float), y.astype(float)))
    pts = pts[np.argsort(pts[:, 0])]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    lower = np.array(lower)
    # Interpolate hull along x
    return np.interp(x, lower[:, 0], lower[:, 1])


def _apply_baseline_1d(
    y: np.ndarray,
    method: Literal['asls', 'arpls', 'airpls', 'modpoly', 'imodpoly', 'rubberband'] = 'asls',
    x: Optional[np.ndarray] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Apply a baseline method to a 1D spectrum and return the estimated baseline."""
    method = method.lower()
    if method == 'asls':
        if _asls is None:
            raise ImportError("pybaselines.whittaker.asls is not available.")
        lam = kwargs.pop('lam', 1e7)
        max_iter = kwargs.pop('max_iter', 30)
        p = kwargs.pop('p', 0.01)
        baseline, _ = _asls(y, lam=lam, p=p, max_iter=max_iter)
        return baseline
    if method == 'arpls':
        if _arpls is None:
            raise ImportError("pybaselines.whittaker.arpls is not available.")
        lam = kwargs.pop('lam', 1e7)
        max_iter = kwargs.pop('max_iter', 30)
        baseline, _ = _arpls(y, lam=lam, max_iter=max_iter)
        return baseline
    if method == 'airpls':
        if _airpls is None:
            raise ImportError("pybaselines.whittaker.airpls is not available.")
        lam = kwargs.pop('lam', 1e7)
        max_iter = kwargs.pop('max_iter', 30)
        order = kwargs.pop('order', 2)
        baseline, _ = _airpls(y, lam=lam, max_iter=max_iter, order=order)
        return baseline
    if method == 'modpoly':
        if _modpoly is None:
            raise ImportError("pybaselines.polynomial.modpoly is not available.")
        poly_order = kwargs.pop('poly_order', 3)
        max_iter = kwargs.pop('max_iter', 50)
        tol = kwargs.pop('tol', 1e-3)
        baseline, _ = _modpoly(y, poly_order=poly_order, max_iter=max_iter, tol=tol)
        return baseline
    if method == 'imodpoly':
        if _imodpoly is None:
            raise ImportError("pybaselines.polynomial.imodpoly is not available.")
        poly_order = kwargs.pop('poly_order', 3)
        max_iter = kwargs.pop('max_iter', 50)
        tol = kwargs.pop('tol', 1e-3)
        baseline, _ = _imodpoly(y, poly_order=poly_order, max_iter=max_iter, tol=tol)
        return baseline
    if method == 'rubberband':
        return _rubberband_baseline(y, x=x)
    raise ValueError(f"Unknown baseline method: {method}")


def baseline_correct(
    X: pd.DataFrame,
    method: Literal['asls', 'arpls', 'airpls', 'modpoly', 'imodpoly', 'rubberband'] = 'asls',
    x: Optional[np.ndarray] = None,
    return_baseline: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply baseline correction across spectra in a DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Rows are spectra; columns are ppm (x-axis) values.
    method : str
        One of: 'asls', 'arpls', 'airpls', 'modpoly', 'imodpoly', 'rubberband'.
    x : np.ndarray, optional
        Explicit x-axis to use (same length as columns). If None, tries to
        infer from `X.columns` by casting to float, else uses index positions.
    return_baseline : bool
        If True, also return a DataFrame of the estimated baselines.
    **kwargs : Any
        Algorithm-specific keyword arguments (e.g., lam, p, max_iter, poly_order, ...).

    Returns
    -------
    corrected : pd.DataFrame
        Baseline-corrected spectra.
    baseline_df : pd.DataFrame (optional)
        Estimated baselines (returned if return_baseline=True).
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame with spectra as rows.")

    if X.isnull().values.any():
        X = X.fillna(0)

    # Prepare x-axis
    if x is None:
        try:
            x = X.columns.astype(float).to_numpy()
        except Exception:
            x = np.arange(X.shape[1], dtype=float)

    baselines = np.empty_like(X.values, dtype=float)
    for i, (_, spec) in enumerate(X.iterrows()):
        y = spec.to_numpy(dtype=float)
        baseline = _apply_baseline_1d(y, method=method, x=x, **kwargs)
        baselines[i, :] = baseline

    corrected = X.values - baselines
    corrected_df = pd.DataFrame(corrected, index=X.index, columns=X.columns)
    baseline_df = pd.DataFrame(baselines, index=X.index, columns=X.columns)

    if return_baseline:
        return corrected_df, baseline_df
    return corrected_df


def bline(X: pd.DataFrame, lam: float = 1e7, max_iter: int = 30) -> pd.DataFrame:
    """
    Backwards-compatible wrapper for ALS baseline correction.

    Parameters
    ----------
    X : pd.DataFrame
        Rows are spectra; columns are ppm values.
    lam : float
        Smoothing parameter for ASLS.
    max_iter : int
        Max iterations for ASLS.
    """
    return baseline_correct(X, method='asls', lam=lam, max_iter=max_iter)
