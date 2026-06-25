"""
_native.py - Compute-backend dispatcher for metbit large-scale kernels.

Auto-dispatch hierarchy (fastest available wins for each dataset size):
  GPU (CuPy/PyTorch CUDA)  ->  large datasets, GPU memory fits
  OpenMP C extension        ->  large CPU-bound datasets (n*p > 10M)
  Single-threaded C         ->  small datasets (n*p <= 10M, fastest cache-warm)
  Chunked NumPy             ->  fallback when no C extension is compiled
  Multiprocessing           ->  used for CPU-parallel STOCSY / variance when
                                the C extension is absent but many cores exist

Environment overrides
---------------------
  METBIT_DISABLE_NATIVE=1   skip the C extension entirely
  METBIT_DISABLE_GPU=1      skip GPU backends (CuPy / PyTorch)
  METBIT_N_JOBS=N           override worker count for multiprocessing paths
  METBIT_CHUNK=N            override default feature chunk size

Thresholds (elements = n_samples * n_features)
-----------------------------------------------
  <= SMALL_THRESH  : single-threaded C  (fits L3 cache, avoids thread overhead)
  <= LARGE_THRESH  : OpenMP C / multiprocessing
  >  LARGE_THRESH  : GPU if available, else OpenMP C / multiprocessing
"""

from __future__ import annotations

import math
import multiprocessing
import os
from functools import partial
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

_DISABLE_NATIVE = os.environ.get("METBIT_DISABLE_NATIVE", "").lower() in {
    "1", "true", "yes",
}
_DISABLE_GPU = os.environ.get("METBIT_DISABLE_GPU", "").lower() in {
    "1", "true", "yes",
}
_ENV_N_JOBS = os.environ.get("METBIT_N_JOBS")
_ENV_CHUNK  = os.environ.get("METBIT_CHUNK")

_N_JOBS_DEFAULT = min(multiprocessing.cpu_count(), 8)
_N_JOBS = int(_ENV_N_JOBS) if _ENV_N_JOBS else _N_JOBS_DEFAULT
_DEFAULT_CHUNK = int(_ENV_CHUNK) if _ENV_CHUNK else 50_000

# Element count thresholds for backend selection
_SMALL_THRESH = 10_000_000    # 10 M  -> single-threaded C
_LARGE_THRESH = 500_000_000   # 500 M -> GPU preferred above this

# ---------------------------------------------------------------------------
# Load optional backends
# ---------------------------------------------------------------------------

try:
    if _DISABLE_NATIVE:
        raise ImportError("native backend disabled by METBIT_DISABLE_NATIVE")
    from . import _native_backend  # pragma: no cover
    _NATIVE_OK = True  # pragma: no cover
except ImportError:
    _native_backend = None  # type: ignore[assignment]
    _NATIVE_OK = False

# Detect number of OpenMP threads reported by the C extension
_OPENMP_THREADS: int = 0
if _NATIVE_OK:  # pragma: no cover
    try:
        _OPENMP_THREADS = _native_backend.openmp_threads()
    except AttributeError:
        _OPENMP_THREADS = 0

# GPU backends: try CuPy first (lighter), then PyTorch
_cupy  = None
_torch = None

if not _DISABLE_GPU:  # pragma: no cover
    try:
        import cupy as _cupy  # type: ignore[import]
        _cupy.cuda.Device(0).use()  # validate at least one GPU
    except Exception:
        _cupy = None
    if _cupy is None:
        try:
            import torch as _torch  # type: ignore[import]
            if not _torch.cuda.is_available():
                _torch = None
        except Exception:
            _torch = None


def native_available() -> bool:
    """Return True when the compiled C extension is active."""
    return _NATIVE_OK


def gpu_available() -> bool:
    """Return True when a CUDA-capable GPU backend is available."""
    return _cupy is not None or _torch is not None


def backend_info() -> dict:
    """Return a dict describing the active compute backends."""
    return {
        "native_c": _NATIVE_OK,
        "openmp_threads": _OPENMP_THREADS,
        "gpu_cupy": _cupy is not None,
        "gpu_torch": _torch is not None and _torch is not None,
        "n_jobs": _N_JOBS,
        "default_chunk": _DEFAULT_CHUNK,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _n_elements(n: int, p: int) -> int:
    return n * p


def _asf64_c(arr: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float64 array (copy only when necessary)."""
    return np.asarray(arr, dtype=np.float64, order="C")


def _asf32_c(arr: np.ndarray) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32, order="C")


# ---------------------------------------------------------------------------
# GPU implementations
# ---------------------------------------------------------------------------

def _pearson_gpu(matrix: np.ndarray, anchor_index: int) -> np.ndarray:  # pragma: no cover
    """Pearson correlation on GPU. Returns numpy float64 array."""
    if _cupy is not None:
        X = _cupy.asarray(matrix, dtype=_cupy.float32)
        anchor = X[:, anchor_index]
        a_c = anchor - anchor.mean()
        X_c = X - X.mean(axis=0)
        num = a_c @ X_c
        denom = _cupy.sqrt(
            float(a_c @ a_c)
            * _cupy.einsum("ij,ij->j", X_c, X_c)
        )
        corr = _cupy.where(denom == 0, _cupy.nan, num / denom)
        corr = _cupy.clip(corr, -1.0, 1.0)
        return _cupy.asnumpy(corr).astype(np.float64)

    if _torch is not None:
        dev = _torch.device("cuda")
        X = _torch.as_tensor(matrix, dtype=_torch.float32, device=dev)
        anchor = X[:, anchor_index]
        a_c = anchor - anchor.mean()
        X_c = X - X.mean(dim=0)
        num = a_c @ X_c
        denom = _torch.sqrt(
            (a_c @ a_c)
            * (X_c * X_c).sum(dim=0)
        )
        corr = _torch.where(denom == 0, _torch.tensor(float("nan"), device=dev), num / denom)
        corr = _torch.clamp(corr, -1.0, 1.0)
        return corr.cpu().numpy().astype(np.float64)

    raise RuntimeError("GPU backend requested but none available")


def _column_variances_gpu(matrix: np.ndarray) -> np.ndarray:  # pragma: no cover
    """Per-column variance on GPU. Returns numpy float64 array."""
    if _cupy is not None:
        X = _cupy.asarray(matrix, dtype=_cupy.float32)
        var = X.var(axis=0, ddof=1)
        return _cupy.asnumpy(var).astype(np.float64)

    if _torch is not None:
        dev = _torch.device("cuda")
        X = _torch.as_tensor(matrix, dtype=_torch.float32, device=dev)
        var = X.var(dim=0, unbiased=True)
        return var.cpu().numpy().astype(np.float64)

    raise RuntimeError("GPU backend requested but none available")


def _vip_scores_gpu(t: np.ndarray, w: np.ndarray, q: np.ndarray) -> np.ndarray:  # pragma: no cover
    """VIP scores on GPU."""
    if _cupy is not None:
        t_g = _cupy.asarray(t, dtype=_cupy.float64)
        w_g = _cupy.asarray(w, dtype=_cupy.float64)
        q_g = _cupy.asarray(q.ravel(), dtype=_cupy.float64)
        n_feat, n_comp = w_g.shape

        t_sq = _cupy.einsum("ij,ij->j", t_g, t_g)  # (h,)
        S = t_sq * q_g ** 2
        total_s = S.sum()

        norms = _cupy.linalg.norm(w_g, axis=0)  # (h,)
        norms[norms == 0] = 1.0
        w_norm = w_g / norms
        vips = _cupy.sqrt(float(n_feat) * ((w_norm ** 2) @ S) / total_s)
        return _cupy.asnumpy(vips).astype(np.float64)

    if _torch is not None:
        dev = _torch.device("cuda")
        t_g = _torch.as_tensor(t, dtype=_torch.float64, device=dev)
        w_g = _torch.as_tensor(w, dtype=_torch.float64, device=dev)
        q_g = _torch.as_tensor(q.ravel(), dtype=_torch.float64, device=dev)
        n_feat = w_g.shape[0]

        S = (t_g ** 2).sum(dim=0) * q_g ** 2
        total_s = S.sum()
        norms = w_g.norm(dim=0)
        norms[norms == 0] = 1.0
        w_norm = w_g / norms
        vips = _torch.sqrt(float(n_feat) * ((w_norm ** 2) @ S) / total_s)
        return vips.cpu().numpy().astype(np.float64)

    raise RuntimeError("GPU backend requested but none available")


# ---------------------------------------------------------------------------
# Multiprocessing helper (used when C extension is absent)
# ---------------------------------------------------------------------------

def _pearson_chunk_worker(args):
    """Worker: compute Pearson r for a column slice. Runs in a subprocess."""
    chunk_data, a_centered, a_sq, col_means_chunk = args
    # chunk_data: (n, chunk_size) float32 or float64
    chunk_f64 = chunk_data.astype(np.float64, copy=False)
    chunk_c = chunk_f64 - col_means_chunk
    num = a_centered @ chunk_c
    sq = np.einsum("ij,ij->j", chunk_c, chunk_c)
    denom = np.sqrt(a_sq * sq)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / denom
    return np.clip(r, -1.0, 1.0)


def _variance_chunk_worker(args):
    """Worker: compute per-column variance for a column slice."""
    chunk_data, col_means_chunk = args
    chunk_f64 = chunk_data.astype(np.float64, copy=False)
    chunk_c = chunk_f64 - col_means_chunk
    ss = np.einsum("ij,ij->j", chunk_c, chunk_c)
    n = chunk_f64.shape[0]
    return ss / (n - 1)


def _pearson_multiprocessing(
    matrix: np.ndarray,
    anchor_index: int,
    chunk_size: int,
    n_jobs: int,
) -> np.ndarray:
    """Chunked Pearson via multiprocessing.Pool (CPU fallback without C ext)."""
    n, p = matrix.shape
    anchor = matrix[:, anchor_index].astype(np.float64)
    a_c = anchor - anchor.mean()
    a_sq = float(np.dot(a_c, a_c))
    col_means = matrix.mean(axis=0).astype(np.float64)

    tasks = []
    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        tasks.append((
            matrix[:, start:end],
            a_c,
            a_sq,
            col_means[start:end],
        ))

    correlations = np.empty(p, dtype=np.float64)
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(_pearson_chunk_worker, tasks)

    idx = 0
    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        correlations[start:end] = results[idx]
        idx += 1
    return correlations


def _variance_multiprocessing(
    matrix: np.ndarray,
    chunk_size: int,
    n_jobs: int,
) -> np.ndarray:
    """Chunked column variance via multiprocessing.Pool."""
    n, p = matrix.shape
    col_means = matrix.mean(axis=0).astype(np.float64)

    tasks = [
        (matrix[:, s:min(s + chunk_size, p)], col_means[s:min(s + chunk_size, p)])
        for s in range(0, p, chunk_size)
    ]

    variances = np.empty(p, dtype=np.float64)
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(_variance_chunk_worker, tasks)

    idx = 0
    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        variances[start:end] = results[idx]
        idx += 1
    return variances


# ---------------------------------------------------------------------------
# NumPy chunked fallback (single-process, memory bounded)
# ---------------------------------------------------------------------------

def _pearson_numpy_chunked(
    matrix: np.ndarray,
    anchor_index: int,
    chunk_size: int,
) -> np.ndarray:
    n, p = matrix.shape
    anchor = matrix[:, anchor_index].astype(np.float64)
    a_c = anchor - anchor.mean()
    a_sq = float(np.dot(a_c, a_c))
    if a_sq == 0.0:
        return np.zeros(p, dtype=np.float64)

    col_means = matrix.mean(axis=0).astype(np.float64)
    correlations = np.empty(p, dtype=np.float64)

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        chunk = matrix[:, start:end].astype(np.float64, copy=False)
        chunk_c = chunk - col_means[start:end]
        num = a_c @ chunk_c
        sq = np.einsum("ij,ij->j", chunk_c, chunk_c)
        denom = np.sqrt(a_sq * sq)
        with np.errstate(invalid="ignore", divide="ignore"):
            correlations[start:end] = num / denom
        del chunk_c

    return np.clip(correlations, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Public API: pearson_columns
# ---------------------------------------------------------------------------

def pearson_columns(
    data,
    anchor_index: int,
    chunk_size: int = _DEFAULT_CHUNK,
    n_jobs: int = _N_JOBS,
) -> np.ndarray:
    """Pearson r between one column and all other columns of a 2D matrix.

    Backend auto-selected based on dataset size and available hardware:

      GPU (cupy/torch)          n*p > LARGE_THRESH and GPU available
      C + OpenMP (parallel)     n*p > SMALL_THRESH and C ext available
      C single-threaded         n*p <= SMALL_THRESH and C ext available
      multiprocessing + NumPy   C ext absent, n_jobs > 1
      chunked NumPy (1 process) absolute fallback

    Parameters
    ----------
    data: array-like, shape (n_samples, n_features)
    anchor_index: int
    chunk_size: int
        Feature chunk for chunked NumPy / multiprocessing paths.
    n_jobs: int
        Worker processes for the multiprocessing path.
    """
    matrix = np.asarray(data, order="C")
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix")
    n, p = matrix.shape
    if n < 2 or p < 1:
        raise ValueError("data must have at least 2 rows and 1 column")
    if not 0 <= anchor_index < p:
        raise IndexError("anchor_index is out of range")

    size = _n_elements(n, p)

    # -- GPU path ----------------------------------------------------------
    if gpu_available() and size > _SMALL_THRESH:  # pragma: no cover
        try:
            return _pearson_gpu(matrix, anchor_index)
        except Exception:
            pass  # fall through to CPU

    # -- C extension paths -------------------------------------------------
    if _NATIVE_OK:  # pragma: no cover
        if matrix.dtype == np.float32 and hasattr(_native_backend, "pearson_columns_f32"):
            mat = _asf32_c(matrix)
            packed = _native_backend.pearson_columns_f32(
                memoryview(mat), n, p, anchor_index
            )
            return np.frombuffer(packed, dtype=np.float64).copy()

        mat = _asf64_c(matrix)
        if size <= _SMALL_THRESH:
            # Single-threaded C: best cache locality for small data
            packed = _native_backend.pearson_columns(
                memoryview(mat), n, p, anchor_index
            )
        else:
            # OpenMP-parallel C: row-parallel with thread-local accumulators
            packed = _native_backend.pearson_columns_par(
                memoryview(mat), n, p, anchor_index
            )
        return np.frombuffer(packed, dtype=np.float64).copy()

    # -- CPU fallback: multiprocessing or single-process NumPy -------------
    if n_jobs > 1 and size > _SMALL_THRESH:
        try:
            return _pearson_multiprocessing(matrix, anchor_index, chunk_size, n_jobs)
        except Exception:  # pragma: no cover
            pass  # multiprocessing may fail in notebooks/subprocesses

    return _pearson_numpy_chunked(matrix, anchor_index, chunk_size)


# ---------------------------------------------------------------------------
# Public API: column_variances
# ---------------------------------------------------------------------------

def column_variances(
    data,
    chunk_size: int = _DEFAULT_CHUNK,
    n_jobs: int = _N_JOBS,
) -> np.ndarray:
    """Per-column sample variance for feature pre-selection.

    Auto-dispatches to GPU, C extension, multiprocessing, or NumPy
    using the same hierarchy as pearson_columns.

    Parameters
    ----------
    data: array-like, shape (n_samples, n_features)
    chunk_size: int
    n_jobs: int

    Returns
    -------
    np.ndarray of shape (n_features,), float64
    """
    matrix = np.asarray(data, order="C")
    n, p = matrix.shape
    size = _n_elements(n, p)

    # -- GPU ---------------------------------------------------------------
    if gpu_available() and size > _SMALL_THRESH:  # pragma: no cover
        try:
            return _column_variances_gpu(matrix)
        except Exception:
            pass

    # -- C extension -------------------------------------------------------
    if _NATIVE_OK:  # pragma: no cover
        if matrix.dtype == np.float32 and hasattr(_native_backend, "column_variances_f32"):
            mat = _asf32_c(matrix)
            packed = _native_backend.column_variances_f32(memoryview(mat), n, p)
        else:
            mat = _asf64_c(matrix)
            packed = _native_backend.column_variances(memoryview(mat), n, p)
        return np.frombuffer(packed, dtype=np.float64).copy()

    # -- Multiprocessing ---------------------------------------------------
    if n_jobs > 1 and size > _SMALL_THRESH:
        try:
            return _variance_multiprocessing(matrix, chunk_size, n_jobs)
        except Exception:  # pragma: no cover
            pass

    # -- NumPy chunked -----------------------------------------------------
    col_means = matrix.mean(axis=0).astype(np.float64)
    variances = np.zeros(p, dtype=np.float64)
    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        chunk = matrix[:, start:end].astype(np.float64, copy=False)
        chunk_c = chunk - col_means[start:end]
        variances[start:end] = np.einsum("ij,ij->j", chunk_c, chunk_c) / (n - 1)
        del chunk_c
    return variances


# ---------------------------------------------------------------------------
# Public API: vip_scores
# ---------------------------------------------------------------------------

def vip_scores(
    t_scores: np.ndarray,
    x_weights: np.ndarray,
    y_loadings: np.ndarray,
) -> np.ndarray:
    """Vectorised VIP scores.

    VIP[i] = sqrt( p * sum_h( S[h] * (w[i,h]/||w[:,h]||)^2 ) / sum(S) )
    where S[h] = ||t[:,h]||^2 * q[h]^2.

    Parameters
    ----------
    t_scores   : (n_samples, n_components) float64
    x_weights  : (n_features, n_components) float64
    y_loadings : (n_components,) or (1, n_components) float64

    Returns
    -------
    np.ndarray of shape (n_features,), float64
    """
    t = np.asarray(t_scores,   dtype=np.float64, order="C")
    w = np.asarray(x_weights,  dtype=np.float64, order="C")
    q = np.asarray(y_loadings, dtype=np.float64).ravel()
    n_samples, n_comp = t.shape
    n_feat = w.shape[0]

    # -- GPU ---------------------------------------------------------------
    if gpu_available() and n_feat > 100_000:  # pragma: no cover
        try:
            return _vip_scores_gpu(t, w, q)
        except Exception:
            pass

    # -- C extension -------------------------------------------------------
    if _NATIVE_OK and hasattr(_native_backend, "vip_scores"):  # pragma: no cover
        q_c = np.ascontiguousarray(q)
        packed = _native_backend.vip_scores(
            memoryview(t), memoryview(w), memoryview(q_c),
            n_samples, n_feat, n_comp,
        )
        return np.frombuffer(packed, dtype=np.float64).copy()

    # -- NumPy vectorised (no Python loop) ---------------------------------
    S = np.einsum("ij,ij->j", t, t) * (q ** 2)   # (h,)
    total_s = S.sum()
    norms = np.linalg.norm(w, axis=0)              # (h,)
    norms[norms == 0.0] = 1.0
    w_norm = w / norms                             # (p, h)
    if total_s == 0.0:
        return np.zeros(n_feat, dtype=np.float64)
    return np.sqrt(n_feat * ((w_norm ** 2) @ S) / total_s)


# ---------------------------------------------------------------------------
# Public API: nipals  (NIPALS-PLS1 loop)
# ---------------------------------------------------------------------------

def nipals(
    x: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> tuple:
    """NIPALS-PLS1.  Returns (w, u, c, t).

    Dispatches to the C extension when available; falls back to pure NumPy.

    Parameters
    ----------
    x        : (n, p) float64
    y        : (n,)   float64
    tol      : convergence tolerance
    max_iter : maximum iterations

    Returns
    -------
    w : (p,) weights
    u : (n,) y-scores
    c : float y-weight
    t : (n,) x-scores
    """
    x_c = _asf64_c(x)
    y_c = np.ascontiguousarray(y, dtype=np.float64)
    n, p = x_c.shape

    # C NIPALS wins only for small matrices (n*p <= 50k) where BLAS call overhead
    # dominates. For larger matrices numpy BLAS DGEMV with SIMD is faster.
    if _NATIVE_OK and hasattr(_native_backend, "nipals_full") and n * p <= 50_000:  # pragma: no cover
        w_b, u_b, c, t_b, _ = _native_backend.nipals_full(
            memoryview(x_c), memoryview(y_c), n, p, tol, max_iter
        )
        w = np.frombuffer(w_b, dtype=np.float64).copy()
        u = np.frombuffer(u_b, dtype=np.float64).copy()
        t = np.frombuffer(t_b, dtype=np.float64).copy()
        return w, u, c, t

    # pure-NumPy fallback
    import numpy.linalg as _la
    u = y_c.copy()
    w = np.zeros(p)
    t = np.zeros(n)
    c = 0.0
    d = tol * 10.0 + 1.0
    for _ in range(max_iter):
        utu = np.dot(u, u)
        if utu < 1e-300:
            break
        w = x_c.T @ u / utu
        wnorm = _la.norm(w)
        if wnorm < 1e-300:
            break
        w /= wnorm
        t = x_c @ w
        ttt = np.dot(t, t)
        if ttt < 1e-300:
            break
        c = np.dot(t, y_c) / ttt
        if abs(c) < 1e-300:
            break
        u_new = y_c / c
        unorm = _la.norm(u_new)
        d = _la.norm(u_new - u) / unorm if unorm > 1e-300 else 0.0
        u = u_new
        if d <= tol:
            break
    return w, u, c, t


# ---------------------------------------------------------------------------
# Public API: scale_transform  (pareto / standard scaler)
# ---------------------------------------------------------------------------

def scale_transform(
    X: np.ndarray,
    mean: np.ndarray,
    s: np.ndarray,
) -> np.ndarray:
    """Element-wise (X - mean) / s.

    s is std for standard scaling or sqrt(std) for pareto scaling.
    Dispatches to the C extension when available; otherwise NumPy broadcast.

    Parameters
    ----------
    X    : (n, p) float64
    mean : (p,)   float64  column means
    s    : (p,)   float64  divisors (std or sqrt(std))

    Returns
    -------
    np.ndarray (n, p) float64
    """
    X_c    = _asf64_c(X)
    mean_c = np.ascontiguousarray(mean, dtype=np.float64)
    s_c    = np.ascontiguousarray(s,    dtype=np.float64)
    squeeze = X_c.ndim == 1
    if squeeze:
        X_c = X_c.reshape(1, -1)
    n, p   = X_c.shape

    if _NATIVE_OK and hasattr(_native_backend, "scale_transform"):  # pragma: no cover
        raw = _native_backend.scale_transform(
            memoryview(X_c), memoryview(mean_c), memoryview(s_c), n, p
        )
        result = np.frombuffer(raw, dtype=np.float64).reshape(n, p).copy()
        return result.squeeze(0) if squeeze else result

    # NumPy fallback – avoids large temporary via in-place ops on a copy
    out = X_c - mean_c
    inv_s = np.where(s_c != 0.0, 1.0 / s_c, 1.0)
    out *= inv_s
    return out.squeeze(0) if squeeze else out


# ---------------------------------------------------------------------------
# Public API: xcorr_max_shift  (icoshift alignment)
# ---------------------------------------------------------------------------

def xcorr_max_shift(
    template: np.ndarray,
    query: np.ndarray,
    max_shift: int,
) -> tuple:
    """Find the integer shift in [-max_shift, max_shift] maximising cross-correlation.

    Parameters
    ----------
    template  : (n,) float64 reference spectrum window
    query     : (n,) float64 sample spectrum window
    max_shift : int  half-width of the shift search range

    Returns
    -------
    (shift, corr) – best integer shift and its cross-correlation value
    """
    tmpl = np.ascontiguousarray(template, dtype=np.float64)
    qry  = np.ascontiguousarray(query,    dtype=np.float64)
    n    = len(tmpl)

    if _NATIVE_OK and hasattr(_native_backend, "xcorr_max_shift"):  # pragma: no cover
        return _native_backend.xcorr_max_shift(
            memoryview(tmpl), memoryview(qry), n, int(max_shift)
        )

    # NumPy fallback – direct correlation over all shifts
    best_shift, best_corr = 0, -1e300
    for sh in range(-max_shift, max_shift + 1):
        i_start = max(0, -sh)
        i_end   = min(n, n - sh)
        if i_end <= i_start:
            continue
        corr = float(np.dot(tmpl[i_start:i_end], qry[i_start + sh:i_end + sh]))
        if corr > best_corr:
            best_corr, best_shift = corr, sh
    return best_shift, best_corr


# ---------------------------------------------------------------------------
# Public API: pqn_median_quotient  (PQN normalisation)
# ---------------------------------------------------------------------------

def pqn_median_quotient(
    sample: np.ndarray,
    reference: np.ndarray,
) -> float:
    """Median of (sample / reference) over non-zero reference entries.

    Parameters
    ----------
    sample    : (n,) float64
    reference : (n,) float64

    Returns
    -------
    float  –  median quotient (1.0 if no valid entries)
    """
    samp = np.ascontiguousarray(sample,    dtype=np.float64)
    ref  = np.ascontiguousarray(reference, dtype=np.float64)
    n    = len(samp)

    if _NATIVE_OK and hasattr(_native_backend, "pqn_median_quotient"):  # pragma: no cover
        return _native_backend.pqn_median_quotient(memoryview(samp), memoryview(ref), n)

    # NumPy fallback
    mask = ref != 0.0
    if not mask.any():
        return 1.0
    return float(np.median(samp[mask] / ref[mask]))
