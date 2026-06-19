"""Dispatch helpers for optional native numerical kernels."""

from __future__ import annotations

import os

import numpy as np


_DISABLE_NATIVE = os.environ.get("METBIT_DISABLE_NATIVE", "").lower() in {
    "1",
    "true",
    "yes",
}

try:
    if _DISABLE_NATIVE:
        raise ImportError("native backend disabled by METBIT_DISABLE_NATIVE")
    from . import _native_backend
except ImportError:
    _native_backend = None


def native_available() -> bool:
    """Return whether the compiled extension is active."""
    return _native_backend is not None


def pearson_columns(data, anchor_index: int) -> np.ndarray:
    """Correlate one column of a 2D matrix against every matrix column."""
    matrix = np.asarray(data, dtype=np.float64, order="C")
    if matrix.ndim != 2:
        raise ValueError("data must be a two-dimensional matrix")
    rows, columns = matrix.shape
    if rows < 2 or columns < 1:
        raise ValueError("data must contain at least two rows and one column")
    if not 0 <= anchor_index < columns:
        raise IndexError("anchor_index is out of range")

    if _native_backend is not None:
        packed = _native_backend.pearson_columns(
            memoryview(matrix), rows, columns, anchor_index
        )
        return np.frombuffer(packed, dtype=np.float64).copy()

    anchor = matrix[:, anchor_index]
    anchor_centered = anchor - anchor.mean()
    centered = matrix - matrix.mean(axis=0)
    numerator = anchor_centered @ centered
    denominator = np.sqrt(
        np.dot(anchor_centered, anchor_centered)
        * np.einsum("ij,ij->j", centered, centered)
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        correlations = numerator / denominator
    return np.clip(correlations, -1.0, 1.0)
