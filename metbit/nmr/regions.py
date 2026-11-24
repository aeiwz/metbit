"""Spectral regions: ppm windows, binning, and keep / cut / suppress of regions.

Matches the Regions step of the metbit desktop app. Spectra are sample-by-feature
DataFrames whose columns are ppm values.
"""
import re
from typing import Iterable, Sequence, Union

import numpy as np
import pandas as pd

_WINDOW = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$")


def parse_windows(text: Union[str, Iterable[Sequence[float]], None]) -> list:
    """'3.2-2.8, 1.2-1.5' (or pairs) -> [(3.2, 2.8), (1.5, 1.2)] as (high, low) pairs."""
    if text is None:
        return []
    if not isinstance(text, str):
        return [(max(a, b), min(a, b)) for a, b in ((float(a), float(b)) for a, b in text)]
    out = []
    for part in filter(None, (s.strip() for s in text.split(","))):
        m = _WINDOW.match(part)
        if not m:
            raise ValueError(f"Cannot read window '{part}': use start-end, for example 3.2-2.8")
        a, b = float(m.group(1)), float(m.group(2))
        out.append((max(a, b), min(a, b)))
    return out


def in_windows(ppm: np.ndarray, windows) -> np.ndarray:
    """True for every ppm inside any (high, low) window."""
    mask = np.zeros(len(ppm), dtype=bool)
    for high, low in parse_windows(windows):
        mask |= (ppm <= high) & (ppm >= low)
    return mask


def bin_spectra(X: pd.DataFrame, width: float) -> pd.DataFrame:
    """Average columns into `width`-ppm bins labelled by bin centre, high ppm first. 0 returns X."""
    if width <= 0:
        return X
    centers = (np.floor(X.columns.to_numpy(dtype=float) / width) + 0.5) * width
    binned = X.T.groupby(np.round(centers, 6)).mean().T
    return binned[sorted(binned.columns, reverse=True)]


def _flatten(values: np.ndarray, inside: np.ndarray, to: str) -> np.ndarray:
    """Each run of `inside` columns set to 0, or to the straight line between the points just outside it."""
    out = values.copy()
    idx = np.flatnonzero(inside)
    if not len(idx):
        return out
    n = values.shape[1]
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if to == "zero":
            out[:, run] = 0.0
            continue
        left, right = max(run[0] - 1, 0), min(run[-1] + 1, n - 1)
        frac = (run - left) / max(right - left, 1)
        out[:, run] = values[:, [left]] + (values[:, [right]] - values[:, [left]]) * frac
    return out


def select_regions(X: pd.DataFrame, keep=None, cut=None, suppress=None,
                   suppress_to: str = "baseline") -> pd.DataFrame:
    """Keep one ppm range, remove `cut` windows, and flatten `suppress` windows.

    Windows are 'high-low' text or (a, b) pairs. Suppressed regions stay in the matrix
    and are set to zero or to the straight line joining their edges ("baseline").
    """
    if suppress_to not in ("baseline", "zero"):
        raise ValueError("suppress_to must be 'baseline' or 'zero'")
    ppm = X.columns.to_numpy(dtype=float)
    remove = in_windows(ppm, cut)
    keep_w = parse_windows(keep)
    if len(keep_w) > 1:
        raise ValueError("Give one range to keep, such as 0.5-9.5")
    if keep_w:
        remove |= ~in_windows(ppm, keep_w)
    if remove.all():
        raise ValueError("These settings remove every point: check the keep range and the cut regions")
    flat = _flatten(X.to_numpy(dtype=float), in_windows(ppm, suppress), suppress_to)
    return pd.DataFrame(flat[:, ~remove], index=X.index, columns=X.columns[~remove])


def nearest_column(X: pd.DataFrame, ppm: float) -> int:
    """Index of the column closest to `ppm`."""
    return int(np.abs(X.columns.to_numpy(dtype=float) - ppm).argmin())


__all__ = ["parse_windows", "in_windows", "bin_spectra", "select_regions", "nearest_column"]
