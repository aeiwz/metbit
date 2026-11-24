"""Peak picking and integral regions as in TopSpin, integrated in every sample.

Matches the Integrate step of the metbit desktop app: regions are found on a reference
(for example the median) spectrum, then every region is integrated by the trapezoidal rule.
"""
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from metbit.nmr.regions import parse_windows

MAX_PEAK_WIDTH = 0.05  # ppm at half height: broader is a hump, not a peak
EDGE_FRACTION = 0.01  # an integral ends where its peak falls to 1% of its height
MULTIPLET_VALLEY = 0.1  # touching regions merge when the valley stays above 10% of the smaller peak
MAX_AUTO_PEAKS = 500  # ponytail: the tallest 500; page through a ranked list if a spectrum needs more


def noise_level(y: np.ndarray) -> float:
    """Robust noise: 1.4826 MAD of point-to-point steps / sqrt(2), blind to peaks and offset.
    Zero steps (flattened or zero-filled stretches) carry no noise and are left out."""
    def mad(v):
        return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if len(v) else 0.0
    d = np.diff(np.asarray(y, dtype=float))
    return mad(d[d != 0]) / np.sqrt(2) or mad(np.asarray(y, dtype=float)) or 1.0


def _walk(side: np.ndarray, base: float, noise: float) -> int:
    """Points from a peak (side[0]) to its edge: where the signal falls to 1% of the peak height or
    2 noise levels, or the valley where it starts rising again."""
    floor = np.minimum.accumulate(side)
    stop = (side <= base + max(2 * noise, EDGE_FRACTION * (side[0] - base))) | (side > floor + 3 * noise)
    if not stop.any():
        return len(side) - 1
    k = int(np.argmax(stop))
    return k if side[k] <= floor[k] else int(np.argmin(side[: k + 1]))


def pick_peaks(ppm, y, snr: float = 10.0, at: Optional[list] = None):
    """Peaks `snr` noise levels above the baseline (or the given ppm positions `at`), and an integral
    region around each, merged so a multiplet's lines share one region.

    Returns (peak ppm list, [(high, low), ...] regions sorted from low to high ppm).
    """
    ppm, y = np.asarray(ppm, dtype=float), np.asarray(y, dtype=float)
    noise, base = noise_level(y), float(np.median(y))
    if at:
        idx = np.unique(np.abs(ppm[:, None] - np.asarray(at, dtype=float)[None, :]).argmin(axis=0))
    else:
        step = float(np.median(np.abs(np.diff(ppm)))) or 1.0
        idx, props = find_peaks(y, height=base + snr * noise, prominence=snr * noise,
                                width=(None, max(MAX_PEAK_WIDTH / step, 4)))
        if len(idx) > MAX_AUTO_PEAKS:
            idx = np.sort(idx[np.argsort(props["peak_heights"])[-MAX_AUTO_PEAKS:]])
    spans = [(i - _walk(y[i::-1], base, noise), i + _walk(y[i:], base, noise)) for i in idx]
    regions: list = []  # [left, right, smallest peak height above baseline]
    for (left, right), i in sorted(zip(spans, idx)):
        h = float(y[i] - base)
        prev = regions[-1] if regions else None
        if prev and (left < prev[1] or (left == prev[1] and y[left] - base > MULTIPLET_VALLEY * min(prev[2], h))):
            prev[1], prev[2] = max(prev[1], right), min(prev[2], h)
        else:
            regions.append([left, right, h])
    out = [(float(max(ppm[a], ppm[b])), float(min(ppm[a], ppm[b]))) for a, b, _ in regions]
    return [float(v) for v in ppm[idx]], sorted(out, key=lambda r: r[1])


def integrate_regions(X, ppm=None, windows=(), chord: bool = False) -> np.ndarray:
    """Trapezoidal area of each (high, low) window in each sample (samples x windows).

    chord=True subtracts the straight line joining each window's ends (peaks on a broad hump).
    X may be a DataFrame with ppm columns, in which case `ppm` can be omitted.
    """
    if ppm is None:
        ppm = X.columns.to_numpy(dtype=float)
    X, ppm = np.asarray(X, dtype=float), np.asarray(ppm, dtype=float)
    windows = parse_windows(windows)
    out = np.empty((len(X), len(windows)))
    for k, (hi, lo) in enumerate(windows):
        m = (ppm <= hi) & (ppm >= lo)
        if m.sum() < 2:
            raise ValueError(f"The integral {lo:.3f}-{hi:.3f} ppm holds fewer than two points: widen it")
        y = X[:, m]
        out[:, k] = np.sum((y[:, 1:] + y[:, :-1]) / 2 * np.abs(np.diff(ppm[m])), axis=1)
        if chord:
            out[:, k] -= (y[:, 0] + y[:, -1]) / 2 * abs(ppm[m][-1] - ppm[m][0])
    return out


def calibrate_integrals(areas: np.ndarray, scale: str = "raw", reference: int = 1, value: float = 1.0) -> np.ndarray:
    """Raw areas, or each row scaled so integral `reference` (1-based, scale='reference') or the
    row sum (scale='sum') equals `value`, e.g. 9 for the nine TSP protons."""
    areas = np.asarray(areas, dtype=float)
    if scale == "raw" or not areas.shape[1]:
        return areas
    if scale not in ("reference", "sum"):
        raise ValueError("scale must be 'raw', 'reference' or 'sum'")
    by = areas[:, reference - 1] if scale == "reference" else areas.sum(axis=1)
    if np.any(by == 0):
        raise ValueError("The reference integral is zero in a sample" if scale == "reference" else "The integrals add up to zero in a sample")
    return areas / by[:, None] * value


def integral_table(X: pd.DataFrame, windows=None, chord: bool = False, scale: str = "raw",
                   reference: int = 1, value: float = 1.0, snr: float = 10.0) -> pd.DataFrame:
    """Integrals of every sample as a feature table (samples x regions).

    Without `windows`, regions are picked automatically on the median spectrum.
    """
    ppm = X.columns.to_numpy(dtype=float)
    if windows is None:
        _, windows = pick_peaks(ppm, np.median(X.to_numpy(dtype=float), axis=0), snr=snr)
    windows = sorted(parse_windows(windows), key=lambda r: r[1])
    areas = calibrate_integrals(integrate_regions(X, ppm, windows, chord), scale, reference, value)
    return pd.DataFrame(areas, index=X.index, columns=[f"integral {lo:.3f}-{hi:.3f}" for hi, lo in windows])


__all__ = ["noise_level", "pick_peaks", "integrate_regions", "calibrate_integrals", "integral_table"]
