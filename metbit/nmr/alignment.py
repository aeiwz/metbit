"""
Automated NMR peak alignment with basic multiplet detection.

Provides:
- detect_multiplets: detect and classify multiplets on a reference spectrum
- icoshift_align: interval-correlation-optimized shifting against a reference
- PeakAligner: convenience wrapper combining detection + alignment

Notes:
- Classification uses simple spacing and height ratio heuristics; intended to
  identify singlet, doublet, triplet, quartet; otherwise 'multiplet'.
- Alignment uses integer-point shifts per window (fast and robust). For
  sub-point precision, interpolate before/after alignment (not included here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks


def _binomial_ratios(n: int) -> np.ndarray:
    """Return normalized binomial coefficient ratios for n peaks (n-1 order)."""
    from math import comb
    k = n - 1
    coeffs = np.array([comb(k, i) for i in range(n)], dtype=float)
    return coeffs / coeffs.max()


def _classify_pattern(centers_ppm: np.ndarray, heights: np.ndarray, sf_mhz: float) -> str:
    """Classify a cluster by peak count, equal spacing, and height ratios.

    - singlet: N=1
    - doublet: N=2, similar heights, consistent spacing (trivial)
    - triplet: N=3, roughly equal spacing and ~1:2:1 heights
    - quartet: N=4, equal spacing and ~1:3:3:1 heights
    - else: 'multiplet'
    """
    n = len(centers_ppm)
    if n == 0:
        return "unknown"
    if n == 1:
        return "singlet"

    # equal spacing check in Hz to be instrument independent
    centers_hz = np.array(centers_ppm) * sf_mhz  # ppm * MHz ~ Hz offset (relative spacing)
    diffs = np.diff(np.sort(centers_hz))
    if len(diffs) > 0:
        rel_var = np.std(diffs) / (np.mean(diffs) + 1e-12)
    else:
        rel_var = 0.0

    # normalize heights and compare to binomial expectations
    h = np.array(heights, dtype=float)
    if h.max() > 0:
        h_norm = h / h.max()
    else:
        h_norm = h

    def close(a, b, tol=0.35):
        return np.mean(np.abs(a - b)) <= tol

    if n == 2 and rel_var < 0.25 and close(h_norm, np.array([1.0, 1.0])):
        return "doublet"
    if n == 3 and rel_var < 0.25 and close(h_norm, _binomial_ratios(3)):
        return "triplet"
    if n == 4 and rel_var < 0.25 and close(h_norm, _binomial_ratios(4)):
        return "quartet"
    return "multiplet"


@dataclass
class Multiplet:
    start_ppm: float
    end_ppm: float
    center_ppm: float
    n_peaks: int
    pattern: str


def detect_multiplets(
    spectrum: pd.Series,
    ppm: np.ndarray,
    sf_mhz: float,
    smooth_window: int = 11,
    smooth_poly: int = 2,
    prominence: float = 0.01,
    width: int = 3,
    max_group_width_ppm: float = 0.03,
) -> List[Multiplet]:
    """Detect and classify multiplets on a single spectrum.

    Parameters
    - spectrum: 1D intensities indexed like ppm
    - ppm: ppm axis as float array, same length as spectrum
    - sf_mhz: spectrometer frequency in MHz (e.g., 600 for 600 MHz)
    - smooth_window, smooth_poly: Savitzky-Golay smoothing params
    - prominence, width: find_peaks parameters (tune to data scale)
    - max_group_width_ppm: group peaks within this span as a multiplet
    """
    y = np.asarray(spectrum, dtype=float)
    if smooth_window and smooth_window > 2:
        y = savgol_filter(y, window_length=smooth_window, polyorder=smooth_poly)

    # detect candidate peaks
    peaks, props = find_peaks(y, prominence=prominence, width=width)
    if len(peaks) == 0:
        return []
    pk_ppm = ppm[peaks]
    pk_h = y[peaks]

    # group peaks into multiplets by proximity on ppm
    order = np.argsort(pk_ppm)
    pk_ppm = pk_ppm[order]
    pk_h = pk_h[order]

    groups: List[List[int]] = []
    current = [0]
    for i in range(1, len(pk_ppm)):
        if abs(pk_ppm[i] - pk_ppm[i - 1]) <= max_group_width_ppm:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)

    # build multiplets
    out: List[Multiplet] = []
    for idxs in groups:
        centers = pk_ppm[idxs]
        heights = pk_h[idxs]
        pattern = _classify_pattern(centers, heights, sf_mhz)
        start_ppm = float(centers.min())
        end_ppm = float(centers.max())
        center_ppm = float(np.average(centers, weights=heights))
        out.append(Multiplet(start_ppm, end_ppm, center_ppm, len(idxs), pattern))
    return out


def icoshift_align(
    spectra: pd.DataFrame,
    ppm: np.ndarray,
    windows: List[Tuple[float, float]],
    reference: str = 'median',
    max_shift_ppm: float = 0.02,
) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
    """Interval-correlation optimized shifting (icoshift-like).

    Parameters
    - spectra: rows=spectra, cols=ppm
    - ppm: ppm vector matching columns (ascending or descending OK)
    - windows: list of (ppm_min, ppm_max) intervals to align independently
    - reference: 'median' or 'mean' for the reference spectrum
    - max_shift_ppm: max allowed shift magnitude within each window

    Returns
    - aligned spectra (DataFrame)
    - per-sample list of applied integer-point shifts for each window
    """
    X = spectra.copy()
    cols = np.asarray(spectra.columns.astype(float))
    ppm_vec = np.asarray(ppm, dtype=float)
    if not np.allclose(cols, ppm_vec):
        # coerce columns to provided ppm for safety
        X.columns = ppm_vec

    # build reference
    ref = np.median(X.values, axis=0) if reference == 'median' else X.values.mean(axis=0)

    aligned = X.values.copy()
    shifts: Dict[str, List[int]] = {str(i): [] for i in X.index}

    for wmin, wmax in windows:
        # map ppm interval to indices
        if ppm_vec[0] > ppm_vec[-1]:
            # descending axis
            mask = (ppm_vec <= wmin) & (ppm_vec >= wmax)
        else:
            mask = (ppm_vec >= wmin) & (ppm_vec <= wmax)
        idx = np.where(mask)[0]
        if idx.size < 5:
            # skip too narrow window
            for sid in shifts:
                shifts[sid].append(0)
            continue

        max_pts = max(1, int(round(abs(max_shift_ppm) / (abs(ppm_vec[1] - ppm_vec[0]) + 1e-12))))
        ref_seg = ref[idx]

        # pre-normalize each segment (zero-mean, unit-norm) to emphasize shape
        ref_z = ref_seg - ref_seg.mean()
        ref_norm = np.linalg.norm(ref_z) + 1e-12

        for r, sid in enumerate(X.index):
            y = aligned[r, idx]
            best_shift = 0
            best_corr = -np.inf
            y_z = y - y.mean()
            denom = (np.linalg.norm(y_z) + 1e-12) * ref_norm
            # try integer shifts in [-max_pts, max_pts]
            for s in range(-max_pts, max_pts + 1):
                if s == 0:
                    corr = float(np.dot(y_z, ref_z) / denom)
                elif s > 0:
                    corr = float(np.dot(y_z[s:], ref_z[:-s]) / ((np.linalg.norm(y_z[s:]) + 1e-12) * (np.linalg.norm(ref_z[:-s]) + 1e-12)))
                else:
                    s_ = -s
                    corr = float(np.dot(y_z[:-s_], ref_z[s_:]) / ((np.linalg.norm(y_z[:-s_]) + 1e-12) * (np.linalg.norm(ref_z[s_:]) + 1e-12)))
                if corr > best_corr:
                    best_corr = corr
                    best_shift = s
            # apply best shift to the aligned matrix within this window
            if best_shift != 0:
                seg = aligned[r, idx]
                aligned[r, idx] = np.roll(seg, best_shift)
            shifts[str(sid)].append(int(best_shift))

    aligned_df = pd.DataFrame(aligned, index=X.index, columns=ppm_vec)
    return aligned_df, shifts


class PeakAligner:
    """High-level helper to detect multiplets and align spectra around them.

    Typical use:
        pa = PeakAligner(spectra, ppm, sf_mhz=600)
        windows, mptable = pa.auto_windows(top_n=30)
        X_aligned, shifts = pa.align(windows)
    """

    def __init__(self, spectra: pd.DataFrame, ppm: np.ndarray, sf_mhz: float) -> None:
        self.spectra = spectra
        self.ppm = np.asarray(ppm, dtype=float)
        self.sf_mhz = float(sf_mhz)

    def auto_windows(self, top_n: int = 30, max_group_width_ppm: float = 0.03) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
        """Generate non-overlapping windows centered on strongest multiplets of the median spectrum."""
        ref = self.spectra.median(axis=0)
        mps = detect_multiplets(ref, self.ppm, self.sf_mhz, max_group_width_ppm=max_group_width_ppm)
        # score by integrated intensity within group
        windows: List[Tuple[float, float]] = []
        rows = []
        for mp in mps:
            i0 = np.argmin(np.abs(self.ppm - mp.start_ppm))
            i1 = np.argmin(np.abs(self.ppm - mp.end_ppm))
            if i0 > i1:
                i0, i1 = i1, i0
            area = float(np.trapz(ref.iloc[i0:i1+1], self.ppm[i0:i1+1]))
            rows.append({**mp.__dict__, 'area': area})
        df = pd.DataFrame(rows).sort_values('area', ascending=False).head(top_n)
        # expand a bit to capture full shape
        for _, r in df.iterrows():
            span = max(0.005, (r['end_ppm'] - r['start_ppm']) * 1.5)
            windows.append((r['center_ppm'] - span/2, r['center_ppm'] + span/2))
        # deduplicate/merge overlapping windows
        windows = sorted(windows, key=lambda w: w[0])
        merged: List[Tuple[float, float]] = []
        for w in windows:
            if not merged or w[0] > merged[-1][1]:
                merged.append(list(w))
            else:
                merged[-1][1] = max(merged[-1][1], w[1])
        return [(float(a), float(b)) for a, b in merged], df

    def align(self, windows: List[Tuple[float, float]], reference: str = 'median', max_shift_ppm: float = 0.02) -> Tuple[pd.DataFrame, Dict[str, List[int]]]:
        return icoshift_align(self.spectra, self.ppm, windows, reference=reference, max_shift_ppm=max_shift_ppm)

