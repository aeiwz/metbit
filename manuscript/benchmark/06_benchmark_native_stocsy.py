#!/usr/bin/env python3
"""Benchmark the native STOCSY correlation kernel against scalar SciPy calls."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np
from scipy.stats import pearsonr

from metbit._native import native_available, pearson_columns


def measure(function, repeats):
    durations = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - start)
    return value, {
        "median_s": statistics.median(durations),
        "min_s": min(durations),
        "max_s": max(durations),
        "repeats": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=132)
    parser.add_argument("--variables", type=int, default=50_029)
    parser.add_argument("--native-repeats", type=int, default=7)
    parser.add_argument("--scipy-repeats", type=int, default=1)
    args = parser.parse_args()

    rng = np.random.default_rng(20260619)
    matrix = rng.normal(size=(args.samples, args.variables))
    anchor_index = args.variables // 3
    matrix[:, anchor_index + 1] = (
        matrix[:, anchor_index] * 0.8
        + rng.normal(scale=0.2, size=args.samples)
    )

    optimized_result, optimized_timing = measure(
        lambda: pearson_columns(matrix, anchor_index),
        args.native_repeats,
    )
    scipy_result, scipy_timing = measure(
        lambda: np.array([
            pearsonr(matrix[:, anchor_index], matrix[:, column]).statistic
            for column in range(matrix.shape[1])
        ]),
        args.scipy_repeats,
    )

    np.testing.assert_allclose(
        optimized_result, scipy_result, rtol=1e-12, atol=1e-12
    )
    result = {
        "native_available": native_available(),
        "shape": [args.samples, args.variables],
        "optimized": optimized_timing,
        "scalar_scipy": scipy_timing,
        "speedup": (
            scipy_timing["median_s"] / optimized_timing["median_s"]
        ),
        "max_abs_error": float(
            np.max(np.abs(optimized_result - scipy_result))
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
