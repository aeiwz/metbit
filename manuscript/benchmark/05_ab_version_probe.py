#!/usr/bin/env python3
"""Run the same deterministic QA probe against a selected metbit source tree."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import json
import platform
import statistics
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def timed(repeats, function):
    durations = []
    value = None
    for _ in range(repeats):
        start = time.perf_counter()
        value = function()
        durations.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(durations),
        "min_s": min(durations),
        "max_s": max(durations),
        "repeats": repeats,
    }, value


def capture_case(name, function, results):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            results[name] = {"status": "pass", "result": function()}
    except Exception as exc:
        results[name] = {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(limit=5),
        }


def deterministic_spectra():
    rng = np.random.default_rng(20260619)
    ppm = np.linspace(0.0, 10.0, 1200)
    base = (
        5.0 * np.exp(-((ppm - 1.25) / 0.035) ** 2)
        + 3.0 * np.exp(-((ppm - 3.40) / 0.055) ** 2)
        + 2.0 * np.exp(-((ppm - 7.15) / 0.075) ** 2)
    )
    rows = []
    for index in range(24):
        scale = 0.75 + 0.03 * index
        rows.append(scale * base + rng.normal(0.0, 0.01, ppm.size))
    return pd.DataFrame(rows, columns=ppm)


def deterministic_classification():
    rng = np.random.default_rng(1701)
    x = rng.normal(size=(60, 80))
    y = np.array(["control"] * 30 + ["case"] * 30)
    x[30:, 8:14] += 0.9
    x[30:, 40:45] -= 0.6
    return pd.DataFrame(x, columns=np.linspace(0.0, 9.0, x.shape[1])), pd.Series(y)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fid-dir")
    args = parser.parse_args()

    import_start = time.perf_counter()
    import metbit
    import_duration = time.perf_counter() - import_start

    results = {
        "label": args.label,
        "reported_version": getattr(metbit, "__version__", "missing"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "import_time_s": import_duration,
        "cases": {},
    }

    capture_case(
        "public_api",
        lambda: {
            name: {
                "available": hasattr(metbit, name),
                "module": getattr(getattr(metbit, name, None), "__module__", None),
                "signature": str(inspect.signature(getattr(metbit, name)))
                if hasattr(metbit, name)
                else None,
            }
            for name in [
                "Normalise",
                "pca",
                "opls_da",
                "STOCSY",
                "nmr_preprocessing",
                "calibrate",
                "PeakAligner",
                "Normalization",
                "UnivarStats",
            ]
        },
        results["cases"],
    )

    def legacy_import_case():
        modules = {}
        for module_name in [
                "metbit.metbit",
                "metbit.utility",
                "metbit.nmr_preprocess",
                "metbit.alignment",
                "metbit.STOCSY",
                "metbit.spec_norm",
        ]:
            try:
                importlib.import_module(module_name)
                modules[module_name] = "available"
            except Exception as exc:
                modules[module_name] = f"{type(exc).__name__}: {exc}"
        return modules

    capture_case("legacy_imports", legacy_import_case, results["cases"])

    spectra = deterministic_spectra()

    def run_pqn():
        model = metbit.Normalise(spectra.copy(), compute_missing=True)
        output = model.pqn_normalise(plot=False)
        return output

    def pqn_case():
        timing, output = timed(7, run_pqn)
        return {
            "timing": timing,
            "shape": list(output.shape),
            "finite": bool(np.isfinite(output.to_numpy()).all()),
            "mean": float(output.to_numpy().mean()),
            "std": float(output.to_numpy().std()),
            "checksum": float(output.to_numpy().sum()),
        }

    capture_case("pqn", pqn_case, results["cases"])

    ppm = spectra.columns.to_numpy(dtype=float)
    shifted = spectra.iloc[:4].copy()
    shifted.iloc[1] = np.roll(shifted.iloc[1].to_numpy(), 2)
    shifted.iloc[2] = np.roll(shifted.iloc[2].to_numpy(), -3)
    windows = [(1.10, 1.40), (3.20, 3.60), (6.90, 7.40)]

    def alignment_case():
        def run():
            return metbit.PeakAligner(shifted.copy(), ppm, sf_mhz=600).align(
                windows, reference="median", max_shift_ppm=0.04
            )

        timing, (aligned, shifts) = timed(7, run)
        return {
            "timing": timing,
            "shape": list(aligned.shape),
            "checksum": float(aligned.to_numpy().sum()),
            "shifts": shifts,
        }

    capture_case("alignment", alignment_case, results["cases"])

    def calibration_case():
        calibrated = metbit.calibrate(
            spectra.iloc[:4].copy(), ppm, calib_type="custom",
            custom_range=(1.0, 1.5), custom_target=1.20,
        )
        return {
            "shape": list(calibrated.shape),
            "checksum": float(calibrated.to_numpy().sum()),
            "changed_from_input": bool(
                not np.allclose(calibrated.to_numpy(), spectra.iloc[:4].to_numpy())
            ),
        }

    capture_case("calibration", calibration_case, results["cases"])

    x_class, y_class = deterministic_classification()

    def fit_opls(auto_ncomp):
        model = metbit.opls_da(
            X=x_class,
            y=y_class,
            n_components=2,
            scaling_method="pareto",
            kfold=5,
            estimator="opls",
            random_state=42,
            auto_ncomp=auto_ncomp,
        )
        model.fit()
        return model

    def summarize_opls(auto_ncomp):
        timing, model = timed(5, lambda: fit_opls(auto_ncomp))
        model.vip_scores()
        scores = model.get_oplsda_scores()
        vip = model.get_vip_scores(filter_=False)
        return {
            "timing": timing,
            "R2Xcorr": float(model.R2Xcorr),
            "R2Y": float(model.R2y),
            "Q2": float(model.q2),
            "score_checksum": float(
                scores[["t_scores", "t_ortho", "t_pred"]].to_numpy().sum()
            ),
            "vip_checksum": float(vip["VIP"].sum()),
            "vip_max": float(vip["VIP"].max()),
        }

    capture_case(
        "opls_da_manual_components",
        lambda: summarize_opls(False),
        results["cases"],
    )
    capture_case(
        "opls_da_auto_components",
        lambda: summarize_opls(True),
        results["cases"],
    )

    def invalid_inputs_case():
        checks = {}
        for label, x_value, y_value in [
            ("missing", None, y_class),
            ("row_mismatch", x_class.iloc[:-1], y_class),
            ("multiclass", x_class, pd.Series(["a", "b", "c"] * 20)),
        ]:
            try:
                model = metbit.opls_da(X=x_value, y=y_value)
                if label == "multiclass":
                    model.fit()
                checks[label] = "accepted"
            except Exception as exc:
                checks[label] = type(exc).__name__
        return checks

    capture_case("invalid_inputs", invalid_inputs_case, results["cases"])

    if args.fid_dir:
        fid_dir = Path(args.fid_dir)

        def raw_preprocessing_case():
            def run(calibration):
                model = metbit.nmr_preprocessing(
                    str(fid_dir),
                    bin_size=0.0002,
                    auto_phasing=False,
                    baseline_correction=False,
                    calibration=calibration,
                    calib_type="tsp",
                    align=False,
                )
                return model.get_data()

            timing_on, calibrated = timed(1, lambda: run(True))
            timing_off, uncalibrated = timed(1, lambda: run(False))
            return {
                "calibration_on_timing": timing_on,
                "calibration_off_timing": timing_off,
                "shape": list(calibrated.shape),
                "calibration_changed_output": bool(
                    not np.allclose(
                        calibrated.to_numpy(), uncalibrated.to_numpy(), equal_nan=True
                    )
                ),
                "calibrated_checksum": float(calibrated.to_numpy().sum()),
                "uncalibrated_checksum": float(uncalibrated.to_numpy().sum()),
            }

        capture_case("raw_preprocessing", raw_preprocessing_case, results["cases"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(output)


if __name__ == "__main__":
    main()
