# metbit v8.7.7 vs 9.0.0-dev QA Report

Test date: 2026-06-19

## Scope

The comparison used the `v8.7.7` Git tag as variant A and the current working
tree at commit `8d94c9488c92e5d6433edb623d58193d1de3233f` as variant B. The B
variant reports version `9.0.0-dev` and includes the current uncommitted NumPy
compatibility fixes in `opls_da.py` and `alignment.py`.

Both variants ran in the same isolated environment:

- Python 3.11.15
- NumPy 2.1.3
- pandas 2.2.3
- SciPy 1.14.1
- scikit-learn 1.5.2
- Identical random seeds, data, parameters, and hardware

The reusable probe is `manuscript/benchmark/05_ab_version_probe.py`.

## Summary

| Area | v8.7.7 | 9.0.0-dev | Assessment |
|---|---:|---:|---|
| Package tests | 23 passed | 211 passed | B has substantially broader coverage |
| Wheel build and install | Pass | Pass | Parity |
| Top-level public imports | Pass | Pass | Parity |
| Legacy module imports | 6/6 pass | 0/6 pass | Breaking change in B |
| PQN numerical output | Pass | Pass | Exact parity |
| Calibration function output | Pass | Pass | Exact parity |
| Alignment output | Pass | Pass | Exact parity |
| OPLS automatic selection | Pass | Pass | Exact parity |
| OPLS manual selection | Incorrect component index | Corrected | B fixes v8 off-by-one defect |
| Raw preprocessing | Pass with defect | Pass with same defect | Calibration result discarded |

## Numerical Results

PQN, calibration, alignment, VIP scores, and automatically selected OPLS
statistics were identical to floating-point precision.

For manual two-component OPLS, outputs differed:

| Metric | v8.7.7 | 9.0.0-dev |
|---|---:|---:|
| R2Y | 0.979887 | 0.957116 |
| Q2 | 0.475898 | 0.471956 |
| R2Xcorr | 0.980458 | 0.985440 |

This is expected after correcting `CrossValidation.reset_optimal_num_component`.
Version 8.7.7 stored `k` as a zero-based array index and therefore selected
component `k + 1`. Version 9 stores `k - 1`, so requesting two components now
selects two components.

Median timings showed no material regression:

| Operation | v8.7.7 | 9.0.0-dev |
|---|---:|---:|
| Import | 0.697 s | 0.571 s |
| PQN, 24 x 1,200 | 0.00966 s | 0.00981 s |
| OPLS automatic selection | 0.02229 s | 0.02231 s |
| Raw processing, four FIDs | 1.043 s | 1.039 s |

The alignment probe completed below one millisecond, so its percentage
difference is dominated by timer noise and is not interpreted.

## Findings

### High: legacy imports break in 9.0.0-dev

The following v8 imports fail with `ModuleNotFoundError`:

- `metbit.metbit`
- `metbit.utility`
- `metbit.nmr_preprocess`
- `metbit.alignment`
- `metbit.STOCSY`
- `metbit.spec_norm`

Top-level imports such as `from metbit import opls_da` remain functional.
Provide compatibility shim modules or document 9.0 as a breaking major release
with a migration table.

### High: raw preprocessing discards calibration in both versions

Enabling and disabling calibration produced byte-equivalent numerical output.
The preprocessing class assigns the calibrated matrix and then overwrites
`self.nmr_data` with the original matrix. Results described as calibrated
cannot be attributed to the calibration operation until this is fixed and the
benchmark is rerun.

### Important: path names containing `fid` are corrupted

Both versions remove the text `fid` from the complete discovered path. A parent
directory named `metbit-ab-fids` was converted to `metbit-ab-s`, causing file
loading to fail. Path handling should remove only the terminal `/fid` component.

### Important: dependency compatibility is insufficiently constrained

Under the machine's Python 3.14 environment, v8.7.7 failed during import through
the eager `lingress`/`statsmodels` dependency chain. Version 9 imported only
after disabling a Numba cache problem triggered by `pybaselines`. Neither setup
defines `python_requires`, and dependency ranges are largely unconstrained.

### Important: v8.7.7 omits a required dependency

The v8 package imports `dash_bootstrap_components` eagerly, but it is absent
from `install_requires`. A clean environment needed this additional package
before v8 could import and run its tests.

## Release Confidence

Confidence in 9.0 core matrix operations is moderate: numerical parity is strong,
automatic OPLS output is unchanged, manual component selection is corrected,
and test coverage has expanded substantially.

Confidence in a drop-in upgrade is low because legacy module imports all break.
Confidence in raw-FID claims is low until calibration persistence and path
handling are corrected and covered by regression tests.
