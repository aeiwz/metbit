# metbit

[![PyPI version](https://img.shields.io/pypi/v/metbit?color=green&style=for-the-badge)](https://pypi.org/project/metbit/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/metbit?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/metbit)
[![Python](https://img.shields.io/pypi/pyversions/metbit?style=for-the-badge)](https://pypi.org/project/metbit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Open Documentation](https://img.shields.io/badge/Docs-metbit--docs.vercel.app-2563EB?style=for-the-badge&logo=readthedocs&logoColor=white)](https://metbit-docs.vercel.app)

An open-source Python package for reproducible 1H NMR metabolomics - from raw FID preprocessing through normalization, chemometrics, and interactive visualization, in a single scriptable workflow.

metbit v9.0.0 adds a four-tier auto-dispatch compute backend (GPU - C+OpenMP - multiprocessing - chunked NumPy), a native C extension with optional OpenMP parallelism, memory-bounded algorithms for biobank-scale cohorts, and a 305-test validation suite.

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [NMR Preprocessing](#nmr-preprocessing)
- [Normalization and Alignment](#normalization-and-alignment)
- [Principal Component Analysis](#principal-component-analysis)
- [OPLS-DA](#opls-da)
- [STOCSY](#stocsy)
- [Large-Scale Compute Backend](#large-scale-compute-backend)
- [Interactive Dash Applications](#interactive-dash-applications)
- [Performance](#performance)
- [Testing](#testing)
- [Citation](#citation)

---

## Installation

```bash
pip install metbit
```

The C extension (VIP kernel, Pearson correlation, column variance) is compiled automatically at install time when a C compiler is available. OpenMP parallelism is enabled when the compiler supports `-fopenmp`.

```bash
# Verify the active compute backend after installation
python -c "import metbit; print(metbit.backend_info())"
# {'native_c': True, 'openmp_threads': 8, 'gpu_cupy': False, 'gpu_torch': False, ...}
```

**Optional GPU acceleration** - install CuPy or PyTorch with CUDA support:

```bash
pip install cupy-cuda12x   # or torch with CUDA
```

---

## Quick Start

```python
import pandas as pd
from metbit import nmr_preprocessing, pca, opls_da, STOCSY

# 1. Load a preprocessed spectral matrix (samples x variables)
df = pd.read_csv("nmr_data.csv")
X = df.iloc[:, 2:]          # spectral matrix
y = df["Group"]             # binary class label
ppm = X.columns.astype(float).tolist()

# 2. PCA - exploratory overview
pca_mod = pca(X=X, label=y, features_name=ppm, n_components=3)
pca_mod.fit()
pca_mod.plot_pca_scores(pc=["PC1", "PC2"])

# 3. OPLS-DA - supervised discrimination
model = opls_da(X=X, y=y, features_name=ppm, scale="uv", auto_ncomp=True)
model.fit()
model.plot_oplsda_scores()
model.vip_scores()
model.vip_plot(threshold=2)

# 4. STOCSY - correlation from a VIP anchor peak
stocsy = STOCSY(X=X, ppm=ppm)
stocsy.fit(driver=3.05)
stocsy.plot()
```

---

## Workflow Overview

```
Raw Bruker FID
      │
      ▼
nmr_preprocessing          # digital filter removal, zero-fill, FFT,
      │                    # ACME phase correction, arPLS baseline correction,
      │                    # TSP/DSS calibration, region exclusion
      ▼
Normalization              # PQN / MSC / TSP-area
      │
      ▼
icoshift_align             # segment-wise spectral alignment
      │
      ▼
pca                        # exploratory variance overview
      │
      ▼
opls_da                    # supervised binary classification
      │                    # VIP scoring, permutation test, CV-ANOVA
      ▼
STOCSY / ChunkedSTOCSY     # structural correlation from anchor peaks
      │
      ▼
Interactive Dash apps      # local browser exploration, no data upload
```

---

## NMR Preprocessing

```python
from metbit import nmr_preprocessing

# Process a folder of Bruker FID experiments
proc = nmr_preprocessing(
    path="path/to/bruker/experiments",
    ppm_range=(-0.5, 10.0),
    water_region=(4.7, 4.9),
    reference_peak=0.0,   # TSP/DSS at 0 ppm
)
spectra_df = proc.process()   # returns samples x ppm DataFrame
```

The preprocessing pipeline applies, in order:

| Step | Method |
|------|--------|
| Digital filter removal | FID truncation at group delay |
| Zero filling | Next power of two |
| Fourier transform | FFT with apodization |
| Phase correction | ACME entropy minimization (Chen et al., 2002) |
| Baseline correction | arPLS (Baek et al., 2015) |
| Chemical shift calibration | TSP/DSS reference at 0 ppm |
| Region exclusion | Water, solvent, user-defined windows |

---

## Normalization and Alignment

```python
from metbit import Normalise, icoshift_align

# Probabilistic Quotient Normalization
normalised = Normalise(spectra_df).pqn()

# Multiplicative Scatter Correction
normalised = Normalise(spectra_df).msc()

# Icoshift spectral alignment
aligned = icoshift_align(normalised, n_intervals="whole")
```

v9.0.0: `icoshift_align` now allocates a single output array instead of two full-matrix copies, halving peak memory at large cohort sizes.

---

## Principal Component Analysis

```python
from metbit import pca

X = df.iloc[:, 2:]
ppm = X.columns.astype(float).tolist()
color_ = df["Group"]
symbol_ = df["Time point"]
time_order = {1: 0, 2: 1, 3: 2, 4: 3}

pca_mod = pca(X=X, label=color_, features_name=ppm, n_components=3)
pca_mod.fit()

pca_mod.plot_cumulative_observed()
pca_mod.plot_pca_scores(pc=["PC1", "PC2"], symbol_=symbol_)
pca_mod.plot_pca_scores(pc=["PC1", "PC3"], symbol_=symbol_)
pca_mod.plot_3d_pca(marker_size=10, symbol_=symbol_)
pca_mod.plot_pca_trajectory(time_=symbol_, time_order=time_order, pc=["PC1", "PC2"])
```

All plots are returned as interactive Plotly HTML figures.

---

## OPLS-DA

```python
from metbit import opls_da

model = opls_da(
    X=X,
    y=y,
    features_name=ppm,
    scale="uv",          # unit-variance scaling
    auto_ncomp=True,     # automatic component selection via cross-validation
)
model.fit()
```

### Scores and loadings

```python
model.plot_oplsda_scores()   # predictive vs orthogonal scores
model.plot_loading()         # loading plot (color by correlation)
model.plot_s_scores()        # S-plot (covariance vs correlation)
```

### Model validation

```python
# Permutation test
model.permutation_test(n_permutations=1000, n_jobs=-1)
model.plot_hist()            # R2 / Q2 null distribution

# VIP scores - identifies discriminating variables
model.vip_scores()
model.vip_plot(threshold=2)  # highlight features with VIP > threshold
```

### Model metrics

```python
print(model.R2Y)    # variance explained in Y
print(model.Q2)     # cross-validated predictive ability
print(model.AUROC)  # area under the ROC curve
```

### Large-cohort option (v9.0.0)

```python
# Halve peak memory with float32 (auto-selected when n*p > 5,000,000)
model = opls_da(X=X, y=y, features_name=ppm, scale="uv", dtype="float32")
```

The `vip_scores()` method in v9.0.0 dispatches to the native C kernel, replacing the previous O(p) Python loop over features (up to 2,680x faster at large feature counts).

---

## STOCSY

Statistical Total Correlation Spectroscopy identifies structurally related resonances by correlating every spectral variable against an anchor peak.

```python
from metbit import STOCSY

stocsy = STOCSY(X=spectra_df, ppm=ppm)
stocsy.fit(driver=3.05)   # anchor ppm - typically a high-VIP peak
stocsy.plot()             # interactive spectrum colored by Pearson r
```

### ChunkedSTOCSY - for large feature counts (v9.0.0)

`ChunkedSTOCSY` processes correlations in feature batches, bounding peak memory to O(n × chunk_size) regardless of total feature count. Suitable for datasets exceeding available RAM.

```python
from metbit import ChunkedSTOCSY

cs = ChunkedSTOCSY(X=spectra_df, ppm=ppm, chunk_size=10_000)
cs.fit(driver=3.05)
cs.plot()

# Inspect which backend is active
print(ChunkedSTOCSY.active_backend())
```

---

## Large-Scale Compute Backend

v9.0.0 introduces a four-tier auto-dispatch compute backend. The backend is selected automatically based on dataset size and available hardware - no code changes required.

```
GPU (CuPy / PyTorch CUDA)     >500 M elements, GPU memory fits
        ↓
C + OpenMP                     >10 M elements, C extension compiled
        ↓
Multiprocessing + NumPy        C extension absent, many cores available
        ↓
Chunked NumPy                  universal fallback, bounded memory
```

```python
import metbit

# Inspect active backends
print(metbit.backend_info())
# {'native_c': True, 'openmp_threads': 8, 'gpu_cupy': False, 'gpu_torch': False,
#  'n_jobs': 8, 'default_chunk': 50000}

print(metbit.native_available())   # True if C extension compiled
print(metbit.gpu_available())      # True if CuPy or PyTorch+CUDA present
```

### Environment overrides

| Variable | Effect |
|----------|--------|
| `METBIT_DISABLE_NATIVE=1` | Skip the C extension, use NumPy paths |
| `METBIT_DISABLE_GPU=1` | Skip CuPy and PyTorch GPU backends |
| `METBIT_N_JOBS=N` | Override worker count for multiprocessing |
| `METBIT_CHUNK=N` | Override feature chunk size (default 50,000) |

### Memory estimation

```python
from metbit import MemoryEstimator, memory_report

# Estimate peak RAM before loading
est = MemoryEstimator()
info = est.estimate(n_samples=10_000, n_features=65_536, dtype="float64")
print(info)
# {'peak_gb': 4.88, 'recommended_dtype': 'float32', 'float32_peak_gb': 2.44}

# Quick one-liner for a DataFrame
memory_report(X)
```

### Variance-based feature pre-selection

Reduces feature count before expensive downstream modeling by removing low-variance spectral bins (typically instrument noise).

```python
from metbit import feature_preselection

X_reduced = feature_preselection(
    X,
    threshold_percentile=20,  # remove bottom 20% by variance
)
```

### LargeScaleAlignment

```python
from metbit import LargeScaleAlignment

aligner = LargeScaleAlignment(memory_limit_gb=8.0)
aligned = aligner.fit_transform(spectra_df)
```

---

## Interactive Dash Applications

Four local Dash applications ship with metbit for browser-based exploration. All run locally - no data is uploaded to external servers.

```python
from metbit.apps import stocsy_app, opls_app, pca_app, spectra_app

# Launch STOCSY explorer
stocsy_app(X=spectra_df, ppm=ppm, port=8050)

# Launch OPLS-DA explorer
opls_app(model=model, port=8051)
```

Apps operate on the same Python objects used in scripted analysis, so results are always consistent with the command-line workflow.

---

## Performance

Benchmarks measured on Apple M5 Pro, single-threaded C extension, no GPU (minimum of 5 wall-clock repetitions, one warm-up discarded).

| Kernel | Dataset (n x p) | Implementation | Speedup | Memory reduction |
|--------|----------------|----------------|---------|-----------------|
| VIP scores | 80 x 5,000 | C vs Python loop | **2,680x** | - |
| VIP scores | 80 x 5,000 | NumPy vec. vs Python loop | 817x | - |
| Pearson correlation | 500 x 30,000 | C vs full-copy NumPy | 1.2x | **126x** |
| Column variance | 500 x 100,000 | C float32 vs NumPy | 2.0x | **251x** |
| ChunkedSTOCSY | 100 x 5,000 | Chunked vs standard | **47x** | bounded |

Reproduce these numbers locally:

```bash
python scripts/perf_report.py
# Writes reports/benchmark_results.json and reports/PERFORMANCE.md
```

---

## Testing

metbit ships with a 305-test suite organized into four collections:

| Collection | Tests | Covers |
|------------|-------|--------|
| `test_e2e_pipeline.py` | 44 | Full pipeline: preprocessing - OPLS-DA - STOCSY - alignment |
| `test_ab_aa.py` | 22 | Statistical validity: AB (must discriminate), AA (must not discriminate on noise) |
| `test_large_scale.py` | 23 | Backend dispatch, memory efficiency via `tracemalloc` |
| `test_performance.py` | 24 | Speedup ratios and performance regression guards |

```bash
# Run full suite
pytest

# Run performance benchmarks
pytest -m perf

# Run without performance tests (faster)
pytest -m "not perf"
```

---

## Links

- Documentation: https://metbit-docs.vercel.app
- Getting Started: https://metbit-docs.vercel.app/docs/getting-started
- API Reference: https://metbit-docs.vercel.app/docs/api
- PyPI: https://pypi.org/project/metbit/
- Source: https://github.com/aeiwz/metbit
- Changelog: [CHANGELOG.md](CHANGELOG.md)

---

## License

MIT License. See [LICENSE](LICENSE) for details.
