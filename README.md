# metbit

[![PyPI version](https://img.shields.io/pypi/v/metbit?color=green&style=for-the-badge)](https://pypi.org/project/metbit/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/metbit?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/metbit)
[![Python](https://img.shields.io/pypi/pyversions/metbit?style=for-the-badge)](https://pypi.org/project/metbit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Open Documentation](https://img.shields.io/badge/Docs-metbit--docs.vercel.app-2563EB?style=for-the-badge&logo=readthedocs&logoColor=white)](https://metbit-docs.vercel.app)

An open-source Python package for reproducible <sup>1</sup>H NMR metabolomics - from raw FID preprocessing through normalization, chemometrics, and interactive visualization, in a single scriptable workflow.

metbit v9.1.0 extends the v9.0.0 compute backend with a full machine-learning and deep-learning layer, extended multivariate analysis (LDA, PLSR, ICA, HCA), volcano-plot and ANOVA/Kruskal statistics, flexible train/test and cross-validation utilities, and an interactive spectra viewer - all behind the same four-tier auto-dispatch backend (GPU - C+OpenMP - multiprocessing - chunked NumPy).

---

## Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Workflow Overview](#workflow-overview)
- [NMR Preprocessing](#nmr-preprocessing)
- [Normalization and Alignment](#normalization-and-alignment)
- [Principal Component Analysis](#principal-component-analysis)
- [OPLS-DA](#opls-da)
- [Extended Multivariate Analysis](#extended-multivariate-analysis)
- [Statistical Testing](#statistical-testing)
- [Machine Learning Classifiers](#machine-learning-classifiers)
- [Deep Learning Models](#deep-learning-models)
- [Train/Test and Cross-Validation](#traintest-and-cross-validation)
- [Spectra Visualization](#spectra-visualization)
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

**Optional extras:**

```bash
pip install metbit[ml]   # XGBoost classifier support
pip install metbit[dl]   # PyTorch deep-learning models
pip install metbit[all]  # both of the above
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
pca / lda / plsr / ica / hca    # exploratory and supervised multivariate
      │
      ▼
opls_da                    # supervised binary classification
      │                    # VIP scoring, permutation test, CV-ANOVA
      ▼
VolcanoPlot / ANOVAStats / KruskalStats   # group comparison statistics
      │
      ▼
MLClassifier               # RF / SVM / XGBoost / ElasticNet with CV
SpectralAutoencoder        # PyTorch deep-learning models
      │
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

---

## Principal Component Analysis

```python
from metbit import pca

pca_mod = pca(X=X, label=y, features_name=ppm, n_components=3)
pca_mod.fit()

pca_mod.plot_cumulative_observed()
pca_mod.plot_pca_scores(pc=["PC1", "PC2"], symbol_=symbol_)
pca_mod.plot_3d_pca(marker_size=10, symbol_=symbol_)
pca_mod.plot_pca_trajectory(time_=symbol_, time_order=time_order, pc=["PC1", "PC2"])
```

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
model.permutation_test(n_permutations=1000, n_jobs=-1)
model.plot_hist()            # R2 / Q2 null distribution

model.vip_scores()
model.vip_plot(threshold=2)
```

### Model metrics

```python
print(model.R2Y)    # variance explained in Y
print(model.Q2)     # cross-validated predictive ability
print(model.AUROC)  # area under the ROC curve
```

---

## Extended Multivariate Analysis

v9.1.0 adds four additional multivariate methods behind the same interface.

```python
from metbit import lda, plsr, ica, hca

# Linear Discriminant Analysis
lda_mod = lda(X=X, y=y, features_name=ppm)
lda_mod.fit()
lda_mod.plot_scores()

# Partial Least Squares Regression
plsr_mod = plsr(X=X, y=y_continuous, features_name=ppm, n_components=5)
plsr_mod.fit()
plsr_mod.plot_scores()

# Independent Component Analysis
ica_mod = ica(X=X, features_name=ppm, n_components=5)
ica_mod.fit()
ica_mod.plot_components()

# Hierarchical Cluster Analysis
hca_mod = hca(X=X, label=y, features_name=ppm)
hca_mod.fit()
hca_mod.plot_dendrogram()
```

---

## Statistical Testing

v9.1.0 adds volcano-plot and multi-group statistical tests directly to the public API.

```python
from metbit import VolcanoPlot, ANOVAStats, KruskalStats

# Volcano plot - fold change vs significance
vp = VolcanoPlot(X=X, y=y, ppm=ppm)
vp.fit()
vp.plot(fc_threshold=1.0, p_threshold=0.05)

# One-way ANOVA across groups
anova = ANOVAStats(X=X, y=y, features_name=ppm)
anova.fit()
print(anova.summary())

# Non-parametric Kruskal-Wallis (use when normality is not assured)
kruskal = KruskalStats(X=X, y=y, features_name=ppm)
kruskal.fit()
print(kruskal.summary())
```

---

## Machine Learning Classifiers

`MLClassifier` wraps four model families behind a single sklearn-compatible interface with built-in cross-validation, feature importance, and Plotly visualizations.

```bash
pip install metbit[ml]   # adds XGBoost support
```

```python
from metbit import MLClassifier

# Random Forest (default)
clf = MLClassifier(X=X, y=y, model="rf")
clf.fit(cv=5)

# Support Vector Machine
clf = MLClassifier(X=X, y=y, model="svm")
clf.fit(cv=5)

# XGBoost (requires metbit[ml])
clf = MLClassifier(X=X, y=y, model="xgb")
clf.fit(cv=5)

# ElasticNet logistic regression
clf = MLClassifier(X=X, y=y, model="elasticnet")
clf.fit(cv=5)
```

### Metrics and plots

```python
print(clf.cv_results_)          # accuracy, balanced_accuracy, roc_auc

clf.plot_feature_importance()   # top features by permutation importance
clf.plot_confusion_matrix()     # confusion matrix heatmap
clf.plot_roc()                  # per-class ROC curves with AUC

# Predict on new data
labels = clf.predict(X_new)
proba  = clf.predict_proba(X_new)

# Top N important features as a DataFrame
top_features = clf.get_feature_importance(top_n=20)
```

| Model | `model=` | Notes |
|-------|---------|-------|
| Random Forest | `"rf"` | No extra install needed |
| SVM | `"svm"` | No extra install needed |
| XGBoost | `"xgb"` | Requires `metbit[ml]` |
| ElasticNet | `"elasticnet"` | No extra install needed |

---

## Deep Learning Models

Three PyTorch-based models for spectral classification and representation learning. Requires `metbit[dl]`.

```bash
pip install metbit[dl]
```

```python
from metbit import SpectralAutoencoder, SpectralMLP, SpectralCNN

# Autoencoder - unsupervised representation learning
ae = SpectralAutoencoder(input_dim=X.shape[1], latent_dim=32)
ae.fit(X, epochs=100)
latent = ae.transform(X)          # compressed representation
reconstructed = ae.reconstruct(X)

# MLP classifier
mlp = SpectralMLP(input_dim=X.shape[1], n_classes=len(y.unique()))
mlp.fit(X, y, epochs=50)
labels = mlp.predict(X_new)
proba  = mlp.predict_proba(X_new)

# 1-D CNN classifier
cnn = SpectralCNN(input_dim=X.shape[1], n_classes=len(y.unique()))
cnn.fit(X, y, epochs=50)
labels = cnn.predict(X_new)
```

---

## Train/Test and Cross-Validation

```python
from metbit import TrainTestSplit, CrossValidator, available_cv_strategies

# Simple stratified train/test split
splitter = TrainTestSplit(X=X, y=y, test_size=0.2, stratify=True, random_state=42)
X_train, X_test, y_train, y_test = splitter.split()
splitter.plot_split()           # bar chart of class distribution in each split
print(splitter.get_summary())   # DataFrame: class, train_n, test_n, train_pct, test_pct

# Cross-validation with any sklearn-compatible estimator
print(available_cv_strategies())   # ["kfold", "stratified", "loo", "repeatedstratified"]

from sklearn.ensemble import RandomForestClassifier
cv = CrossValidator(
    X=X, y=y,
    estimator=RandomForestClassifier(n_estimators=100),
    strategy="stratified",
    n_splits=5,
)
cv.fit()
print(cv.get_scores())   # DataFrame with per-fold accuracy, balanced_accuracy, roc_auc
```

---

## Spectra Visualization

`SpectraPlot` renders NMR spectra as interactive Plotly figures with overlay, mean±SD, stacked, and single-sample views.

```python
from metbit import SpectraPlot

sp = SpectraPlot(X=spectra_df, ppm=ppm, label=y)

sp.overlay()      # all spectra overlaid, colored by group
sp.mean_sd()      # group mean with shaded ±1 SD ribbon
sp.stacked()      # vertically stacked spectra
sp.single(idx=0)  # single sample spectrum
```

---

## STOCSY

Statistical Total Correlation Spectroscopy identifies structurally related resonances by correlating every spectral variable against an anchor peak.

```python
from metbit import STOCSY

stocsy = STOCSY(X=spectra_df, ppm=ppm)
stocsy.fit(driver=3.05)   # anchor ppm - typically a high-VIP peak
stocsy.plot()             # interactive spectrum colored by Pearson r
```

### ChunkedSTOCSY - for large feature counts

```python
from metbit import ChunkedSTOCSY

cs = ChunkedSTOCSY(X=spectra_df, ppm=ppm, chunk_size=10_000)
cs.fit(driver=3.05)
cs.plot()

print(ChunkedSTOCSY.active_backend())
```

---

## Large-Scale Compute Backend

The four-tier auto-dispatch backend is inherited by all v9.1.0 modules including `MLClassifier`, `SpectralAutoencoder`, and the new multivariate methods.

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

print(metbit.backend_info())
# {'native_c': True, 'openmp_threads': 8, 'gpu_cupy': False, 'gpu_torch': False,
#  'n_jobs': 8, 'default_chunk': 50000}

print(metbit.native_available())
print(metbit.gpu_available())
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

est = MemoryEstimator()
info = est.estimate(n_samples=10_000, n_features=65_536, dtype="float64")
print(info)
# {'peak_gb': 4.88, 'recommended_dtype': 'float32', 'float32_peak_gb': 2.44}

memory_report(X)
```

---

## Interactive Dash Applications

```python
from metbit.apps import stocsy_app, opls_app, pca_app, spectra_app

stocsy_app(X=spectra_df, ppm=ppm, port=8050)
opls_app(model=model, port=8051)
```

All apps run locally - no data is uploaded to external servers.

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

```bash
python scripts/perf_report.py
```

---

## Testing

```bash
# Run full suite
pytest

# Run without slow/performance tests (faster)
pytest -m "not slow and not perf"

# Run performance benchmarks only
pytest -m perf
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
