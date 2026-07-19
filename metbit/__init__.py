# -*- coding: utf-8 -*-

# Package metadata
__author__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__maintainer__ = "aeiwz"
__status__ = "Development"
__copyright__ = "Copyright 2024, Theerayut"
__version__ = "9.1.0"

# Core analysis
from metbit.analysis.opls_da import opls_da
from metbit.analysis.pca import pca
from metbit.analysis.stocsy import STOCSY
from metbit.analysis.large_scale import (
    ChunkedSTOCSY,
    MemoryEstimator,
    LargeScaleAlignment,
    feature_preselection,
    memory_report,
)
from metbit._native import backend_info, gpu_available, gpu_installed, native_available

# NMR processing
from metbit.nmr.preprocess import nmr_preprocessing
from metbit.nmr.calibrate import calibrate
from metbit.nmr.alignment import detect_multiplets, icoshift_align, PeakAligner
from metbit.nmr.peaks import peak_chops

# Statistics
from metbit.stats.normalise import Normalise
from metbit.stats.univariate import UnivarStats

# Preprocessing
from metbit.preprocessing.baseline import baseline_correct, bline
from metbit.preprocessing.normalize import Normalization

# Extended statistics
from metbit.stats.multitest import VolcanoPlot, ANOVAStats, KruskalStats

# Extended multivariate analysis
from metbit.analysis.multivariate import lda, plsr, ica, hca

# Validation
from metbit.validation.metrics import ModelValidator
from metbit.validation.splitter import TrainTestSplit, CrossValidator, available_cv_strategies

# Machine learning classifiers (optional: requires xgboost)
try:
    from metbit.ml.classifiers import MLClassifier
except Exception:  # pragma: no cover
    pass

# Deep learning models (optional: requires torch)
try:
    from metbit.dl.models import SpectralAutoencoder, SpectralMLP, SpectralCNN
except Exception:  # pragma: no cover
    pass

# Dash apps (optional heavy deps)
try:
    from metbit.apps.annotate import annotate_peak
except Exception:  # pragma: no cover
    pass

try:
    from metbit.apps.stocsy_app import STOCSY_app
except Exception:  # pragma: no cover
    pass

try:
    from metbit.apps.peak_picker import pickie_peak
except Exception:  # pragma: no cover
    pass

# Visualization - spectra
from metbit.viz.spectra import SpectraPlot

# Visualization - statistical summaries
from metbit.viz.summary import FeatureHeatmap, CorrelationMatrix, PValueTable

# Visualization - model interpretation
from metbit.viz.interpretation import Biplot, CoefficientPlot, FeatureImportancePlot

# Visualization - metabolite profiling
from metbit.viz.profiling import FoldChangePlot, GroupComparison

# Dash app: metabolite dashboard (optional heavy deps)
try:
    from metbit.viz.profiling import MetaboliteDashboard
except Exception:  # pragma: no cover
    pass

# Legacy flat-layout class not yet moved to a sub-package
try:
    from metbit.lazy_opls_da import lazy_opls_da
except Exception:  # pragma: no cover
    pass

_always_exported = [
    "opls_da",
    "pca",
    "STOCSY",
    "ChunkedSTOCSY",
    "MemoryEstimator",
    "LargeScaleAlignment",
    "feature_preselection",
    "memory_report",
    "backend_info",
    "gpu_available",
    "gpu_installed",
    "native_available",
    "nmr_preprocessing",
    "calibrate",
    "detect_multiplets",
    "icoshift_align",
    "PeakAligner",
    "peak_chops",
    "Normalise",
    "UnivarStats",
    "baseline_correct",
    "bline",
    "Normalization",
    # Extended stats
    "VolcanoPlot",
    "ANOVAStats",
    "KruskalStats",
    # Extended multivariate
    "lda",
    "plsr",
    "ica",
    "hca",
    # Validation
    "ModelValidator",
    "TrainTestSplit",
    "CrossValidator",
    "available_cv_strategies",
    # Visualization - spectra
    "SpectraPlot",
    # Visualization - statistical summaries
    "FeatureHeatmap",
    "CorrelationMatrix",
    "PValueTable",
    # Visualization - model interpretation
    "Biplot",
    "CoefficientPlot",
    "FeatureImportancePlot",
    # Visualization - metabolite profiling
    "FoldChangePlot",
    "GroupComparison",
]
_optional = [
    "annotate_peak", "STOCSY_app", "pickie_peak", "lazy_opls_da",
    "MLClassifier",
    "SpectralAutoencoder", "SpectralMLP", "SpectralCNN",
    "MetaboliteDashboard",
]
__all__ = _always_exported + [n for n in _optional if n in globals()]
