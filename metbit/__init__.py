# -*- coding: utf-8 -*-

# Package metadata
__author__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__maintainer__ = "aeiwz"
__status__ = "Development"
__copyright__ = "Copyright 2024, Theerayut"
__version__ = "9.0.0"

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
from metbit._native import backend_info, gpu_available, native_available

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
]
_optional = ["annotate_peak", "STOCSY_app", "pickie_peak", "lazy_opls_da"]
__all__ = _always_exported + [n for n in _optional if n in globals()]
