# -*- coding: utf-8 -*-

# Public classes/functions re-exported for convenience
from .metbit import opls_da, pca
from .utility import UnivarStats, Normalise
from .lazy_opls_da import lazy_opls_da

# Backwards-compat star imports (intended public surface)
from .spec_norm import *  # noqa: F401,F403
from .peak_processe import peak_chops
from .STOCSY import STOCSY
from .ui_stocsy import STOCSY_app
from .ui_picky_peak import pickie_peak
from .take_intensity import *  # noqa: F401,F403
from .nmr_preprocess import nmr_preprocessing
from .calibrate import calibrate
from .annotate_peak import annotate_peak
from .baseline import baseline_correct, bline
from .alignment import detect_multiplets, icoshift_align, PeakAligner

# Package metadata
__author__ = "aeiwz"
__email__ = "theerayut_aeiw_123@hotmail.com"
__maintainer__ = "aeiwz"
__status__ = "Development"
__copyright__ = "Copyright 2024, Theerayut"

# Optional version string to keep in sync with setup.py
__version__ = "8.7.6"

# Explicit export list for primary API (star imports remain for compatibility)
__all__ = [
    "opls_da",
    "pca",
    "UnivarStats",
    "Normalise",
    "lazy_opls_da",
    "peak_chops",
    "STOCSY",
    "STOCSY_app",
    "pickie_peak",
    "nmr_preprocessing",
    "calibrate",
    "annotate_peak",
    "baseline_correct",
    "bline",
    "detect_multiplets",
    "icoshift_align",
    "PeakAligner",
]
