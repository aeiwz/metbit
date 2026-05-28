# Backwards-compatibility re-exports from sub-package paths.
# Importing from old flat paths (e.g. `from metbit.nmr_preprocess import ...`)
# still works via the originals; this module provides the NEW canonical paths
# so external consumers can start migrating to the sub-package imports.

# --- nmr/ ---
from .nmr.preprocess import (  # noqa: F401
    nmr_preprocessing,
    read_fid,
    remove_digital_filter,
    generate_ppm_scale,
    phasing,
)
from .nmr.calibrate import calibrate  # noqa: F401
from .nmr.alignment import (  # noqa: F401
    detect_multiplets,
    icoshift_align,
    PeakAligner,
    Multiplet,
)
from .nmr.denoise import Denoise  # noqa: F401
from .nmr.peaks import peak_chops  # noqa: F401

# --- models/ ---
from .models.base import nipals  # noqa: F401
from .models.pls import PLS  # noqa: F401
from .models.opls import OPLS  # noqa: F401
from .models.vip import vip_scores  # noqa: F401
from .models.cross_validation import CrossValidation  # noqa: F401

# --- preprocessing/ ---
from .preprocessing.scaler import Scaler as SimpleScaler  # noqa: F401
from .preprocessing.scaler_ext import Scaler  # noqa: F401
from .preprocessing.normalize import Normalization  # noqa: F401
from .preprocessing.baseline import baseline_correct, bline  # noqa: F401

# --- analysis/ ---
from .analysis.stocsy import STOCSY  # noqa: F401

# --- viz/ ---
from .viz.ellipse import confidence_ellipse  # noqa: F401

# --- higher-level classes (canonical sub-package paths) ---
from .analysis.opls_da import opls_da  # noqa: F401
from .analysis.pca import pca  # noqa: F401

try:
    from .lazy_opls_da import lazy_opls_da  # noqa: F401
except Exception:
    pass

# --- utility classes (canonical sub-package paths) ---
from .stats.univariate import UnivarStats  # noqa: F401
from .stats.normalise import Normalise  # noqa: F401

# --- Dash apps (optional heavy deps) ---
try:
    from .apps.annotate import annotate_peak  # noqa: F401
except Exception:
    pass

try:
    from .apps.stocsy_app import STOCSY_app  # noqa: F401
except Exception:
    pass

try:
    from .apps.peak_picker import pickie_peak  # noqa: F401
except Exception:
    pass
