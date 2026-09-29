"""Stage registry, in rail order."""
from .annotate import ANNOTATE_STAGES, AnnotateParams
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, BrukerSource, ImportParams, NormalizeParams, ProcessParams, parse_windows
from .multivariate import MULTIVARIATE_STAGES, HCAParams, ICAParams, LDAParams, PLSRParams
from .stats import STATS_STAGES, GroupTestParams, GroupTestResult, VolcanoParams, bin_matrix

_analysis = {s.id: s for s in ANALYSIS_STAGES}

STAGES: list[Stage] = [
    *DATA_STAGES,
    _analysis["pca"],
    _analysis["opls"],
    *MULTIVARIATE_STAGES,
    _analysis["stocsy"],
    *STATS_STAGES,
    *ANNOTATE_STAGES,
]
