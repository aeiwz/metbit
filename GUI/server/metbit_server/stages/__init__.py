"""Stage registry, in rail order."""
from .analysis import ANALYSIS_STAGES, OPLSParams, OPLSResult, PCAParams, PCAResult, STOCSYParams
from .base import DATA_ORDER, Dataset, Progress, Stage, resolve_schema
from .data import DATA_STAGES, AlignParams, ImportParams, NormalizeParams, parse_windows

STAGES: list[Stage] = [*DATA_STAGES, *ANALYSIS_STAGES]
