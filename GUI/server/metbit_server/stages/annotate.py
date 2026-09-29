"""Annotate stage: spectra to label peaks on, and the intensity table of the labelled peaks."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ..annotations import intensity_table
from ..session import Session
from .base import DATA_ORDER, Dataset, Progress, Stage, median_spectra_figure, spectra_figure, working
from .style import colors


class AnnotateParams(BaseModel):
    mode: Literal["single", "median"] = Field(
        "single", title="Spectra", description="Single draws every spectrum; median draws one line per group."
    )


@dataclass
class AnnotateResult:
    data: Dataset
    mode: str
    annotations: list  # the session's live list: new labels show up without re-running


def run_annotate(p: AnnotateParams, session: Session, progress: Progress) -> AnnotateResult:
    progress(1.0)
    return AnnotateResult(data=working(session), mode=p.mode, annotations=session.annotations)


def annotate_spectra(r: AnnotateResult, style=None) -> go.Figure:
    if r.mode == "single":
        return spectra_figure(r.data.X, r.data.labels, "Annotated spectra", style)
    return median_spectra_figure(r.data.X, r.data.labels, "Median spectra", style)


ANNOTATE_STAGES: list[Stage] = [
    Stage(
        id="annotate",
        title="Annotate",
        section="Statistics",
        params=AnnotateParams,
        run=run_annotate,
        table=lambda r: intensity_table(r.data.X, r.data.meta, r.annotations),
        figures={"spectra": annotate_spectra},
        requires="import",
        after=list(DATA_ORDER),
        ppm_views={"spectra": "x"},
    ),
]
