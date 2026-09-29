"""Annotate stage: spectra to label peaks on, and the intensity table of the labelled peaks."""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ..annotations import intensity_table
from ..session import Session
from .base import DATA_ORDER, MAX_PLOT_POINTS, Dataset, Progress, Stage, spectra_figure, working
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
    X, labels = r.data.X, r.data.labels
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)  # display-only decimation, as other spectrum views
    ppm = X.columns.to_numpy(dtype=float)[::step]
    palette = colors(style) if style else {}
    fig = go.Figure()
    for group in sorted(set(labels)):
        median = np.median(X[(labels == group).to_numpy()].to_numpy(), axis=0)[::step]
        line = {"color": palette.get(group), "width": style.line_width if style else 1.2}
        fig.add_trace(go.Scatter(x=ppm, y=median, name=group, mode="lines", line=line))
    fig.update_layout(
        title="Median spectra",
        template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"},
        yaxis={"title": "Intensity"},
    )
    return fig


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
