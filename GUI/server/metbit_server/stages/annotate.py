"""Annotate stage: spectra to label peaks on, and the intensity table of the labelled peaks."""
from dataclasses import dataclass, field
from typing import Literal, Optional

import plotly.graph_objects as go
from pydantic import BaseModel, Field

from ..annotations import apply_line_styles, intensity_table
from ..session import Session
from .base import DATA_ORDER, MAX_PLOT_POINTS, Dataset, Progress, Stage, median_spectra_figure, working
from .style import colors

MODES = ("single", "median")


class AnnotateParams(BaseModel):
    mode: Literal["single", "median"] = Field(
        "single", title="Spectra", description="Starting view; the toolbar switches between single and median instantly."
    )


@dataclass
class AnnotateResult:
    data: Dataset
    mode: str
    annotations: list  # the session's live list: new labels show up without re-running
    settings: dict = field(default_factory=dict)  # the session's live annotator settings (line styles)


def run_annotate(p: AnnotateParams, session: Session, progress: Progress) -> AnnotateResult:
    progress(1.0)
    return AnnotateResult(
        data=working(session), mode=p.mode, annotations=session.annotations, settings=session.annotation_settings
    )


def sample_names(d) -> list[str]:
    """First metadata column (sample IDs such as Sample or fid) when there is one, else the row index."""
    if d.meta.shape[1]:
        return [str(v) for v in d.meta.iloc[:, 0]]
    return [str(v) for v in d.X.index]


def sample_spectra_figure(d: Dataset, title: str, style=None) -> go.Figure:
    """One trace per sample, named by sample and grouped by group in the legend, as metbit's annotator draws."""
    X = d.X
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)  # display-only decimation, as other spectrum views
    ppm = X.columns.to_numpy(dtype=float)[::step]
    palette = colors(style) if style else {}
    width = style.line_width if style else 1.2
    fig = go.Figure()
    titled: set = set()
    for i, (name, group) in enumerate(zip(sample_names(d), d.labels)):
        fig.add_trace(
            go.Scatter(
                x=ppm,
                y=X.iloc[i, ::step].to_numpy(),
                name=name,
                mode="lines",
                legendgroup=group,
                legendgrouptitle_text=None if group in titled else group,
                line={"color": palette.get(group), "width": width},
            )
        )
        titled.add(group)
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"},
        yaxis={"title": "Intensity"},
    )
    return fig


def annotate_spectra(r: AnnotateResult, style=None, mode: Optional[str] = None) -> go.Figure:
    mode = mode if mode in MODES else r.mode
    if mode == "median":
        fig = median_spectra_figure(r.data.X, r.data.labels, "Median spectra", style)
    else:
        fig = sample_spectra_figure(r.data, "Spectra", style)
    return apply_line_styles(fig, r.settings.get("line_styles", {}))


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
