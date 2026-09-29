"""Time series stages: PCA trajectory, feature time profile, ASCA and per-feature mixed models."""
import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, field_validator, model_validator

from metbit import pca

from ..session import Session
from .analysis import Scaling
from .base import DATA_ORDER, Dataset, Progress, Stage, median_spectra_figure, not_run_figure, working
from .style import colors

NONE = "(none)"
SECTION = "Time series"


def natural_key(text: str) -> list:
    """Sort key that puts 2-wk before 10-wk."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", text)]


class TimeParams(BaseModel):
    time_column: str = Field("", title="Time column", json_schema_extra={"options": "time_columns"})
    time_order: str = Field(
        "",
        title="Time order",
        description="Comma-separated, e.g. 1-wk, 2-wk, 4-wk. Empty sorts numbers by value and text naturally.",
    )
    subject_column: str = Field(
        "",
        title="Subject column",
        description="Sample donor ID for repeated measures.",
        json_schema_extra={"options": "subject_columns"},
    )


@dataclass
class Design:
    time: pd.Series                   # time label per sample (str)
    order: list[str]                  # time labels in time order
    subject: Optional[pd.Series]      # subject per sample (str), None without a subject column
    labels: pd.Series                 # group per sample (str)
    column: str                       # time column name, for tables

    def position(self) -> np.ndarray:
        return self.time.map({t: i for i, t in enumerate(self.order)}).to_numpy()


def _column(meta: pd.DataFrame, name: str, what: str) -> pd.Series:
    if name not in meta.columns:
        raise ValueError(f"Unknown {what} column '{name}'. Columns: {', '.join(map(str, meta.columns))}")
    values = meta[name]
    if values.isna().any() or (values.astype(str).str.strip() == "").any():
        raise ValueError(f"The {what} column '{name}' has empty values")
    return values.astype(str).str.strip()


def design(session: Session, p: TimeParams) -> tuple[Design, Dataset]:
    d = working(session)
    meta = d.meta.reset_index(drop=True)
    time = _column(meta, p.time_column, "time")
    values = sorted(set(time))
    if p.time_order.strip():
        given = list(dict.fromkeys(t.strip() for t in p.time_order.split(",") if t.strip()))
        missing = sorted(set(values) - set(given), key=natural_key)
        if missing:
            raise ValueError(f"Time order is missing: {', '.join(missing)}")
        order = [t for t in given if t in values]
    else:
        numbers = pd.to_numeric(pd.Series(values), errors="coerce")
        order = [values[i] for i in numbers.argsort()] if numbers.notna().all() else sorted(values, key=natural_key)
    if len(order) < 2:
        raise ValueError("A time series needs at least two time points")
    subject = None if p.subject_column in (NONE, "") else _column(meta, p.subject_column, "subject")
    return Design(time, order, subject, d.labels.reset_index(drop=True), p.time_column), d


def need_subject(des: Design, why: str) -> None:
    if des.subject is None:
        raise ValueError(f"{why} needs a subject column")


def summarize(y: np.ndarray, des: Design, center: str, error: str) -> pd.DataFrame:
    """center and error of `y` per group and time point, in time order."""
    df = pd.DataFrame({"Group": des.labels.to_numpy(), "Time": des.time.to_numpy(), "y": y})
    g = df.groupby(["Group", "Time"])["y"]
    out = pd.DataFrame({"n": g.size(), "center": g.mean() if center == "mean" else g.median()})
    out["error"] = g.sem() if error == "sem" else g.std()
    out = out.reset_index()
    out["_pos"] = out["Time"].map({t: i for i, t in enumerate(des.order)})
    return out.sort_values(["Group", "_pos"]).drop(columns="_pos").reset_index(drop=True)


def profile_figure(y: np.ndarray, des: Design, center: str, error: str, title: str, style=None, ytitle="Intensity"):
    """Faint line per subject, then center ± error per group across time (the per-subject trajectories view)."""
    palette = colors(style) if style else {}
    fig = go.Figure()
    if des.subject is not None:
        df = pd.DataFrame({"g": des.labels, "s": des.subject, "pos": des.position(), "t": des.time, "y": y})
        for (g, s), part in df.sort_values("pos").groupby(["g", "s"], sort=True):
            fig.add_trace(go.Scatter(
                x=part["t"], y=part["y"], mode="lines", name=s, showlegend=False, opacity=0.3,
                line={"color": palette.get(g, "#888888"), "width": 1}, legendgroup=g, hoverinfo="name+y",
            ))
    width = style.line_width * 2 if style else 2.4
    for g, part in summarize(y, des, center, error).groupby("Group", sort=True):
        fig.add_trace(go.Scatter(
            x=part["Time"], y=part["center"], name=g, legendgroup=g, mode="lines+markers",
            error_y={"type": "data", "array": part["error"].fillna(0).to_numpy(), "visible": True},
            line={"color": palette.get(g), "width": width},
            marker={"size": style.marker_size / 2 if style else 8},
        ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"title": des.column, "type": "category", "categoryorder": "array", "categoryarray": des.order},
        yaxis={"title": ytitle},
    )
    return fig


# Trajectory -----------------------------------------------------------------

class TrajectoryParams(TimeParams):
    multilevel: bool = Field(
        False,
        title="Multilevel",
        description="Centers each subject on its own mean first, so the plot shows change within subjects.",
    )
    scaling: Scaling = Field("pareto", title="Scaling")
    center: Literal["mean", "median"] = Field("mean", title="Center")
    error: Literal["sem", "std"] = Field("sem", title="Error bars")
    pc_x: int = Field(1, ge=1, le=10, title="X axis PC")
    pc_y: int = Field(2, ge=1, le=10, title="Y axis PC")

    @model_validator(mode="after")
    def _axes_differ(self) -> "TrajectoryParams":
        if self.pc_x == self.pc_y:
            raise ValueError("X and Y axis PCs must differ")
        return self


@dataclass
class TrajectoryResult:
    model: Any
    pcs: list[str]
    des: Design
    center: str
    error: str
    multilevel: bool


def run_trajectory(p: TrajectoryParams, session: Session, progress: Progress) -> TrajectoryResult:
    des, d = design(session, p)
    X = d.X.reset_index(drop=True)
    if p.multilevel:
        need_subject(des, "Multilevel")
        X = X - X.groupby(des.subject.to_numpy()).transform("mean")  # within-subject variation only
    model = pca(X=X, label=des.labels, features_name=list(X.columns), n_components=max(p.pc_x, p.pc_y), scaling_method=p.scaling)
    model.fit()
    progress(1.0)
    return TrajectoryResult(model, [f"PC{p.pc_x}", f"PC{p.pc_y}"], des, p.center, p.error, p.multilevel)


def _scores(r: TrajectoryResult) -> pd.DataFrame:
    df = r.model.df_scores_[r.pcs].reset_index(drop=True).copy()
    df.insert(0, "Group", r.des.labels.to_numpy())
    df.insert(1, "Time", r.des.time.to_numpy())
    if r.des.subject is not None:
        df.insert(2, "Subject", r.des.subject.to_numpy())
    return df


def _trajectory(r: TrajectoryResult, style=None) -> go.Figure:
    extra = {}
    if style:
        extra = {"color_dict": colors(style), "marker_size": style.marker_size, "marker_opacity": style.marker_opacity,
                 "font_size": style.font_size, "title_font_size": style.font_size}
    time = pd.Series(r.des.time.to_numpy(), index=r.model.df_scores_.index)
    fig = r.model.plot_pca_trajectory(
        time_=time, time_order={t: i for i, t in enumerate(r.des.order)}, stat_=[r.center, r.error], pc=r.pcs, **extra
    )
    kind = "multilevel PCA" if r.multilevel else "PCA"
    return fig.update_layout(title_text=f"Trajectory ({kind}, {r.center} ± {r.error})")


def _subjects(r: TrajectoryResult, style=None) -> go.Figure:
    if r.des.subject is None:
        return not_run_figure("Choose a subject column to draw each subject's path.")
    palette = colors(style) if style else {}
    df = _scores(r)
    df["_pos"] = r.des.position()
    x, y = r.pcs
    fig = go.Figure()
    for (g, s), part in df.sort_values("_pos").groupby(["Group", "Subject"]):
        fig.add_trace(go.Scatter(
            x=part[x], y=part[y], mode="lines+markers", name=s, showlegend=False, opacity=0.35,
            text=part["Time"], hoverinfo="name+text", line={"color": palette.get(g, "#888888"), "width": 1},
        ))
    for g, part in df.groupby("Group"):
        mean = part.groupby("_pos")[[x, y]].mean()
        fig.add_trace(go.Scatter(
            x=mean[x], y=mean[y], mode="lines+markers", name=g,
            text=[r.des.order[i] for i in mean.index], hoverinfo="name+text",
            line={"color": palette.get(g), "width": 3}, marker={"size": 10},
        ))
    return fig.update_layout(title="Subject paths", template="plotly_white", xaxis_title=x, yaxis_title=y)


# Time profile ---------------------------------------------------------------

class ProfileParams(TimeParams):
    center: Literal["mean", "median"] = Field("mean", title="Center")
    error: Literal["sem", "std"] = Field("sem", title="Error bars")
    feature_ppm: Optional[float] = Field(
        None,
        title="Feature (ppm)",
        description="Click a peak on the spectrum below the plot. Empty shows the most variable feature.",
        json_schema_extra={"pick_ppm": True},
    )

    @field_validator("feature_ppm", mode="before")
    @classmethod
    def _blank_is_none(cls, v):
        return None if v == "" else v


@dataclass
class ProfileResult:
    data: Dataset
    des: Design
    center: str
    error: str
    feature: float


def run_profile(p: ProfileParams, session: Session, progress: Progress) -> ProfileResult:
    des, d = design(session, p)
    ppm = d.X.columns.to_numpy(dtype=float)
    feature = ppm[int(d.X.var().to_numpy().argmax())] if p.feature_ppm is None else p.feature_ppm
    progress(1.0)
    return ProfileResult(d, des, p.center, p.error, float(feature))


def nearest(X: pd.DataFrame, ppm: float) -> int:
    return int(np.abs(X.columns.to_numpy(dtype=float) - ppm).argmin())


def _profile(r: ProfileResult, style=None, ppm: Optional[float] = None) -> go.Figure:
    j = nearest(r.data.X, r.feature if ppm is None else ppm)
    at = float(r.data.X.columns[j])
    return profile_figure(r.data.X.iloc[:, j].to_numpy(), r.des, r.center, r.error, f"{at:.4f} ppm over time", style)


def _profile_spectra(r: ProfileResult, style=None, ppm: Optional[float] = None) -> go.Figure:
    fig = median_spectra_figure(r.data.X, r.data.labels, "Click a peak to show its time profile", style)
    fig.add_vline(x=float(r.data.X.columns[nearest(r.data.X, r.feature if ppm is None else ppm)]), line_dash="dot", line_color="#888888")
    return fig


def _profile_table(r: ProfileResult) -> pd.DataFrame:
    return summarize(r.data.X.iloc[:, nearest(r.data.X, r.feature)].to_numpy(), r.des, r.center, r.error)


def time_stages() -> list[Stage]:
    from .longitudinal import LONGITUDINAL_STAGES  # ASCA and mixed models build on the helpers above

    common = {"section": SECTION, "requires": "import", "after": list(DATA_ORDER)}
    return [
        Stage(id="trajectory", title="Trajectory", params=TrajectoryParams, run=run_trajectory, table=_scores,
              figures={"trajectory": _trajectory, "subjects": _subjects}, **common),
        Stage(id="timeprofile", title="Time profile", params=ProfileParams, run=run_profile, table=_profile_table,
              figures={"profile": _profile, "spectra": _profile_spectra}, companions={"profile": "spectra"},
              ppm_views={"spectra": "x"}, **common),
        *LONGITUDINAL_STAGES,
    ]
