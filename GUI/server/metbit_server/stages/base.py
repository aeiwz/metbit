"""Shared stage types and helpers for reading the working matrix."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from metbit.viz.spectra import SpectraPlot

from ..session import Session
from .style import PlotStyle, colors

Progress = Callable[[float], None]

DATA_ORDER = ["import", "process", "normalize", "align"]
MAX_PLOT_POINTS = 8000


@dataclass
class Stage:
    id: str
    title: str
    params: type[BaseModel]
    run: Callable[[Any, Session, Progress], Any]
    section: str = "Analysis"
    table: Optional[Callable[[Any], pd.DataFrame]] = None
    figures: dict[str, Callable[[Any], go.Figure]] = field(default_factory=dict)
    requires: Optional[str] = None
    after: list[str] = field(default_factory=list)
    # view name -> where a clicked point keeps its ppm: "x" or "customdata" (index 0)
    ppm_views: dict[str, str] = field(default_factory=dict)
    # reason the stage cannot run in this session (shown on the disabled rail item), or None
    blocked: Optional[Callable[[Session], Optional[str]]] = None
    # CSV export; defaults to `table` when None
    export: Optional[Callable[[Any], pd.DataFrame]] = None
    # views drawn from the working matrix before the stage has a result (called with the Dataset)
    pre_views: dict[str, Callable[..., go.Figure]] = field(default_factory=dict)
    # main view -> companion view drawn under it (hidden from the tab bar); a click on it sets ppm for both
    companions: dict[str, str] = field(default_factory=dict)


@dataclass
class Dataset:
    X: pd.DataFrame                        # samples x features, float ppm column labels
    meta: pd.DataFrame                     # every non-ppm column
    labels: pd.Series                      # group name per sample (str)
    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views
    # per-matrix result cache (STOCSY): new data is a new Dataset, so nothing stale survives
    cache: dict = field(default_factory=dict, repr=False, compare=False)


def not_run_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def matrix_frame(d: Dataset) -> pd.DataFrame:
    """Metadata then ppm columns, one row per sample: re-importable as a CSV."""
    return pd.concat([d.meta.reset_index(drop=True), d.X.reset_index(drop=True)], axis=1)


def working(session: Session, upto: Optional[str] = None) -> Dataset:
    """Newest `done` data stage holding a matrix, considering only stages earlier than `upto`."""
    order = DATA_ORDER if upto is None else DATA_ORDER[: DATA_ORDER.index(upto)]
    for sid in reversed(order):
        out = session.outputs.get(sid)
        if session.status.get(sid) == "done" and isinstance(out, Dataset):
            return out
    if session.status.get("import") == "done":
        raise ValueError("Run Process first: Bruker data must be processed before analysis")
    raise ValueError("Import data first")


def groups(session: Session) -> pd.Series:
    return session.output("import").labels


def two_groups(session: Session, a: str, b: str) -> tuple[pd.DataFrame, pd.Series]:
    """Working matrix and labels restricted to groups `a` and `b`."""
    if a == b:
        raise ValueError("Choose two different groups")
    g = groups(session)
    missing = [x for x in (a, b) if x not in set(g)]
    if missing:
        raise ValueError(f"Unknown group(s): {missing}. Groups: {sorted(set(g))}")
    mask = g.isin([a, b]).to_numpy()
    X = working(session).X
    return X[mask].reset_index(drop=True), g[mask].reset_index(drop=True)


def spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str, style: Optional[PlotStyle] = None) -> go.Figure:
    # ponytail: display-only decimation keeps figure JSON small; full data stays in the session
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    plot = SpectraPlot(X.iloc[:, ::step], label=list(labels), color_dict=colors(style) if style else None)
    if style is None:
        return plot.overlay(title=title)
    return plot.overlay(title=title, linewidth=style.line_width, font_size=style.font_size)


def median_spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str, style: Optional[PlotStyle] = None) -> go.Figure:
    """One median line per group, display-decimated like the other spectrum views."""
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    ppm = X.columns.to_numpy(dtype=float)[::step]
    palette = colors(style) if style else {}
    fig = go.Figure()
    for group in sorted(set(labels)):
        median = np.median(X[(labels == group).to_numpy()].to_numpy(), axis=0)[::step]
        line = {"color": palette.get(group), "width": style.line_width if style else 1.2}
        fig.add_trace(go.Scatter(x=ppm, y=median, name=group, mode="lines", line=line))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"autorange": "reversed", "title": "δ¹H (ppm)"},
        yaxis={"title": "Intensity"},
    )
    return fig


def mark_feature(fig: go.Figure, center: float, bin_width: float = 0.0) -> go.Figure:
    """Dotted line at the tested feature; with binning, also shade the whole bin it averages."""
    if bin_width > 0:
        fig.add_vrect(x0=center - bin_width / 2, x1=center + bin_width / 2, fillcolor="#888888", opacity=0.18, line_width=0)
        fig.update_layout(title_text=f"{fig.layout.title.text} · {bin_width:g} ppm bins")
    fig.add_vline(x=center, line_dash="dot", line_color="#888888")
    return fig


def _numeric_columns(meta: pd.DataFrame) -> list[str]:
    return [str(c) for c in meta.columns if pd.to_numeric(meta[c], errors="coerce").notna().all()]


TIME_HINTS = ("time", "day", "week", "wk", "visit", "hour", "month")
SUBJECT_HINTS = ("subject", "patient", "animal", "mouse", "donor", "individual", "id")


def option_values(session: Session, kind: str) -> Optional[list[str]]:
    if session.status.get("import") != "done":
        return None
    meta = session.output("import").meta
    if kind == "groups":
        return sorted(set(groups(session)))
    if kind == "columns":
        return [str(c) for c in meta.columns]
    if kind in ("time_columns", "subject_columns"):
        # likely columns first, so the default pick is usually right
        hint = TIME_HINTS if kind == "time_columns" else SUBJECT_HINTS
        cols = [str(c) for c in meta.columns]
        likely = [c for c in cols if any(h in c.lower() for h in hint)]
        rest = [c for c in cols if c not in likely]
        # subject: a likely column is the default, else "(none)"
        return likely + rest if kind == "time_columns" else [*likely, "(none)", *rest]
    if kind == "numeric_columns":
        return _numeric_columns(meta) or None
    return None


def _flatten_nullable(prop: dict) -> None:
    """Optional[X] arrives as anyOf [X, null]; give the form the plain X schema."""
    options = prop.get("anyOf")
    if options and len(options) == 2 and {"type": "null"} in options:
        prop.update(next(o for o in options if o != {"type": "null"}))
        del prop["anyOf"]


def resolve_schema(stage: Stage, session: Session) -> dict:
    """Param JSON schema with data-driven `enum` and `default` for tagged fields."""
    schema = stage.params.model_json_schema()
    for prop in schema.get("properties", {}).values():
        _flatten_nullable(prop)
        values = option_values(session, prop.get("options", ""))
        if values:
            prop["enum"] = values
            if prop.get("default") not in values:
                prop["default"] = values[min(prop.get("pick", 0), len(values) - 1)]
    return schema
