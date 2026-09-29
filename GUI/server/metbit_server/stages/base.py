"""Shared stage types and helpers for reading the working matrix."""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel

from metbit.viz.spectra import SpectraPlot

from ..session import Session

Progress = Callable[[float], None]

DATA_ORDER = ["import", "normalize", "align"]
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


@dataclass
class Dataset:
    X: pd.DataFrame                        # samples x features, float ppm column labels
    meta: pd.DataFrame                     # every non-ppm column
    labels: pd.Series                      # group name per sample (str)
    before: Optional[pd.DataFrame] = None  # input matrix, for before / after views


def working(session: Session, upto: Optional[str] = None) -> Dataset:
    """Newest `done` data stage, considering only stages earlier than `upto`."""
    order = DATA_ORDER if upto is None else DATA_ORDER[: DATA_ORDER.index(upto)]
    for sid in reversed(order):
        if session.status.get(sid) == "done":
            return session.outputs[sid]
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


def spectra_figure(X: pd.DataFrame, labels: pd.Series, title: str) -> go.Figure:
    # ponytail: display-only decimation keeps figure JSON small; full data stays in the session
    step = max(1, X.shape[1] // MAX_PLOT_POINTS)
    return SpectraPlot(X.iloc[:, ::step], label=list(labels)).overlay(title=title)


def _numeric_columns(meta: pd.DataFrame) -> list[str]:
    return [str(c) for c in meta.columns if pd.to_numeric(meta[c], errors="coerce").notna().all()]


def option_values(session: Session, kind: str) -> Optional[list[str]]:
    if session.status.get("import") != "done":
        return None
    meta = session.output("import").meta
    if kind == "groups":
        return sorted(set(groups(session)))
    if kind == "columns":
        return [str(c) for c in meta.columns]
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
