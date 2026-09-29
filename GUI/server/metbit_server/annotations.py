"""Peak annotations: sanitizing, overlaying on spectra and the intensity table.

Label and CSV rules follow metbit.apps.annotate; they are reimplemented here because
that module imports Dash, which the sidecar does not need.
"""
import html
import json
import uuid
from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pydantic import BaseModel, Field, ValidationError, field_validator

MAX_LABEL = 256
ANNOTATED_VIEWS = {"spectra", "before", "correlation"}


def sanitize_label(value: Any) -> str:
    """Trimmed, at most 256 characters, HTML-escaped. Unescaping first makes it idempotent."""
    text = html.unescape(str(value or "")).strip()[:MAX_LABEL].strip()
    return html.escape(text, quote=True)


def csv_safe(value: Any) -> str:
    """Neutralize spreadsheet formula injection, as metbit's intensity export does."""
    text = str(value if value is not None else "")
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t") else text


class AnnotationIn(BaseModel):
    ppm: float
    label: str
    angle: float = Field(0, ge=-180, le=180)

    @field_validator("label")
    @classmethod
    def _clean(cls, v: str) -> str:
        clean = sanitize_label(v)
        if not clean:
            raise ValueError("Label is empty")
        return clean


def new_annotation(a: AnnotationIn) -> dict:
    return {"id": uuid.uuid4().hex[:8], **a.model_dump()}


def parse_annotation_list(items: list) -> list[dict]:
    """Our shape or metbit's annotator JSON (x, text, textangle); invalid entries are skipped."""
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw = {
            "ppm": item.get("ppm", item.get("x")),
            "label": item.get("label", item.get("text")),
            "angle": item.get("angle", item.get("textangle")) or 0,
        }
        try:
            out.append(new_annotation(AnnotationIn.model_validate(raw)))
        except ValidationError:
            continue
    return out


def parse_annotation_file(text: str) -> list[dict]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list of annotations")
    return parse_annotation_list(data)


def _peak_height(fig: go.Figure, ppm: float) -> Optional[float]:
    """Tallest y among the figure's curves at the point nearest `ppm`."""
    best = None
    for trace in fig.data:
        x, y = getattr(trace, "x", None), getattr(trace, "y", None)
        if x is None or y is None or len(x) == 0:
            continue
        try:
            xs, ys = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        except (TypeError, ValueError):
            continue
        value = ys[int(np.nanargmin(np.abs(xs - ppm)))]
        if np.isfinite(value):
            best = value if best is None else max(best, value)
    return best


def overlay(fig: go.Figure, annotations: list[dict], font_size: int) -> go.Figure:
    for a in annotations:
        y = _peak_height(fig, a["ppm"])
        fig.add_annotation(
            x=a["ppm"],
            y=y if y is not None else 1,
            xref="x",
            yref="y" if y is not None else "paper",
            text=a["label"],
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            textangle=a["angle"],
            font={"size": font_size},
        )
    return fig


def intensity_table(X: pd.DataFrame, meta: pd.DataFrame, annotations: list[dict]) -> pd.DataFrame:
    """Metadata, then one column per label with each sample's value at the nearest ppm (metbit's export)."""
    ppm = X.columns.to_numpy(dtype=float)
    columns = {}
    for a in annotations:
        j = int(np.argmin(np.abs(ppm - a["ppm"])))
        columns[f"{csv_safe(html.unescape(a['label']))}_{ppm[j]:.3f}"] = X.iloc[:, j].to_numpy()
    safe_meta = meta.reset_index(drop=True).apply(lambda col: col.map(csv_safe) if col.dtype == object else col)
    return pd.concat([safe_meta, pd.DataFrame(columns)], axis=1)
