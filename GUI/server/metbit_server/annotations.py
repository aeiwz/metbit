"""Peak annotations: sanitizing, overlaying on spectra and the intensity table.

Label and CSV rules follow metbit.apps.annotate; they are reimplemented here because
that module imports Dash, which the sidecar does not need.
"""
import html
import json
import uuid
from typing import Any, Literal, Optional

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
    ax: float = Field(0, ge=-600, le=600)  # label offset from its peak, in pixels (metbit's default)
    ay: float = Field(-40, ge=-600, le=600)

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
            "ax": item.get("ax", 0),
            "ay": item.get("ay", -40),
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


def overlay(fig: go.Figure, annotations: list[dict], font_size: int, settings: Optional[dict] = None) -> go.Figure:
    settings = settings or {}
    font = {"size": font_size}
    font.update({k: settings[f"font_{k}"] for k in ("family", "weight", "style") if settings.get(f"font_{k}")})
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
            ax=a.get("ax", 0),
            ay=a.get("ay", -40),
            name=a["id"],  # lets the renderer map a dragged label back to its annotation
            textangle=a["angle"],
            captureevents=True,  # a click on the label reaches the renderer (to edit it)
            font=font,
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


Dash = Literal["solid", "dash", "dot", "dashdot"]


class AnnotationPatch(BaseModel):
    label: Optional[str] = None
    angle: Optional[float] = Field(None, ge=-180, le=180)
    ax: Optional[float] = Field(None, ge=-600, le=600)
    ay: Optional[float] = Field(None, ge=-600, le=600)

    @field_validator("label")
    @classmethod
    def _clean(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        clean = sanitize_label(v)
        if not clean:
            raise ValueError("Label is empty")
        return clean


class LineStyle(BaseModel):
    color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    dash: Optional[Dash] = None
    width: Optional[float] = Field(None, ge=0.1, le=10)


FONT_FAMILIES = ("Arial", "Helvetica", "Times New Roman", "Georgia", "Courier New", "Verdana")


class AnnotatorSettings(BaseModel):
    font_size: Optional[int] = Field(None, ge=6, le=48)
    font_family: Optional[Literal[FONT_FAMILIES]] = None  # None: the app font
    font_weight: Optional[Literal["normal", "bold"]] = None
    font_style: Optional[Literal["normal", "italic"]] = None
    line_styles: dict[str, LineStyle] = {}


def settings_dict(s: AnnotatorSettings) -> dict:
    d = s.model_dump(exclude_none=True)
    d["line_styles"] = {k: v for k, v in d.get("line_styles", {}).items() if v}
    return d


def apply_line_styles(fig: go.Figure, line_styles: dict) -> go.Figure:
    """Per-line overrides like metbit's annotator: a sample's own style goes on top of its group's."""
    for trace in fig.data:
        if trace.type not in ("scatter", "scattergl"):
            continue
        style = {**line_styles.get(str(trace.legendgroup or ""), {}), **line_styles.get(str(trace.name), {})}
        if style.get("color"):
            trace.line.color = style["color"]
        if style.get("dash"):
            trace.line.dash = style["dash"]
        if style.get("width"):
            trace.line.width = style["width"]
    return fig
