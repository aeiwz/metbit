"""Plot styling: a global PlotStyle, per-stage overrides, and applying them to figures."""
import inspect
from typing import Any, Callable, Iterable, Literal, Optional

import plotly.graph_objects as go
from pydantic import BaseModel, Field

Symbol = Literal["circle", "square", "diamond", "triangle-up", "triangle-down", "cross", "x", "star"]
SYMBOLS: list[str] = list(Symbol.__args__)
# Okabe-Ito (colorblind-safe); black last because it disappears on dark backgrounds
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
HEX = r"^#[0-9a-fA-F]{6}$"


class GroupStyle(BaseModel):
    color: str = Field(pattern=HEX)
    symbol: Symbol = "circle"


class PlotStyle(BaseModel):
    groups: dict[str, GroupStyle] = {}
    marker_size: int = Field(12, ge=4, le=60)
    marker_opacity: float = Field(0.8, ge=0.1, le=1)
    font_size: int = Field(14, ge=8, le=32)
    line_width: float = Field(1.2, ge=0.5, le=5)
    show_ellipses: bool = True


class GroupOverride(BaseModel):
    color: Optional[str] = Field(None, pattern=HEX)
    symbol: Optional[Symbol] = None


class StyleOverride(BaseModel):
    groups: dict[str, GroupOverride] = {}
    marker_size: Optional[int] = Field(None, ge=4, le=60)
    marker_opacity: Optional[float] = Field(None, ge=0.1, le=1)
    font_size: Optional[int] = Field(None, ge=8, le=32)
    line_width: Optional[float] = Field(None, ge=0.5, le=5)
    show_ellipses: Optional[bool] = None


def with_groups(style: PlotStyle, names: Iterable[str]) -> PlotStyle:
    """Give every group a style; new groups get unused palette colors in sorted order."""
    groups = dict(style.groups)
    used = {g.color.lower() for g in groups.values()}
    free = [c for c in PALETTE if c.lower() not in used] or PALETTE
    new = [n for n in sorted(set(names)) if n not in groups]
    for i, name in enumerate(new):
        groups[name] = GroupStyle(color=free[i % len(free)])
    return style.model_copy(update={"groups": groups})


def merge(base: PlotStyle, override: StyleOverride) -> PlotStyle:
    data = base.model_dump()
    for key, value in override.model_dump(exclude={"groups"}).items():
        if value is not None:
            data[key] = value
    for name, g in override.groups.items():
        if name in data["groups"]:
            data["groups"][name].update({k: v for k, v in g.model_dump().items() if v is not None})
    return PlotStyle.model_validate(data)


def override_dict(o: StyleOverride) -> dict:
    d = o.model_dump(exclude_none=True)
    if not d.get("groups"):
        d.pop("groups", None)
    return d


def colors(style: PlotStyle) -> dict[str, str]:
    return {name: g.color for name, g in style.groups.items()}


def symbols(style: PlotStyle) -> dict[str, str]:
    return {name: g.symbol for name, g in style.groups.items()}


def apply_style(fig: go.Figure, style: PlotStyle, recolor: bool = True) -> go.Figure:
    fig.update_layout(font_size=style.font_size)
    if not recolor:
        return fig
    for trace in fig.data:
        g = style.groups.get(str(trace.name)) or style.groups.get(str(getattr(trace, "legendgroup", "") or ""))
        if g is None:
            continue
        if trace.type in ("scatter", "scattergl"):
            mode = trace.mode or "markers"
            if "markers" in mode:
                trace.marker.color = g.color
                trace.marker.symbol = g.symbol
            if "lines" in mode:
                trace.line.color = g.color
        elif trace.type in ("box", "violin", "bar", "histogram"):
            trace.marker.color = g.color
            if trace.type in ("box", "violin"):
                trace.line.color = g.color
    return fig


def lift_paper_annotations(fig: go.Figure, font_size: int) -> go.Figure:
    """Re-stack notes metbit places above the plot (paper y > 1) at fixed pixel offsets.

    metbit sizes those offsets for its fixed figure height; in an autosized plot they scale with the
    window, drift into the title and get clipped. Pixel offsets plus a matching top margin keep them
    readable at any size, on screen and in exports.
    """
    above = sorted(
        (a for a in fig.layout.annotations if a.yref == "paper" and a.y is not None and a.y > 1),
        key=lambda a: a.y,
    )
    if not above:
        return fig
    step = round(font_size * 1.6)
    for i, a in enumerate(above):
        a.update(y=1, yanchor="bottom", yshift=6 + i * step)
    needed = 6 + len(above) * step + 2 * font_size  # top note plus room for the title line
    fig.update_layout(margin_t=max(fig.layout.margin.t or 0, needed))
    return fig


def _takes_style(make: Callable) -> bool:
    try:
        return len(inspect.signature(make).parameters) >= 2
    except (TypeError, ValueError):
        return False


def render(make: Callable, result: Any, style: PlotStyle) -> go.Figure:
    """Build a figure with the effective style: native metbit arguments when supported, else the fallback."""
    native = _takes_style(make)
    fig = make(result, style) if native else make(result)
    apply_style(fig, style, recolor=not native)
    return lift_paper_annotations(fig, style.font_size)
