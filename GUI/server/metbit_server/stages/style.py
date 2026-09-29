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
    """Re-stack metbit's paper-positioned notes at fixed pixel offsets.

    metbit spaces its notes (Q2 / R2X blocks, permutation summaries) in paper units sized for its
    fixed figure height. In an autosized plot those gaps scale with the window: notes above the plot
    drift into the title and get clipped, and corner blocks inside the plot collide when it is short.
    Pixel offsets keep them readable at any size, on screen and in exports.
    """
    notes = [a for a in fig.layout.annotations if a.yref == "paper" and a.y is not None]
    step = round(font_size * 1.6)

    above = sorted((a for a in notes if a.y > 1), key=lambda a: a.y)
    if above:
        for i, a in enumerate(above):
            a.update(y=1, yanchor="bottom", yshift=6 + i * step)
        title = fig.layout.title
        title_room = 0
        if title.text:
            # the title goes to the very top so the stacked notes below it can never meet it
            fig.update_layout(title_yref="container", title_y=1, title_yanchor="top", title_pad_t=8)
            title_room = round((title.font.size or font_size * 1.4) * 1.3) + 16
        needed = 6 + len(above) * step + title_room + 8
        fig.update_layout(margin_t=max(fig.layout.margin.t or 0, needed))

    # blocks of notes sharing a corner inside the plot: keep the lowest in place, stack the rest on it
    # (lifted notes now sit at y=1 too; they are already stacked and must not be stacked again)
    lifted = {id(a) for a in above}
    corners: dict = {}
    for a in notes:
        if a.y <= 1 and id(a) not in lifted:
            corners.setdefault((round(a.x or 0, 2), a.xanchor), []).append(a)
    for block in corners.values():
        block.sort(key=lambda a: a.y)
        if len(block) < 2 or block[-1].y - block[0].y > 0.25:
            continue
        base = block[0].y
        for i, a in enumerate(block):
            a.update(y=base, yanchor="bottom", yshift=i * step)
    return fig


def declutter_point_labels(fig: go.Figure, x_frac: float = 0.1, y_frac: float = 0.05) -> go.Figure:
    """Drop data-point labels that would sit on a more significant one (higher y wins).

    metbit labels the top-n features; neighbouring points of one peak are all in the top n, so their
    labels pile up. Keeps one label per crowded spot, sized in fractions of the data range.
    """
    xs, ys = [], []
    for trace in fig.data:
        try:
            xs.extend(float(v) for v in (trace.x if trace.x is not None else ()))
            ys.extend(float(v) for v in (trace.y if trace.y is not None else ()))
        except (TypeError, ValueError):
            continue
    xr = (max(xs) - min(xs)) if xs else 1.0
    yr = (max(ys) - min(ys)) if ys else 1.0
    points = [a for a in fig.layout.annotations if a.showarrow and a.xref in (None, "x") and a.yref in (None, "y")]
    kept: list = []
    for a in sorted(points, key=lambda a: -(a.y or 0)):
        if not any(abs(a.x - k.x) < x_frac * xr and abs(a.y - k.y) < y_frac * yr for k in kept):
            kept.append(a)
    others = [a for a in fig.layout.annotations if a not in points]
    fig.layout.annotations = others + kept
    return fig


def _takes_style(make: Callable) -> bool:
    try:
        return len(inspect.signature(make).parameters) >= 2
    except (TypeError, ValueError):
        return False


def _declared(make: Callable, extra: dict) -> dict:
    """The extras (such as ppm) this figure function names in its signature; None means not given."""
    try:
        names = inspect.signature(make).parameters
    except (TypeError, ValueError):
        return {}
    return {k: v for k, v in extra.items() if v is not None and k in names}


def render(make: Callable, result: Any, style: PlotStyle, **extra: Any) -> go.Figure:
    """Build a figure with the effective style: native metbit arguments when supported, else the fallback."""
    native = _takes_style(make)
    kwargs = _declared(make, extra)
    fig = make(result, style, **kwargs) if native else make(result, **kwargs)
    apply_style(fig, style, recolor=not native)
    return lift_paper_annotations(fig, style.font_size)
