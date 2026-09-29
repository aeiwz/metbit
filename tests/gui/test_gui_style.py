import plotly.graph_objects as go
import pytest
from pydantic import ValidationError

from metbit_server.stages.style import (
    PALETTE,
    PlotStyle,
    StyleOverride,
    apply_style,
    merge,
    override_dict,
    render,
    with_groups,
)


def test_defaults_and_validation():
    s = PlotStyle()
    assert (s.marker_size, s.marker_opacity, s.font_size, s.line_width, s.show_ellipses) == (12, 0.8, 14, 1.2, True)
    with pytest.raises(ValidationError):
        PlotStyle(groups={"A": {"color": "red"}})
    with pytest.raises(ValidationError):
        PlotStyle(marker_size=100)
    with pytest.raises(ValidationError):
        PlotStyle(groups={"A": {"color": "#112233", "symbol": "hexagon-open-dot"}})


def test_with_groups_assigns_palette_in_sorted_order_and_keeps_choices():
    s = with_groups(PlotStyle(groups={"B": {"color": "#0072B2"}}), ["C", "B", "A"])
    assert s.groups["B"].color == "#0072B2"
    assert s.groups["A"].color == PALETTE[1]  # PALETTE[0] is already B's
    assert s.groups["C"].color == PALETTE[2]
    assert s.groups["A"].symbol == "circle"


def test_merge_prefers_override_and_ignores_unknown_groups():
    base = with_groups(PlotStyle(), ["A", "B"])
    o = StyleOverride(marker_size=30, groups={"A": {"symbol": "square"}, "Z": {"color": "#000000"}})
    eff = merge(base, o)
    assert (eff.marker_size, eff.font_size) == (30, 14)
    assert (eff.groups["A"].symbol, eff.groups["A"].color) == ("square", base.groups["A"].color)
    assert "Z" not in eff.groups
    assert override_dict(StyleOverride()) == {}
    assert override_dict(o) == {"marker_size": 30, "groups": {"A": {"symbol": "square"}, "Z": {"color": "#000000"}}}


def test_apply_style_colors_traces_named_after_groups():
    style = PlotStyle(groups={"A": {"color": "#112233", "symbol": "diamond"}}, font_size=18)
    fig = go.Figure([
        go.Scatter(name="A", mode="markers", x=[1], y=[1]),
        go.Scatter(name="A", mode="lines", x=[1], y=[1]),
        go.Box(name="A", y=[1, 2]),
        go.Scatter(name="other", mode="markers", x=[1], y=[1], marker_color="#999999"),
    ])
    apply_style(fig, style)
    assert (fig.data[0].marker.color, fig.data[0].marker.symbol) == ("#112233", "diamond")
    assert fig.data[1].line.color == "#112233"
    assert fig.data[2].marker.color == "#112233"
    assert fig.data[3].marker.color == "#999999"
    assert fig.layout.font.size == 18


def test_render_passes_style_to_native_figures_and_skips_their_recolor():
    style = PlotStyle(groups={"A": {"color": "#112233"}}, font_size=20)
    seen = []
    plain = render(lambda r: go.Figure([go.Scatter(name="A", mode="lines")]), "result", style)
    assert plain.data[0].line.color == "#112233"
    native = render(
        lambda r, st=None: seen.append(st) or go.Figure([go.Scatter(name="A", mode="lines", line_color="rgba(1,2,3,0.5)")]),
        "result",
        style,
    )
    assert seen == [style]
    assert native.data[0].line.color == "rgba(1,2,3,0.5)"
    assert native.layout.font.size == 20
