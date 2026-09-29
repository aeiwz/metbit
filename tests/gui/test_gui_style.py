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


def test_notes_above_the_plot_are_restacked_in_pixels_with_room_for_them():
    from metbit_server.stages.style import lift_paper_annotations

    fig = go.Figure()
    for y, text in [(1.25, "Number of permutation: 10"), (1.18, "Accuracy score"), (1.11, "p-value"), (0.5, "inside")]:
        fig.add_annotation(text=text, xref="paper", yref="paper", x=0, y=y, showarrow=False)
    lift_paper_annotations(fig, font_size=14)
    notes = {a.text: a for a in fig.layout.annotations}
    stacked = [notes["p-value"], notes["Accuracy score"], notes["Number of permutation: 10"]]
    assert all(a.y == 1 and a.yanchor == "bottom" for a in stacked)
    assert [a.yshift for a in stacked] == sorted(a.yshift for a in stacked)  # original order kept, bottom up
    assert notes["inside"].y == 0.5 and notes["inside"].yshift is None
    assert fig.layout.margin.t >= stacked[-1].yshift + 2 * 14  # the top note fits inside the margin


def test_render_lifts_notes_for_every_figure():
    style = PlotStyle()
    fig = render(lambda r: go.Figure(layout={"annotations": [{"text": "Q2", "xref": "paper", "yref": "paper", "x": 0, "y": 1.2}]}), None, style)
    assert fig.layout.annotations[0].y == 1 and fig.layout.margin.t > 0


def test_lifted_notes_push_the_title_to_the_top_of_the_figure():
    from metbit_server.stages.style import lift_paper_annotations

    fig = go.Figure(layout={"title": {"text": "OPLS-DA Scores Plot"}})
    fig.add_annotation(text="Q2", xref="paper", yref="paper", x=0, y=1.1, showarrow=False)
    lift_paper_annotations(fig, font_size=14)
    assert (fig.layout.title.yref, fig.layout.title.y, fig.layout.title.yanchor) == ("container", 1, "top")


def test_corner_notes_inside_the_plot_are_stacked_in_pixels():
    from metbit_server.stages.style import lift_paper_annotations

    fig = go.Figure()
    fig.add_annotation(text="Q2X", xref="paper", yref="paper", x=1, y=0.0, showarrow=False, xanchor="right")
    fig.add_annotation(text="R2X", xref="paper", yref="paper", x=1, y=0.05, showarrow=False, xanchor="right")
    fig.add_annotation(text="lonely", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    lift_paper_annotations(fig, font_size=14)
    notes = {a.text: a for a in fig.layout.annotations}
    assert notes["Q2X"].y == notes["R2X"].y == 0.0
    assert notes["R2X"].yshift > notes["Q2X"].yshift
    assert notes["lonely"].y == 0.5 and notes["lonely"].yshift is None


def test_crowded_point_labels_keep_only_the_most_significant():
    from metbit_server.stages.style import declutter_point_labels

    fig = go.Figure(go.Scatter(x=[-2, 2], y=[0, 20], mode="markers"))
    for x, y, text in [(1.0, 18.5, "8.0"), (1.05, 19.0, "8.0167"), (1.02, 18.6, "7.9833"), (-1.5, 10.0, "3.3")]:
        fig.add_annotation(x=x, y=y, text=text, showarrow=True)
    fig.add_annotation(text="p=0.05", xref="paper", yref="paper", x=1, y=0.1, showarrow=False)
    declutter_point_labels(fig)
    assert sorted(a.text for a in fig.layout.annotations) == ["3.3", "8.0167", "p=0.05"]


def test_lifted_notes_keep_their_order_when_nothing_else_shares_the_corner():
    from metbit_server.stages.style import lift_paper_annotations

    fig = go.Figure()  # exactly metbit's permutation plot: three notes above the plot, nothing inside
    for y, text in [(1.25, "Number of permutation"), (1.18, "Accuracy score"), (1.11, "p-value")]:
        fig.add_annotation(text=text, xref="paper", yref="paper", x=0, y=y, showarrow=False)
    lift_paper_annotations(fig, font_size=14)
    shift = {a.text: a.yshift for a in fig.layout.annotations}
    assert shift["p-value"] < shift["Accuracy score"] < shift["Number of permutation"]
