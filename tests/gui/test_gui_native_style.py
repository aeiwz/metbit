import pandas as pd

from metbit_server.stages import STAGES
from metbit_server.stages.style import PlotStyle, with_groups

BY_ID = {s.id: s for s in STAGES}
STYLE = with_groups(
    PlotStyle(
        groups={"A": {"color": "#112233", "symbol": "square"}},
        marker_size=20,
        marker_opacity=0.5,
        line_width=2,
        show_ellipses=False,
    ),
    ["A", "B"],
)


def traces(fig, group):
    """Traces for a group, marker traces first (ellipses are line traces with the same name)."""
    found = [t for t in fig.data if str(t.name).split(",")[0].strip() == group]
    return sorted(found, key=lambda t: "markers" not in (getattr(t, "mode", None) or "markers"))


def paint(t) -> set:
    return {getattr(t.marker, "color", None), getattr(t.line, "color", None), getattr(t, "fillcolor", None)}


def test_scores_use_native_color_symbol_size_and_ellipse_toggle(run_all):
    s = run_all(("pca", {}), ("opls", {"group_a": "A", "group_b": "B"}))
    for sid in ("pca", "opls"):
        styled = BY_ID[sid].figures["scores"](s.output(sid), STYLE)
        a = traces(styled, "A")[0]
        assert (a.marker.color, a.marker.symbol, a.marker.size, a.marker.opacity) == ("#112233", "square", 20, 0.5)
        with_ellipses = BY_ID[sid].figures["scores"](s.output(sid), STYLE.model_copy(update={"show_ellipses": True}))
        count = lambda f: len(f.data) + len(f.layout.shapes or ())
        assert count(styled) < count(with_ellipses)


def test_vip_uses_marker_size(run_all):
    r = run_all(("opls", {"group_a": "A", "group_b": "B"})).output("opls")
    fig = BY_ID["opls"].figures["vip"](r, STYLE)
    assert any(t.marker.size == 20 for t in fig.data)


def test_spectra_keep_transparency_and_take_group_color_and_width(run_all):
    fig = BY_ID["import"].figures["spectra"](run_all().output("import"), STYLE)
    a = traces(fig, "A")[0]
    assert a.line.color.startswith("rgba(17,34,51")
    assert a.line.width == 2


def test_lda_ica_and_boxplot_use_group_colors(run_all, csv_path, tmp_path):
    s = run_all(("lda", {}), ("ica", {"n_components": 3}), ("grouptests", {"bin_width": 0}))
    lda = BY_ID["lda"].figures["scores"](s.output("lda"), STYLE)
    ica = BY_ID["ica"].figures["components"](s.output("ica"), STYLE)
    box = BY_ID["grouptests"].figures["boxplot"](s.output("grouptests"), STYLE)
    for fig in (lda, ica, box):
        assert any("#112233" in paint(t) for t in traces(fig, "A")), fig


def test_figures_still_work_without_a_style(run_all):
    s = run_all(("pca", {}))
    assert BY_ID["pca"].figures["scores"](s.output("pca")) is not None
    assert BY_ID["import"].figures["spectra"](s.output("import")) is not None
