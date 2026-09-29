import plotly.graph_objects as go
import pytest

from metbit_server.session import Session
from metbit_server.stages import STAGES, OPLSParams, resolve_schema

BY_ID = {s.id: s for s in STAGES}


def test_group_options_are_injected_after_import(run_all):
    fresh = Session([st.id for st in STAGES])
    assert "enum" not in resolve_schema(BY_ID["opls"], fresh)["properties"]["group_a"]
    props = resolve_schema(BY_ID["opls"], run_all())["properties"]
    assert props["group_a"]["enum"] == ["A", "B"]
    assert (props["group_a"]["default"], props["group_b"]["default"]) == ("A", "B")


def test_opls_views_table_and_default_permutation_placeholder(run_all):
    r = run_all(("opls", {"group_a": "A", "group_b": "B"})).output("opls")
    for view in ("scores", "s-plot", "vip", "permutation"):
        assert isinstance(BY_ID["opls"].figures[view](r), go.Figure)
    assert "not run" in BY_ID["opls"].figures["permutation"](r).layout.annotations[0].text
    vip = BY_ID["opls"].table(r)
    assert list(vip["VIP"]) == sorted(vip["VIP"], reverse=True)
    assert BY_ID["opls"].ppm_views == {"s-plot": "customdata", "vip": "x"}


def test_opls_permutation_test(run_all):
    r = run_all(("opls", {"group_a": "A", "group_b": "B", "permutations": 5})).output("opls")
    assert r.permuted
    assert isinstance(BY_ID["opls"].figures["permutation"](r), go.Figure)


@pytest.mark.parametrize("a,b,msg", [("A", "A", "two different"), ("A", "Z", "Unknown group")])
def test_two_group_validation(run_all, a, b, msg):
    with pytest.raises(ValueError, match=msg):
        run_all(("opls", {"group_a": a, "group_b": b}))


def test_opls_params_bounds():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OPLSParams(kfold=1)
    with pytest.raises(ValidationError):
        OPLSParams(permutations=-1)


def test_stocsy_runs_and_checks_anchor_range(run_all):
    fig = run_all(("stocsy", {"anchor_ppm": 5.0})).output("stocsy")
    assert isinstance(BY_ID["stocsy"].figures["correlation"](fig), go.Figure)
    assert BY_ID["stocsy"].table is None
    assert BY_ID["stocsy"].params.model_json_schema()["properties"]["anchor_ppm"]["pick_ppm"] is True
    with pytest.raises(ValueError, match="outside the spectrum"):
        run_all(("stocsy", {"anchor_ppm": 50.0}))
