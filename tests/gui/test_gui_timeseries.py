import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from conftest import AUTH, TOKEN
from metbit_server.app import create_app
from metbit_server.session import Session
from metbit_server.stages import STAGES
from metbit_server.stages.timeseries import NONE, TimeParams, design, natural_key

BY_ID = {s.id: s for s in STAGES}
TIMES = ["1-wk", "2-wk", "10-wk"]  # natural order differs from plain string order
SPIKE = 5  # a peak (columns 5-9) that rises over time in group B only
PEAK = range(SPIKE, SPIKE + 5)


@pytest.fixture
def ts_path(tmp_path):
    rng = np.random.default_rng(1)
    ppm = np.linspace(10, 0, 40)
    rows, meta = [], []
    for g in ("A", "B"):
        for s in range(4):
            offset = rng.normal(scale=0.2)  # subject baseline
            for t, time in enumerate(TIMES):
                x = np.abs(rng.normal(size=40)) * 0.2 + 1 + offset
                if g == "B":
                    x[SPIKE : SPIKE + 5] += 3 * t
                rows.append(x)
                meta.append((f"{g}{s}-{time}", g, time, f"{g}{s}"))
    df = pd.DataFrame(rows, columns=[f"{v:.4f}" for v in ppm])
    for i, name in enumerate(["Sample", "Group", "Time", "Subject"]):
        df.insert(i, name, [m[i] for m in meta])
    path = tmp_path / "ts.csv"
    df.to_csv(path, index=False)
    return str(path)


def session_with(ts_path, *runs):
    s = Session([st.id for st in STAGES], {st.id: st.after for st in STAGES})
    for sid, params in [("import", {"path": ts_path}), *runs]:
        stage = BY_ID[sid]
        p = stage.params.model_validate(params)
        s.complete(sid, p.model_dump(), stage.run(p, s, lambda _pct: None))
    return s


BASE = {"time_column": "Time", "subject_column": "Subject"}
SPIKE_PPM = 10 - SPIKE * 10 / 39


def test_time_order_is_natural_explicit_or_rejected(ts_path):
    s = session_with(ts_path)
    des, _ = design(s, TimeParams(time_column="Time"))
    assert des.order == TIMES and des.subject is None
    des, _ = design(s, TimeParams(time_column="Time", time_order="10-wk, 2-wk, 1-wk", subject_column="Subject"))
    assert des.order == ["10-wk", "2-wk", "1-wk"] and des.subject.iloc[0] == "A0"
    with pytest.raises(ValueError, match="missing: 10-wk"):
        design(s, TimeParams(time_column="Time", time_order="1-wk, 2-wk"))
    with pytest.raises(ValueError, match="Unknown time column"):
        design(s, TimeParams(time_column="Nope"))
    assert sorted(["t10", "t2", "t1"], key=natural_key) == ["t1", "t2", "t10"]


def test_column_pickers_default_to_time_and_subject_like_columns(ts_path):
    c = TestClient(create_app(TOKEN))
    job = c.post("/stages/import/run", json={"path": ts_path}, headers=AUTH).json()["job_id"]
    c.get(f"/jobs/{job}/events", headers=AUTH)
    props = {s["id"]: s["schema"]["properties"] for s in c.get("/stages", headers=AUTH).json()}["trajectory"]
    assert props["time_column"]["default"] == "Time"
    assert props["subject_column"]["default"] == "Subject" and NONE in props["subject_column"]["enum"]


def test_trajectory_and_multilevel(ts_path):
    s = session_with(ts_path, ("trajectory", {**BASE, "multilevel": True}))
    r = s.output("trajectory")
    stage = BY_ID["trajectory"]
    fig = stage.figures["trajectory"](r)
    assert "multilevel" in fig.layout.title.text
    subjects = stage.figures["subjects"](r)
    assert len([t for t in subjects.data if t.showlegend is False]) == 8  # one faint path per subject
    table = stage.table(r)
    assert {"Group", "Time", "Subject", "PC1", "PC2"} <= set(table.columns)
    with pytest.raises(ValueError, match="subject column"):
        session_with(ts_path, ("trajectory", {"time_column": "Time", "multilevel": True}))
    no_subject = session_with(ts_path, ("trajectory", {"time_column": "Time"})).output("trajectory")
    assert "subject column" in stage.figures["subjects"](no_subject).layout.annotations[0].text


def test_time_profile_follows_the_picked_ppm(ts_path):
    s = session_with(ts_path, ("timeprofile", BASE))
    r, stage = s.output("timeprofile"), BY_ID["timeprofile"]
    fig = stage.figures["profile"](r, None, SPIKE_PPM)
    b = next(t for t in fig.data if t.name == "B")
    assert list(b.x) == TIMES and b.y[2] > b.y[0] + 4  # B rises by about 6
    assert len([t for t in fig.data if t.showlegend is False]) == 8
    assert stage.companions == {"profile": "spectra"} and stage.ppm_views == {"spectra": "x"}
    assert list(stage.table(r).columns) == ["Group", "Time", "n", "center", "error"]


def test_asca_finds_the_time_by_group_interaction(ts_path):
    s = session_with(ts_path, ("asca", {"time_column": "Time", "permutations": 99, "bin_width": 0}))
    r, stage = s.output("asca"), BY_ID["asca"]
    table = stage.table(r).set_index("effect")
    assert list(table.index) == ["Time", "Group", "Time × Group", "Residual"]
    assert table.loc["Time × Group", "p_value"] <= 0.05
    assert abs(table["ssq_percent"].sum() - 100) < 1e-6  # balanced design: effects partition the variation
    assert list(stage.figures) == ["effects", "scores", "loadings"]
    for view in stage.figures:
        assert stage.figures[view](r).data
    paired = session_with(ts_path, ("asca", {**BASE, "permutations": 99, "bin_width": 0})).output("asca")
    assert paired.table.set_index("effect").loc["Time × Group", "p_value"] <= 0.05  # within-subject shuffles
    top = np.abs(r.effects["Time × Group"]["loadings"][0]).argmax()
    assert top in PEAK


def test_mixed_model_and_friedman(ts_path):
    s = session_with(ts_path, ("mixedmodel", {**BASE, "bin_width": 0}))
    r, stage = s.output("mixedmodel"), BY_ID["mixedmodel"]
    table = stage.table(r)
    assert {"p_time", "p_group", "p_interaction", "p_interaction_adj"} <= set(table.columns)
    peak_ppm = [10 - j * 10 / 39 for j in PEAK]
    assert np.isclose(table["ppm"].iloc[0], peak_ppm).any()  # the peak ranks first
    assert table["p_interaction_adj"].iloc[0] < 0.001
    assert stage.figures["pvalues"](r).data and stage.figures["profile"](r, None, SPIKE_PPM).data
    binned = session_with(ts_path, ("mixedmodel", {**BASE, "bin_width": 0.5})).output("mixedmodel")
    assert [sh for sh in stage.figures["pvalues"](binned).layout.shapes if sh.type == "rect"]
    from metbit_server.stages.longitudinal import _short_p

    assert (_short_p(0.0), _short_p(0.01234), _short_p(float("nan"))) == ("<1e-16", "0.012", "n/a")
    f = session_with(ts_path, ("mixedmodel", {**BASE, "test": "friedman", "bin_width": 0})).output("mixedmodel")
    assert list(stage.table(f).columns[:3]) == ["ppm", "p_time", "p_time_adj"]
    with pytest.raises(ValueError, match="subject column"):
        session_with(ts_path, ("mixedmodel", {"time_column": "Time"}))


def test_mixed_model_reports_when_nothing_fits(ts_path, monkeypatch):
    from statsmodels.regression.mixed_linear_model import MixedLM

    def broken(*args, **kwargs):
        raise TypeError("optimizer exploded")

    monkeypatch.setattr(MixedLM, "fit", broken)
    with pytest.raises(ValueError, match="could not be fitted: optimizer exploded"):
        session_with(ts_path, ("mixedmodel", {**BASE, "bin_width": 0}))
