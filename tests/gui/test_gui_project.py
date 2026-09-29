import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from metbit_server.project import ProjectError, load_project, save_project
from metbit_server.session import Session
from metbit_server.stages import STAGES, BrukerSource


def fresh() -> Session:
    return Session([s.id for s in STAGES], {s.id: s.after for s in STAGES})


def test_roundtrip_restores_data_and_reruns_analyses(run_all, tmp_path):
    s = run_all(
        ("normalize", {"method": "pqn"}),
        ("pca", {"n_components": 3}),
        ("opls", {"group_a": "A", "group_b": "B", "permutations": 5}),
        ("stocsy", {"anchor_ppm": 5.0}),
    )
    s.status["stocsy"] = "stale"  # as if its input changed after it ran
    path = tmp_path / "study.metbit"
    save_project(s, str(path))

    t = fresh()
    load_project(t, STAGES, str(path))
    assert (t.status["import"], t.status["normalize"], t.status["pca"]) == ("done", "done", "done")
    assert t.output("normalize").X.equals(s.output("normalize").X)
    assert t.output("normalize").before.equals(s.output("normalize").before)
    assert t.output("import").meta.equals(s.output("import").meta)
    assert list(t.output("import").labels) == list(s.output("import").labels)
    scores = lambda sess: sess.output("pca").model.get_scores().select_dtypes("number").to_numpy()
    assert np.allclose(scores(t), scores(s))
    # the slow permutation test is not repeated on load
    assert t.status["opls"] == "stale" and not t.output("opls").permuted
    assert t.params["opls"]["permutations"] == 5
    # stale analyses keep their params but no result
    assert t.status["stocsy"] == "empty" and t.params["stocsy"]["anchor_ppm"] == 5.0
    assert t.status["volcano"] == "empty" and "volcano" not in t.params


def test_project_is_plain_json_and_parquet(run_all, tmp_path):
    save_project(run_all(), str(tmp_path / "p.metbit"))
    with zipfile.ZipFile(tmp_path / "p.metbit") as zf:
        assert sorted(zf.namelist()) == ["import/X.parquet", "import/labels.parquet", "import/meta.parquet", "params.json"]
        manifest = json.loads(zf.read("params.json"))
    assert (manifest["format"], manifest["version"]) == ("metbit-gui-project", 1)
    assert manifest["stages"]["import"]["status"] == "done"


def test_bruker_import_roundtrip(run_all, tmp_path):
    for n in ["61020601", "61030601"]:
        (tmp_path / "exps" / n).mkdir(parents=True)
        (tmp_path / "exps" / n / "fid").write_bytes(b"")
    pd.DataFrame({"fid": [61020601, 61030601], "Group": ["sham", "nx"]}).to_csv(tmp_path / "map.csv", index=False)
    s = run_all(("import", {
        "source": "bruker", "folder": str(tmp_path / "exps"),
        "metadata_path": str(tmp_path / "map.csv"), "sample_column": "fid",
    }))
    save_project(s, str(tmp_path / "b.metbit"))
    t = fresh()
    load_project(t, STAGES, str(tmp_path / "b.metbit"))
    src = t.output("import")
    assert isinstance(src, BrukerSource)
    assert src.experiments == s.output("import").experiments
    assert list(src.labels) == ["sham", "nx"]


def test_rejects_files_that_are_not_projects(tmp_path):
    bad = tmp_path / "text.metbit"
    bad.write_text("nope")
    other = tmp_path / "other.metbit"
    with zipfile.ZipFile(other, "w") as zf:
        zf.writestr("params.json", json.dumps({"format": "something-else"}))
    newer = tmp_path / "newer.metbit"
    with zipfile.ZipFile(newer, "w") as zf:
        zf.writestr("params.json", json.dumps({"format": "metbit-gui-project", "version": 99, "stages": {}}))
    t = fresh()
    for path, msg in [(bad, "Not a metbit project"), (other, "Not a metbit project"),
                      (tmp_path / "missing.metbit", "Not a metbit project"), (newer, "newer metbit GUI")]:
        with pytest.raises(ProjectError, match=msg):
            load_project(t, STAGES, str(path))
    assert set(t.status.values()) == {"empty"}  # a failed load leaves the session untouched
