import plotly.graph_objects as go
import pytest

from metbit_server.stages import STAGES, parse_windows
from metbit_server.stages.base import working

BY_ID = {s.id: s for s in STAGES}


def test_data_stage_registration():
    assert [s.id for s in STAGES][:4] == ["import", "process", "normalize", "align"]
    assert BY_ID["normalize"].after == ["import", "process"]
    assert BY_ID["align"].after == ["import", "process", "normalize"]
    assert BY_ID["align"].ppm_views == {"spectra": "x", "before": "x"}


@pytest.mark.parametrize("method", ["pqn", "msc", "snv", "snv_pqn"])
def test_normalize_uses_import_and_becomes_working_matrix(run_all, method):
    s = run_all(("normalize", {"method": method}))
    out = s.output("normalize")
    assert out.X.shape == (12, 40)
    assert out.before is s.output("import").X
    assert not out.X.equals(out.before)
    assert list(out.X.columns) == list(out.before.columns)
    assert working(s) is out
    for view in ("spectra", "before"):
        assert isinstance(BY_ID["normalize"].figures[view](out), go.Figure)


def test_align_reads_normalized_matrix_and_defaults_to_whole_spectrum(run_all):
    s = run_all(("normalize", {"method": "snv"}), ("align", {}))
    out = s.output("align")
    assert out.before is s.output("normalize").X
    assert list(out.X.columns) == list(out.before.columns)
    assert working(s) is out


def test_align_with_explicit_windows(run_all):
    s = run_all(("align", {"windows": "9-7, 3-1", "reference": "mean", "max_shift_ppm": 0.1}))
    assert s.output("align").X.shape == (12, 40)


def test_parse_windows():
    assert parse_windows("") == []
    assert parse_windows("3.2-2.8, 1.2 - 1.5") == [(3.2, 2.8), (1.5, 1.2)]
    assert parse_windows("0.5--0.2") == [(0.5, -0.2)]
    with pytest.raises(ValueError, match="Cannot read window 'abc'"):
        parse_windows("abc")


def test_bad_windows_are_a_validation_error():
    from pydantic import ValidationError

    from metbit_server.stages import AlignParams

    with pytest.raises(ValidationError, match="Cannot read window"):
        AlignParams(windows="3.2 to nowhere")


def test_stale_normalize_is_skipped_by_working(run_all, csv_path):
    s = run_all(("normalize", {"method": "pqn"}), ("import", {"path": csv_path}))
    assert s.status["normalize"] == "stale"
    assert working(s) is s.output("import")
