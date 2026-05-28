import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch


def _make_nmrglue_stub():
    """Return a minimal nmrglue stub so module-level functions can be imported."""
    ng = types.ModuleType("nmrglue")
    ng.bruker = MagicMock()
    ng.process = MagicMock()
    ng.proc_base = MagicMock()
    return ng


@pytest.fixture(autouse=True)
def _stub_nmrglue():
    """Inject a fake nmrglue before every test in this module."""
    stub = _make_nmrglue_stub()
    sys.modules.setdefault("nmrglue", stub)
    yield
    # leave the stub in place so subsequent imports in the same session see it


class TestGeneratePPMScale:
    def test_length_matches_data(self):
        from metbit.nmr.preprocess import generate_ppm_scale

        dic = {
            "acqus": {"SW": 20.0, "SFO1": 600.0},
            "procs": {"OFFSET": 10.0},
        }
        data = [0.0] * 1024
        ppm = generate_ppm_scale(dic, data)

        assert len(ppm) == 1024

    def test_ppm_is_descending_from_offset(self):
        from metbit.nmr.preprocess import generate_ppm_scale

        dic = {
            "acqus": {"SW": 12.0, "SFO1": 600.0},
            "procs": {"OFFSET": 6.0},
        }
        data = [0.0] * 512
        ppm = generate_ppm_scale(dic, data)

        assert ppm[0] == pytest.approx(6.0)
        assert ppm[-1] == pytest.approx(6.0 - 12.0)
        assert ppm[0] > ppm[-1]

    def test_sweep_width_span(self):
        from metbit.nmr.preprocess import generate_ppm_scale

        sw = 15.0
        offset = 8.0
        dic = {"acqus": {"SW": sw, "SFO1": 500.0}, "procs": {"OFFSET": offset}}
        data = [0.0] * 256
        ppm = generate_ppm_scale(dic, data)

        assert ppm[0] - ppm[-1] == pytest.approx(sw)


class TestNMRPreprocessingErrors:
    def test_raises_file_not_found_for_missing_path(self):
        from metbit.nmr.preprocess import nmr_preprocessing

        with pytest.raises(FileNotFoundError):
            nmr_preprocessing("/nonexistent/path/that/does/not/exist")

    def test_raises_value_error_when_no_fid_found(self, tmp_path):
        from metbit.nmr.preprocess import nmr_preprocessing

        with pytest.raises(ValueError, match="No 'fid' files"):
            nmr_preprocessing(str(tmp_path))

    def test_raises_value_error_for_nested_empty_dir(self, tmp_path):
        from metbit.nmr.preprocess import nmr_preprocessing

        (tmp_path / "sample1").mkdir()
        (tmp_path / "sample2").mkdir()

        with pytest.raises(ValueError, match="No 'fid' files"):
            nmr_preprocessing(str(tmp_path))
