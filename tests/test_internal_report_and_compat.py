import os

import pytest

import metbit._compat as compat
from metbit._internal.report import gen_page, gen_page_legacy, oplsda_path


def _build_report_input(root, vip_folder):
    element = root / "element"
    folders = [
        "hist_plot",
        "Lingress",
        "loading_plot",
        "s_plot",
        "score_plot",
        vip_folder,
    ]
    for folder in folders:
        target = element / folder
        target.mkdir(parents=True, exist_ok=True)
        (target / "Permutation_scores_Group A.html").write_text(
            f"<html>{folder}</html>", encoding="utf-8"
        )

    (root / "main").mkdir(parents=True, exist_ok=True)
    (root / "placeholder.txt").write_text("x", encoding="utf-8")
    return str(root)


def test_oplsda_path_validates_inputs(tmp_path):
    with pytest.raises(ValueError):
        oplsda_path(123)

    with pytest.raises(ValueError):
        oplsda_path(str(tmp_path / "missing"))

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        oplsda_path(str(empty_dir))


def test_oplsda_path_make_path_creates_expected_dirs(tmp_path):
    data_dir = tmp_path / "input"
    data_dir.mkdir()
    (data_dir / "seed.txt").write_text("seed", encoding="utf-8")

    op = oplsda_path(str(data_dir).replace("/", "\\"))
    path_map = op.make_path()

    assert op.get_path() == path_map
    for key in [
        "main",
        "element",
        "hist_plot",
        "Lingress",
        "loading_plot",
        "s_plot",
        "score_plot",
        "VIP_score",
    ]:
        assert os.path.isdir(path_map[key])


def test_gen_page_validates_missing_required_folder(tmp_path):
    data_dir = tmp_path / "report"
    _build_report_input(data_dir, "VIP_score")

    vip_dir = data_dir / "element" / "VIP_score"
    for item in vip_dir.iterdir():
        item.unlink()
    vip_dir.rmdir()

    with pytest.raises(ValueError, match="VIP_score"):
        gen_page(str(data_dir))


def test_gen_page_get_files_generates_html(tmp_path):
    data_dir = tmp_path / "report"
    root = _build_report_input(data_dir, "VIP_score")
    start_cwd = os.getcwd()
    try:
        gp = gen_page(root + "/")
        gp.get_files()
    finally:
        os.chdir(start_cwd)

    output = data_dir / "main" / "oplsda_Group_A.html"
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "../element/score_plot/Permutation_scores_Group A.html" in content
    assert "../element/VIP_score/Permutation_scores_Group A.html" in content


def test_gen_page_legacy_get_files_generates_html(tmp_path):
    data_dir = tmp_path / "legacy_report"
    root = _build_report_input(data_dir, "VIP_scores")
    start_cwd = os.getcwd()
    try:
        gp = gen_page_legacy(root)
        gp.get_files()
    finally:
        os.chdir(start_cwd)

    output = data_dir / "main" / "oplsda_Group_A.html"
    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert "../element/Lingress/Permutation_scores_Group A.html" in content


def test_compat_reexports_expected_symbols():
    from metbit.analysis.opls_da import opls_da
    from metbit.models.pls import PLS
    from metbit.models.opls import OPLS
    from metbit.models.cross_validation import CrossValidation
    from metbit.preprocessing.normalize import Normalization
    from metbit.stats.normalise import Normalise

    assert compat.opls_da is opls_da
    assert compat.PLS is PLS
    assert compat.OPLS is OPLS
    assert compat.CrossValidation is CrossValidation
    assert compat.Normalization is Normalization
    assert compat.Normalise is Normalise
