import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metbit.metbit import _resolve_scale_power, pca


def test_resolve_scale_power_mapping_and_errors():
    assert _resolve_scale_power("pareto") == 0.5
    assert _resolve_scale_power("mean") == 0
    assert _resolve_scale_power("uv") == 1
    with pytest.raises(ValueError):
        _resolve_scale_power("unknown-scale")


def test_pca_fit_produces_scores_and_variance():
    X = pd.DataFrame(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.5],
            [3.0, 4.5, 6.0],
            [4.0, 6.0, 8.0],
        ],
        columns=["a", "b", "c"],
    )
    labels = pd.Series(["A", "A", "B", "B"])

    model = pca(X=X, label=labels, n_components=2, scaling_method="mean", random_state=0, test_size=0.5)
    model.fit()

    scores = model.get_scores()
    loadings = model.get_loadings()
    q2 = model.get_q2_test()

    assert list(scores.columns) == ["PC1", "PC2", "Group"]
    assert set(loadings.columns) == {"PC1", "PC2", "Features"}
    assert scores.shape[0] == X.shape[0]
    assert loadings.shape[0] == X.shape[1]
    assert -1 <= q2 <= 1
