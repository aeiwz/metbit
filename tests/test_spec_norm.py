import numpy as np
import pandas as pd
import pytest

from metbit.spec_norm import Normalization


def test_pqn_normalization_rescales_against_reference_median():
    data = pd.DataFrame([[1, 2], [2, 4], [4, 8]], columns=["a", "b"])

    result = Normalization.pqn_normalization(data)

    expected = pd.DataFrame([[2, 4], [2, 4], [2, 4]], columns=["a", "b"])
    np.testing.assert_allclose(result.values, expected.values)
    pd.testing.assert_index_equal(result.columns, expected.columns)


def test_snv_normalization_centers_and_scales_columns():
    data = pd.DataFrame([[1, 2, 3], [2, 3, 4], [3, 4, 5]], columns=list("abc"))

    result = Normalization.snv_normalization(data)

    np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(result.std(axis=0, ddof=0), 1, atol=1e-12)


def test_snv_msc_pqn_normalization_removes_row_and_column_bias():
    data = pd.DataFrame([[1, 1, 2], [2, 2, 4], [3, 3, 6]], columns=list("abc"))

    result = Normalization.snv_msc_pqn_normalization(data)

    assert not result.isna().any().any()
    np.testing.assert_allclose(result.mean(axis=0), 0, atol=1e-12)


def test_normalization_rejects_unconvertible_input():
    class Uncoercible:
        def __iter__(self):
            raise TypeError("no iteration")

    with pytest.raises(TypeError):
        Normalization.pqn_normalization(Uncoercible())


def test_pqn_normalization_handles_zero_columns_gracefully():
    data = pd.DataFrame([[0, 0], [1, 1]], columns=["a", "b"])

    result = Normalization.pqn_normalization(data)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == data.shape
