import numpy as np
import pandas as pd

from metbit.utility import Normalise


def test_normalise_pqn_imputes_and_scales():
    data = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 4.0, 6.0]})

    normaliser = Normalise(data, compute_missing=True)
    pqn = normaliser.pqn_normalise(plot=False)

    assert not pqn.isna().any().any()
    pd.testing.assert_index_equal(pqn.columns, data.columns)


def test_normalise_rounding_and_zscore_behaviour():
    data = pd.DataFrame({"a": [1.111, 2.222, 3.333], "b": [4.444, 5.555, 6.666]})
    normaliser = Normalise(data, compute_missing=False)

    rounded = normaliser.decimal_place_normalisation(decimals=1)
    pd.testing.assert_frame_equal(rounded, data.round(1))

    zscore = normaliser.z_score_normalisation()
    np.testing.assert_allclose(zscore.mean(axis=0), 0, atol=1e-12)
    np.testing.assert_allclose(zscore.std(axis=0, ddof=0), 1, atol=1e-12)


def test_normalise_linear_normalisation_bounds():
    data = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    normaliser = Normalise(data, compute_missing=False)

    linear = normaliser.linear_normalisation()

    assert ((linear >= 0) & (linear <= 1)).all().all()


def test_normalise_pqn_respects_reference_indices():
    data = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
    normaliser = Normalise(data, compute_missing=False)

    # Use first two rows as reference; row 2 should scale relative to that median
    pqn = normaliser.pqn_normalise(ref_index=[0, 1], plot=False)

    # Third row should remain proportional after scaling
    ratio = pqn.iloc[2] / pqn.iloc[0]
    assert np.allclose(ratio, ratio.iloc[0])
